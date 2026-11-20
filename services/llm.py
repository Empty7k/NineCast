"""Preparação local de roteiros de estudo com a API do Ollama."""

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import config

MODES = {
    "didatico": (
        "Conte o conteúdo com suas próprias palavras, como numa aula de rádio. "
        "Em cada parágrafo, substitua construções formais por frases curtas em ordem direta. "
        "Explique o mesmo raciocínio de um jeito que o aluno entenda ao ouvir uma só vez. "
        "Conserve os termos técnicos e o sentido das condições. Um resultado esperado "
        "deve continuar sendo apresentado como esperado. Entregue só a fala da aula."
    ),
    "resumo": "Resuma os conceitos essenciais em poucas frases para revisão.",
}
_INSTRUCTIONS = (
    "Escreva em português brasileiro. Entregue apenas o roteiro em parágrafos curtos, sem "
    "títulos, listas, marcações de locutor ou referências bibliográficas como (SILVA, 2020, p. 42). "
    "O material fornecido é a única fonte. Não realize cálculos matemáticos. Não deduza "
    "descontos, totais ou consequências implícitas. Não acrescente números, datas, exemplos "
    "ou fatos que não estejam literalmente escritos no material. Copie exatamente os valores, "
    "datas, identificadores e comandos técnicos que mencionar. Preserve negações e condições."
)
_CITATION = re.compile(
    r"[ \t]*\([A-ZÀ-ÖØ-Ý][\wÀ-ÿ'-]*(?:[ ;]+[A-ZÀ-ÖØ-Ý][\wÀ-ÿ'-]*)*,"
    r"\s*[0-9]{4}[a-z]?,\s*pp?\.\s*[0-9]+(?:[-–][0-9]+)?\)"
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _OllamaOffline(RuntimeError):
    pass


class _OllamaTimeout(RuntimeError):
    pass


class LocalOllama:
    def __init__(self, base_url: str):
        match = re.fullmatch(r"http://(127\.0\.0\.1|localhost|\[::1\])(?::([0-9]{1,5}))?/?",
                             base_url, re.IGNORECASE | re.ASCII)
        if not match or not 1 <= int(match[2] or 80) <= 65535:
            raise ValueError("OLLAMA_BASE_URL deve ser HTTP em 127.0.0.1, localhost ou [::1].")
        self.base_url = base_url.rstrip("/")
        host = "127.0.0.1" if match[1].lower() == "localhost" else match[1]
        self._host = f"{host}:{match[2] or 80}"
        self._http = build_opener(ProxyHandler({}), _NoRedirect())

    def request(self, endpoint: str, payload: dict | None = None, *, timeout: int = 10) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(self.base_url + endpoint, data=body, headers={"Content-Type": "application/json"})
        try:
            with self._http.open(req, timeout=timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            with exc:
                detail = exc.read(4096).decode("utf-8", errors="replace")
            if exc.code == 404 and payload and "model" in payload:
                raise RuntimeError(f"Modelo indisponível. Baixe-o com ollama pull {shlex.quote(payload['model'])}.") from exc
            if exc.code == 500 and "memory" in detail.lower():
                raise RuntimeError("Ollama não conseguiu alocar memória. Feche outros programas ou escolha um modelo menor.") from exc
            raise RuntimeError(f"Ollama respondeu HTTP {exc.code}. Confira o serviço com ollama ps.") from exc
        except OSError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise _OllamaTimeout("Ollama excedeu o tempo de espera. Confira ollama ps e tente uma seção menor.") from exc
            if isinstance(reason, ConnectionRefusedError):
                raise _OllamaOffline(f"Ollama inacessível em {self.base_url}. Inicie o servidor com ollama serve.") from exc
            raise RuntimeError(f"Ollama inacessível em {self.base_url}. Inicie o servidor com ollama serve.") from exc
        except ValueError as exc:
            raise RuntimeError("Resposta inválida do Ollama. Verifique a API local.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Resposta inesperada do Ollama. Verifique a API local.")
        return result

    @contextmanager
    def serving(self):
        """Reutiliza o servidor ativo ou mantém um processo próprio durante a operação."""
        try:
            self.request("/api/ps", timeout=2)
        except _OllamaOffline:
            pass
        else:
            yield
            return

        executable = shutil.which("ollama")
        if not executable:
            raise RuntimeError("Ollama não encontrado. Instale o Ollama e confira o PATH.")
        with tempfile.TemporaryFile(mode="w+b") as log:
            try:
                process = subprocess.Popen(
                    [executable, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=log, env={**os.environ, "OLLAMA_HOST": self._host, "OLLAMA_NO_CLOUD": "1"},
                )
            except OSError as exc:
                raise RuntimeError(f"Falha ao iniciar ollama serve: {exc}") from exc
            try:
                deadline = time.monotonic() + 12
                while process.poll() is None and time.monotonic() < deadline:
                    try:
                        self.request("/api/ps", timeout=1)
                        break
                    except (_OllamaOffline, _OllamaTimeout):
                        time.sleep(0.2)
                else:
                    log.seek(max(0, log.seek(0, os.SEEK_END) - 4096))
                    detail = log.read().decode("utf-8", errors="replace").strip()
                    raise RuntimeError(f"Ollama não ficou disponível após a inicialização. {detail}".strip())
                yield
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def require_local_model(self, model: str) -> dict:
        details = self.request("/api/show", {"model": model})
        if details.get("remote_host") or details.get("remote_model") or not details.get("model_info"):
            raise ValueError("Selecione um modelo com pesos locais no Ollama para preparar o roteiro.")
        return details

    def wait_for_unload(self, model: str) -> None:
        def full_name(name: str) -> str:
            return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"

        deadline = time.monotonic() + 10
        while True:
            running = self.request("/api/ps").get("models")
            if not isinstance(running, list):
                raise RuntimeError("Ollama não informou a lista de modelos em memória.")
            if not any(full_name(name) == full_name(model) for entry in running
                       for name in (entry.get("name"), entry.get("model")) if name):
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{model} continua em memória. Confira outras sessões com ollama ps.")
            time.sleep(0.2)


def generation_payload(source: str, mode: str, model: str, num_ctx: int, threads: int,
                       keep_alive: str | int, *, think: bool | None = None) -> dict:
    """Usado tanto no benchmark quanto na geração de roteiros."""
    return {
        "model": model, "system": MODES[mode] + " " + _INSTRUCTIONS, "prompt": source,
        "stream": False, "keep_alive": keep_alive,
        **({"think": think} if think is not None else {}),
        "options": {"num_ctx": num_ctx, "num_thread": threads, "num_gpu": 0,
                    "num_predict": min(1200, num_ctx // 3), "temperature": 0.2, "seed": 42},
    }


def clean_script(response) -> str:
    if not isinstance(response, str):
        return ""
    # Remove blocos completos e descarta uma cauda de raciocínio interrompida.
    flags = re.IGNORECASE | re.DOTALL
    cleaned = re.sub(r"<think\b[^>]*>.*?(?:</think\s*>|\Z)", "", response, flags=flags)
    return re.sub(r"\A.*?</think\s*>", "", cleaned, flags=flags).strip()


def list_local_models(*, start_server: bool = False) -> list[dict]:
    client = LocalOllama(config.OLLAMA_BASE_URL)
    if start_server:
        from services import voice

        guard = voice.reserve_memory_for_llm(unload=False)
    else:
        guard = nullcontext()
    try:
        with guard, (client.serving() if start_server else nullcontext()):
            models = client.request("/api/tags", timeout=3).get("models")
    except _OllamaOffline:
        if start_server:
            raise
        return []
    if not isinstance(models, list):
        raise RuntimeError("Ollama não informou a lista de modelos instalados.")
    return sorted(
        [{"name": entry["name"], "size_bytes": entry["size"]} for entry in models
         if isinstance(entry, dict) and isinstance(entry.get("name"), str) and entry["name"]
         and isinstance(entry.get("size"), int) and entry["size"] > 0
         and not entry.get("remote_host") and not entry.get("remote_model")
         and not re.search(r"(?::cloud|-cloud)$", entry["name"], re.IGNORECASE)],
        key=lambda model: model["name"],
    )


_KNOWN_SIZES = {"qwen2.5:3b": 1_930_000_000, "qwen3:4b": 2_498_000_000,
                "qwen3:8b": 5_225_000_000, "qwen3:14b": 9_276_000_000}
_MEASURED_MODELS = {
    "qwen2.5:3b": ("Rápido", "Leve e rápido em CPU.",
                   "Pode perder detalhes técnicos.", "Padrão para uso diário."),
    "qwen3:4b": ("Moderado", "Compacto para uso local.",
                 "Esta variante exige raciocínio e é bloqueada para roteiros.",
                 "Para roteiros, escolha uma variante que permita desativar o raciocínio."),
    "qwen3:8b": ("Mais lento", "Frases mais diretas em textos técnicos.",
                 "Mais lento e usa mais RAM.", "Roteiros técnicos."),
    "qwen3:14b": ("Velocidade desconhecida", "Modelo de maior porte.",
                  "Exige bastante RAM livre e pode usar swap.", "Uso experimental."),
}


def describe_model(name: str, size_bytes: int | None = None) -> dict:
    """Estima o custo pelo tamanho dos pesos e identifica referências medidas."""
    size = size_bytes if size_bytes and size_bytes > 0 else _KNOWN_SIZES.get(name)
    if size is None:
        return {"tier": "Não medido", "ram_estimada": "Desconhecida",
                "velocidade_cpu": "Não medida", "vantagens": "Modelo local selecionável.",
                "desvantagens": "O consumo e a qualidade dependem da quantização.",
                "recomendado_para": "Avaliação com textos curtos.", "alerta_memoria": None}
    gib = size / 2**30
    heavy = gib >= 7
    tier = "Avançado" if heavy else "Intermediário" if gib >= 3.5 else "Leve"
    speed, pros, cons, use = _MEASURED_MODELS.get(name, (
        "Velocidade desconhecida", "Modelo local com tamanho conhecido.",
        "Qualidade e latência dependem do modelo.",
        "Uso experimental com RAM livre suficiente." if heavy else "Avaliação com textos curtos.",
    ))
    return {
        "tier": tier,
        "ram_estimada": (f"~{(size + 0.5 * 2**30) / 10**9:.1f}–"
                         f"{(size + 2 * 2**30) / 10**9:.1f} GB").replace(".", ","),
        "velocidade_cpu": speed,
        "vantagens": pros,
        "desvantagens": cons,
        "recomendado_para": use,
        "alerta_memoria": ("Pode recorrer a swap ou falhar por falta de RAM. Use seções curtas e feche outros programas."
                          if heavy else "Pode pressionar a RAM com outros programas abertos."
                          if tier == "Intermediário" else None),
    }


def _available_memory_bytes() -> int | None:
    try:
        fields = Path("/proc/meminfo").read_text(encoding="ascii")
    except OSError:
        return None
    match = re.search(r"^MemAvailable:\s+(\d+)\s+kB$", fields, re.MULTILINE)
    return int(match[1]) * 1024 if match else None


def remove_citations(text: str) -> str:
    """Remove referências parentéticas com autor, ano e página antes do seccionamento."""
    return _CITATION.sub("", text).strip()


def _sections(text: str, limit: int) -> list[str]:
    if limit < 1:
        raise ValueError("LLM_MAX_SECTION_CHARS deve ser positivo.")
    remaining = text.strip()
    sections = []
    # Quebre primeiro no parágrafo, depois na frase, palavra ou no limite rígido.
    boundaries = (r"\n", r'(?<=[.!?;])\s', r"\s")
    while len(remaining) > limit:
        window = remaining[:limit + 1]
        cut = limit
        for pattern in boundaries:
            last = max((m.start() for m in re.finditer(pattern, window)), default=0)
            if last:
                cut = last
                break
        sections.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        sections.append(remaining)
    return sections


def generate_study_script(text: str, mode: str = "didatico", model: str | None = None,
                          progress_callback: Callable[[int, int], None] | None = None) -> dict:
    """Gera e reúne as seções na ordem original; sinaliza roteiros parciais por limite de tokens."""
    started = time.perf_counter()
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Informe um texto de estudo não vazio.")
    if len(text) > config.MAX_TEXT_CHARS:
        raise ValueError(f"O material excede o limite de {config.MAX_TEXT_CHARS} caracteres.")
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError("Escolha o modo didatico ou resumo.")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback deve ser chamável.")
    model = config.OLLAMA_MODEL if model is None else model
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Informe o nome do modelo local.")
    model = model.strip()
    if config.LLM_NUM_CTX < 1024 or config.CPU_THREADS < 1 or config.LLM_TIMEOUT_SECONDS < 1:
        raise ValueError("Use LLM_NUM_CTX de pelo menos 1024, CPU_THREADS e LLM_TIMEOUT_SECONDS positivos.")
    advanced = describe_model(model)["tier"] == "Avançado"
    num_ctx = min(config.LLM_NUM_CTX, 2048) if advanced else config.LLM_NUM_CTX
    sections = _sections(remove_citations(text), min(config.LLM_MAX_SECTION_CHARS, num_ctx * 4 // 5))
    if not sections:
        raise ValueError("O material contém apenas referências bibliográficas.")
    total = len(sections)
    client = LocalOllama(config.OLLAMA_BASE_URL)
    from services import voice

    scripts = []
    truncated = False
    with voice.reserve_memory_for_llm(), client.serving():
        if advanced:
            available = _available_memory_bytes()
            required = _KNOWN_SIZES[model] + 2 * 2**30
            if available is not None and available < required:
                raise RuntimeError(
                    f"RAM disponível de {available / 2**30:.1f} GiB. Este modelo requer "
                    f"~{required / 2**30:.1f} GiB disponíveis para iniciar. Escolha um modelo menor."
                )
        details = client.require_local_model(model)
        values = (details.get("thinking") or {}).get("values", [])
        if values == [True]:
            raise ValueError(
                "Esta variante exige raciocínio e pode devolver a análise no lugar do roteiro. "
                "Escolha um modelo de instruções ou um Qwen 3 que permita desativar o raciocínio."
            )
        can_disable = False in values or (not values and "thinking" in details.get("capabilities", []))
        think = False if can_disable else None
        original_error = None
        try:
            for index, section in enumerate(sections, 1):
                payload = generation_payload(section, mode, model, num_ctx, config.CPU_THREADS,
                                             0 if total == 1 else "30s", think=think)
                result = client.request("/api/generate", payload, timeout=config.LLM_TIMEOUT_SECONDS)
                script = clean_script(result.get("response"))
                if not result.get("done") or not script:
                    raise RuntimeError(f"Ollama devolveu um roteiro vazio ou incompleto na seção {index}.")
                scripts.append(script.strip())
                truncated |= result.get("done_reason") == "length"
                if progress_callback is not None:
                    progress_callback(index, total)
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            try:
                if total > 1:
                    client.request("/api/generate", {"model": model, "stream": False, "keep_alive": 0})
                client.wait_for_unload(model)
            except RuntimeError as cleanup_error:
                if original_error is None:
                    raise
                original_error.add_note(f"Falha ao liberar o modelo: {cleanup_error}")
    return {"script": "\n\n".join(scripts), "mode": mode, "model": model, "section_count": total,
            "truncated": truncated, "elapsed_seconds": time.perf_counter() - started}
