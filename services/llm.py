"""Preparação local de roteiros de estudo com a API do Ollama."""

import json
import re
import shlex
import time
from collections.abc import Callable
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


class LocalOllama:
    def __init__(self, base_url: str):
        match = re.fullmatch(r"http://(127\.0\.0\.1|localhost|\[::1\])(?::([0-9]{1,5}))?/?",
                             base_url, re.IGNORECASE | re.ASCII)
        if not match or not 1 <= int(match[2] or 80) <= 65535:
            raise ValueError("OLLAMA_BASE_URL deve ser HTTP em 127.0.0.1, localhost ou [::1].")
        self.base_url = base_url.rstrip("/")
        self._http = build_opener(ProxyHandler({}), _NoRedirect())

    def request(self, endpoint: str, payload: dict | None = None, *, timeout: int = 10) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(self.base_url + endpoint, data=body, headers={"Content-Type": "application/json"})
        try:
            with self._http.open(req, timeout=timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            exc.close()
            if exc.code == 404 and payload and "model" in payload:
                raise RuntimeError(f"Modelo indisponível. Baixe-o com ollama pull {shlex.quote(payload['model'])}.") from exc
            raise RuntimeError(f"Ollama respondeu HTTP {exc.code}. Confira o serviço com ollama ps.") from exc
        except OSError as exc:
            if isinstance(getattr(exc, "reason", exc), TimeoutError):
                raise RuntimeError("Ollama excedeu o tempo de espera. Confira ollama ps e tente uma seção menor.") from exc
            raise RuntimeError(f"Ollama inacessível em {self.base_url}. Inicie o servidor com ollama serve.") from exc
        except ValueError as exc:
            raise RuntimeError("Resposta inválida do Ollama. Verifique a API local.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Resposta inesperada do Ollama. Verifique a API local.")
        return result

    def require_local_model(self, model: str) -> None:
        details = self.request("/api/show", {"model": model})
        if details.get("remote_host") or details.get("remote_model") or not details.get("model_info"):
            raise ValueError("Selecione um modelo com pesos locais no Ollama para preparar o roteiro.")

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
                       keep_alive: str | int) -> dict:
    """Usado tanto no benchmark quanto na geração de roteiros."""
    return {
        "model": model, "system": MODES[mode] + " " + _INSTRUCTIONS, "prompt": source,
        "stream": False, "keep_alive": keep_alive,
        "options": {"num_ctx": num_ctx, "num_thread": threads, "num_gpu": 0,
                    "num_predict": min(1200, num_ctx // 3), "temperature": 0.2, "seed": 42},
    }


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
    if config.LLM_NUM_CTX < 1024 or config.CPU_THREADS < 1:
        raise ValueError("Use LLM_NUM_CTX de pelo menos 1024 e CPU_THREADS positivo.")
    sections = _sections(remove_citations(text), config.LLM_MAX_SECTION_CHARS)
    if not sections:
        raise ValueError("O material contém apenas referências bibliográficas.")
    total = len(sections)
    client = LocalOllama(config.OLLAMA_BASE_URL)
    from services import voice

    scripts = []
    truncated = False
    with voice.reserve_memory_for_llm():
        client.require_local_model(model)
        original_error = None
        try:
            for index, section in enumerate(sections, 1):
                payload = generation_payload(section, mode, model, config.LLM_NUM_CTX,
                                             config.CPU_THREADS, 0 if total == 1 else "30s")
                result = client.request("/api/generate", payload, timeout=300)
                script = result.get("response")
                if not result.get("done") or not isinstance(script, str) or not script.strip():
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
