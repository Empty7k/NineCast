"""Benchmark do Ollama local em CPU com dois textos sintéticos em PT-BR."""

import argparse
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

_BASE_URL = "http://127.0.0.1:11434"
_HTTP = build_opener(ProxyHandler({}))
_SCENARIOS = {
    "trecho_curto": (
        "Em 18/11/2026, uma turma reservou R$ 1.200,00 para um laboratório de redes. "
        "O desconto foi de 15% sobre o valor inicial. A atividade demonstrou como o DNS "
        "traduz nomes de domínio em endereços IP."
    ),
    "trecho_apostila": (
        "Uma transação agrupa operações relacionadas em uma única unidade lógica. "
        "Em um sistema de matrículas, a inscrição e a atualização do número de vagas "
        "devem ter o mesmo desfecho. Pelo princípio da atomicidade, uma falha na segunda "
        "operação impede que a primeira permaneça gravada (SILVA, 2020, p. 42).\n\n"
        "O exemplo usa uma tabela chamada Matricula e uma coluna chamada turma_id. "
        "O comando SELECT COUNT(*) FROM Matricula WHERE turma_id = 42 conta as matrículas "
        "da turma 42. A contagem serve para conferência, mas executá-la separadamente "
        "da transação não impede dois usuários de tentar ocupar a última vaga.\n\n"
        "No cenário da apostila, havia 30 vagas e 29 matrículas confirmadas em 05/03/2026. "
        "O teste TX-042 verifica se apenas uma das duas inscrições concorrentes é confirmada. "
        "Quando uma transação falha, ROLLBACK desfaz suas alterações ainda não confirmadas. "
        "O teste observa o estado final da tabela, que deve conter 30 matrículas (LIMA, 2022, p. 18)."
    ),
}
_MODES = {
    "didatico": "Reescreva como roteiro didático, com explicações e transições claras para narração.",
    "resumo": "Condense os conceitos essenciais em um roteiro curto de revisão para narração.",
}


def _request(endpoint: str, payload: dict | None = None, *, timeout: int = 5) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(_BASE_URL + endpoint, data=body, headers={"Content-Type": "application/json"})
    with _HTTP.open(request, timeout=timeout) as response:
        return json.load(response)


def _available_mib() -> float:
    with open("/proc/meminfo", encoding="ascii") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1024, 1)
    raise RuntimeError("MemAvailable não está disponível em /proc/meminfo.")


def _model_name(name: str) -> str:
    return name if ":" in name.rsplit("/", 1)[-1] else name + ":latest"


def _wait_for_unload(model: str) -> None:
    deadline = time.monotonic() + 10
    while True:
        running = _request("/api/ps")["models"]
        if not any(
            _model_name(name) == _model_name(model)
            for entry in running
            for name in (entry.get("name"), entry.get("model"))
            if name
        ):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"{model} continua em /api/ps. Feche outras sessões e tente novamente.")
        time.sleep(0.2)


def _xtts_cycle(load_xtts: bool) -> dict:
    from services import voice

    memory_loaded = None
    if load_xtts:
        if _request("/api/ps")["models"]:
            raise RuntimeError("Descarregue os modelos listados em ollama ps antes de carregar o XTTS.")
        voice._get_model()
    try:
        if load_xtts:
            memory_loaded = _available_mib()
    finally:
        unloaded = voice.unload_model()
    return {
        "modelo_carregado_pelo_benchmark": load_xtts,
        "instancia_xtts_liberada_neste_processo": unloaded,
        "ram_disponivel_xtts_carregado_mib": memory_loaded,
        "ram_disponivel_apos_unload_xtts_mib": _available_mib(),
    }


def _run_scenario(name: str, source: str, args: argparse.Namespace) -> dict:
    cycle = _xtts_cycle(args.load_xtts) if args.with_xtts_cycle else None
    payload = {
        "model": args.model,
        "system": (
            "Responda em português brasileiro. " + _MODES[args.mode] +
            " Use somente fatos do material, preserve números, datas e identificadores técnicos. "
            "Mantenha comandos de código exatos, sem traduzi-los, e omita citações bibliográficas "
            "do texto falado. Entregue apenas o roteiro."
        ),
        "prompt": source,
        "stream": False,
        "keep_alive": 0 if args.with_xtts_cycle else "30s",
        "options": {
            "num_ctx": args.num_ctx,
            "num_thread": args.threads,
            "num_gpu": 0,
            "num_predict": min(512, args.num_ctx // 4),
            "temperature": 0.2,
            "seed": 42,
        },
    }
    before = _available_mib()
    result = _request("/api/generate", payload, timeout=300)
    model = result.get("model") or args.model
    try:
        after_generation = _available_mib()
    finally:
        if not args.with_xtts_cycle:
            _request("/api/generate", {"model": model, "stream": False, "keep_alive": 0})
    _wait_for_unload(model)
    after_unload = _available_mib()

    script = result.get("response")
    if not result.get("done") or not isinstance(script, str) or not script.strip():
        raise RuntimeError(f"Ollama devolveu uma geração vazia ou incompleta no cenário {name}.")
    eval_duration = result.get("eval_duration")
    eval_count = result.get("eval_count")
    return {
        "cenario": name,
        "modelo": model,
        "modo": args.mode,
        "num_ctx": args.num_ctx,
        "threads": args.threads,
        "ciclo_xtts": cycle,
        "ram_disponivel_mib": {
            "antes_da_geracao": before,
            "logo_apos_resposta": after_generation,
            "apos_confirmar_liberacao": after_unload,
        },
        "keep_alive_da_geracao": payload["keep_alive"],
        "modelo_liberado_api_ps": True,
        "load_duration_ns": result.get("load_duration"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": eval_count,
        "eval_duration_ns": eval_duration,
        "tokens_por_segundo": round(eval_count * 1e9 / eval_duration, 2)
        if eval_count is not None and eval_duration else None,
        "done_reason": result.get("done_reason"),
        "material": source,
        "roteiro": script.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=config.CPU_THREADS)
    parser.add_argument("--mode", choices=tuple(_MODES), default="didatico")
    parser.add_argument("--with-xttsCycle", dest="with_xtts_cycle", action="store_true",
                        help="Chama unload_model neste processo e gera com keep_alive=0.")
    parser.add_argument("--load-xtts", action="store_true",
                        help="Carrega o XTTS real antes do ciclo. Requer COQUI_TOS_AGREED=1.")
    args = parser.parse_args()
    if args.num_ctx < 1024 or args.threads < 1 or not args.model.strip():
        parser.error("Informe um modelo, --num-ctx >= 1024 e --threads >= 1.")
    if args.load_xtts and not args.with_xtts_cycle:
        parser.error("--load-xtts requer --with-xttsCycle.")
    if args.load_xtts and os.getenv("COQUI_TOS_AGREED") != "1":
        parser.error("Para carregar o XTTS, aceite sua licença e defina COQUI_TOS_AGREED=1.")

    try:
        _request("/api/ps")
        if args.load_xtts:
            _request("/api/show", {"model": args.model})
        for name, source in _SCENARIOS.items():
            print(f"Executando cenário: {name}", file=sys.stderr, flush=True)
            print(json.dumps(_run_scenario(name, source, args), ensure_ascii=False, indent=2), flush=True)
    except HTTPError as exc:
        with exc:
            detail = exc.read(4096).decode("utf-8", errors="replace").strip()
        if exc.code == 404 and exc.url.endswith(("/api/generate", "/api/show")):
            print(f"Modelo indisponível. Baixe com: ollama pull {shlex.quote(args.model)}", file=sys.stderr)
        else:
            print(f"Ollama respondeu HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Benchmark interrompido: {exc}", file=sys.stderr)
        if shutil.which("ollama") is None:
            print("Instale o Ollama no CachyOS com: sudo pacman -S ollama", file=sys.stderr)
        print(f"Confira a API em {_BASE_URL} e inicie com: OLLAMA_NO_CLOUD=1 ollama serve",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
