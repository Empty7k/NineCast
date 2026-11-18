"""Benchmark do Ollama local em CPU com dois textos sintéticos em PT-BR."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from services import llm

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


def _available_mib() -> float:
    with open("/proc/meminfo", encoding="ascii") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) / 1024, 1)
    raise RuntimeError("MemAvailable não está disponível em /proc/meminfo.")


def _xtts_cycle(load_xtts: bool, client: llm.LocalOllama) -> dict:
    from services import voice

    memory_loaded = None
    if load_xtts:
        if client.request("/api/ps")["models"]:
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


def _run_scenario(name: str, source: str, args: argparse.Namespace, client: llm.LocalOllama) -> dict:
    cycle = _xtts_cycle(args.load_xtts, client) if args.with_xtts_cycle else None
    payload = llm.generation_payload(llm.remove_citations(source), args.mode, args.model, args.num_ctx, args.threads,
                                     0 if args.with_xtts_cycle else "30s")
    before = _available_mib()
    result = client.request("/api/generate", payload, timeout=300)
    model = result.get("model") or args.model
    try:
        after_generation = _available_mib()
    finally:
        if not args.with_xtts_cycle:
            client.request("/api/generate", {"model": model, "stream": False, "keep_alive": 0})
    client.wait_for_unload(model)
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
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    parser.add_argument("--num-ctx", type=int, default=config.LLM_NUM_CTX)
    parser.add_argument("--threads", type=int, default=config.CPU_THREADS)
    parser.add_argument("--mode", choices=tuple(llm.MODES), default="didatico")
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
        client = llm.LocalOllama(config.OLLAMA_BASE_URL)
        client.require_local_model(args.model)
        for name, source in _SCENARIOS.items():
            print(f"Executando cenário: {name}", file=sys.stderr, flush=True)
            print(json.dumps(_run_scenario(name, source, args, client), ensure_ascii=False, indent=2), flush=True)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Benchmark interrompido: {exc}", file=sys.stderr)
        if shutil.which("ollama") is None:
            print("Instale o Ollama no CachyOS com: sudo pacman -S ollama", file=sys.stderr)
        print(f"Confira a API em {config.OLLAMA_BASE_URL} e inicie com: OLLAMA_NO_CLOUD=1 ollama serve",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
