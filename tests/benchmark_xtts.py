"""Script de benchmark e validação do modelo XTTS-v2 em CPU.

Mede tempo de carregamento, consumo de RAM, tempo de síntese e Real-Time Factor (RTF).
"""

import argparse
import os
from pathlib import Path
import resource
import sys
import time
import wave

import torch

# Adiciona a raiz do projeto ao path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config


def get_memory_mb() -> float:
    """Retorna o pico de uso de memória (RSS) do processo atual em MB."""
    # No Linux, ru_maxrss é retornado em kilobytes
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage_kb / 1024.0


def get_audio_duration_seconds(file_path: Path) -> float:
    """Calcula a duração de um arquivo WAV em segundos."""
    with wave.open(str(file_path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def find_default_speaker_wav() -> Path | None:
    """Busca o primeiro arquivo .wav na pasta de vozes."""
    config.ensure_directories()
    wav_files = sorted(config.VOICES_DIR.glob("*.wav"))
    return wav_files[0] if wav_files else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark e validação do XTTS-v2 em CPU para o NineCast."
    )
    parser.add_argument(
        "--speaker-wav",
        type=str,
        default=None,
        help="Caminho para o áudio WAV de referência da voz autorizada (6 a 15 segundos).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Olá! Este é um teste de síntese de voz local do projeto NineCast rodando diretamente na CPU.",
        help="Texto em português para validação da síntese.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=f"Quantidade de threads PyTorch para inferência (padrão: {config.CPU_THREADS}).",
    )
    parser.add_argument(
        "--agree-license",
        action="store_true",
        help="Confirma concordância com os termos da licença Coqui Public Model License (CPML).",
    )
    args = parser.parse_args()

    config.ensure_directories()

    # Identifica o arquivo de voz de referência
    speaker_wav = Path(args.speaker_wav) if args.speaker_wav else find_default_speaker_wav()

    if not speaker_wav or not speaker_wav.is_file():
        print("=" * 60)
        print("❌ NENHUM ÁUDIO DE REFERÊNCIA ENCONTRADO!")
        print("=" * 60)
        print(f"O XTTS-v2 necessita de um arquivo .wav com a voz de referência autorizada.")
        print(f"Coloque um arquivo .wav curto (6 a 15 segundos, limpo e sem ruídos) em:")
        print(f"  -> {config.VOICES_DIR.resolve()}")
        print("Ou informe explicitamente via parâmetro:")
        print("  -> python tests/benchmark_xtts.py --speaker-wav /caminho/para/voz.wav")
        print("=" * 60)
        return 1

    # Licença CPML
    if args.agree_license or os.environ.get("COQUI_TOS_AGREED") == "1":
        os.environ["COQUI_TOS_AGREED"] = "1"

    # Configuração de threads do PyTorch
    initial_threads = torch.get_num_threads()
    # Em CPUs como Ryzen 5 5650U (6c/12t), limitar a 6 threads físicas costuma reduzir contenção
    threads_to_use = args.threads if args.threads is not None else config.CPU_THREADS
    torch.set_num_threads(threads_to_use)

    output_path = config.PODCASTS_DIR / "benchmark_test.wav"

    print("=" * 60)
    print("🎙️ INICIANDO BENCHMARK DO XTTS-v2 (NINECAST)")
    print("=" * 60)
    print(f"Hardware detectado/configurado:")
    print(f"  - Threads lógicas disponíveis: {os.cpu_count()}")
    print(f"  - Threads PyTorch configuradas: {threads_to_use} (padrão inicial: {initial_threads})")
    print("  - Dispositivo de inferência:   cpu")
    print(f"  - Modelo alvo:                 {config.TTS_MODEL}")
    print(f"  - Idioma:                      {config.DEFAULT_LANGUAGE}")
    print(f"  - Voz de referência:           {speaker_wav.name}")
    print(f"  - Texto de teste:              '{args.text}'")
    print("-" * 60)

    mem_before_load = get_memory_mb()
    print(f"[1/3] Carregando modelo XTTS-v2 na memória...")
    print(f"      (RAM inicial do processo: {mem_before_load:.1f} MB)")

    load_start = time.perf_counter()
    from TTS.api import TTS

    tts = TTS(model_name=config.TTS_MODEL, progress_bar=True, gpu=False)
    load_time = time.perf_counter() - load_start

    mem_after_load = get_memory_mb()
    print(f"      ✓ Modelo carregado em {load_time:.2f} s")
    print(f"      ✓ RAM após carregamento: {mem_after_load:.1f} MB (aumento de {mem_after_load - mem_before_load:.1f} MB)")
    print("-" * 60)

    print(f"[2/3] Sintetizando áudio de teste...")
    synth_start = time.perf_counter()

    tts.tts_to_file(
        text=args.text,
        speaker_wav=str(speaker_wav),
        language=config.DEFAULT_LANGUAGE,
        file_path=str(output_path),
    )
    synth_time = time.perf_counter() - synth_start

    mem_peak = get_memory_mb()
    audio_duration = get_audio_duration_seconds(output_path)
    rtf = synth_time / audio_duration if audio_duration > 0 else 0.0

    print(f"      ✓ Síntese concluída em {synth_time:.2f} s")
    print(f"      ✓ Áudio gerado em: {output_path}")
    print(f"      ✓ Duração do áudio: {audio_duration:.2f} s")
    print("-" * 60)

    print("[3/3] RESULTADOS DO BENCHMARK:")
    print(f"  - Tempo de carga do modelo:    {load_time:.2f} s")
    print(f"  - Tempo de síntese:           {synth_time:.2f} s")
    print(f"  - Duração do áudio gerado:    {audio_duration:.2f} s")
    print(f"  - Real-Time Factor (RTF):     {rtf:.2f}x")
    print(f"  - Consumo de RAM de pico:     {mem_peak:.1f} MB (~{mem_peak / 1024.0:.2f} GB)")
    print("=" * 60)

    if rtf <= 3.5:
        print("✅ Viabilidade técnica confirmada: RTF dentro da faixa esperada para CPU.")
    else:
        print("⚠️ Alerta: RTF elevado (> 3.5x). Necessário avaliar quantização ou blocos menores.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
