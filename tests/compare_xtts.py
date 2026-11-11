"""Compara três conjuntos de referências com o Perfil B refinado, offline."""

import argparse
import json
import os
import random
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config


TEXT = (
    "Olá, este é um teste de síntese de voz local do projeto NineCast, "
    "rodando diretamente na CPU com parâmetros ajustados."
)
SEED = 42
CONDITIONING = {
    "max_ref_length": 30,
    "gpt_cond_len": -1,
    "gpt_cond_chunk_len": 6,
    "sound_norm_refs": True,
    # O corte em NumPy é feito antes: a implementação Coqui passa Tensor ao librosa.
    "librosa_trim_db": None,
}
REFERENCE_SETS = {
    "conjunto_1_grave_limpo": ["13seg.wav", "2.wav", "3.wav"],
    "conjunto_2_expressivo": ["5.wav", "6.wav"],
    "conjunto_3_todas": ["13seg.wav", "2.wav", "3.wav", "4.wav", "5.wav", "6.wav"],
}
SETTINGS = {
    "temperature": 0.70, "repetition_penalty": 6.0, "length_penalty": 1.0,
    "top_p": 0.85, "top_k": 50, "speed": 1.0, "enable_text_splitting": False,
}


def prepare_reference(path: Path, destination: Path) -> dict:
    """Remove silêncio nas cópias e limita a 30 s, incluindo as micropausas."""
    import librosa

    started = time.perf_counter()
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError(f"Referência vazia, silenciosa ou inválida: {path}")
    original_seconds = audio.size / sample_rate
    intervals = librosa.effects.split(audio, top_db=30)
    remaining = sample_rate * CONDITIONING["max_ref_length"]
    pause = np.zeros(round(sample_rate * 0.15), dtype=np.float32)
    parts = []
    speech_samples = 0
    used_segments = 0
    for start, end in intervals:
        if parts:
            # Só insere pausa se também couber alguma fala depois dela.
            if remaining <= pause.size:
                break
            parts.append(pause)
            remaining -= pause.size
        segment = audio[start : min(end, start + remaining)]
        parts.append(segment)
        speech_samples += segment.size
        used_segments += 1
        remaining -= segment.size
        if remaining == 0:
            break
    if not parts or speech_samples < sample_rate * 0.33:
        raise ValueError(f"Referência sem fala suficiente após corte: {path}")
    prepared = np.concatenate(parts)
    sf.write(destination, prepared, sample_rate, subtype="FLOAT")
    return {
        "original_seconds": original_seconds, "prepared_seconds": prepared.size / sample_rate,
        "speech_seconds": speech_samples / sample_rate, "segments_used": used_segments,
        "seconds": time.perf_counter() - started,
    }


def reset_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agree-license", action="store_true", help="Aceita os termos não comerciais da CPML.")
    args = parser.parse_args()
    if args.agree_license:
        os.environ["COQUI_TOS_AGREED"] = "1"
    if os.environ.get("COQUI_TOS_AGREED") != "1":
        parser.error("Informe --agree-license após aceitar os termos da CPML.")
    references = [config.VOICES_DIR / name for name in REFERENCE_SETS["conjunto_3_todas"]]
    for path in references:
        if not path.is_file():
            raise FileNotFoundError(f"Referência não encontrada: {path}")
        audio, _ = sf.read(path, dtype="float32")
        if audio.size == 0 or not np.isfinite(audio).all() or np.max(np.abs(audio)) == 0:
            raise ValueError(f"Referência vazia, silenciosa ou inválida: {path}")
        del audio

    # Reutiliza o cache local da etapa de benchmark, sem conexão ao Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(config.DATA_DIR / "models" / "huggingface")
    os.environ["XDG_DATA_HOME"] = str(config.DATA_DIR / "models")

    config.PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="comparacao_xtts_", dir=config.PODCASTS_DIR))
    torch.set_num_threads(6)

    from TTS.api import TTS

    print("Carregando XTTS-v2 em CPU...", flush=True)
    started = time.perf_counter()
    tts = TTS(model_name=config.TTS_MODEL, progress_bar=False).to(config.TTS_DEVICE)
    load_seconds = time.perf_counter() - started
    print(f"Modelo carregado em {load_seconds:.2f} s", flush=True)
    model = tts.synthesizer.tts_model
    report = {
        "text": TEXT, "seed": SEED, "threads": 6, "device": config.TTS_DEVICE,
        "model": config.TTS_MODEL, "conditioning": CONDITIONING,
        "load_seconds": load_seconds, "parameters": SETTINGS,
        "preprocessing": {"split_db": 30, "pause_seconds": 0.15, "references": {}},
        "sets": {},
    }
    # Cada referência é preparada uma vez e reutilizada pelos conjuntos.
    with tempfile.TemporaryDirectory(prefix="referencias_", dir=output_dir) as temporary_dir:
        preprocessing_started = time.perf_counter()
        prepared = {}
        for index, path in enumerate(references):
            prepared_path = Path(temporary_dir) / f"referencia_{index}.wav"
            metrics = prepare_reference(path, prepared_path)
            prepared[path.name] = str(prepared_path)
            report["preprocessing"]["references"][path.name] = metrics
            print(f"{path.name}: {metrics['original_seconds']:.2f} s → "
                  f"{metrics['prepared_seconds']:.2f} s preparados", flush=True)
        report["preprocessing"]["seconds"] = time.perf_counter() - preprocessing_started

        for label, names in REFERENCE_SETS.items():
            reset_seed()
            print(f"Calculando latentes de {label}...", flush=True)
            started = time.perf_counter()
            with torch.inference_mode():
                gpt_latents, speaker_embedding = model.get_conditioning_latents(
                    audio_path=[prepared[name] for name in names], **CONDITIONING
                )
            conditioning_seconds = time.perf_counter() - started
            print(f"Latentes: {conditioning_seconds:.2f} s; gerando áudio...", flush=True)
            reset_seed()
            started = time.perf_counter()
            result = model.inference(
                text=TEXT, language=config.TTS_LANGUAGE,
                gpt_cond_latent=gpt_latents, speaker_embedding=speaker_embedding,
                **SETTINGS,
            )
            elapsed = time.perf_counter() - started
            audio = np.asarray(result["wav"])
            if audio.size == 0 or not np.isfinite(audio).all():
                raise ValueError(f"O conjunto {label} produziu áudio vazio ou inválido.")
            duration = audio.size / 24000
            output_path = output_dir / f"{label}.wav"
            sf.write(output_path, audio, 24000, subtype="PCM_16")
            report["sets"][label] = {
                "references": names,
                "preprocessing_seconds": sum(report["preprocessing"]["references"][n]["seconds"] for n in names),
                "conditioning_seconds": conditioning_seconds, "inference_seconds": elapsed,
                "duration_seconds": duration, "rtf": elapsed / duration,
                "process_peak_rss_mib_so_far": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "file": output_path.name,
            }
            print(f"{label}: {elapsed:.2f} s, duração {duration:.2f} s, RTF {elapsed / duration:.2f}x", flush=True)
            print(f"Arquivo: {output_path}", flush=True)
            del result, audio, gpt_latents, speaker_embedding

    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    report["peak_rss_mib"] = peak_mib
    with (output_dir / "metricas.json").open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    print(f"Pico RSS do processo: {peak_mib:.1f} MiB", flush=True)
    print(f"Métricas: {output_dir / 'metricas.json'}", flush=True)


if __name__ == "__main__":
    main()
