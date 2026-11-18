"""Compara conjuntos de referências ou mede similaridade de locutor em CPU, offline."""

import argparse
import json
import os
import random
import resource
import subprocess
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


def reset_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Baselines congelados para permitir repetir a comparação após mudar os padrões.
EQ = ("highpass=f=80,equalizer=f=7000:t=q:w=1.5:g=-3,equalizer=f=3000:t=q:w=2:g=-1.5,"
      "equalizer=f=200:t=q:w=1:g=1,")
DYNAMICS = ("acompressor=threshold=0.125:ratio=2:attack=15:release=180:makeup=1:knee=2.828,"
            "loudnorm=I=-16:TP=-1.5:LRA=9")
FILTERS = {"eq": EQ + DYNAMICS, "transparente": "highpass=f=80," + DYNAMICS}
PROFILES = {"base": SETTINGS, "ancorado": dict(SETTINGS, temperature=0.63, top_p=0.80,
                                               top_k=35, repetition_penalty=7.5)}
TRIALS = (
    (42, "Quando lemos em voz alta, percebemos o ritmo de cada frase."),
    (73, "Antes de alterar os dados, confira o resultado da consulta."),
    (101, "Por que o céu muda de cor? A resposta depende de como a luz atravessa a atmosfera."),
)
VALIDATION_TRIALS = (
    (37, "Um índice ajuda a encontrar o registro certo sem percorrer a tabela inteira."),
    (89, "A primeira ideia parece simples. Vamos ouvir o exemplo antes de tirar uma conclusão."),
)


def weighted_latents(model, paths):
    """Replica o carregamento do Coqui, com pesos em amostras e poda só no GPT."""
    from services import voice
    from TTS.tts.models.xtts import load_audio

    rate = 22050
    signals = []
    for path in paths:
        signal = load_audio(str(path), rate)[:, :30 * rate]
        signals.append(signal / signal.abs().max() * 0.75)
    return voice._pool_reference_latents(model, signals, rate)


def embedding(model, path):
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
    return model.get_speaker_embedding(mono, rate).flatten()


def similarity(candidate, target):
    return torch.nn.functional.cosine_similarity(candidate, target, dim=0).item()


def measure_trial(model, gpt, speaker, settings, targets, output_dir, temporary, label, seed, text):
    from services import audio

    stem = f"{label}_{seed}"
    reset_seed(seed)
    started = time.perf_counter()
    wav = model.inference(text=text, language="pt", gpt_cond_latent=gpt,
                          speaker_embedding=speaker, **settings)["wav"]
    elapsed = time.perf_counter() - started
    if not np.isfinite(wav).all() or not np.any(wav):
        raise ValueError(f"Saída vazia ou inválida: {stem}")
    raw = output_dir / f"{stem}.wav"
    sf.write(raw, wav, 24000, subtype="PCM_16")
    raw_emb = embedding(model, raw)
    mounted = temporary / "montagem.wav"
    audio._concatenate_wavs([raw], [0], mounted)
    row = {"variant": label, "seed": seed, "wav": similarity(raw_emb, targets[0]),
           "wav_holdout": similarity(raw_emb, targets[1]), "duration": len(wav) / 24000,
           "inference_seconds": elapsed, "clip_fraction": float(np.mean(np.abs(wav) >= 1)),
           "speaker_norm": speaker.norm().item(), "mp3": {}}
    row["rtf"] = elapsed / row["duration"]
    for chain, filters in FILTERS.items():
        mp3 = output_dir / f"{stem}_{chain}.mp3"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(mounted),
                        "-af", filters, "-ac", "1", "-ar", "48000", "-c:a", "libmp3lame",
                        "-b:a", "192k", str(mp3)], check=True)
        encoded_emb = embedding(model, mp3)
        row["mp3"][chain] = {"target": similarity(encoded_emb, targets[0]),
                             "holdout": similarity(encoded_emb, targets[1])}
    return row


def summarize(report):
    """Médias pareadas por texto; RTF agregado por duração bruta sintetizada."""
    rows = report["rows"]
    baseline_duration = {r["seed"]: r["duration"] for r in rows if r["variant"] == "base_base"}
    summary = {}
    for variant in dict.fromkeys(r["variant"] for r in rows):
        group = [r for r in rows if r["variant"] == variant]
        duration_ratios = [r["duration"] / baseline_duration[r["seed"]] for r in group]
        summary[variant] = {
            "wav": float(np.mean([r["wav"] for r in group])),
            "wav_holdout": float(np.mean([r["wav_holdout"] for r in group])),
            "rtf": sum(r["inference_seconds"] for r in group) / sum(r["duration"] for r in group),
            "duration_ratio_range": [min(duration_ratios), max(duration_ratios)],
            "max_raw_clip_fraction": max(r["clip_fraction"] for r in group),
            "mp3": {chain: {target: float(np.mean([r["mp3"][chain][target] for r in group]))
                            for target in ("target", "holdout")} for chain in FILTERS},
        }
    return summary


def confirm_finalists(model, report, latents, targets, output_dir, temporary):
    summary = summarize(report)
    finalists = ["base_base"]
    for target in ("target", "holdout"):
        finalists.append(max(summary, key=lambda v: max(s[target] for s in summary[v]["mp3"].values())))
    rows = []
    for label in dict.fromkeys(finalists):
        gpt, speaker, settings = latents[label]
        for seed, text in VALIDATION_TRIALS:
            rows.append(measure_trial(model, gpt, speaker, settings, targets, output_dir,
                                      temporary, label, seed, text))
    report["validation"] = {"trials": VALIDATION_TRIALS, "rows": rows, "summary": summarize({"rows": rows})}


@torch.inference_mode()
def measure_similarity(output_dir, model):
    from torchaudio.functional import highpass_biquad
    from services import voice

    refs = REFERENCE_SETS["conjunto_1_grave_limpo"]
    held_out = ["4.wav", "5.wav", "6.wav"]
    report = {"references": refs, "held_out": held_out, "trials": TRIALS, "profiles": PROFILES,
              "filters": FILTERS, "threads": config.CPU_THREADS, "reference_seconds": {}, "rows": []}
    with tempfile.TemporaryDirectory(dir=output_dir) as directory:
        temporary = Path(directory)
        for name in refs + held_out:
            voice._prepare_reference(config.VOICES_DIR / name, temporary / name)
        paths = [temporary / name for name in refs]
        report["reference_seconds"] = {path.name: sf.info(path).duration for path in paths}
        targets = [torch.stack([embedding(model, temporary / name) for name in group]).mean(0)
                   for group in (refs, held_out)]
        latents = {}
        for hp in (False, True):
            prepared = []
            for path in paths:
                if hp:
                    samples, rate = sf.read(path, dtype="float32")
                    filtered = highpass_biquad(torch.from_numpy(samples), rate, 75)
                    path = temporary / ("hp_" + path.name)
                    sf.write(path, filtered.numpy(), rate, subtype="FLOAT")
                prepared.append(str(path))
            baseline = model.get_conditioning_latents(audio_path=prepared, **CONDITIONING)
            weighted = weighted_latents(model, prepared)
            variants = {"base": baseline, "ponderada": weighted}
            if not hp:
                # Isola os efeitos do vetor do locutor e da poda de estilo no perfil original.
                variants["so_locutor"] = (baseline[0], weighted[1])
                variants["so_poda"] = (weighted[0], baseline[1])
            for weighting, (gpt, speaker) in variants.items():
                profiles = {"base": SETTINGS} if weighting.startswith("so_") else PROFILES
                for profile, settings in profiles.items():
                    label = f"{weighting}{'_hp75' if hp else ''}_{profile}"
                    latents[label] = (gpt, speaker, settings)
                    for seed, text in TRIALS:
                        row = measure_trial(model, gpt, speaker, settings, targets, output_dir,
                                            temporary, label, seed, text)
                        report["rows"].append(row)
                        print(f"{label}_{seed} cos={row['wav']:.3f} RTF={row['rtf']:.2f}", flush=True)
        confirm_finalists(model, report, latents, targets, output_dir, temporary)
    report["summary"] = summarize(report)
    (output_dir / "metricas.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agree-license", action="store_true", help="Aceita os termos não comerciais da CPML.")
    parser.add_argument("--similarity", action="store_true", help="Compara condicionamento, inferência e masterização.")
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
    os.environ["HF_HOME"] = str(config.MODELS_DIR / "huggingface")
    os.environ["XDG_DATA_HOME"] = str(config.MODELS_DIR)

    config.PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="comparacao_xtts_", dir=config.PODCASTS_DIR))
    torch.set_num_threads(config.CPU_THREADS)

    from TTS.api import TTS

    print("Carregando XTTS-v2 em CPU...", flush=True)
    started = time.perf_counter()
    tts = TTS(model_name=config.TTS_MODEL, progress_bar=False).to("cpu")
    load_seconds = time.perf_counter() - started
    print(f"Modelo carregado em {load_seconds:.2f} s", flush=True)
    model = tts.synthesizer.tts_model
    if args.similarity:
        print(f"Comparação: {output_dir}", flush=True)
        measure_similarity(output_dir, model)
        return
    report = {
        "text": TEXT, "seed": SEED, "threads": config.CPU_THREADS, "device": "cpu",
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
                text=TEXT, language=config.DEFAULT_LANGUAGE,
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
