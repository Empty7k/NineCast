"""Worker executado pelo Python isolado do Qwen3-TTS."""

import json
import math
import os
import sys
import time
from pathlib import Path

_GENERATION = {
    "temperature": 0.9,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.05,
    "subtalker_temperature": 0.9,
    "subtalker_top_p": 1.0,
}


def synthesize(job: dict, emit) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(job["threads"]))
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    torch.set_num_threads(job["threads"])
    torch.set_num_interop_threads(1)
    source = job["model"]
    model = Qwen3TTSModel.from_pretrained(
        source, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa",
        local_files_only=Path(source).is_dir(),
    )
    reference, sample_rate = sf.read(job["reference"], dtype="float32")
    with torch.inference_mode():
        prompt = model.create_voice_clone_prompt(
            ref_audio=(reference, sample_rate), ref_text=job["ref_text"], x_vector_only_mode=False,
        )
        for index, (chunk, output) in enumerate(zip(job["chunks"], job["outputs"], strict=True), 1):
            started = time.perf_counter()
            waves, rate = model.generate_voice_clone(
                text=chunk, language="Portuguese", voice_clone_prompt=prompt, **_GENERATION,
            )
            waveform = np.asarray(waves[0])
            if rate != 24000 or waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                raise RuntimeError("Áudio inválido ou fora de 24 kHz.")
            with open(output, "xb") as wav:
                sf.write(wav, waveform, rate, format="WAV", subtype="PCM_16")
            emit({"current": index, "synthesis_seconds": time.perf_counter() - started})


def transcribe(job: dict, emit) -> bool:
    import soundfile as sf
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly

    source = job["model"]
    cached = Path(job["download_root"]) / source
    if (cached / "model.bin").is_file() and (cached / "tokenizer.json").is_file():
        source = str(cached)
    model = WhisperModel(source, device="cpu", compute_type="int8",
                         cpu_threads=job["threads"], download_root=job["download_root"])
    samples, rate = sf.read(job["reference"], dtype="float32")
    # Whisper recebe PCM mono de 16 kHz quando o áudio já vem decodificado.
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor)
    segments, _ = model.transcribe(samples, language=job["language"], vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    emit({"text": text} if text else {"error": "no_speech"})
    return bool(text)


def main() -> int:
    # Bibliotecas do modelo imprimem no stdout durante o import e a inferência.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        def emit(event: dict) -> None:
            print(json.dumps(event), file=protocol, flush=True)

        try:
            job = json.load(sys.stdin)
            if job["action"] == "transcribe":
                return 0 if transcribe(job, emit) else 1
            if job["action"] != "synthesize":
                raise ValueError("Ação desconhecida no worker.")
            synthesize(job, emit)
        except ImportError as exc:
            print(f"Qwen3-TTS: {exc}", file=sys.stderr, flush=True)
            emit({"error": "installation"})
            return 1
        except Exception as exc:
            message = str(exc).lower()
            memory = isinstance(exc, MemoryError) or any(
                phrase in message for phrase in ("out of memory", "allocate memory", "not enough memory")
            )
            print(f"Qwen3-TTS: {type(exc).__name__}", file=sys.stderr, flush=True)
            emit({"error": "memory" if memory else "synthesis"})
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
