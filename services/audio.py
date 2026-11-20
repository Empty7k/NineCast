"""Divisão de roteiros e montagem local de episódios MP3."""

import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import uuid
import wave
from pathlib import Path
from typing import Callable

import config
import numpy as np
from services import qwen_tts, tts_text, voice

normalize_text_for_tts = tts_text.normalize_text_for_tts

TTS_ENGINES = {
    "xtts": {
        "label": "XTTS-v2",
        "perfil": "Mais rápido em CPU",
        "velocidade_cpu": "Rápido (~3×)",
        "ram_estimada": "~3,5 GB RAM",
        "licenca": "CPML (não comercial)",
        "requires_cpml": True,
        "requires_transcript": False,
        "vantagens": "Geração mais rápida; aceita 1 a 5 WAVs sem transcrição.",
        "desvantagens": "Voz um pouco mais robótica; uso não comercial.",
        "recomendado_para": "Episódios longos ou uso rápido.",
    },
    "qwen3_06b": {
        "label": "Qwen3-TTS 0.6B",
        "perfil": "Voz mais natural e fiel",
        "velocidade_cpu": "Moderado (~7×)",
        "ram_estimada": "~4,5 GB RAM",
        "licenca": "Apache-2.0",
        "requires_cpml": False,
        "requires_transcript": True,
        "vantagens": "Timbre mais fiel; licença livre; libera a RAM ao terminar.",
        "desvantagens": "~2× mais lento em CPU; requer instalação isolada.",
        "recomendado_para": "Máxima fidelidade (1 a 3 WAVs com transcrição automática).",
    },
}


def describe_tts_engine(engine_id: str) -> dict:
    try:
        return TTS_ENGINES[engine_id].copy()
    except (KeyError, TypeError) as exc:
        raise ValueError("Selecione um motor de voz conhecido.") from exc


_FILTERS = (
    "highpass=f=80,"
    "acompressor=threshold=0.125:ratio=2:attack=15:release=180:makeup=1:knee=2.828,"
    "loudnorm=I=-16:TP=-1.5:LRA=9"
)
_PEAK_LIMIT = 10 ** (-1 / 20)
_MAX_RMS_GAIN = 10 ** (6 / 20)
_ABBREVIATIONS = {
    "sr.", "sra.", "srta.", "dr.", "dra.", "prof.", "profa.", "ex.",
    "p.ex.", "etc.", "vs.", "av.", "nº.", "no.", "pág.", "fig.",
}
_ENDING = re.compile(r"[.!?;]+[\"'’”»\)\]]*(?=\s|$)")
_INTERMEDIATE = re.compile(r"(?:[, :]|--)[\"'’”»\)\]]*(?=\s|$)")


def _terminal_breaks(text: str) -> list[int]:
    breaks = []
    for match in _ENDING.finditer(text):
        end = match.end()
        word = text[:match.start()].split()[-1] if text[:match.start()].split() else ""
        punctuation = match.group().rstrip("\"'’”»)]")
        token = (word + punctuation).lower().strip("\"'‘“«(")
        if (punctuation == "." and (token in _ABBREVIATIONS
                or re.fullmatch(r"(?:[a-zá-ú]\.){1,}", token, re.IGNORECASE))):
            continue
        breaks.append(end)
    return breaks


def _paragraph_chunks(paragraph: str, max_chars: int) -> list[str]:
    remaining = " ".join(paragraph.split())
    chunks = []
    while len(remaining) > max_chars:
        terminal = [p for p in _terminal_breaks(remaining) if 20 <= p <= max_chars]
        intermediate = [m.end() for m in _INTERMEDIATE.finditer(remaining)
                        if 20 <= m.end() <= max_chars and m.group().strip()]
        whitespace = [m.start() for m in re.finditer(r"\s+", remaining)
                      if 20 <= m.start() <= max_chars]
        # Prioriza o fim de frase, depois pontuação intermediária e limites de palavras.
        cut = (terminal or intermediate or whitespace or [max_chars])[-1]
        chunk, remaining = remaining[:cut].strip(), remaining[cut:].strip()
        if chunk:
            chunks.append(chunk)
    if remaining:
        chunks.append(remaining)
    # Se o último pedaço for muito curto, redistribui palavras do anterior quando possível.
    if len(chunks) > 1 and len(chunks[-1]) < 20:
        previous, tail = chunks[-2:]
        while len(tail) < 20 and " " in previous:
            prefix, last = previous.rsplit(" ", 1)
            candidate = f"{last} {tail}"
            if len(prefix) < 20 or len(candidate) > max_chars:
                break
            previous, tail = prefix, candidate
        chunks[-2:] = [previous, tail]
    return chunks


def _chunks_with_paragraphs(text: str, max_chars: int) -> list[tuple[str, int]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Digite um texto não vazio para gerar o podcast.")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 20:
        raise ValueError("max_chars deve ser um inteiro de pelo menos 20 caracteres.")
    chunks = []
    paragraphs = [p for p in re.split(r"\r\n|\r|\n", text) if p.strip()]
    for index, paragraph in enumerate(paragraphs):
        chunks.extend((chunk, index) for chunk in _paragraph_chunks(paragraph, max_chars))
    return chunks


def split_text_into_chunks(text: str, max_chars: int = 220) -> list[str]:
    """Fatia texto bruto; normalize antes para estimar os blocos reais de TTS."""
    return [chunk for chunk, _ in _chunks_with_paragraphs(text, max_chars)]


def _sanitize_episode_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    return slug[:80].rstrip("-") or "episodio"


def _analyze_wav(source: wave.Wave_read) -> tuple[int, int, float, float]:
    """Mede limites, RMS ativo e pico com buffers de um segundo, sem alterar o WAV."""
    total = source.getnframes()
    rate = source.getframerate()
    if not total:
        raise ValueError("O bloco de áudio está vazio.")
    peak = 0.0
    source.rewind()
    while data := source.readframes(rate):
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768
        peak = max(peak, float(np.max(np.abs(samples))))
    # Piso de -60 dBFS ou 45 dB abaixo do pico: preserva finais fracos da fala.
    threshold = max(10 ** (-60 / 20), peak * 10 ** (-45 / 20))
    first, last = total, -1
    energy, count, offset = 0.0, 0, 0
    source.rewind()
    while data := source.readframes(rate):
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768
        active = np.flatnonzero(np.abs(samples) > threshold)
        if active.size:
            first = min(first, offset + int(active[0]))
            last = offset + int(active[-1])
            energy += float(np.sum(samples[active] ** 2))
            count += active.size
        offset += samples.size
    if not count:
        return 0, total, 0.0, peak  # Silêncio/ruído de fundo nunca recebe ganho.
    pad, long_silence = round(rate * 0.04), round(rate * 0.1)
    start = max(0, first - pad) if first > long_silence else 0
    end = min(total, last + 1 + pad) if total - last - 1 > long_silence else total
    return start, end, float(np.sqrt(energy / count)), peak


def _write_faded_frames(
    source: wave.Wave_read,
    target: wave.Wave_write,
    fade_ms: int = 12,
    *,
    start: int = 0,
    end: int | None = None,
    gain: float = 1.0,
) -> int:
    """Copia o intervalo PCM16 com ganho e fades, sem esticar ou cortar a fala."""
    total = (source.getnframes() if end is None else end) - start
    rate = source.getframerate()
    source.setpos(start)
    fade_frames = min(round(rate * fade_ms / 1000), total // 2)
    offset = 0
    while offset < total:
        data = source.readframes(min(rate, total - offset))
        if not data:
            raise ValueError("O bloco WAV terminou antes da duração informada.")
        samples = np.frombuffer(data, dtype="<i2").astype(np.float64) * gain
        if fade_frames:
            positions = np.arange(offset, offset + samples.size)
            gains = np.ones(samples.size, dtype=np.float32)
            gains = np.minimum(gains, np.clip(positions / fade_frames, 0, 1))
            gains = np.minimum(gains, np.clip((total - 1 - positions) / fade_frames, 0, 1))
            samples *= gains
        pcm = np.clip(np.rint(samples), -32768, 32767).astype("<i2")
        target.writeframesraw(pcm.tobytes())
        offset += samples.size
    return total


def _concatenate_wavs(paths: list[Path], paragraphs: list[int], output: Path) -> float:
    """Apara pontas, nivela e une WAVs com pausas, mantendo buffers pequenos."""
    frames = 0
    rate = 24000
    analyses = []
    for path in paths:
        with wave.open(str(path), "rb") as part:
            if (part.getnchannels(), part.getsampwidth(), part.getframerate(), part.getcomptype()) != (1, 2, rate, "NONE"):
                raise ValueError(f"Formato de áudio incompatível: {path}")
            analyses.append(_analyze_wav(part))
    active_levels = [rms for _, _, rms, _ in analyses if rms > 0]
    target_rms = float(np.median(active_levels)) if active_levels else 0.0
    with wave.open(str(output), "wb") as combined:
        combined.setnchannels(1)
        combined.setsampwidth(2)
        combined.setframerate(rate)
        for index, path in enumerate(paths):
            if index:
                pause_seconds = 0.6 if paragraphs[index] != paragraphs[index - 1] else 0.3
                pause_frames = round(rate * pause_seconds)
                combined.writeframesraw(b"\0\0" * pause_frames)
                frames += pause_frames
            with wave.open(str(path), "rb") as part:
                start, end, rms, peak = analyses[index]
                # Corrige metade da diferença em dB e limita a +/-6 dB por bloco.
                gain = 1.0
                if rms:
                    gain = float(np.clip(np.sqrt(target_rms / rms),
                                         1 / _MAX_RMS_GAIN, _MAX_RMS_GAIN))
                if peak:
                    gain = min(gain, _PEAK_LIMIT / peak)
                frames += _write_faded_frames(part, combined, start=start, end=end, gain=gain)
    return frames / rate


def generate_podcast(
    text: str,
    voice_name: str,
    output_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    episode_title: str | None = None,
    engine: str = "xtts",
) -> dict:
    """Sintetiza blocos e salva um MP3; nunca sobrescreve uma saída existente.

    O callback recebe o número de blocos já concluídos e o total. A duração é
    medida no WAV de montagem, incluindo as pausas, antes da codificação MP3.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Digite um texto não vazio para gerar o podcast.")
    if len(text) > config.MAX_TEXT_CHARS:
        raise ValueError(f"O roteiro excede o limite de {config.MAX_TEXT_CHARS} caracteres.")
    if episode_title is not None and not isinstance(episode_title, str):
        raise TypeError("episode_title deve ser texto ou None.")
    clean_title = episode_title.strip() if episode_title else ""
    describe_tts_engine(engine)
    normalized_text = tts_text.normalize_text_for_tts(text)
    max_chars = config.QWEN_TTS_MAX_CHARS if engine == "qwen3_06b" else config.XTTS_MAX_CHARS
    chunks = _chunks_with_paragraphs(normalized_text, max_chars)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback deve ser uma função ou None.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg não encontrado. Instale-o e confira o PATH.")
    if output_path is None:
        config.PODCASTS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        prefix = _sanitize_episode_title(clean_title) if clean_title else "podcast"
        while True:
            output_path = config.PODCASTS_DIR / f"{prefix}-{uuid.uuid4().hex[:10]}.mp3"
            if not output_path.exists():
                break
    else:
        output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp3":
        raise ValueError("O arquivo de saída deve ter extensão .mp3.")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Diretório de saída não encontrado: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(f"O arquivo de saída já existe: {output_path}")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".ninecast_audio_", dir=output_path.parent) as temporary:
        temp_dir = Path(temporary)
        wavs = [temp_dir / f"trecho_{index:04d}.wav" for index in range(len(chunks))]
        if engine == "qwen3_06b":
            results = qwen_tts.synthesize_chunks(
                [chunk for chunk, _ in chunks], voice_name, wavs, progress_callback,
            )
        else:
            results = []
            for index, ((chunk, _), wav_path) in enumerate(zip(chunks, wavs), 1):
                results.append(voice.synthesize_speech(chunk, voice_name, wav_path))
                if progress_callback is not None:
                    progress_callback(index, len(chunks))
        synthesis_seconds = sum(result["synthesis_seconds"] for result in results)
        combined = temp_dir / "montagem.wav"
        duration = _concatenate_wavs(wavs, [paragraph for _, paragraph in chunks], combined)
        encoded = temp_dir / "episodio.mp3"
        metadata_title = clean_title or output_path.stem
        # MPEG-2 a 24 kHz limita MP3 a 160 kbps; 48 kHz permite 192 kbps reais.
        # Os WAVs dos motores e a montagem continuam em 24 kHz.
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", str(combined),
                   "-af", _FILTERS, "-ac", "1", "-ar", "48000", "-c:a", "libmp3lame", "-b:a", "192k",
                   "-metadata", f"title={metadata_title}", "-metadata", f"artist={voice_name}", str(encoded)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Falha ao exportar MP3 com FFmpeg: {exc.stderr.strip()}") from exc
        # Reserva o nome exclusivamente antes de mover, sem exigir suporte a hard links.
        with output_path.open("xb") as reserved:
            reservation = os.fstat(reserved.fileno())
        try:
            shutil.move(str(encoded), str(output_path))
        except Exception:
            try:
                current = output_path.lstat()
                if (current.st_dev, current.st_ino) == (reservation.st_dev, reservation.st_ino):
                    output_path.unlink()
            except FileNotFoundError:
                pass
            raise
    elapsed = time.perf_counter() - started
    return {"output_path": str(output_path), "chunk_count": len(chunks),
            "audio_duration": duration, "synthesis_seconds": synthesis_seconds,
            "total_seconds": elapsed, "rtf": synthesis_seconds / duration}
