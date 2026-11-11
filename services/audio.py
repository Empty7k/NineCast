"""Divisão de roteiros e montagem local de episódios MP3."""

import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Callable

import config
from services import voice

_FILTERS = (
    "equalizer=f=8000:t=q:w=1.5:g=-5,"
    "equalizer=f=3000:t=q:w=2:g=-2,"
    "equalizer=f=200:t=q:w=1:g=2,"
    "loudnorm=I=-16:TP=-1.5:LRA=11"
)
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
    """Divide o texto por parágrafo, frase e palavra, preservando sua ordem."""
    return [chunk for chunk, _ in _chunks_with_paragraphs(text, max_chars)]


def _concatenate_wavs(paths: list[Path], paragraphs: list[int], output: Path) -> float:
    """Une WAVs PCM mono com silêncio entre blocos, sem carregá-los todos na RAM."""
    frames = 0
    rate = 24000
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
                if (part.getnchannels(), part.getsampwidth(), part.getframerate(), part.getcomptype()) != (1, 2, rate, "NONE"):
                    raise ValueError(f"Formato de áudio incompatível: {path}")
                while data := part.readframes(rate):
                    combined.writeframesraw(data)
                    frames += len(data) // 2
    return frames / rate


def generate_podcast(
    text: str,
    voice_name: str,
    output_path: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict:
    """Sintetiza blocos e salva um MP3; nunca sobrescreve uma saída existente.

    O callback recebe o número de blocos já concluídos e o total. A duração é
    medida no WAV de montagem, incluindo as pausas, antes da codificação MP3.
    """
    chunks = _chunks_with_paragraphs(text, 220)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback deve ser uma função ou None.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg não encontrado. Instale-o e confira o PATH.")
    if output_path is None:
        config.PODCASTS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        output_path = config.PODCASTS_DIR / f"podcast-{uuid.uuid4().hex}.mp3"
    else:
        output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp3":
        raise ValueError("O arquivo de saída deve ter extensão .mp3.")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"Diretório de saída não encontrado: {output_path.parent}")
    if output_path.exists():
        raise FileExistsError(f"O arquivo de saída já existe: {output_path}")

    started = time.perf_counter()
    synthesis_seconds = 0.0
    with tempfile.TemporaryDirectory(prefix=".ninecast_audio_", dir=output_path.parent) as temporary:
        temp_dir = Path(temporary)
        wavs = []
        for index, (chunk, _) in enumerate(chunks):
            wav_path = temp_dir / f"trecho_{index:04d}.wav"
            metrics = voice.synthesize_speech(chunk, voice_name, wav_path)
            synthesis_seconds += metrics["synthesis_seconds"]
            wavs.append(wav_path)
            if progress_callback is not None:
                progress_callback(index + 1, len(chunks))
        combined = temp_dir / "montagem.wav"
        duration = _concatenate_wavs(wavs, [paragraph for _, paragraph in chunks], combined)
        encoded = temp_dir / "episodio.mp3"
        command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-i", str(combined),
                   "-af", _FILTERS, "-ac", "1", "-ar", "24000", "-c:a", "libmp3lame", "-b:a", "128k", str(encoded)]
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
