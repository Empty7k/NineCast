"""Perfis de voz locais e execução isolada do Qwen3-TTS."""

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import soundfile as sf

import config
from services import voice

_INSTALL_HINT = "Execute ./ninecast.sh --install-qwen-tts para instalar o Qwen3-TTS."
_DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def _python() -> Path:
    return config.BASE_DIR / config.QWEN_TTS_VENV_DIR / "bin/python"


def is_available() -> bool:
    """Verifica a instalação do pacote sem importar ou carregar o modelo."""
    python = _python()
    return python.is_file() and os.access(python, os.X_OK) and any(
        (python.parent.parent / "lib").glob("python*/site-packages/qwen_tts/__init__.py")
    )


def _profiles_dir() -> Path:
    root = config.VOICES_DIR / "qwen"
    if root.is_symlink():
        raise ValueError("O diretório de vozes Qwen não pode ser um link simbólico.")
    return root


def _profile_path(name: str) -> Path:
    if (not isinstance(name, str) or not name or name != name.strip()
            or name.startswith(".") or any(c in name for c in "/\\")
            or any(ord(c) < 32 for c in name) or len(name.encode("utf-8")) > 255):
        raise ValueError("Informe um nome de voz válido, sem caminhos ou barras.")
    return _profiles_dir() / name


def _has_profile(path: Path) -> bool:
    return not path.is_symlink() and path.is_dir() and all(
        (path / filename).is_file() and not (path / filename).is_symlink()
        for filename in ("reference.wav", "profile.json")
    )


def list_available_voices() -> list[str]:
    """Lista apenas perfis publicados e completos do Qwen."""
    root = _profiles_dir()
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir()
                  if not path.name.startswith(".") and _has_profile(path))


def _wav_source(reference_wav: Path) -> Path:
    source = Path(reference_wav)
    if source.suffix.lower() != ".wav":
        raise ValueError("Selecione um arquivo WAV de referência.")
    return source


def register_voice(voice_name: str, reference_wav: Path, ref_text: str | None = None) -> str:
    """Publica a referência e a transcrição informada ou obtida localmente."""
    path = _profile_path(voice_name)
    if ref_text is not None and not isinstance(ref_text, str):
        raise ValueError("A transcrição deve ser um texto.")
    source = _wav_source(reference_wav)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Já existe uma voz com esse nome. Escolha outro nome.")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with source.open("rb") as raw:
        fingerprint = hashlib.file_digest(raw, "sha256").hexdigest()
    with tempfile.TemporaryDirectory(prefix=".cadastro-", dir=path.parent) as temporary:
        staged = Path(temporary)
        reference = staged / "reference.wav"
        metadata = staged / "profile.json"
        voice._prepare_reference(source, reference, highpass_hz=75)
        reference.chmod(0o600)
        transcript = (ref_text or "").strip() or _transcribe_prepared(reference)
        metadata.write_text(json.dumps({"ref_text": transcript, "sha256": fingerprint},
                                       ensure_ascii=False) + "\n", encoding="utf-8")
        metadata.chmod(0o600)
        # Diretórios publicados são não vazios, inclusive se outro cadastro vencer a corrida.
        staged.rename(path)
    return transcript


def _model_source() -> str:
    model = config.QWEN_TTS_MODEL
    if model == _DEFAULT_MODEL:
        cached = config.DATA_DIR / "benchmarks/tts-cloning-v05/cache/qwen"
        if all((cached / filename).is_file() for filename in (
            "config.json", "model.safetensors", "speech_tokenizer/model.safetensors",
        )):
            return str(cached)
    return model


def _run_worker(job: dict, on_event: Callable[[dict], None]) -> None:
    transcribing = job["action"] == "transcribe"
    engine = "Whisper" if transcribing else "Qwen3-TTS"
    command = [str(_python()), "-I", "-u", str(Path(__file__).with_name("qwen_tts_worker.py"))]
    payload = json.dumps(job)
    with voice.reserve_memory_for_llm():
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       text=True, encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc
        with process:
            try:
                try:
                    with process.stdin:
                        process.stdin.write(payload)
                except BrokenPipeError:
                    pass
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise ValueError
                    except ValueError as exc:
                        raise RuntimeError(f"Resposta inválida do processo do {engine}.") from exc
                    error = event.get("error")
                    if error == "memory":
                        raise MemoryError(f"Memória insuficiente para executar o {engine}.")
                    if error == "installation":
                        if transcribing:
                            raise RuntimeError("faster-whisper indisponível. " + _INSTALL_HINT)
                        raise RuntimeError(_INSTALL_HINT)
                    if error == "no_speech":
                        raise ValueError("A referência não contém fala transcritível. Escolha outro WAV ou informe a transcrição.")
                    if "error" in event:
                        raise RuntimeError(f"Falha no {engine}. Confira o cache local do modelo "
                                           "ou a conexão para o primeiro download.")
                    on_event(event)
                code = process.wait()
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    if code == -9:
        raise MemoryError(f"O processo do {engine} foi interrompido. Confira a memória disponível.")
    if code:
        raise RuntimeError(f"O processo do {engine} terminou com erro.")


def _transcribe_prepared(reference: Path) -> str:
    result = ""

    def save_text(event: dict) -> None:
        nonlocal result
        text = event.get("text")
        if not isinstance(text, str) or not text.strip() or result:
            raise RuntimeError("Resposta inválida do processo do Whisper.")
        result = text.strip()

    download_root = config.MODELS_DIR / "whisper"
    job = {
        "action": "transcribe", "reference": str(reference.absolute()),
        "model": config.WHISPER_MODEL, "language": config.DEFAULT_LANGUAGE,
        "threads": config.CPU_THREADS, "download_root": str(download_root),
    }
    _run_worker(job, save_text)
    if not result:
        raise RuntimeError("O processo do Whisper terminou sem retornar a transcrição.")
    return result


def transcribe_reference(reference_wav: Path) -> str:
    """Transcreve uma cópia preparada da referência no processo isolado."""
    source = _wav_source(reference_wav)
    with tempfile.TemporaryDirectory(prefix="ninecast_whisper_") as temporary:
        reference = Path(temporary) / "reference.wav"
        voice._prepare_reference(source, reference, highpass_hz=75)
        return _transcribe_prepared(reference)


def synthesize_chunks(
    chunks: list[str],
    voice_name: str,
    output_paths: list[Path],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    """Executa um único worker por episódio e encaminha o progresso."""
    if not chunks or len(chunks) != len(output_paths):
        raise ValueError("Informe um caminho de saída para cada trecho do Qwen3-TTS.")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback deve ser uma função ou None.")
    if not is_available():
        raise RuntimeError(_INSTALL_HINT)
    profile = _profile_path(voice_name)
    if not _has_profile(profile):
        raise FileNotFoundError("Cadastre uma referência para essa voz do Qwen3-TTS.")
    try:
        ref_text = json.loads((profile / "profile.json").read_text(encoding="utf-8"))["ref_text"]
        if not isinstance(ref_text, str) or not ref_text.strip():
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Transcrição inválida no perfil do Qwen3-TTS.") from exc
    paths = [Path(path).absolute() for path in output_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Cada trecho do Qwen3-TTS precisa de uma saída distinta.")
    if any(path.exists() or path.is_symlink() for path in paths):
        raise FileExistsError("Um arquivo de saída do Qwen3-TTS já existe.")
    job = {
        "action": "synthesize", "model": _model_source(), "threads": config.CPU_THREADS,
        "reference": str((profile / "reference.wav").absolute()), "ref_text": ref_text,
        "chunks": chunks, "outputs": [str(path) for path in paths],
    }
    metrics = []

    def update_progress(event: dict) -> None:
        elapsed = event.get("synthesis_seconds")
        if (event.get("current") != len(metrics) + 1 or len(metrics) == len(chunks)
                or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
            raise RuntimeError("Evento de progresso inválido do Qwen3-TTS.")
        metrics.append({"synthesis_seconds": elapsed})
        if progress_callback is not None:
            progress_callback(len(metrics), len(chunks))

    _run_worker(job, update_progress)
    if len(metrics) != len(chunks):
        raise RuntimeError("O processo do Qwen3-TTS terminou antes de gerar todos os trechos.")
    for path, result in zip(paths, metrics):
        try:
            info = sf.info(path)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError("O Qwen3-TTS não produziu um WAV legível para um dos trechos.") from exc
        if (info.format, info.subtype, info.samplerate, info.channels) != ("WAV", "PCM_16", 24000, 1) or not info.frames:
            raise RuntimeError("O Qwen3-TTS gerou um trecho WAV incompatível.")
        result.update(audio_duration=info.duration, output_path=str(path))
    return metrics
