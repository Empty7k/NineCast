"""Clonagem de voz local com XTTS-v2; o modelo é carregado sob demanda."""

import ctypes
import gc
import hashlib
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import config

_model = None
_lock = threading.RLock()
_SAMPLE_RATE = 24000
_CACHE_VERSION = 1
_REFERENCE_PREPARATION_VERSION = 3
_MAX_REF_SECONDS = 30
_STYLE_CHUNK_SECONDS = 6
_GENERATION = {
    "temperature": 0.70, "repetition_penalty": 6.0, "length_penalty": 1.0,
    "top_p": 0.85, "top_k": 50, "speed": 1.0, "enable_text_splitting": False,
}


def _is_memory_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(phrase in message for phrase in (
        "out of memory", "cannot allocate memory", "can't allocate memory", "not enough memory",
    ))


def _cache_path(voice_name: str) -> Path:
    if (not isinstance(voice_name, str) or not voice_name.strip()
            or voice_name != voice_name.strip() or voice_name in {".", ".."}
            or "/" in voice_name or "\\" in voice_name or "\x00" in voice_name):
        raise ValueError("Informe um nome de voz válido, sem caminhos ou barras.")
    return config.VOICES_DIR / f"{voice_name}.latents.pt"


def list_available_voices() -> list[str]:
    """Lista perfis com latentes salvos; WAVs isolados não são perfis ainda."""
    suffix = ".latents.pt"
    return sorted(p.name[:-len(suffix)] for p in config.VOICES_DIR.glob(f"*{suffix}") if p.is_file())


def _get_model():
    global _model
    with _lock:
        if _model is None:
            # Definidos antes da importação do Coqui/Hugging Face, sem impor download.
            os.environ.setdefault("HF_HOME", str(config.MODELS_DIR / "huggingface"))
            os.environ.setdefault("XDG_DATA_HOME", str(config.MODELS_DIR))
            torch.set_num_threads(config.CPU_THREADS)
            try:
                from TTS.api import TTS

                tts = TTS(model_name=config.TTS_MODEL, progress_bar=False).to("cpu")
                _model = tts.synthesizer.tts_model
            except MemoryError as exc:
                raise MemoryError("Memória insuficiente para carregar o XTTS-v2 em CPU.") from exc
            except RuntimeError as exc:
                if _is_memory_error(exc):
                    raise MemoryError("Memória insuficiente para carregar o XTTS-v2 em CPU.") from exc
                raise RuntimeError(f"Não foi possível carregar o XTTS-v2: {exc}") from exc
            except Exception as exc:
                raise RuntimeError(f"Não foi possível carregar o XTTS-v2: {exc}") from exc
        return _model


def unload_model() -> bool:
    """Libera a instância XTTS deste processo sem apagar os latentes em disco."""
    global _model
    with _lock:
        if _model is None:
            return False
        _model = None
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass  # O alocador pode não oferecer malloc_trim.
        return True


@contextmanager
def reserve_memory_for_llm():
    """Impede que outra sessão recarregue o XTTS durante a geração do roteiro."""
    with _lock:
        unload_model()
        yield


def _fingerprints(paths: list[Path]) -> list[str]:
    fingerprints = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de referência não encontrado: {path}")
        with path.open("rb") as audio_file:
            fingerprints.append(hashlib.file_digest(audio_file, "sha256").hexdigest())
    return fingerprints


def _prepare_reference(source: Path, destination: Path) -> None:
    """Remove DC, reduz pausas e normaliza uma cópia para pico de -1 dBFS."""
    import librosa

    try:
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Não foi possível ler o áudio de referência: {source}") from exc
    audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError(f"Áudio de referência vazio, silencioso ou inválido: {source}")
    # Antes da detecção de fala, pois um offset pode esconder os silêncios.
    audio = audio - np.mean(audio, dtype=np.float64)
    if not np.any(audio):
        raise ValueError(f"Áudio de referência sem sinal útil após remover DC: {source}")
    intervals = librosa.effects.split(audio, top_db=30)
    pad = round(rate * 0.03)
    minimum_gap = round(rate * 0.35)
    merged = []
    for start, end in intervals:
        start, end = int(start), int(end)
        if merged and start - merged[-1][1] < minimum_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    parts = []
    remaining = rate * _MAX_REF_SECONDS
    pause = np.zeros(round(rate * 0.15), dtype=np.float32)
    for start, end in merged:
        start, end = max(0, start - pad), min(len(audio), end + pad)
        if parts:
            if remaining <= len(pause):
                break
            parts.append(pause)
            remaining -= len(pause)
        segment = audio[start:min(end, start + remaining)]
        parts.append(segment)
        remaining -= len(segment)
        if remaining == 0:
            break
    if not parts or sum(len(p) for p in parts) < rate * 0.33:
        raise ValueError(f"Áudio de referência curto demais após o preparo: {source}")
    prepared = np.concatenate(parts)
    # Os cortes podem mudar a média; recentra antes de definir o pico final.
    prepared = prepared - np.mean(prepared, dtype=np.float64)
    peak = np.max(np.abs(prepared))
    if peak <= np.finfo(np.float32).tiny:
        raise ValueError(f"Áudio de referência sem sinal útil após o preparo: {source}")
    prepared = (prepared / peak) * 10 ** (-1 / 20)
    sf.write(destination, prepared, rate, subtype="FLOAT")


def _load_cache(path: Path) -> dict:
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(data, dict) or data.get("version") != _CACHE_VERSION
                or data.get("model") != config.TTS_MODEL
                or not isinstance(data.get("gpt_cond_latent"), torch.Tensor)
                or not isinstance(data.get("speaker_embedding"), torch.Tensor)):
            raise ValueError("formato ou versão incompatível")
        return data
    except Exception as exc:
        raise ValueError(f"Cache de voz inválido em {path}; recrie a voz com as referências. Detalhes: {exc}") from exc


def _pool_reference_latents(model, signals, rate):
    """Pondera o locutor pela duração preparada e poda apenas a cauda curta do estilo."""
    weighted = [model.get_speaker_embedding(signal, rate) * signal.size(-1) for signal in signals]
    speaker = torch.stack(weighted).sum(0)
    norm = torch.linalg.vector_norm(speaker, dim=1, keepdim=True)
    if not torch.isfinite(norm).all() or (norm <= 1e-8).any():
        raise ValueError("As referências não produziram um embedding de locutor válido.")
    full = torch.cat(signals, dim=-1)
    window = _STYLE_CHUNK_SECONDS * rate
    tail = full.size(-1) % window
    if full.size(-1) >= window and 0 < tail < 2 * rate:
        full = full[:, :-tail]
    style = model.get_gpt_cond_latents(full, rate, length=-1, chunk_length=_STYLE_CHUNK_SECONDS)
    return style, speaker / norm


def _extract_latents(model, prepared):
    from torchaudio.functional import highpass_biquad, resample

    rate = 22050  # Frequência usada pelo condicionamento GPT do XTTS.
    signals = []
    for path in prepared:
        samples, native_rate = sf.read(path, dtype="float32")
        filtered = highpass_biquad(torch.from_numpy(samples), native_rate, 75)
        signal = resample(filtered.unsqueeze(0), native_rate, rate).clamp_(-1, 1)
        signal = signal[:, :rate * _MAX_REF_SECONDS]
        peak = signal.abs().max()
        if not torch.isfinite(peak) or peak <= torch.finfo(signal.dtype).tiny:
            raise ValueError("Áudio de referência sem sinal útil após a filtragem.")
        signals.append(signal / peak * 0.75)
    return _pool_reference_latents(model, signals, rate)


def get_or_create_voice_latents(voice_name: str, speaker_wavs=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Carrega os latentes de uma voz ou cria-os a partir de WAVs autorizados.

    Se referências forem informadas para uma voz existente, alterações de conteúdo
    ou de ordem invalidam o cache. Sem referências, somente um cache válido é usado.
    """
    cache = _cache_path(voice_name)
    if isinstance(speaker_wavs, (str, Path)):
        raise ValueError("Informe as referências como uma lista de caminhos WAV.")
    paths = [Path(p) for p in speaker_wavs] if speaker_wavs is not None else []
    with _lock:
        try:
            hashes = _fingerprints(paths) if paths else None
            if cache.is_file():
                try:
                    data = _load_cache(cache)
                except ValueError:
                    if not paths:
                        raise
                else:
                    if hashes is None or (
                        data.get("reference_hashes") == hashes
                        and data.get("preparation_version") == _REFERENCE_PREPARATION_VERSION
                    ):
                        return data["gpt_cond_latent"], data["speaker_embedding"]
            if not paths:
                raise FileNotFoundError(f"A voz '{voice_name}' não tem cache nem referências. Informe arquivos WAV para criá-la.")
            if any(p.suffix.lower() != ".wav" for p in paths):
                raise ValueError("As referências de voz devem ser arquivos WAV.")
            config.VOICES_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tempfile.TemporaryDirectory(prefix="ninecast_refs_", dir=config.VOICES_DIR) as temp_dir:
                prepared = []
                for index, path in enumerate(paths):
                    destination = Path(temp_dir) / f"{index}.wav"
                    _prepare_reference(path, destination)
                    prepared.append(str(destination))
                model = _get_model()
                with torch.inference_mode():
                    gpt, speaker = _extract_latents(model, prepared)
                data = {"version": _CACHE_VERSION, "model": config.TTS_MODEL,
                        "preparation_version": _REFERENCE_PREPARATION_VERSION,
                        "reference_hashes": hashes, "gpt_cond_latent": gpt.cpu(),
                        "speaker_embedding": speaker.cpu()}
                # O arquivo temporário privado evita deixar um cache incompleto.
                temporary_cache = Path(temp_dir) / "latents.pt"
                torch.save(data, temporary_cache)
                os.chmod(temporary_cache, 0o600)
                os.replace(temporary_cache, cache)
            return data["gpt_cond_latent"], data["speaker_embedding"]
        except MemoryError as exc:
            raise MemoryError("Memória insuficiente para preparar a voz. Feche outros aplicativos e tente novamente.") from exc
        except RuntimeError as exc:
            if _is_memory_error(exc):
                raise MemoryError("Memória insuficiente para preparar a voz. Feche outros aplicativos e tente novamente.") from exc
            raise


def synthesize_speech(text: str, voice_name: str, output_path: Path) -> dict:
    """Sintetiza um trecho curto em WAV mono 24 kHz sem sobrescrever arquivos."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Digite um texto não vazio para gerar a narração.")
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".wav":
        raise ValueError("A saída desta etapa deve ser um arquivo .wav.")
    if output_path.exists():
        raise FileExistsError(f"O arquivo de saída já existe: {output_path}")
    with _lock:
        try:
            gpt, speaker = get_or_create_voice_latents(voice_name)
            model = _get_model()
            tokens = model.tokenizer.encode(text.strip().lower(), lang=config.DEFAULT_LANGUAGE)
            if len(tokens) >= model.args.gpt_max_text_tokens:
                raise ValueError("Texto longo demais para um único trecho XTTS; divida-o em frases menores.")
            started = time.perf_counter()
            with torch.inference_mode():
                result = model.inference(text=text.strip(), language=config.DEFAULT_LANGUAGE,
                                         gpt_cond_latent=gpt, speaker_embedding=speaker, **_GENERATION)
            elapsed = time.perf_counter() - started
            audio = np.asarray(result["wav"])
            if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                raise ValueError("O modelo retornou áudio vazio ou inválido.")
            duration = audio.size / _SAMPLE_RATE
            # 'xb' previne sobrescrita mesmo se o arquivo surgir durante a geração.
            with output_path.open("xb") as output:
                sf.write(output, audio, _SAMPLE_RATE, format="WAV", subtype="PCM_16")
            return {"synthesis_seconds": elapsed, "audio_duration": duration,
                    "rtf": elapsed / duration, "output_path": str(output_path)}
        except MemoryError as exc:
            raise MemoryError("Memória insuficiente para sintetizar o áudio. Feche outros aplicativos e tente novamente.") from exc
        except RuntimeError as exc:
            if _is_memory_error(exc):
                raise MemoryError("Memória insuficiente para sintetizar o áudio. Feche outros aplicativos e tente novamente.") from exc
            raise
