"""Configurações compartilhadas do NineCast."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
PODCASTS_DIR = DATA_DIR / "podcasts"
UPLOADS_DIR = DATA_DIR / "uploads"
SCRIPTS_DIR = DATA_DIR / "scripts"

TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE = "pt"
TTS_DEVICE = "cpu"


def ensure_directories() -> None:
    """Garante a existência dos diretórios locais de trabalho."""
    for directory in (VOICES_DIR, PODCASTS_DIR, UPLOADS_DIR, SCRIPTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

