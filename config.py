"""Configurações compartilhadas do NineCast."""

import os
from pathlib import Path

from dotenv import load_dotenv


_CONFIG_DIR = Path(__file__).resolve().parent
load_dotenv(_CONFIG_DIR / ".env")

BASE_DIR = _CONFIG_DIR
DATA_DIR = BASE_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
PODCASTS_DIR = DATA_DIR / "podcasts"
MODELS_DIR = DATA_DIR / "models"

TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "pt")
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "50000"))
XTTS_MAX_CHARS = int(os.getenv("XTTS_MAX_CHARS", "220"))
CPU_THREADS = int(os.getenv("CPU_THREADS", "6"))


def ensure_directories() -> None:
    """Garante a existência dos diretórios locais de trabalho."""
    for directory in (VOICES_DIR, PODCASTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

