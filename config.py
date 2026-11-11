"""Configurações compartilhadas do NineCast."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
VOICES_DIR = DATA_DIR / "voices"
PODCASTS_DIR = DATA_DIR / "podcasts"

TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
TTS_LANGUAGE = "pt"
TTS_DEVICE = "cpu"
