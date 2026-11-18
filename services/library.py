"""Biblioteca local de episódios MP3."""

from datetime import datetime
import math
from pathlib import Path
import shutil
import stat
import subprocess

import config


def _formatted_duration(path: Path) -> str:
    """Retorna a duração como mm:ss sem impedir a listagem em caso de falha."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return "--:--"
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        seconds = float(result.stdout.strip())
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError
        total_seconds = round(seconds)
        return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "--:--"


def list_episodes() -> list[dict]:
    episodes = []
    for path in config.PODCASTS_DIR.glob("*"):
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if path.suffix.lower() != ".mp3" or not stat.S_ISREG(info.st_mode):
            continue
        episodes.append({"filename": path.name, "path": path.absolute(),
                         "size_mb": round(info.st_size / 1_000_000, 2),
                         "created_at": datetime.fromtimestamp(info.st_mtime).strftime("%d/%m/%Y %H:%M"),
                         "duration": _formatted_duration(path),
                         "_mtime": info.st_mtime_ns})
    episodes.sort(key=lambda episode: episode["_mtime"], reverse=True)
    for episode in episodes:
        del episode["_mtime"]
    return episodes


def delete_episode(filename: str) -> bool:
    if (not isinstance(filename, str) or not filename or filename in {".", ".."}
            or "/" in filename or "\\" in filename or "\x00" in filename
            or Path(filename).suffix.lower() != ".mp3"):
        raise ValueError("Nome de episódio inválido.")
    path = config.PODCASTS_DIR / filename
    try:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("O episódio deve ser um arquivo MP3 comum, não um link.")
        path.unlink()
        return True
    except FileNotFoundError:
        return False
