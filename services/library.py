"""Biblioteca local de episódios MP3."""

from datetime import datetime
from pathlib import Path
import stat

import config


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
