import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
from services import library


class LibraryTests(unittest.TestCase):
    def test_list_order_and_safe_delete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "podcasts"
            folder.mkdir()
            old, new = folder / "antigo.mp3", folder / "novo.mp3"
            old.write_bytes(b"a")
            new.write_bytes(b"bb")
            os.utime(old, (1000000000, 1000000000))
            outside = root / "outside.mp3"
            outside.write_bytes(b"privado")
            (folder / "link.mp3").symlink_to(outside)
            (folder / "texto.txt").write_text("teste")
            with mock.patch.object(config, "PODCASTS_DIR", folder):
                episodes = library.list_episodes()
                self.assertEqual([e["filename"] for e in episodes], ["novo.mp3", "antigo.mp3"])
                self.assertEqual(episodes[0]["path"], new)
                self.assertEqual(episodes[0]["duration"], "--:--")
                for bad in ("../outside.mp3", "/tmp/outside.mp3", "x\\outside.mp3", "texto.txt", "link.mp3"):
                    with self.assertRaises(ValueError):
                        library.delete_episode(bad)
                self.assertTrue(library.delete_episode("novo.mp3"))
                self.assertFalse(library.delete_episode("novo.mp3"))
                self.assertEqual(outside.read_bytes(), b"privado")

    def test_duration_format_and_safe_fallback(self):
        completed = mock.Mock(stdout="125.4\n")
        with mock.patch.object(library.shutil, "which", return_value="/usr/bin/ffprobe"), \
             mock.patch.object(library.subprocess, "run", return_value=completed):
            self.assertEqual(library._formatted_duration(Path("episodio.mp3")), "02:05")
        with mock.patch.object(library.shutil, "which", return_value=None):
            self.assertEqual(library._formatted_duration(Path("episodio.mp3")), "--:--")
        with mock.patch.object(library.shutil, "which", return_value="ffprobe"), \
             mock.patch.object(library.subprocess, "run", side_effect=OSError("falha")):
            self.assertEqual(library._formatted_duration(Path("episodio.mp3")), "--:--")


if __name__ == "__main__":
    unittest.main()
