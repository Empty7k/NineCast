"""Testes do lançador e da instalação isolada do atalho."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LauncherTests(unittest.TestCase):
    def test_install_desktop_for_project_with_spaces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "Projeto NineCast"
            project.mkdir()
            launcher = project / "ninecast.sh"
            shutil.copy2(Path(__file__).resolve().parent.parent / "ninecast.sh", launcher)
            env = os.environ.copy()
            env["XDG_DATA_HOME"] = str(root / "dados")
            result = subprocess.run([str(launcher), "--install-desktop"], env=env,
                                    capture_output=True, text=True, check=True)
            desktop = root / "dados" / "applications" / "NineCast.desktop"
            self.assertIn(str(desktop), result.stdout)
            content = desktop.read_text()
            self.assertIn(f'Exec="{launcher}"', content)
            self.assertIn(f"Path={project}", content)
            self.assertIn("Terminal=true", content)
            self.assertEqual(desktop.stat().st_mode & 0o777, 0o755)
            validator = shutil.which("desktop-file-validate")
            if validator:
                subprocess.run([validator, str(desktop)], check=True, capture_output=True, text=True)
            before = desktop.read_bytes()
            retry = subprocess.run([str(launcher), "--install-desktop"], env=env, capture_output=True)
            self.assertNotEqual(retry.returncode, 0)
            self.assertEqual(desktop.read_bytes(), before)

    def test_missing_venv_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as temporary:
            launcher = Path(temporary) / "ninecast.sh"
            shutil.copy2(Path(__file__).resolve().parent.parent / "ninecast.sh", launcher)
            result = subprocess.run([str(launcher)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("uv sync --locked", result.stderr)


if __name__ == "__main__":
    unittest.main()
