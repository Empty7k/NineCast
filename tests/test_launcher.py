"""Testes do lançador e da instalação isolada do atalho."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class LauncherTests(unittest.TestCase):
    def test_qwen_installer_uses_separate_python_and_cpu_wheels(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "Projeto NineCast"
            project.mkdir()
            launcher = project / "ninecast.sh"
            shutil.copy2(Path(__file__).resolve().parent.parent / "ninecast.sh", launcher)
            uv = project / "uv"
            log = project / "uv.calls"
            uv.write_text('#!/bin/bash\nprintf "%s\\n" "$@" >> "$UV_CALLS"\n')
            uv.chmod(0o755)
            env = dict(os.environ, PATH=f"{project}:{os.environ['PATH']}", UV_CALLS=str(log))
            result = subprocess.run([str(launcher), "--install-qwen-tts"], env=env,
                                    capture_output=True, text=True, check=True)
            args = log.read_text().splitlines()
            venv = project / ".venv-qwen-tts"
            self.assertEqual(args[:4], ["venv", "--python", "3.12", str(venv)])
            self.assertEqual(args.count(str(venv / "bin/python")), 2)
            self.assertIn("https://download.pytorch.org/whl/cpu", args)
            for package in ("torch==2.8.0+cpu", "torchaudio==2.8.0+cpu", "qwen-tts==0.1.1",
                            "soundfile", "numpy", "faster-whisper==1.2.1"):
                self.assertIn(package, args)
            self.assertIn(str(venv), result.stdout)

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
