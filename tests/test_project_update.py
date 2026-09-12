import contextlib
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.update_project import USE_LOCAL, update_project


class ProjectUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="moegal update ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote.git"
        self.source = self.root / "source"
        self.local = self.root / "local project"
        self.git(self.root, "init", "--bare", "--initial-branch=main", str(self.remote))
        self.git(self.root, "clone", str(self.remote), str(self.source))
        (self.source / ".gitignore").write_text(".env\n.venv/\nsaved/\n", encoding="utf-8")
        (self.source / "app.txt").write_text("old", encoding="utf-8")
        self.commit(self.source, "initial")
        self.git(self.source, "push", "-u", "origin", "main")
        self.git(self.root, "clone", str(self.remote), str(self.local))
        self.old_head = self.git(self.local, "rev-parse", "HEAD")
        # 模拟已经能运行的环境与用户配置，更新过程不能碰它们。
        (self.local / ".env").write_text("CUSTOM_API_KEY=keep", encoding="utf-8")
        (self.local / ".venv").mkdir()
        (self.local / ".venv" / "marker").write_text("working", encoding="utf-8")

    def git(self, cwd, *args):
        return subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.test", *args],
            cwd=cwd, check=True, capture_output=True, text=True,
        ).stdout.strip()

    def commit(self, cwd, message):
        self.git(cwd, "add", ".")
        self.git(cwd, "commit", "-m", message)

    def publish(self):
        (self.source / "app.txt").write_text("new", encoding="utf-8")
        self.commit(self.source, "new version")
        self.git(self.source, "push")

    def update(self, root=None):
        with contextlib.redirect_stdout(io.StringIO()):
            result = update_project(root or self.local)
        self.assertEqual("CUSTOM_API_KEY=keep", (self.local / ".env").read_text())
        self.assertEqual("working", (self.local / ".venv" / "marker").read_text())
        return result

    def assert_unchanged(self):
        self.assertEqual(self.old_head, self.git(self.local, "rev-parse", "HEAD"))
        self.assertEqual("old", (self.local / "app.txt").read_text())

    def test_fast_forward_updates_code_and_preserves_runtime(self):
        self.publish()
        self.assertEqual(0, self.update())
        self.assertEqual("new", (self.local / "app.txt").read_text())
        self.assertEqual(self.git(self.source, "rev-parse", "HEAD"), self.git(self.local, "rev-parse", "HEAD"))

    def test_no_update_uses_existing_environment(self):
        self.assertEqual(USE_LOCAL, self.update())
        self.assert_unchanged()

    def test_fetch_failure_preserves_old_version(self):
        self.git(self.local, "remote", "set-url", "origin", str(self.root / "missing.git"))
        self.assertEqual(USE_LOCAL, self.update())
        self.assert_unchanged()

    def test_dirty_and_untracked_files_are_not_overwritten(self):
        self.publish()
        (self.local / "app.txt").write_text("my changes")
        self.assertEqual(USE_LOCAL, self.update())
        self.assertEqual("my changes", (self.local / "app.txt").read_text())
        (self.local / "app.txt").write_text("old")
        (self.local / "notes.txt").write_text("my notes")
        self.assertEqual(USE_LOCAL, self.update())
        self.assert_unchanged()
        self.assertEqual("my notes", (self.local / "notes.txt").read_text())

    def test_diverged_history_is_not_reset(self):
        self.publish()
        (self.local / "local.txt").write_text("local commit")
        self.commit(self.local, "local change")
        local_head = self.git(self.local, "rev-parse", "HEAD")
        self.assertEqual(USE_LOCAL, self.update())
        self.assertEqual(local_head, self.git(self.local, "rev-parse", "HEAD"))
        self.assertEqual("old", (self.local / "app.txt").read_text())

    def test_remote_cannot_overwrite_ignored_config(self):
        (self.source / ".env").write_text("remote config")
        self.git(self.source, "add", "-f", ".env")
        self.commit(self.source, "tracked config")
        self.git(self.source, "push")
        self.assertEqual(USE_LOCAL, self.update())
        self.assert_unchanged()

    def test_detached_head_and_missing_upstream_are_skipped(self):
        self.git(self.local, "checkout", "--detach")
        self.assertEqual(USE_LOCAL, self.update())
        self.git(self.local, "checkout", "main")
        self.git(self.local, "branch", "--unset-upstream")
        self.assertEqual(USE_LOCAL, self.update())
        self.assert_unchanged()

    def test_zip_directory_inside_repository_does_not_update_parent(self):
        nested = self.local / "unzipped"
        nested.mkdir()
        self.publish()
        self.assertEqual(USE_LOCAL, self.update(nested))
        self.assert_unchanged()

    def test_missing_git_and_fetch_timeout_fall_back(self):
        with patch("scripts.update_project.subprocess.run", side_effect=FileNotFoundError):
            self.assertEqual(USE_LOCAL, self.update())
        run = subprocess.run

        def timeout_fetch(args, **kwargs):
            if args[1] == "fetch":
                raise subprocess.TimeoutExpired(args, 60)
            return run(args, **kwargs)

        with patch("scripts.update_project.subprocess.run", side_effect=timeout_fetch):
            self.assertEqual(USE_LOCAL, self.update())
        self.assert_unchanged()


@unittest.skipUnless(sys.platform == "darwin", "验证 macOS 启动入口")
class MacUpdateLauncherTests(unittest.TestCase):
    def test_failed_update_starts_local_python_without_uv(self):
        project = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix="moegal launcher ") as directory:
            root = Path(directory)
            for name in ("start.command", "update.command", ".env.example"):
                shutil.copy(project / name, root / name)
            (root / "scripts").mkdir()
            shutil.copy(project / "scripts/update_project.py", root / "scripts/update_project.py")
            (root / ".env").write_text("CUSTOM_API_KEY=keep")
            python = root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            # 更新逻辑使用真实 Python，启动服务处记录调用，避免下载模型或占用端口。
            python.write_text(
                '#!/bin/bash\nif [ "$1" = "-m" ]; then\n'
                '  printf "%s\\n" "$*" > "$LAUNCH_LOG"\n'
                '  exit 0\nfi\n'
                f'exec {shlex.quote(sys.executable)} "$@"\n'
            )
            python.chmod(0o755)
            result = subprocess.run(
                ["bash", str(root / "update.command")], cwd="/tmp",
                env=dict(os.environ, LAUNCH_LOG=str(root / "launch.log")),
                capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertIn("-m uvicorn app.main:app", (root / "launch.log").read_text())
            self.assertEqual("CUSTOM_API_KEY=keep", (root / ".env").read_text())
            self.assertFalse((root / ".tools").exists())

            # 普通启动也必须复用环境，不运行更新器、uv 或任何下载工具。
            (root / "scripts/update_project.py").write_text("raise RuntimeError('must not update')")
            result = subprocess.run(
                ["bash", str(root / "start.command")], cwd="/tmp",
                env=dict(os.environ, LAUNCH_LOG=str(root / "launch.log")),
                capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(0, result.returncode, result.stdout + result.stderr)
            self.assertFalse((root / ".tools").exists())

    def test_successful_update_explicitly_syncs_dependencies(self):
        project = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix="moegal update launch ") as directory:
            root = Path(directory)
            shutil.copy(project / "update.command", root)
            (root / "scripts").mkdir()
            (root / "scripts/update_project.py").write_text("raise SystemExit(0)")
            (root / "start.command").write_text('printf "%s" "$*" > launch.args\n')
            python = root / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
            result = subprocess.run(["bash", str(root / "update.command")], capture_output=True, timeout=20)
            self.assertEqual(0, result.returncode)
            self.assertEqual("--sync", (root / "launch.args").read_text())


if __name__ == "__main__":
    unittest.main()
