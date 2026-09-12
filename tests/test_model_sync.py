import os
from pathlib import Path
import runpy
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Event, Thread
import unittest
from unittest.mock import patch

from app.core import model_sync


class ModelSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = [
            "comic-text-and-bubble-detector/config.json",
            "manga-ocr-base/config.json",
        ]
        self.manifest = self.root / "manifest.txt"
        self.manifest.write_text("\n".join(self.files))
        for name, value in {
            "MODELS_DIR": self.root / "models",
            "SYNC_LOCK_PATH": self.root / "models/.sync.lock",
            "MODELS_MANIFEST_PATH": self.manifest,
            "HF_ENDPOINT": "https://primary.example",
            "HF_FALLBACK_ENDPOINT": "https://backup.example",
        }.items():
            patcher = patch.object(model_sync, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env_patch = patch.dict(os.environ, {"MODEL_DOWNLOAD_WORKERS": "2"})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def write_download(self, **kwargs):
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("downloaded")
        return str(path)

    def test_metadata_connection_interruption_is_retried(self):
        class Handler(BaseHTTPRequestHandler):
            attempts = 0

            def do_HEAD(self):
                Handler.attempts += 1
                if Handler.attempts == 1:
                    self.close_connection = True
                    return
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with model_sync._download_session() as session:
                session.trust_env = False  # 本地测试不依赖机器的代理设置。
                response = session.head(f"http://127.0.0.1:{server.server_port}/model", timeout=3)
                self.assertEqual(200, response.status_code)
                self.assertTrue(session.verify)
            self.assertEqual(2, Handler.attempts)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_files_download_concurrently(self):
        # 两个任务必须同时进入下载；串行实现会超时，不能仅靠耗时猜测并发。
        barrier = Barrier(2)

        def download(**kwargs):
            barrier.wait(timeout=5)
            return self.write_download(**kwargs)

        with patch.object(model_sync, "hf_hub_download", side_effect=download) as hub:
            model_sync.ensure_models_ready()
        self.assertEqual(2, hub.call_count)
        for filename in self.files:
            self.assertTrue((model_sync.MODELS_DIR / filename).is_file())

    def test_failed_primary_is_skipped_for_later_files(self):
        def download(**kwargs):
            if kwargs["endpoint"] == model_sync.HF_ENDPOINT:
                raise OSError("primary unavailable")
            return self.write_download(**kwargs)

        fallback = Event()
        with patch.object(model_sync, "hf_hub_download", side_effect=download) as hub:
            for filename in self.files:
                model_sync._download_single_file(filename, fallback)
        self.assertEqual(
            [model_sync.HF_ENDPOINT, model_sync.HF_FALLBACK_ENDPOINT, model_sync.HF_FALLBACK_ENDPOINT],
            [call.kwargs["endpoint"] for call in hub.call_args_list],
        )

    def test_complete_models_do_not_contact_network(self):
        with patch.object(model_sync, "hf_hub_download", side_effect=self.write_download):
            model_sync.ensure_models_ready()
        with patch.object(model_sync, "hf_hub_download") as hub:
            model_sync.ensure_models_ready()
        hub.assert_not_called()

    def test_retry_only_downloads_missing_files(self):
        def download(**kwargs):
            if kwargs["repo_id"] == model_sync.MANGA_OCR_REPO_ID:
                raise OSError("download interrupted")
            return self.write_download(**kwargs)

        with patch.dict(os.environ, {"MODEL_DOWNLOAD_WORKERS": "1"}):
            with patch.object(model_sync, "hf_hub_download", side_effect=download):
                with self.assertRaises(OSError):
                    model_sync.ensure_models_ready()
        self.assertTrue((model_sync.MODELS_DIR / self.files[0]).exists())
        with patch.object(model_sync, "hf_hub_download", side_effect=self.write_download) as hub:
            model_sync.ensure_models_ready()
        hub.assert_called_once()
        self.assertEqual(model_sync.MANGA_OCR_REPO_ID, hub.call_args.kwargs["repo_id"])
        # 新一轮下载重新尝试主源，不把前一次的失败状态永久保存。
        self.assertEqual(model_sync.HF_ENDPOINT, hub.call_args.kwargs["endpoint"])

    def test_no_duplicate_retry_without_distinct_fallback(self):
        for endpoint in ("", model_sync.HF_ENDPOINT):
            with self.subTest(endpoint=endpoint):
                with patch.object(model_sync, "HF_FALLBACK_ENDPOINT", endpoint):
                    with patch.object(model_sync, "hf_hub_download", side_effect=OSError) as hub:
                        with self.assertRaises(OSError):
                            model_sync._download_single_file(self.files[0], Event())
                self.assertEqual(1, hub.call_count)

    def test_dotenv_config_is_read_before_endpoint_constants(self):
        (self.root / ".env").write_text("HF_ENDPOINT=https://dotenv.example\n")
        env = {key: value for key, value in os.environ.items() if key != "HF_ENDPOINT"}
        with patch("app.core.paths.PROJECT_ROOT", self.root):
            with patch.dict(os.environ, env, clear=True):
                values = runpy.run_path(model_sync.__file__)
                self.assertEqual("https://dotenv.example", values["HF_ENDPOINT"])
            with patch.dict(os.environ, {"HF_ENDPOINT": "https://shell.example"}):
                values = runpy.run_path(model_sync.__file__)
                self.assertEqual("https://shell.example", values["HF_ENDPOINT"])


if __name__ == "__main__":
    unittest.main()
