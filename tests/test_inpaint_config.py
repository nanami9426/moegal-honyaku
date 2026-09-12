import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
import torch

from app.api.routes import update_conf as conf_routes
from app.core.custom_conf import (
    CustomConf,
    DEFAULT_INPAINT_BACKEND,
    _inpaint_backend_from_env,
    custom_conf,
)
from app.services import inpainting


class CustomConfInpaintTests(unittest.TestCase):
    def test_startup_default_uses_normalized_env_and_falls_back_for_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual("opencv", _inpaint_backend_from_env())
        for value, expected in (("lama", "lama"), (" LAMA ", "lama"), ("opencv", "opencv"), ("", "opencv"), ("unknown", "opencv")):
            with self.subTest(value=value), patch.dict(os.environ, {"INPAINT_BACKEND": value}):
                self.assertEqual(expected, _inpaint_backend_from_env())

    def test_new_config_uses_startup_default_even_if_environment_changes(self):
        other = "lama" if DEFAULT_INPAINT_BACKEND == "opencv" else "opencv"
        with patch.dict(os.environ, {"INPAINT_BACKEND": other}):
            self.assertEqual(DEFAULT_INPAINT_BACKEND, CustomConf().inpaint_backend)

    def test_update_accepts_only_supported_backends(self):
        conf = CustomConf(inpaint_backend="opencv")
        for backend in ("lama", "opencv"):
            self.assertEqual({"inpaint_backend": backend, "status": "success"}, conf.update_conf("inpaint_backend", backend))
        for value in ("auto", "LAMA", "", True, 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "inpaint_backend 必须"):
                conf.update_conf("inpaint_backend", value)
        self.assertEqual("opencv", conf.inpaint_backend)


class InpaintConfigApiTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model_path = Path(temp.name) / "big-lama.pt"
        self.model_path.write_bytes(b"mock model")

        self.reset_models = Mock()
        fake_ocr = types.ModuleType("app.services.ocr")
        fake_ocr.reset_models = self.reset_models
        fake_ocr.get_gpu_status = lambda: {"requested": custom_conf.use_gpu, "device": "cpu"}
        patches = [
            patch.dict(custom_conf.__dict__, {
                "translate_api_type": "custom", "translate_mode": "parallel",
                "use_gpu": False, "inpaint_backend": "opencv",
            }, clear=True),
            patch.dict(os.environ, {"INPAINT_BACKEND": "opencv", "INPAINT_LAMA_MODEL_PATH": str(self.model_path)}),
            patch.dict(sys.modules, {"app.services.ocr": fake_ocr}),
            patch.object(conf_routes, "get_provider_status", return_value={}),
            patch.object(inpainting, "_model", None),
            patch.object(inpainting, "_model_file", None),
            patch.object(inpainting, "_model_device", None),
            patch.object(inpainting, "_failed_devices", set()),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        # 只挂载配置路由，验证真实 HTTP 请求且不启动 OCR 模型预热。
        app = FastAPI()
        app.include_router(conf_routes.update_conf_router)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_query_and_options_expose_current_backend(self):
        response = self.client.get("/conf/query")
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("opencv", payload["inpaint_backend"])
        self.assertEqual("opencv", payload["inpaint_status"]["requested"])
        self.assertTrue(payload["inpaint_status"]["available"])
        options = self.client.get("/conf/options")
        self.assertEqual(200, options.status_code)
        self.assertEqual(["opencv", "lama"], options.json()["inpaint_backend"])

    def test_update_and_query_take_effect_without_loading_or_resetting_models(self):
        with patch.object(torch.jit, "load") as load:
            response = self.client.post("/conf/update", json={"attr": "inpaint_backend", "v": "lama"})
            self.assertEqual(200, response.status_code)
            payload = response.json()
            self.assertEqual("lama", payload["inpaint_backend"])
            self.assertEqual("lama", payload["inpaint_status"]["effective_backend"])
            self.assertFalse(payload["inpaint_status"]["model_loaded"])
            self.assertEqual(payload, self.client.get("/conf/query").json())
            self.assertEqual("opencv", os.environ["INPAINT_BACKEND"])
            response = self.client.post("/conf/update", json={"attr": "inpaint_backend", "v": "opencv"})
            self.assertEqual(200, response.status_code)
            self.assertEqual("opencv", response.json()["inpaint_status"]["effective_backend"])
        load.assert_not_called()
        self.reset_models.assert_not_called()

    def test_invalid_update_is_rejected_without_changing_selection(self):
        for value in ("auto", "LAMA", "", True, 1, None):
            with self.subTest(value=value):
                response = self.client.post("/conf/update", json={"attr": "inpaint_backend", "v": value})
                self.assertEqual(400, response.status_code)
                self.assertIn("inpaint_backend 必须", response.json()["detail"])
                self.assertEqual("opencv", custom_conf.inpaint_backend)

    def test_init_restores_startup_default_and_returns_status(self):
        for default in ("opencv", "lama"):
            with self.subTest(default=default), patch.object(conf_routes, "DEFAULT_INPAINT_BACKEND", default):
                custom_conf.inpaint_backend = "lama" if default == "opencv" else "opencv"
                response = self.client.post("/conf/init")
                self.assertEqual(200, response.status_code)
                self.assertEqual(default, response.json()["inpaint_backend"])
                self.assertEqual(default, response.json()["inpaint_status"]["requested"])

    def test_missing_model_preserves_choice_and_exposes_fallback(self):
        self.model_path.unlink()
        with patch.object(torch.jit, "load") as load:
            response = self.client.post("/conf/update", json={"attr": "inpaint_backend", "v": "lama"})
            self.assertEqual(200, response.status_code)
            payload = response.json()
            self.assertEqual("lama", payload["inpaint_backend"])
            self.assertEqual("lama", payload["inpaint_status"]["requested"])
            self.assertEqual("opencv", payload["inpaint_status"]["effective_backend"])
            self.assertFalse(payload["inpaint_status"]["available"])
            self.assertIn("OpenCV", payload["inpaint_status"]["message"])
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
