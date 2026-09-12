import sys
import types
import unittest
from unittest.mock import Mock, patch

from app.api.routes.update_conf import UpdateItem, update_conf
from app.core.custom_conf import CustomConf, custom_conf


class CustomConfGpuTests(unittest.TestCase):
    def test_use_gpu_accepts_boolean(self):
        conf = CustomConf(use_gpu=False)

        result = conf.update_conf("use_gpu", True)

        self.assertTrue(conf.use_gpu)
        self.assertEqual({"use_gpu": True, "status": "success"}, result)

    def test_use_gpu_rejects_non_boolean_values(self):
        conf = CustomConf(use_gpu=False)

        for value in ("true", 1, 0.0, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "use_gpu 必须是布尔值"):
                    conf.update_conf("use_gpu", value)

    def test_update_item_preserves_boolean_type(self):
        item = UpdateItem(attr="use_gpu", v=True)

        self.assertIs(item.v, True)


class UpdateConfGpuTests(unittest.TestCase):
    def setUp(self):
        self.old_use_gpu = custom_conf.use_gpu

    def tearDown(self):
        custom_conf.use_gpu = self.old_use_gpu

    def test_device_change_resets_models_and_returns_gpu_status(self):
        custom_conf.use_gpu = False
        reset_models = Mock()
        fake_ocr = types.ModuleType("app.services.ocr")
        fake_ocr.reset_models = reset_models
        fake_ocr.get_gpu_status = lambda: {
            "requested": custom_conf.use_gpu,
            "device": "gpu" if custom_conf.use_gpu else "cpu",
        }

        with patch.dict(sys.modules, {"app.services.ocr": fake_ocr}):
            payload = update_conf(UpdateItem(attr="use_gpu", v=True))

        reset_models.assert_called_once_with()
        self.assertTrue(payload["use_gpu"])
        self.assertEqual("gpu", payload["gpu_status"]["device"])

    def test_unchanged_device_does_not_reset_models(self):
        custom_conf.use_gpu = False
        reset_models = Mock()
        fake_ocr = types.ModuleType("app.services.ocr")
        fake_ocr.reset_models = reset_models
        fake_ocr.get_gpu_status = lambda: {"requested": False, "device": "cpu"}

        with patch.dict(sys.modules, {"app.services.ocr": fake_ocr}):
            update_conf(UpdateItem(attr="use_gpu", v=False))

        reset_models.assert_not_called()


if __name__ == "__main__":
    unittest.main()
