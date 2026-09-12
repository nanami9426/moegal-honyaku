import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from app.core.custom_conf import custom_conf
from app.services import inpainting


class ColorModel(torch.nn.Module):
    """用固定 RGB 结果验证预处理和回贴，不依赖联网或真实大模型。"""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, image, mask):
        self.inputs.append((image.clone(), mask.clone()))
        color = torch.tensor([0.9, 0.2, 0.1], dtype=image.dtype).view(1, 3, 1, 1)
        return color.expand_as(image)


class InpaintingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model_path = self.root / "big-lama.pt"
        self.model_path.write_bytes(b"mock model")
        self.model = ColorModel()
        patches = [
            patch.dict(os.environ, {"INPAINT_LAMA_MODEL_PATH": str(self.model_path)}),
            patch.object(custom_conf, "inpaint_backend", "lama"),
            patch.object(custom_conf, "use_gpu", False),
            patch.object(inpainting, "_model", None),
            patch.object(inpainting, "_model_file", None),
            patch.object(inpainting, "_model_device", None),
            patch.object(inpainting, "_failed_devices", set()),
            patch.object(inpainting, "_warnings", set()),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.image = np.full((400, 500, 3), (7, 61, 211), dtype=np.uint8)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.mask[180:186, 200:209] = 255

    def assert_preserved(self, result, original=None, mask=None):
        original = self.image if original is None else original
        mask = self.mask if mask is None else mask
        self.assertEqual(original.shape, result.shape)
        self.assertEqual(np.uint8, result.dtype)
        np.testing.assert_array_equal(result[mask == 0], original[mask == 0])

    def test_empty_mask_does_not_load_or_modify_input(self):
        empty = np.zeros_like(self.mask)
        with patch.object(torch.jit, "load") as load:
            result = inpainting.inpaint_image(self.image, empty)
        load.assert_not_called()
        np.testing.assert_array_equal(result, self.image)

    def test_default_opencv_backend_does_not_load_lama(self):
        damaged = self.image.copy()
        damaged[self.mask != 0] = 0
        with patch.object(custom_conf, "inpaint_backend", "opencv"), patch.object(torch.jit, "load") as load:
            result = inpainting.inpaint_image(damaged, self.mask)
        load.assert_not_called()
        self.assert_preserved(result, damaged)
        self.assertLess(np.mean(np.abs(result.astype(float) - self.image.astype(float))), 0.1)

    def test_runtime_switch_uses_selected_backend_and_reuses_lama_model(self):
        damaged = self.image.copy()
        damaged[self.mask != 0] = 0
        # 环境变量保持 OpenCV，运行时选择仍应立即生效。
        with patch.dict(os.environ, {"INPAINT_BACKEND": "opencv"}), patch.object(torch.jit, "load", return_value=self.model) as load:
            custom_conf.update_conf("inpaint_backend", "lama")
            first = inpainting.inpaint_image(damaged, self.mask)
            custom_conf.update_conf("inpaint_backend", "opencv")
            opencv = inpainting.inpaint_image(damaged, self.mask)
            self.assertFalse(inpainting.get_inpaint_status()["model_loaded"])
            custom_conf.update_conf("inpaint_backend", "lama")
            second = inpainting.inpaint_image(damaged, self.mask)
            self.assertEqual("opencv", os.environ["INPAINT_BACKEND"])
        load.assert_called_once()
        self.assertEqual(2, len(self.model.inputs))
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first[self.mask != 0], opencv[self.mask != 0]))
        self.assert_preserved(opencv, damaged)

    def test_status_checks_local_file_without_loading_model(self):
        with patch.object(torch.jit, "load") as load:
            status = inpainting.get_inpaint_status()
        load.assert_not_called()
        self.assertEqual("lama", status["requested"])
        self.assertEqual("lama", status["effective_backend"])
        self.assertTrue(status["available"])
        self.assertFalse(status["model_loaded"])
        self.assertIn("首次", status["message"])

    def test_status_reports_missing_model_fallback_and_recovers_when_created(self):
        self.model_path.unlink()
        with patch.object(torch.jit, "load") as load:
            missing = inpainting.get_inpaint_status()
            self.model_path.mkdir()
            directory = inpainting.get_inpaint_status()
            self.model_path.rmdir()
            self.model_path.write_bytes(b"new model")
            ready = inpainting.get_inpaint_status()
        load.assert_not_called()
        for status in (missing, directory):
            self.assertEqual("lama", status["requested"])
            self.assertEqual("opencv", status["effective_backend"])
            self.assertFalse(status["available"])
            self.assertFalse(status["model_loaded"])
            self.assertIn("OpenCV", status["message"])
        self.assertTrue(ready["available"])
        self.assertEqual("lama", ready["effective_backend"])

    def test_status_reflects_failure_and_model_replacement_without_loading(self):
        with patch.object(torch.jit, "load", side_effect=RuntimeError("invalid model")) as load, patch.object(inpainting.logger, "warning"):
            inpainting.inpaint_image(self.image, self.mask)
            failed = inpainting.get_inpaint_status()
            self.model_path.write_bytes(b"replacement model with different size")
            ready = inpainting.get_inpaint_status()
        load.assert_called_once()
        self.assertEqual("opencv", failed["effective_backend"])
        self.assertFalse(failed["available"])
        self.assertIn("失败", failed["message"])
        self.assertTrue(ready["available"])
        self.assertFalse(ready["model_loaded"])
        self.assertEqual("lama", ready["effective_backend"])

    def test_rgb_conversion_context_crop_and_padding(self):
        original = self.image.copy()
        with patch.object(torch.jit, "load", return_value=self.model) as load:
            result = inpainting.inpaint_image(self.image, self.mask)
        load.assert_called_once_with(str(self.model_path), map_location="cpu")
        image_tensor, mask_tensor = self.model.inputs[0]
        self.assertEqual((1, 3, 136, 144), tuple(image_tensor.shape))
        self.assertEqual((1, 1, 136, 144), tuple(mask_tensor.shape))
        np.testing.assert_allclose(image_tensor[0, :, 0, 0].numpy(), np.array([211, 61, 7]) / 255)
        self.assertEqual(torch.float32, mask_tensor.dtype)
        self.assertEqual({0.0, 1.0}, set(mask_tensor.unique().tolist()))
        np.testing.assert_array_equal(result[self.mask != 0], np.tile([26, 51, 230], (54, 1)))
        np.testing.assert_array_equal(self.image, original)
        self.assert_preserved(result)

    def test_distant_regions_keep_local_resolution_and_share_cached_model(self):
        image = np.zeros((600, 900, 3), dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[50:60, 50:70] = 255
        mask[450:460, 750:770] = 255
        with patch.object(torch.jit, "load", return_value=self.model) as load:
            result = inpainting.inpaint_image(image, mask)
            inpainting.inpaint_image(image, mask)
        load.assert_called_once()
        self.assertEqual(4, len(self.model.inputs))
        self.assertTrue(all(max(tensor.shape[-2:]) < 200 for tensor, _ in self.model.inputs))
        self.assert_preserved(result, image, mask)

    def test_large_region_scales_back_and_does_not_lose_thin_mask(self):
        self.mask[:] = 0
        self.mask[10:390, 201] = 255
        with patch.object(torch.jit, "load", return_value=self.model), patch.object(inpainting, "LAMA_MAX_SIZE", 64):
            result = inpainting.inpaint_image(self.image, self.mask)
        image_tensor, mask_tensor = self.model.inputs[0]
        self.assertLessEqual(max(image_tensor.shape[-2:]), 64)
        self.assertGreater(int(mask_tensor.sum()), 0)
        np.testing.assert_array_equal(result[self.mask != 0], np.tile([26, 51, 230], (380, 1)))
        self.assert_preserved(result)

    def test_missing_model_falls_back_and_logs_once_then_recovers_when_created(self):
        self.model_path.unlink()
        with patch.object(torch.jit, "load", return_value=self.model) as load, patch.object(inpainting.logger, "warning") as warning:
            result = inpainting.inpaint_image(self.image, self.mask)
            inpainting.inpaint_image(self.image, self.mask)
            load.assert_not_called()
            warning.assert_called_once()
            self.model_path.write_bytes(b"model now available")
            inpainting.inpaint_image(self.image, self.mask)
        load.assert_called_once()
        self.assert_preserved(result)

    def test_relative_model_path_uses_project_root(self):
        with patch.dict(os.environ, {"INPAINT_LAMA_MODEL_PATH": "big-lama.pt"}), patch.object(inpainting, "PROJECT_ROOT", self.root):
            with patch.object(torch.jit, "load", return_value=self.model) as load:
                inpainting.inpaint_image(self.image, self.mask)
        load.assert_called_once_with(str(self.model_path), map_location="cpu")

    def test_available_cuda_is_ignored_when_gpu_is_disabled(self):
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.jit, "load", return_value=self.model) as load:
            inpainting.inpaint_image(self.image, self.mask)
        self.assertEqual("cpu", load.call_args.kwargs["map_location"])

    def test_cuda_load_failure_retries_cpu_and_caches_fallback(self):
        def load_model(path, map_location):
            if map_location == "cuda":
                raise RuntimeError("CUDA unavailable")
            return self.model

        with patch.object(custom_conf, "use_gpu", True), patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.jit, "load", side_effect=load_model) as load, patch.object(inpainting.logger, "warning") as warning:
                result = inpainting.inpaint_image(self.image, self.mask)
                inpainting.inpaint_image(self.image, self.mask)
                status = inpainting.get_inpaint_status()
        self.assertEqual(["cuda", "cpu"], [call.kwargs["map_location"] for call in load.call_args_list])
        warning.assert_called_once()
        self.assert_preserved(result)
        self.assertEqual("lama", status["effective_backend"])
        self.assertTrue(status["available"])
        self.assertTrue(status["model_loaded"])
        self.assertIn("CPU", status["message"])

    def test_cuda_inference_failure_retries_cpu(self):
        run_lama = inpainting._run_lama
        devices = []

        def run(image, mask, model, device):
            devices.append(device)
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            return run_lama(image, mask, model, device)

        with patch.object(custom_conf, "use_gpu", True), patch.object(torch.cuda, "is_available", return_value=True):
            with patch.object(torch.jit, "load", return_value=self.model), patch.object(inpainting, "_run_lama", side_effect=run):
                with patch.object(inpainting.logger, "warning"):
                    result = inpainting.inpaint_image(self.image, self.mask)
        self.assertEqual(["cuda", "cpu"], devices)
        self.assert_preserved(result)

    def test_failed_cpu_model_is_not_reloaded_until_file_changes(self):
        with patch.object(torch.jit, "load", side_effect=RuntimeError("invalid model")) as load, patch.object(inpainting.logger, "warning") as warning:
            result = inpainting.inpaint_image(self.image, self.mask)
            inpainting.inpaint_image(self.image, self.mask)
            load.assert_called_once()
            warning.assert_called_once()
            self.model_path.write_bytes(b"replaced model with different size")
            inpainting.inpaint_image(self.image, self.mask)
            self.assertEqual(2, load.call_count)
        self.assert_preserved(result)

    def test_invalid_model_output_falls_back_without_changing_unmasked_pixels(self):
        class InvalidModel(torch.nn.Module):
            def forward(self, image, mask):
                return torch.full_like(image, float("nan"))

        with patch.object(torch.jit, "load", return_value=InvalidModel()), patch.object(inpainting.logger, "warning") as warning:
            result = inpainting.inpaint_image(self.image, self.mask)
        warning.assert_called_once()
        expected = cv2.inpaint(self.image, self.mask, 3, cv2.INPAINT_TELEA)
        np.testing.assert_array_equal(expected, result)
        self.assert_preserved(result)


if __name__ == "__main__":
    unittest.main()
