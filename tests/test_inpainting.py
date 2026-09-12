import unittest

import numpy as np

from app.services.inpainting import inpaint_image


class InpaintingTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((400, 500, 3), (7, 61, 211), dtype=np.uint8)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.mask[180:186, 200:209] = 255

    def test_empty_mask_leaves_input_unchanged(self):
        result = inpaint_image(self.image, np.zeros_like(self.mask))
        np.testing.assert_array_equal(result, self.image)
        self.assertFalse(np.shares_memory(result, self.image))

    def test_repairs_masked_pixels_without_changing_surroundings(self):
        damaged = self.image.copy()
        damaged[self.mask != 0] = 0
        original = damaged.copy()

        result = inpaint_image(damaged, self.mask)

        self.assertEqual(self.image.shape, result.shape)
        self.assertEqual(np.uint8, result.dtype)
        np.testing.assert_array_equal(damaged, original)
        np.testing.assert_array_equal(result[self.mask == 0], damaged[self.mask == 0])
        # 只检查受损区域的恢复误差，避免大面积完好背景稀释残留文字的影响。
        error = np.abs(result.astype(float) - self.image.astype(float))[self.mask != 0]
        self.assertLess(float(error.mean()), 3)

    def test_nonzero_mask_values_select_the_same_region(self):
        damaged = self.image.copy()
        damaged[self.mask != 0] = 0
        expected = inpaint_image(damaged, self.mask)
        for mask in (self.mask != 0, (self.mask != 0).astype(np.uint8)):
            with self.subTest(dtype=mask.dtype):
                np.testing.assert_array_equal(inpaint_image(damaged, mask), expected)


if __name__ == "__main__":
    unittest.main()
