from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.services.text_erasure import erase_text_regions


FONT_PATH = Path(__file__).resolve().parents[1] / "assets/fonts/LXGWWenKai-Regular.ttf"


def make_text_sample(background, foreground, text="文字。!", font_size=30):
    """用独立的透明文字层生成样本，保留擦除前的干净背景作为真值。"""
    alpha_image = Image.new("L", (background.shape[1], background.shape[0]))
    draw = ImageDraw.Draw(alpha_image)
    font = ImageFont.truetype(str(FONT_PATH), font_size)
    draw.text((44, 35), text, font=font, fill=255, anchor="lt")
    alpha = np.asarray(alpha_image)
    opacity = alpha[..., None].astype(np.float32) / 255
    image = np.rint(
        background * (1 - opacity) + np.asarray(foreground) * opacity
    ).astype(np.uint8)
    ys, xs = np.nonzero(alpha)
    bbox = (int(xs.min()) - 5, int(ys.min()) - 5,
            int(xs.max()) + 6, int(ys.max()) + 6)
    return image, alpha, bbox


def make_outlined_text_sample(
    background, text, position, font_size=42, stroke_width=3, foreground=(105, 32, 34),
):
    """分别保存彩色字芯和白描边真值，防止只擦字芯的掩码误通过检查。"""
    size = (background.shape[1], background.shape[0])
    body_image, outline_image = Image.new("L", size), Image.new("L", size)
    body_draw, outline_draw = ImageDraw.Draw(body_image), ImageDraw.Draw(outline_image)
    font = ImageFont.truetype(str(FONT_PATH), font_size)
    x, y = position
    for row, content in enumerate(text.split("\n")):
        origin = (x, y + row * (font_size + 7))
        body_draw.text(origin, content, font=font, fill=255, anchor="lt")
        outline_draw.text(origin, content, font=font, fill=255, anchor="lt",
                          stroke_width=stroke_width, stroke_fill=255)
    body, outline = np.asarray(body_image), np.asarray(outline_image)
    opacity = outline[..., None].astype(np.float32) / 255
    image = background * (1 - opacity) + 255 * opacity
    opacity = body[..., None].astype(np.float32) / 255
    image = np.rint(image * (1 - opacity) + np.asarray(foreground) * opacity).astype(np.uint8)
    ys, xs = np.nonzero(outline)
    bbox = (int(xs.min()) - 4, int(ys.min()) - 4,
            int(xs.max()) + 5, int(ys.max()) + 5)
    return image, body, outline, bbox


class TextErasureTests(unittest.TestCase):
    def assert_output_contract(self, image, original, erased, mask):
        np.testing.assert_array_equal(image, original)
        self.assertEqual(erased.shape, image.shape)
        self.assertEqual(erased.dtype, np.uint8)
        self.assertEqual(mask.shape, image.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(np.all((mask == 0) | (mask == 255)))
        # 修复后只允许修改掩码覆盖的像素。
        np.testing.assert_array_equal(erased[mask == 0], image[mask == 0])

    def test_removes_antialiased_text_on_light_dark_and_colored_backgrounds(self):
        colors = (
            ((255, 255, 255), (20, 20, 20)),
            ((25, 32, 40), (245, 245, 245)),
            # 两种颜色的灰度几乎一致，不能只靠亮度识别文字。
            ((40, 180, 160), (210, 175, 105)),
        )
        for background_color, foreground_color in colors:
            with self.subTest(background=background_color, foreground=foreground_color):
                background = np.full((105, 180, 3), background_color, dtype=np.uint8)
                image, alpha, bbox = make_text_sample(background, foreground_color)
                original = image.copy()

                erased, mask = erase_text_regions(image, [bbox])

                self.assert_output_contract(image, original, erased, mask)
                # 连抗锯齿边缘一起检查，避免只擦掉笔画中心却留下灰边。
                error = np.abs(erased.astype(np.int16) - background.astype(np.int16))
                text_error = error[alpha > 0]
                self.assertLessEqual(float(text_error.mean()), 2)
                self.assertLessEqual(float(np.percentile(text_error, 99)), 6)

    def test_keeps_small_punctuation_in_the_erasure_mask(self):
        background = np.full((105, 180, 3), 250, dtype=np.uint8)
        image, _, bbox = make_text_sample(background, (10, 10, 10), text="文字")
        punctuation = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.circle(punctuation, (112, 61), 1, 255, -1, lineType=cv2.LINE_AA)
        opacity = punctuation[..., None].astype(np.float32) / 255
        image = np.rint(image * (1 - opacity) + 10 * opacity).astype(np.uint8)
        bbox = (bbox[0], bbox[1], 121, max(bbox[3], 67))

        erased, mask = erase_text_regions(image, [bbox])

        error = np.abs(erased.astype(np.int16) - background.astype(np.int16))
        self.assertLessEqual(float(error[punctuation > 0].mean()), 2)
        self.assertTrue(np.all(mask[punctuation >= 128] == 255))

    def test_preserves_gradient_instead_of_replacing_it_with_one_color(self):
        x = np.arange(180, dtype=np.float32)
        row = np.stack((65 + x * 0.7, 110 + x * 0.4, 205 - x * 0.3), axis=-1)
        background = np.broadcast_to(row[None, ...], (105, 180, 3)).astype(np.uint8).copy()
        image, alpha, bbox = make_text_sample(background, (12, 18, 24))
        original = image.copy()

        erased, mask = erase_text_regions(image, [bbox])

        self.assert_output_contract(image, original, erased, mask)
        before = np.abs(image.astype(np.int16) - background.astype(np.int16))[alpha > 0]
        after = np.abs(erased.astype(np.int16) - background.astype(np.int16))[alpha > 0]
        self.assertLess(float(after.mean()), float(before.mean()) * 0.15)
        self.assertLess(float(after.mean()), 6)
        # 没有文字的渐变仍应保留，不能把整个检测矩形涂成一种颜色。
        away_from_text = cv2.dilate((alpha > 0).astype(np.uint8), np.ones((13, 13), np.uint8)) == 0
        np.testing.assert_array_equal(erased[away_from_text], background[away_from_text])

    def test_erases_white_outlines_of_colored_text_on_a_gradient(self):
        yy, xx = np.indices((250, 200))
        background = np.stack(
            (180 + xx * 0.15, 150 + yy * 0.25, 210 - yy * 0.15), axis=2
        ).astype(np.uint8)
        image, body, outline, bbox = make_outlined_text_sample(
            background, "あ\nっ\nた\n？", (72, 30)
        )
        original = image.copy()

        erased, mask = erase_text_regions(image, [bbox])

        self.assert_output_contract(image, original, erased, mask)
        outline_only = (outline >= 192) & (body == 0)
        self.assertGreaterEqual(float(np.mean(mask[body >= 192] > 0)), 0.99)
        self.assertGreaterEqual(float(np.mean(mask[outline_only] > 0)), 0.99)
        # 少量白边残留就会被修复算法带回字芯，必须同时检查擦后颜色误差。
        error = np.abs(erased.astype(np.int16) - background.astype(np.int16))
        self.assertLess(float(error[body > 0].mean()), 10)
        self.assertLess(float(error[outline_only].mean()), 10)

    def test_preserves_a_shallow_gradient_behind_outlined_text(self):
        yy, xx = np.indices((250, 200))
        background = np.stack(
            (255 - yy * 16 / 249, 255 - yy * 16 * 0.8 / 249,
             np.full_like(xx, 255)), axis=2,
        ).astype(np.uint8)
        image, body, outline, bbox = make_outlined_text_sample(
            background, "あ\nっ\nた\n？", (72, 30),
        )
        original = image.copy()

        erased, mask = erase_text_regions(image, [bbox])

        self.assert_output_contract(image, original, erased, mask)
        # 浅渐变误刷成单色时平均误差仍很小，还需检查上下行原有的明暗变化。
        upper_text = (body > 0) & (yy < 100)
        lower_text = (body > 0) & (yy > 170)
        contrast = float(erased[upper_text, 0].mean() - erased[lower_text, 0].mean())
        self.assertGreater(contrast, 5)
        error = np.abs(erased.astype(np.int16) - background.astype(np.int16))
        self.assertLess(float(error[outline > 0].mean()), 3)
        self.assertLessEqual(float(np.percentile(error[outline > 0], 99)), 6)

    def test_erases_outlined_text_touching_a_dark_background_line(self):
        background = np.full((140, 180, 3), 180, dtype=np.uint8)
        cv2.line(background, (0, 60), (179, 60), (90, 90, 90), 1)
        image, body, outline, bbox = make_outlined_text_sample(
            background, "あ？", (50, 40), font_size=44
        )
        original = image.copy()
        self.assertTrue(np.any(outline[60] >= 192))

        erased, mask = erase_text_regions(image, [bbox])

        self.assert_output_contract(image, original, erased, mask)
        outline_only = (outline >= 192) & (body == 0)
        # 明暗响应不能先合并成触边连通域，否则白边会把字芯接到背景线并一起保护。
        self.assertGreaterEqual(float(np.mean(mask[body >= 192] > 0)), 0.99)
        self.assertGreaterEqual(float(np.mean(mask[outline_only] > 0)), 0.99)
        distance = cv2.distanceTransform(
            (outline == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        far_from_text = distance > 4
        self.assertTrue(np.any(far_from_text[60]))
        np.testing.assert_array_equal(erased[far_from_text], background[far_from_text])
        error = np.abs(erased.astype(np.int16) - background.astype(np.int16))
        self.assertLess(float(error[body > 0].mean()), 10)
        self.assertLess(float(error[outline_only].mean()), 10)

    def test_erases_dense_small_outlined_text_without_black_inpainting_spots(self):
        background = np.full((120, 170, 3), (254, 234, 247), dtype=np.uint8)
        image, body, outline, bbox = make_outlined_text_sample(
            background, "着開続く\nた拓けれ\nらをてる？", (30, 20),
            font_size=19, stroke_width=2, foreground=(10, 10, 10),
        )
        original = image.copy()

        erased, mask = erase_text_regions(image, [bbox])

        self.assert_output_contract(image, original, erased, mask)
        # 密集小字的内部笔画未必单独被白色包围，漏掉它们会让修复算法扩散黑点。
        self.assertGreaterEqual(float(np.mean(mask[body >= 192] > 0)), 0.99)
        self.assertFalse(np.any(erased[outline > 0].max(axis=1) < 100))
        error = np.abs(erased.astype(np.int16) - background.astype(np.int16))
        self.assertLess(float(error[outline > 0].mean()), 5)

    def test_preserves_repeated_background_lines_under_outlined_text(self):
        background = np.full((140, 180, 3), (240, 220, 250), dtype=np.uint8)
        lines = np.zeros(background.shape[:2], dtype=np.uint8)
        for x in range(0, background.shape[1], 20):
            lines[:, x:x + 6] = 255
        background[lines > 0] = 30
        image, body, outline, bbox = make_outlined_text_sample(
            background, "あ？", (40, 45), font_size=30,
            stroke_width=2, foreground=(10, 10, 10),
        )
        original = image.copy()

        erased, mask = erase_text_regions(image, [bbox])

        self.assert_output_contract(image, original, erased, mask)
        self.assertTrue(np.all(mask[body >= 192] == 255))
        # 多条背景线各自很细，但合起来不是气泡边框，不能全部排除后刷成粉底。
        covered_lines = (lines > 0) & (outline >= 192)
        self.assertGreater(int(np.count_nonzero(covered_lines)), 100)
        error = np.abs(erased.astype(np.int16) - background.astype(np.int16))
        # OpenCV 可近似接续被文字遮住的线条；整片填底色会产生超过 200 的误差。
        self.assertLess(float(error[covered_lines].mean()), 100)
        far_from_text = cv2.distanceTransform(
            (outline == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        ) > 4
        np.testing.assert_array_equal(erased[far_from_text], background[far_from_text])

    def test_erases_outlined_text_near_a_border_without_copying_its_black_color(self):
        background = np.full((130, 190, 3), (237, 235, 255), dtype=np.uint8)
        border = np.zeros(background.shape[:2], dtype=np.uint8)
        cv2.line(border, (0, 35), (189, 35), 255, 2)
        background[border > 0] = 10
        image, body, outline, bbox = make_outlined_text_sample(
            background, "あ？", (50, 40), font_size=30,
            stroke_width=2, foreground=(10, 10, 10),
        )
        original = image.copy()

        erased, mask = erase_text_regions(image, [bbox])

        self.assert_output_contract(image, original, erased, mask)
        self.assertFalse(np.any(mask[border > 0]))
        np.testing.assert_array_equal(erased[border > 0], background[border > 0])
        self.assertTrue(np.all(mask[body >= 192] == 255))
        # 即使字芯完整、边框未被圈入，也不能把邻近黑边作为颜色来源扩散回文字区。
        self.assertFalse(np.any(erased[outline > 0].max(axis=1) < 100))
        error = np.abs(erased.astype(np.int16) - background.astype(np.int16))
        self.assertLess(float(error[outline > 0].mean()), 5)

    def test_preserves_bubble_border_and_pixels_outside_bbox(self):
        background = np.full((105, 180, 3), (160, 115, 90), dtype=np.uint8)
        cv2.rectangle(background, (24, 20), (158, 87), (250, 250, 250), -1)
        cv2.rectangle(background, (24, 20), (158, 87), (15, 15, 15), 2)
        image, alpha, _ = make_text_sample(background, (15, 15, 15))
        bbox = (27, 23, 156, 85)
        original = image.copy()

        erased, mask = erase_text_regions(image, [bbox])

        self.assert_output_contract(image, original, erased, mask)
        outside = np.ones(image.shape[:2], dtype=bool)
        outside[bbox[1]:bbox[3], bbox[0]:bbox[2]] = False
        np.testing.assert_array_equal(erased[outside], image[outside])
        self.assertFalse(np.any(mask[outside]))
        border = np.all(background == (15, 15, 15), axis=-1)
        np.testing.assert_array_equal(erased[border], background[border])
        text_error = np.abs(erased.astype(np.int16) - background.astype(np.int16))[alpha > 0]
        self.assertLess(float(text_error.mean()), 3)

    def test_preserves_short_ellipse_arcs_while_erasing_text_in_pale_bubbles(self):
        samples = (
            ((250, 250, 250), "文字。!", 30),
            # 密集汉字的浅色字间隙不能与深色笔画连成整块，从而被当作边框保留。
            ((242, 225, 245), "翻譯翻譯", 22),
        )
        for bubble_color, text, font_size in samples:
            with self.subTest(bubble=bubble_color, text=text):
                background = np.full((105, 180, 3), (100, 135, 185), dtype=np.uint8)
                cv2.ellipse(background, (90, 54), (68, 43), 0, 0, 360,
                            bubble_color, -1, cv2.LINE_AA)
                cv2.ellipse(background, (90, 54), (68, 43), 0, 0, 360,
                            (15, 15, 15), 2, cv2.LINE_AA)
                border = np.zeros(background.shape[:2], dtype=np.uint8)
                cv2.ellipse(border, (90, 54), (68, 43), 0, 0, 360, 255, 2, cv2.LINE_AA)
                bubble = np.zeros_like(border)
                cv2.ellipse(bubble, (90, 54), (68, 43), 0, 0, 360, 255, -1, cv2.LINE_AA)
                image, alpha, _ = make_text_sample(
                    background, (15, 15, 15), text=text, font_size=font_size
                )
                original = image.copy()
                # 矩形检测框的角落跨过椭圆轮廓，裁剪区域内只露出几段短弧线。
                bbox = (32, 27, 149, 80)
                self.assertTrue(np.any(border[bbox[1]:bbox[3], bbox[0]:bbox[2]]))

                erased, mask = erase_text_regions(image, [bbox])

                self.assert_output_contract(image, original, erased, mask)
                np.testing.assert_array_equal(erased[border > 0], background[border > 0])
                np.testing.assert_array_equal(erased[bubble == 0], background[bubble == 0])
                self.assertFalse(np.any(mask[border > 0]))
                text_error = np.abs(
                    erased.astype(np.int16) - background.astype(np.int16)
                )[alpha > 0]
                self.assertLess(float(text_error.mean()), 3)

    def test_erases_text_touching_gray_lines_that_cross_the_detection_roi(self):
        for background_color in ((255, 255, 255), (250, 240, 247)):
            with self.subTest(background=background_color):
                background = np.full((105, 180, 3), background_color, dtype=np.uint8)
                lines = np.zeros(background.shape[:2], dtype=np.uint8)
                cv2.line(lines, (0, 44), (179, 44), 255, 1, cv2.LINE_AA)
                cv2.line(lines, (110, 0), (110, 104), 255, 1, cv2.LINE_AA)
                opacity = lines[..., None].astype(np.float32) / 255
                background = np.rint(background * (1 - opacity) + 220 * opacity).astype(np.uint8)
                image, alpha, bbox = make_text_sample(background, (15, 15, 15), text="文字?")
                original = image.copy()
                # 模拟透明气泡下的头发线：浅灰线跨过 ROI，并实际接触问号和文字笔画。
                question = np.indices(alpha.shape)[1] >= 104
                self.assertTrue(np.any((alpha >= 192) & (lines > 0) & question))

                erased, mask = erase_text_regions(image, [bbox])

                self.assert_output_contract(image, original, erased, mask)
                self.assertTrue(np.all(mask[alpha >= 192] == 255))
                distance = cv2.distanceTransform(
                    (alpha == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
                )
                far_from_text = distance > 4
                self.assertTrue(np.any((lines > 0) & far_from_text))
                np.testing.assert_array_equal(erased[far_from_text], background[far_from_text])
                text_error = np.abs(
                    erased.astype(np.int16) - background.astype(np.int16)
                )[alpha > 0]
                self.assertLess(float(text_error.mean()), 15)

    def test_preserves_faint_background_details_around_a_lower_row_question_mark(self):
        background = np.full((145, 180, 3), (251, 245, 249), dtype=np.uint8)
        lines = np.zeros(background.shape[:2], dtype=np.uint8)
        # 斜线穿过下排问号，旁边再放一段与文字分离的浅灰线，二者都不是待擦除的字。
        cv2.line(lines, (65, 78), (120, 114), 255, 1, cv2.LINE_AA)
        cv2.line(lines, (106, 67), (106, 80), 255, 1, cv2.LINE_AA)
        opacity = lines[..., None].astype(np.float32) / 255
        background = np.rint(background * (1 - opacity) + 220 * opacity).astype(np.uint8)
        alpha_image = Image.new("L", (180, 145))
        draw = ImageDraw.Draw(alpha_image)
        font = ImageFont.truetype(str(FONT_PATH), 30)
        draw.text((44, 35), "文字", font=font, fill=255, anchor="lt")
        draw.text((72, 80), "?", font=font, fill=255, anchor="lt")
        alpha = np.asarray(alpha_image)
        opacity = alpha[..., None].astype(np.float32) / 255
        image = np.rint(background * (1 - opacity) + 15 * opacity).astype(np.uint8)
        original = image.copy()
        self.assertTrue(np.any((alpha[80:] >= 192) & (lines[80:] > 0)))

        erased, mask = erase_text_regions(image, [(38, 25, 117, 117)])

        self.assert_output_contract(image, original, erased, mask)
        self.assertTrue(np.all(mask[alpha >= 192] == 255))
        distance = cv2.distanceTransform(
            (alpha == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        far_from_text = distance > 4
        np.testing.assert_array_equal(erased[far_from_text], background[far_from_text])
        self.assertFalse(np.any(mask[(lines > 0) & far_from_text]))
        text_error = np.abs(erased.astype(np.int16) - background.astype(np.int16))[alpha > 0]
        self.assertLess(float(text_error.mean()), 15)

    def test_faint_bridge_does_not_remove_protection_of_a_strong_border(self):
        background = np.full((105, 180, 3), 255, dtype=np.uint8)
        cv2.line(background, (66, 0), (66, 104), (15, 15, 15), 1)
        cv2.line(background, (63, 50), (66, 50), (220, 220, 220), 1)
        background[:, 66] = 15
        image = background.copy()
        cv2.rectangle(image, (60, 35), (64, 65), (15, 15, 15), -1)
        original = image.copy()

        erased, mask = erase_text_regions(image, [(55, 30, 70, 71)])

        self.assert_output_contract(image, original, erased, mask)
        self.assertTrue(np.all(mask[35:66, 60:65] == 255))
        # 灰线把字和边框接成同一弱连通域，也不能撤销强轮廓保护，让膨胀擦掉边框。
        self.assertFalse(np.any(mask[:, 66]))
        np.testing.assert_array_equal(erased[:, 66], background[:, 66])

    def test_handles_clipped_boxes_and_tiny_images_without_mutating_input(self):
        for height, width in ((1, 1), (2, 3), (12, 16)):
            with self.subTest(shape=(height, width)):
                image = np.full((height, width, 3), 235, dtype=np.uint8)
                image[height // 2, width // 2] = 30
                original = image.copy()

                erased, mask = erase_text_regions(
                    image, [(-10, -20, width + 10, height + 20), (0, 0, width, height)]
                )

                self.assert_output_contract(image, original, erased, mask)

    def test_overlapping_boxes_are_independent_of_order_and_duplicates(self):
        background = np.full((105, 180, 3), (235, 245, 250), dtype=np.uint8)
        image, _, bbox = make_text_sample(background, (20, 20, 20))
        overlap = (bbox[0] + 20, bbox[1] - 2, bbox[2] + 8, bbox[3] + 2)
        original = image.copy()

        erased, mask = erase_text_regions(image, [bbox, overlap])
        repeated, repeated_mask = erase_text_regions(image, [overlap, bbox, overlap])

        self.assert_output_contract(image, original, erased, mask)
        np.testing.assert_array_equal(repeated, erased)
        np.testing.assert_array_equal(repeated_mask, mask)

    def test_empty_boxes_leave_image_unchanged(self):
        image = np.random.default_rng(7).integers(0, 256, (27, 39, 3), dtype=np.uint8)
        original = image.copy()

        erased, mask = erase_text_regions(image, [])

        self.assert_output_contract(image, original, erased, mask)
        np.testing.assert_array_equal(erased, original)
        self.assertFalse(np.any(mask))


if __name__ == "__main__":
    unittest.main()
