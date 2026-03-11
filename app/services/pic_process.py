import asyncio
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.core.font_conf import FontConfig
from app.core.paths import SAVED_DIR
from app.services.ocr import get_mocr

OCR_MAX_CONCURRENCY = max(1, int(os.getenv("OCR_MAX_CONCURRENCY", "2")))
INPAINT_RADIUS = 2


def _sanitize_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def _build_text_mask(cropped_cv: np.ndarray) -> np.ndarray:
    if cropped_cv.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    gray = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 黑帽突出深色细节，可避免把半透明气泡整体选进掩码。
    kernel_size = max(3, min(11, (min(cropped_cv.shape[:2]) // 10) * 2 + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    _, bh_mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 兜底保留极深像素，兼容描边文字。
    p15 = float(np.percentile(blur, 15))
    dark_threshold = int(max(30, min(120, p15)))
    dark_mask = cv2.inRange(blur, 0, dark_threshold)

    merged = cv2.bitwise_or(bh_mask, dark_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    filtered = np.zeros_like(merged)
    area = cropped_cv.shape[0] * cropped_cv.shape[1]
    min_area = max(6, int(area * 0.0002))
    max_area = max(20, int(area * 0.2))
    for idx in range(1, num_labels):
        component_area = int(stats[idx, cv2.CC_STAT_AREA])
        if min_area <= component_area <= max_area:
            filtered[labels == idx] = 255

    return cv2.dilate(filtered, np.ones((2, 2), dtype=np.uint8), iterations=1)


async def get_text_masked_pic(image_pil, image_cv, bboxes, inpaint=True):
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    if len(bboxes) == 0:
        return [], image_cv

    height, width = image_cv.shape[:2]
    mocr = get_mocr()
    semaphore = asyncio.Semaphore(min(OCR_MAX_CONCURRENCY, len(bboxes)))

    async def ocr_and_mask(bbox):
        x1, y1, x2, y2 = _sanitize_bbox(bbox, width, height)
        cropped_image = image_pil.crop((x1, y1, x2, y2))
        async with semaphore:
            text = await asyncio.to_thread(mocr, cropped_image)
        local_mask = _build_text_mask(image_cv[y1:y2, x1:x2])
        return text, (x1, y1, x2, y2), local_mask

    tasks = [ocr_and_mask(bbox) for bbox in bboxes]
    results = await asyncio.gather(*tasks)
    all_text = []
    for text, (x1, y1, x2, y2), local_mask in results:
        all_text.append(text)
        if local_mask.size == 0:
            continue
        target = mask[y1:y2, x1:x2]
        np.maximum(target, local_mask, out=target)

    if inpaint and np.any(mask):
        image_cv = cv2.inpaint(image_cv, mask, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
    return all_text, image_cv


def wrap_text_by_width(draw, text, font, max_width):
    """
    将文字根据实际像素宽度换行，返回行列表
    """
    lines = []
    line = ''
    for char in text:
        test_line = line + char
        w = draw.textlength(test_line, font=font)
        if w <= max_width:
            line = test_line
        else:
            if line:
                lines.append(line)
            line = char
    if line:
        lines.append(line)
    return lines


def _pick_text_style(cropped_cv: np.ndarray):
    if cropped_cv.size == 0:
        return (20, 20, 20), (245, 245, 245)
    gray = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2GRAY)
    median_luma = float(np.median(gray))
    if median_luma >= 145:
        return (20, 20, 20), (245, 245, 245)
    return (245, 245, 245), (20, 20, 20)


def draw_text_on_boxes(image: np.ndarray, boxes: list, texts: list) -> np.ndarray:
    height, width = image.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    line_spacing = 4  # 行间距
    for box, text in zip(boxes, texts):
        x1, y1, x2, y2 = _sanitize_bbox(box, width, height)
        if not text:
            continue
        box_width = x2 - x1
        box_height = y2 - y1
        font_config = FontConfig(box_height, box_width, text)
        font = font_config.font
        lines = wrap_text_by_width(draw, text, font, box_width)
        line_height = font.getbbox("中")[3] - font.getbbox("中")[1]
        total_height = line_height * len(lines) + line_spacing * max(0, len(lines) - 1)
        start_y = y1 + max(0, (box_height - total_height) // 2)
        fill_color, stroke_color = _pick_text_style(image[y1:y2, x1:x2])
        stroke_width = max(1, int(font.size * 0.12))
        for i, line in enumerate(lines):
            y = start_y + i * (line_height + line_spacing)
            line_width = draw.textlength(line, font=font)
            x = x1 + max(0, int((box_width - line_width) / 2))
            draw.text(
                (x, y),
                line,
                font=font,
                fill=fill_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color,
            )
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def save_img(file_bytes, pre: str, file_name: str):
    folder_path = SAVED_DIR / pre
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(folder_path / file_name, "wb") as f:
        f.write(file_bytes)
