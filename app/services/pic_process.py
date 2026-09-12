import asyncio
import os
from typing import Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.font_conf import FontConfig
from app.core.paths import SAVED_DIR
from app.services.ocr import get_mocr
from app.services.text_erasure import erase_text_regions

OCR_MAX_CONCURRENCY = max(1, int(os.getenv("OCR_MAX_CONCURRENCY", "2")))
TextDirection = Literal["horizontal", "vertical"]
MIN_FONT_SIZE = 1


def _sanitize_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


async def get_text_masked_pic(image_pil, image_cv, bboxes, inpaint=True):
    if len(bboxes) == 0:
        return [], image_cv

    height, width = image_cv.shape[:2]
    mocr = get_mocr()
    semaphore = asyncio.Semaphore(min(OCR_MAX_CONCURRENCY, len(bboxes)))

    async def recognize(bbox):
        x1, y1, x2, y2 = _sanitize_bbox(bbox, width, height)
        cropped_image = image_pil.crop((x1, y1, x2, y2))
        async with semaphore:
            return await asyncio.to_thread(mocr, cropped_image)

    all_text = await asyncio.gather(*(recognize(bbox) for bbox in bboxes))
    if inpaint:
        # 图像处理和可选模型推理在线程中执行，避免阻塞异步接口。
        image_cv, _ = await asyncio.to_thread(erase_text_regions, image_cv, bboxes)
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


def _draw_horizontal_text(
    draw: ImageDraw.ImageDraw,
    image: np.ndarray,
    box: tuple[int, int, int, int],
    text: str,
):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    font = FontConfig(box_height, box_width, text).font
    lines = wrap_text_by_width(draw, text, font, box_width)
    line_spacing = 4
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


def _vertical_layout(text: str, font, box_height: int):
    gap = max(2, int(font.size * 0.1))
    bbox = font.getbbox("中")
    char_width = max(1, bbox[2] - bbox[0])
    char_height = max(1, bbox[3] - bbox[1])
    rows_per_col = max(1, (box_height + gap) // (char_height + gap))
    columns = [text[i:i + rows_per_col] for i in range(0, len(text), rows_per_col)]
    max_rows = max((len(column) for column in columns), default=0)
    total_width = len(columns) * char_width + gap * max(0, len(columns) - 1)
    total_height = max_rows * char_height + gap * max(0, max_rows - 1)
    return {
        "gap": gap,
        "char_width": char_width,
        "char_height": char_height,
        "rows_per_col": rows_per_col,
        "columns": columns,
        "total_width": total_width,
        "total_height": total_height,
    }


def _fit_vertical_font(box_height: int, box_width: int, text: str):
    font_config = FontConfig(box_height, box_width, text)
    font_size = max(MIN_FONT_SIZE, int(font_config.font_size))
    while font_size >= MIN_FONT_SIZE:
        font = ImageFont.truetype(font_config.font_path, font_size)
        layout = _vertical_layout(text, font, box_height)
        if layout["total_width"] <= box_width and layout["total_height"] <= box_height:
            return font, layout
        font_size -= 1
    font = ImageFont.truetype(font_config.font_path, MIN_FONT_SIZE)
    return font, _vertical_layout(text, font, box_height)


def _draw_vertical_text(
    draw: ImageDraw.ImageDraw,
    image: np.ndarray,
    box: tuple[int, int, int, int],
    text: str,
):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    font, layout = _fit_vertical_font(box_height, box_width, text)
    gap = layout["gap"]
    columns = layout["columns"]
    char_width = layout["char_width"]
    char_height = layout["char_height"]
    total_width = layout["total_width"]
    total_height = layout["total_height"]
    start_x = x1 + max(0, (box_width - total_width) // 2)
    start_y = y1 + max(0, (box_height - total_height) // 2)
    fill_color, stroke_color = _pick_text_style(image[y1:y2, x1:x2])
    stroke_width = max(1, int(font.size * 0.12))

    for col_idx, column in enumerate(columns):
        x = start_x + (len(columns) - col_idx - 1) * (char_width + gap)
        for row_idx, char in enumerate(column):
            y = start_y + row_idx * (char_height + gap)
            char_bbox = font.getbbox(char)
            actual_width = max(1, char_bbox[2] - char_bbox[0])
            actual_height = max(1, char_bbox[3] - char_bbox[1])
            draw_x = x + max(0, (char_width - actual_width) // 2)
            draw_y = y + max(0, (char_height - actual_height) // 2)
            draw.text(
                (draw_x, draw_y),
                char,
                font=font,
                fill=fill_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color,
            )


def draw_text_on_boxes(
    image: np.ndarray,
    boxes: list,
    texts: list,
    text_direction: TextDirection = "horizontal",
) -> np.ndarray:
    height, width = image.shape[:2]
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for box, text in zip(boxes, texts):
        x1, y1, x2, y2 = _sanitize_bbox(box, width, height)
        if not text:
            continue
        sanitized_box = (x1, y1, x2, y2)
        if text_direction == "vertical":
            _draw_vertical_text(draw, image, sanitized_box, text)
        elif text_direction == "horizontal":
            _draw_horizontal_text(draw, image, sanitized_box, text)
        else:
            raise ValueError("text_direction 必须是 ('horizontal', 'vertical')")
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def save_img(file_bytes, pre: str, file_name: str):
    folder_path = SAVED_DIR / pre
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(folder_path / file_name, "wb") as f:
        f.write(file_bytes)
