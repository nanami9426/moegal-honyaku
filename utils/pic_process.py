import cv2
import numpy as np
from PIL import Image, ImageDraw
import asyncio
from api.api_v1 import MOCR, FONT, LINE_HEIGHT


async def get_text_masked_pic(image_pil, image_cv, bboxes, inpaint=True):
    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    async def ocr_and_mask(bbox):
        # 识别文字
        cropped_image = image_pil.crop(bbox)
        text = await asyncio.to_thread(MOCR, cropped_image)
        # 创建掩码
        x1, y1, x2, y2 = map(int, bbox)
        mask[y1:y2, x1:x2] = 255
        if not inpaint:
            image_cv[y1:y2, x1:x2] = (255, 255, 255)
        return text
    
    tasks = [ocr_and_mask(bbox) for bbox in bboxes]
    all_text = await asyncio.gather(*tasks)
    if inpaint:
        image_cv = cv2.inpaint(image_cv, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
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

def draw_text_on_boxes(image: np.ndarray, boxes: list, texts: list) -> np.ndarray:
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    line_spacing = 4  # 行间距
    for box, text in zip(boxes, texts):
        x1, y1, x2, y2 = map(int, box)
        box_width = x2 - x1
        lines = wrap_text_by_width(draw, text, FONT, box_width)
        for i, line in enumerate(lines):
            y = y1 + i * (LINE_HEIGHT + line_spacing)
            draw.text((x1, y), line, font=FONT, fill=(0, 64, 0))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)