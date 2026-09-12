"""使用 OpenCV 修复文字掩码内的背景。"""

import cv2
import numpy as np


def inpaint_image(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """修复非零掩码内的像素，保持 BGR uint8、原始尺寸和掩码外像素。"""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("背景修复需要 BGR uint8 图像")
    if mask.ndim != 2 or mask.shape != image_bgr.shape[:2]:
        raise ValueError("背景修复掩码必须与图像尺寸一致")
    mask = np.where(mask != 0, 255, 0).astype(np.uint8)
    result = image_bgr.copy()
    if np.any(mask):
        repaired = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)
        # 只回贴文字区域，保留掩码外的气泡轮廓和画面细节。
        result[mask != 0] = repaired[mask != 0]
    return result
