"""生成文字掩码，并按局部背景选择纯色填充或图像修复。"""

import cv2
import numpy as np

from app.services.inpainting import inpaint_image


def _dominant_color(samples: np.ndarray) -> np.ndarray:
    bins = samples.astype(np.int32) // 16
    keys = bins[:, 0] * 256 + bins[:, 1] * 16 + bins[:, 2]
    dominant = np.argmax(np.bincount(keys, minlength=4096))
    return np.median(samples[keys == dominant], axis=0).astype(np.uint8)


def _uniform_background(
    crop: np.ndarray, core: np.ndarray, text_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    # 从文字框外圈采样，避免把文字颜色当背景。贴着图像边缘时仍使用可用的外圈。
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    if text_mask is None:
        interior = cv2.erode(core.astype(np.uint8), np.ones((5, 5), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        # 扩边可能越过气泡轮廓；外圈不可靠时，再检查文字框内侧的窄边。
        rings = (~core, core & ~interior)
    else:
        # 初步找到文字后，笔画周围的干净像素比矩形框角更能代表气泡底色。
        nearby = cv2.dilate(text_mask, np.ones((5, 5), np.uint8)) > 0
        rings = (nearby & (text_mask == 0) & core,)
    for ring in rings:
        if np.count_nonzero(ring) < 8:
            continue
        color = _dominant_color(crop[ring])
        color_lab = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
        distance = np.linalg.norm(lab - color_lab, axis=2)
        # 采样圈和框内都要有足够一致的背景；渐变、半透明和网点不能直接刷成纯色。
        percentile = 99 if text_mask is not None else 90
        if np.percentile(distance[ring], percentile) <= 8 and np.mean(distance[core] <= 8) >= 0.5:
            return np.median(crop[ring & (distance <= 8)], axis=0).astype(np.uint8)
    return None


def _text_mask(crop: np.ndarray, core: np.ndarray, background: np.ndarray | None) -> np.ndarray:
    if background is not None:
        # 自适应颜色差区分文字主体和透出的浅色背景线，也支持深底白字及等亮度彩字。
        response = np.max(np.abs(crop.astype(np.int16) - background), axis=2).astype(np.uint8)
        threshold, _ = cv2.threshold(response[core], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        strong = response > max(16, threshold)
        weak = response > 3
    else:
        # Lab 三通道的黑帽/顶帽同时寻找暗字、亮字和颜色差异，避免只看灰度。
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        size = 2 * max(2, min(10, min(crop.shape[:2]) // 8)) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        dark = cv2.morphologyEx(lab, cv2.MORPH_BLACKHAT, kernel)
        light = cv2.morphologyEx(lab, cv2.MORPH_TOPHAT, kernel)
        response = np.maximum(dark, light).max(axis=2)
        threshold, _ = cv2.threshold(response[core], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        color = _dominant_color(crop[core])
        difference = np.max(np.abs(crop.astype(np.int16) - color), axis=2)
        # 顶帽也会响应深色文字之间的浅色空隙；排除主背景色，防止把整段文字
        # 与空隙连成一大块，再误当作触边结构。这里仅辅助掩码，不据此刷纯色。
        strong = (response > max(12, threshold)) & (difference > 16)
        weak = (response > max(4, threshold * 0.25)) & (difference > 3)

    # 只用笔画主体判断连通性，避免浅灰背景线把文字接到裁剪边缘而整段漏擦。
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), connectivity=8)
    mask = np.zeros(core.shape, dtype=np.uint8)
    height, width = core.shape
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        component = labels[y:y + h, x:x + w] == idx
        local_core = core[y:y + h, x:x + w]
        if not np.any(component & local_core & strong[y:y + h, x:x + w]):
            continue
        # 不按整框面积删除小连通域，标点和细笔画同样需要擦除。
        # 连到扩边区域边缘的细节通常是气泡边框或画面轮廓；弧线在框角
        # 可能只露出很短一截，不能根据长度决定是否保护。
        touches_edge = x == 0 or y == 0 or x + w == width or y + h == height
        if touches_edge:
            continue
        mask[y:y + h, x:x + w][component] = 255

    if not np.any(mask):
        return mask
    # 强像素比完整笔画窄一圈；额外扩一像素覆盖缩放造成的灰边和轻微投影。
    stroke = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    radius = int(np.clip(np.ceil(np.percentile(stroke[mask > 0], 75)) + 1, 1, 4))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    expanded = cv2.dilate(mask, kernel)
    # 低对比度像素只负责补足笔画灰边，不能沿着它们无限扩张。
    # 不含文字主体的弱连通域仍保留，防止膨胀重新圈入气泡边框的抗锯齿边缘。
    _, weak_labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    text_labels = np.unique(weak_labels[mask > 0])
    protected = (weak & ~np.isin(weak_labels, text_labels)) | (strong & (mask == 0))
    expanded[protected] = 0
    return expanded


def erase_text_regions(image_bgr: np.ndarray, bboxes) -> tuple[np.ndarray, np.ndarray]:
    """返回擦除图和全图掩码；只修改掩码内像素，不改动输入原图。"""
    height, width = image_bgr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    repair_mask = np.zeros_like(mask)
    result = image_bgr.copy()
    if height == 0 or width == 0:
        return result, mask

    for bbox in bboxes:
        coords = np.asarray(bbox, dtype=np.float64)
        if coords.shape != (4,) or not np.all(np.isfinite(coords)):
            continue
        # 右下角向上取整，避免浮点检测框截掉最后一列/行文字。
        x1, y1 = np.floor(coords[:2]).astype(int)
        x2, y2 = np.ceil(coords[2:]).astype(int)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        padding = int(np.clip(np.ceil(min(x2 - x1, y2 - y1) * 0.08), 3, 12))
        left, top = max(0, x1 - padding), max(0, y1 - padding)
        right, bottom = min(width, x2 + padding), min(height, y2 + padding)
        crop = image_bgr[top:bottom, left:right]
        core = np.zeros(crop.shape[:2], dtype=bool)
        core[y1 - top:y2 - top, x1 - left:x2 - left] = True
        background = _uniform_background(crop, core)
        local_mask = _text_mask(crop, core, background)
        if np.any(local_mask):
            # 外圈看起来纯白时，框内仍可能透出稀疏背景线；填色前必须重新检查笔画周围。
            verified_background = _uniform_background(crop, core, local_mask)
            if background is None and verified_background is not None:
                background = verified_background
                local_mask = _text_mask(crop, core, background)
                verified_background = _uniform_background(crop, core, local_mask)
            background = verified_background
        selected = local_mask > 0
        target = mask[top:bottom, left:right]
        np.maximum(target, local_mask, out=target)
        if background is not None:
            result[top:bottom, left:right][selected] = background
        else:
            target = repair_mask[top:bottom, left:right]
            np.maximum(target, local_mask, out=target)

    if np.any(repair_mask):
        # 已填好的纯色区域也作为干净上下文；只把仍需修复的区域交给后端。
        result = inpaint_image(result, repair_mask)
    return result, mask
