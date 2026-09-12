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
        interior = cv2.erode(core.astype(np.uint8), np.ones((7, 7), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        rings = (nearby & (text_mask == 0) & interior,)
    for ring in rings:
        if np.count_nonzero(ring) < 8:
            continue
        color = _dominant_color(crop[ring])
        color_lab = cv2.cvtColor(color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB)[0, 0]
        distance = np.linalg.norm(lab - color_lab, axis=2)
        check = core
        if text_mask is not None:
            # 排除少量连到裁剪边缘的深色轮廓，避免气泡边框污染取色。
            # 大片色块和浅灰纹理仍参与检查，不能把半透明或复杂背景误判为纯色。
            high = (distance > 64) & (text_mask == 0)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(high.astype(np.uint8), connectivity=8)
            excluded = np.zeros(core.shape, dtype=np.uint8)
            height, width = core.shape
            max_border_area = np.count_nonzero(core) * 0.1
            for idx in range(1, count):
                x, y, w, h, _ = stats[idx]
                if x != 0 and y != 0 and x + w != width and y + h != height:
                    continue
                component = labels[y:y + h, x:x + w] == idx
                if np.count_nonzero(component & core[y:y + h, x:x + w]) <= max_border_area:
                    excluded[y:y + h, x:x + w][component] = 1
            # 多条细线也可能累计成大片纹理，不能逐条忽略后将背景刷平。
            if np.count_nonzero((excluded > 0) & core) > max_border_area:
                excluded[:] = 0
            # 一并避开轮廓外侧一像素的抗锯齿灰边。
            excluded = cv2.dilate(excluded, np.ones((3, 3), np.uint8)) > 0
            ring = ring & ~excluded
            check = core & (text_mask == 0) & ~excluded
            if np.count_nonzero(ring) < 8 or np.count_nonzero(check) < 8:
                continue
        # 采样圈和框内都要有足够一致的背景；渐变、半透明和网点不能直接刷成纯色。
        # 找到掩码后只检查剩余背景，密集文字本身不应降低背景一致性。
        percentile = 99 if text_mask is not None else 90
        required_ratio = 0.9 if text_mask is not None else 0.5
        # 少量抗锯齿噪点可容忍，但大部分背景必须更接近底色，以保留浅色渐变。
        if text_mask is not None and np.percentile(distance[ring], 95) > 3:
            continue
        if np.percentile(distance[ring], percentile) <= 8 and np.mean(distance[check] <= 8) >= required_ratio:
            return np.median(crop[ring & (distance <= 8)], axis=0).astype(np.uint8)
    return None


def _outlined_text_mask(crop: np.ndarray, core: np.ndarray) -> np.ndarray | None:
    """识别白描边的深色字；证据不足时交回普通文字处理。"""
    if min(crop.shape[:2]) < 7 or not np.any(core):
        return None
    lightness = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[:, :, 0]
    size = 2 * max(2, min(10, min(crop.shape[:2]) // 8)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    # 字芯与亮描边分别判断，避免两者通过背景线连到裁剪边缘后整字被保护。
    dark = cv2.morphologyEx(lightness, cv2.MORPH_BLACKHAT, kernel)
    threshold, _ = cv2.threshold(dark[core], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strong = dark > max(16, threshold)
    if np.count_nonzero(strong & core) < 8:
        return None
    # 黑帽也会响应白描边旁的灰底，再从高响应像素中分出较暗的字芯。
    samples = lightness[strong & core]
    threshold, _ = cv2.threshold(samples, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strong &= lightness <= max(threshold, np.percentile(samples, 10))
    near_white = crop.min(axis=2) >= 238
    count, labels, stats, _ = cv2.connectedComponentsWithStats(strong.astype(np.uint8), connectivity=8)
    mask = np.zeros(core.shape, dtype=np.uint8)
    height, width = core.shape
    eligible = np.zeros(core.shape, dtype=np.uint8)
    for idx in range(1, count):
        x, y, w, h, _ = stats[idx]
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        component = labels[y:y + h, x:x + w] == idx
        # 保留小笔画和单像素标点，只排除触边结构及文字框外的孤立细节。
        if np.any(component & core[y:y + h, x:x + w]):
            eligible[y:y + h, x:x + w][component] = 1
    if not np.any(eligible):
        return None
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # 密集小字的内部碎笔画未必各自带白边，先把相距很近的字芯分组判断。
    # 膨胀只用于分组，连接笔画的空隙不会直接加入文字掩码。
    groups = cv2.dilate(eligible, near_kernel)
    group_count, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(groups, connectivity=8)
    text_neighborhood = cv2.dilate(strong.astype(np.uint8), ring_kernel) > 0
    outlined_components = 0
    for idx in range(1, group_count):
        x, y, w, h, _ = group_stats[idx]
        # 只在连通域附近采样，避免每个字都对整张裁剪图重复膨胀。
        left, top = max(0, x - 7), max(0, y - 7)
        right, bottom = min(width, x + w + 7), min(height, y + h + 7)
        component = ((group_labels[top:bottom, left:right] == idx)
                     & (eligible[top:bottom, left:right] > 0)).astype(np.uint8)
        inner = cv2.dilate(component, near_kernel) > 0
        near = cv2.dilate(component, ring_kernel) > 0
        outer = cv2.dilate(component, outer_kernel) > 0
        ring = near & ~inner
        # 外圈还要避开邻字，不能把密集黑字当成白描边外侧的有色背景。
        surroundings = outer & ~near & ~text_neighborhood[top:bottom, left:right]
        white = near_white[top:bottom, left:right]
        # 字芯大部分周长应被白色包围，只有一侧发白的气泡弧线不能当作文字。
        if not np.any(ring) or np.mean(white[ring]) <= 0.6:
            continue
        mask[top:bottom, left:right][component > 0] = 255
        # 白底黑字没有独立描边：需多个字同时呈现“近处白、远处有底色”。
        if np.any(surroundings) and np.mean(~white[surroundings]) > 0.5:
            outlined_components += len(np.unique(labels[top:bottom, left:right][component > 0]))
    if outlined_components < 2:
        return None

    grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    limit = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    adjacency = np.ones((3, 3), dtype=np.uint8)
    # 仅沿相邻白色像素补足描边，且限制离字芯的距离，避免吞掉整片白背景。
    for _ in range(5):
        grown |= (cv2.dilate(grown.astype(np.uint8), adjacency) > 0) & near_white & limit
    grown = cv2.dilate(grown.astype(np.uint8), adjacency) > 0
    # 补回已被描边掩码完整覆盖的内部笔画，防止保护逻辑挖出黑洞再被修复算法扩散。
    # 不做任意孔洞填充；仍保护触边轮廓和只有一部分落入掩码的背景线。
    for idx in range(1, count):
        x, y, w, h, _ = stats[idx]
        component = labels[y:y + h, x:x + w] == idx
        if (np.any(eligible[y:y + h, x:x + w][component])
                and np.all(grown[y:y + h, x:x + w][component])):
            mask[y:y + h, x:x + w][component] = 255
    grown[strong & (mask == 0)] = False
    return grown.astype(np.uint8) * 255


def _text_mask(crop: np.ndarray, core: np.ndarray, background: np.ndarray | None) -> np.ndarray:
    outlined = _outlined_text_mask(crop, core)
    if outlined is not None:
        return outlined
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
        # 已填好的纯色区域也作为干净上下文；其余文字区域使用 OpenCV 修复。
        result = inpaint_image(result, repair_mask)
    return result, mask
