"""文字区域背景修复；LaMa 是可选的本地 TorchScript 后端。"""

import os
from pathlib import Path
from stat import S_ISREG
from threading import Lock

import cv2
import numpy as np
import torch

from app.core.custom_conf import custom_conf
from app.core.logger import logger
from app.core.paths import MODELS_DIR, PROJECT_ROOT


LAMA_CONTEXT = 64
LAMA_MAX_SIZE = 1024
_inference_lock = Lock()
_model = None
_model_file = None
_model_device = None
_failed_devices = set()
_warnings = set()


def _warn_once(key, message):
    # 一页可能有很多文字区域，模型不可用时只提示一次，避免刷屏。
    if key not in _warnings:
        _warnings.add(key)
        logger.warning(message)


def _lama_model_path():
    configured_path = os.getenv("INPAINT_LAMA_MODEL_PATH", "").strip()
    path = Path(configured_path).expanduser() if configured_path else MODELS_DIR / "lama/big-lama.pt"
    return path if path.is_absolute() else PROJECT_ROOT / path


def _lama_file_key(model_path):
    stat = model_path.stat()
    if not S_ISREG(stat.st_mode) or not os.access(model_path, os.R_OK):
        raise OSError("模型路径必须是可读取的文件")
    return str(model_path), stat.st_mtime_ns, stat.st_size


def get_inpaint_status() -> dict:
    """查询所选擦除后端及回退状态，只检查本地文件，不加载或下载模型。"""
    requested = custom_conf.inpaint_backend
    status = {
        "requested": requested,
        "effective_backend": "opencv",
        "available": True,
        "model_loaded": False,
        "message": "使用 OpenCV 修复背景，纯色气泡优先填充背景色。",
    }
    if requested != "lama":
        return status

    with _inference_lock:
        try:
            file_key = _lama_file_key(_lama_model_path())
        except OSError:
            status.update(
                available=False,
                message="LaMa 本地模型缺失或不可读取，当前将使用 OpenCV；请安装模型并检查 INPAINT_LAMA_MODEL_PATH。",
            )
            return status

        devices = ["cuda", "cpu"] if custom_conf.use_gpu and torch.cuda.is_available() else ["cpu"]
        # 文件被替换后，旧文件的加载失败不应阻止下一次尝试。
        same_file = _model_file == file_key
        failed_devices = _failed_devices if same_file else set()
        if all(device in failed_devices for device in devices):
            status.update(
                available=False,
                message="LaMa 模型加载或修复失败，当前已回退 OpenCV；请检查或替换模型文件后重试。",
            )
            return status

        loaded = same_file and _model is not None and _model_device in devices
        status.update(effective_backend="lama", model_loaded=loaded)
        if loaded:
            status["message"] = "LaMa 模型已加载，纯色气泡仍优先填充背景色。"
            if custom_conf.use_gpu and _model_device == "cpu":
                status["message"] = "LaMa 当前使用 CPU 修复背景，纯色气泡仍优先填充背景色。"
        else:
            status["message"] = "LaMa 本地模型已找到，将在首次擦除复杂背景时加载。"
        return status


def _opencv_inpaint(image_bgr, mask):
    repaired = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)
    result = image_bgr.copy()
    result[mask != 0] = repaired[mask != 0]
    return result


def _lama_regions(mask):
    # 相邻笔画和文字共用上下文；相距很远的气泡分别修复，避免整页缩小后丢失细节。
    size = LAMA_CONTEXT * 2 + 1
    nearby = cv2.dilate(mask, np.ones((size, size), dtype=np.uint8))
    contours, _ = cv2.findContours(nearby, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.boundingRect(contour) for contour in contours]


def _run_lama(image_bgr, mask, model, device):
    result = image_bgr.copy()
    for x, y, width, height in _lama_regions(mask):
        cropped = image_bgr[y:y + height, x:x + width]
        local_mask = mask[y:y + height, x:x + width]
        scale = min(1.0, LAMA_MAX_SIZE / max(width, height))
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        if scale < 1.0:
            cropped = cv2.resize(cropped, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            # 最近邻缩小可能让细笔画消失，面积采样后保留所有与掩码相交的像素。
            model_mask = cv2.resize(
                local_mask.astype(np.float32), (resized_width, resized_height), interpolation=cv2.INTER_AREA
            ) > 0
        else:
            model_mask = local_mask != 0

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pad_height, pad_width = -resized_height % 8, -resized_width % 8
        rgb = np.pad(rgb, ((0, pad_height), (0, pad_width), (0, 0)), mode="symmetric")
        model_mask = np.pad(model_mask, ((0, pad_height), (0, pad_width)), mode="symmetric")
        image_tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(model_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(image_tensor, mask_tensor)
        if not isinstance(output, torch.Tensor) or output.shape != image_tensor.shape:
            raise ValueError("LaMa 模型应返回与输入图像同尺寸的 RGB NCHW 张量")
        output = output[0, :, :resized_height, :resized_width].detach().float().cpu().numpy()
        if not np.isfinite(output).all():
            raise ValueError("LaMa 模型输出包含无效像素")
        repaired_rgb = np.clip(np.rint(output.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
        repaired = cv2.cvtColor(repaired_rgb, cv2.COLOR_RGB2BGR)
        if scale < 1.0:
            repaired = cv2.resize(repaired, (width, height), interpolation=cv2.INTER_LINEAR)
        # 即使裁剪区域经过缩放、补边，掩码外的边框和画面也保持原始像素。
        target = result[y:y + height, x:x + width]
        target[local_mask != 0] = repaired[local_mask != 0]
    return result


def inpaint_image(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """修复非零掩码内的像素，保持 BGR uint8、原始尺寸和掩码外像素。"""
    global _model, _model_file, _model_device

    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("背景修复需要 BGR uint8 图像")
    if mask.ndim != 2 or mask.shape != image_bgr.shape[:2]:
        raise ValueError("背景修复掩码必须与图像尺寸一致")
    mask = np.where(mask != 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        return image_bgr.copy()

    # 每次擦除读取运行时配置，前端切换后下一次处理即可生效。
    if custom_conf.inpaint_backend != "lama":
        return _opencv_inpaint(image_bgr, mask)

    # 模型只从本地读取，不在翻译请求中下载；第一次需要复杂背景修复时才加载。
    model_path = _lama_model_path()
    with _inference_lock:
        try:
            file_key = _lama_file_key(model_path)
        except OSError as exc:
            _warn_once(("missing", str(model_path)), f"LaMa 模型不可用，回退 OpenCV：{model_path}（{exc}）")
            return _opencv_inpaint(image_bgr, mask)

        if _model_file != file_key:
            _model, _model_device = None, None
            _model_file = file_key
            _failed_devices.clear()
        devices = ["cuda", "cpu"] if custom_conf.use_gpu and torch.cuda.is_available() else ["cpu"]
        for device in devices:
            if device in _failed_devices:
                continue
            try:
                if _model is None or _model_device != device:
                    # 切换设备时先释放旧引用，防止同时保留两份大模型。
                    _model, _model_device = None, None
                    _model = torch.jit.load(str(model_path), map_location=device).eval()
                    _model_device = device
                return _run_lama(image_bgr, mask, _model, device)
            except Exception as exc:
                _model, _model_device = None, None
                _failed_devices.add(device)
                fallback = "CPU" if device == "cuda" else "OpenCV"
                _warn_once((file_key, device), f"LaMa {device} 修复失败，回退 {fallback}：{exc}")
                # 同一文件失败后不反复加载；替换模型文件或重启服务后会重新尝试。
        return _opencv_inpaint(image_bgr, mask)
