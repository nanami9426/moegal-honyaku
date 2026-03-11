import os
from threading import Lock

import torch
from dotenv import load_dotenv
from manga_ocr import MangaOcr
from ultralytics import YOLO

from app.core.logger import logger
from app.core.paths import MODELS_DIR

DET_MODEL_PATH = MODELS_DIR / "comic-text-segmenter.pt"
MOCR_MODEL_PATH = MODELS_DIR / "manga-ocr-base"
load_dotenv()

_MODEL_LOCK = Lock()
_DET_MODEL: YOLO | None = None
_MOCR: MangaOcr | None = None


def _is_true_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_cuda_related_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "cuda",
            "cudnn",
            "no kernel image",
            "driver",
            "device-side assert",
        )
    )


def _is_cuda_runtime_usable() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() = False"
    try:
        a = torch.tensor([1, 2, 3], device="cuda")
        b = torch.tensor([2], device="cuda")
        _ = torch.isin(a, b)
        torch.cuda.synchronize()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _resolve_device() -> tuple[torch.device, bool]:
    gpu_enabled = _is_true_env("MOEGAL_USE_GPU", default=False)
    cuda_usable, cuda_fail_reason = _is_cuda_runtime_usable() if gpu_enabled else (False, "")
    use_cuda = gpu_enabled and cuda_usable

    if gpu_enabled and not use_cuda:
        logger.warning(f"检测到 GPU 已启用但 CUDA 不可用，自动回退 CPU。原因：{cuda_fail_reason}")

    return torch.device("cuda:0") if use_cuda else torch.device("cpu"), use_cuda


def warmup_models() -> tuple[YOLO, MangaOcr]:
    global _DET_MODEL, _MOCR

    with _MODEL_LOCK:
        if _DET_MODEL is not None and _MOCR is not None:
            return _DET_MODEL, _MOCR

        device, use_cuda = _resolve_device()

        if _DET_MODEL is None:
            _DET_MODEL = YOLO(str(DET_MODEL_PATH)).to(device)
            logger.info(f"气泡检测模型加载成功，使用：{_DET_MODEL.device}")

        if _MOCR is None:
            if use_cuda:
                try:
                    _MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=False)
                    logger.info("MangaOCR 加载成功，使用：cuda")
                except Exception as exc:
                    if not _is_cuda_related_error(exc):
                        raise
                    logger.warning(f"MangaOCR CUDA 初始化失败，自动回退 CPU。原因：{exc}")
                    _MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=True)
                    logger.info("MangaOCR 加载成功，使用：cpu")
            else:
                _MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=True)
                logger.info("MangaOCR 加载成功，使用：cpu")

        return _DET_MODEL, _MOCR


def get_det_model() -> YOLO:
    det_model, _ = warmup_models()
    return det_model


def get_mocr() -> MangaOcr:
    _, mocr = warmup_models()
    return mocr
