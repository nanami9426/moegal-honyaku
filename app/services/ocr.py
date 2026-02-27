import os

import torch
from dotenv import load_dotenv
from manga_ocr import MangaOcr
from ultralytics import YOLO

from app.core.logger import logger
from app.core.paths import MODELS_DIR

DET_MODEL_PATH = MODELS_DIR / "comic-text-segmenter.pt"
MOCR_MODEL_PATH = MODELS_DIR / "manga-ocr-base"
load_dotenv()


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


GPU_ENABLED = _is_true_env("MOEGAL_USE_GPU", default=False)
CUDA_USABLE, CUDA_FAIL_REASON = _is_cuda_runtime_usable() if GPU_ENABLED else (False, "")
USE_CUDA = GPU_ENABLED and CUDA_USABLE

if GPU_ENABLED and not USE_CUDA:
    logger.warning(f"检测到 GPU 已启用但 CUDA 不可用，自动回退 CPU。原因：{CUDA_FAIL_REASON}")

DEVICE = torch.device("cuda:0") if USE_CUDA else torch.device("cpu")
DET_MODEL = YOLO(str(DET_MODEL_PATH)).to(DEVICE)
logger.info(f"气泡检测模型加载成功，使用：{DET_MODEL.device}")

if USE_CUDA:
    try:
        MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=False)
        logger.info("MangaOCR 加载成功，使用：cuda")
    except Exception as exc:
        if not _is_cuda_related_error(exc):
            raise
        logger.warning(f"MangaOCR CUDA 初始化失败，自动回退 CPU。原因：{exc}")
        MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=True)
        logger.info("MangaOCR 加载成功，使用：cpu")
else:
    MOCR = MangaOcr(pretrained_model_name_or_path=str(MOCR_MODEL_PATH), force_cpu=True)
    logger.info("MangaOCR 加载成功，使用：cpu")
