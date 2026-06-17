from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch
from dotenv import load_dotenv
from manga_ocr import MangaOcr
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

from app.core.logger import logger
from app.core.paths import MODELS_DIR

TEXT_BUBBLE_MODEL_PATH = MODELS_DIR / "comic-text-and-bubble-detector"
TEXT_BUBBLE_LABEL = "text_bubble"
TEXT_BUBBLE_CONFIDENCE = 0.8
MOCR_MODEL_PATH = MODELS_DIR / "manga-ocr-base"
load_dotenv()

_MODEL_LOCK = Lock()
_DET_MODEL: TextBubbleDetector | None = None
_MOCR: MangaOcr | None = None


@dataclass(frozen=True)
class TextBubbleDetector:
    processor: Any
    model: torch.nn.Module
    device: torch.device
    label_id: int


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


def _resolve_label_id(model: torch.nn.Module) -> int:
    label2id = getattr(model.config, "label2id", {}) or {}
    if TEXT_BUBBLE_LABEL in label2id:
        return int(label2id[TEXT_BUBBLE_LABEL])

    id2label = getattr(model.config, "id2label", {}) or {}
    for raw_id, label in id2label.items():
        if label == TEXT_BUBBLE_LABEL:
            return int(raw_id)

    raise RuntimeError(f"Detector label not found: {TEXT_BUBBLE_LABEL}")


def warmup_models() -> tuple[TextBubbleDetector, MangaOcr]:
    global _DET_MODEL, _MOCR

    with _MODEL_LOCK:
        if _DET_MODEL is not None and _MOCR is not None:
            return _DET_MODEL, _MOCR

        device, use_cuda = _resolve_device()

        if _DET_MODEL is None:
            processor = AutoImageProcessor.from_pretrained(
                str(TEXT_BUBBLE_MODEL_PATH),
                local_files_only=True,
                use_fast=False,
            )
            model = AutoModelForObjectDetection.from_pretrained(
                str(TEXT_BUBBLE_MODEL_PATH),
                local_files_only=True,
            ).to(device)
            model.eval()
            _DET_MODEL = TextBubbleDetector(
                processor=processor,
                model=model,
                device=device,
                label_id=_resolve_label_id(model),
            )
            logger.info(f"Text bubble detector loaded on {_DET_MODEL.device}")

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


def get_det_model() -> TextBubbleDetector:
    det_model, _ = warmup_models()
    return det_model


def detect_text_bubbles(image_cv: np.ndarray) -> np.ndarray:
    detector = get_det_model()
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    inputs = detector.processor(images=image, return_tensors="pt")
    inputs = {
        key: value.to(detector.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = detector.model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=detector.device)
    result = detector.processor.post_process_object_detection(
        outputs,
        threshold=TEXT_BUBBLE_CONFIDENCE,
        target_sizes=target_sizes,
    )[0]

    labels = result["labels"].detach().cpu().numpy()
    boxes = result["boxes"].detach().cpu().numpy()
    text_bubble_boxes = boxes[labels == detector.label_id]
    return text_bubble_boxes.astype(np.float32, copy=False)


def get_mocr() -> MangaOcr:
    _, mocr = warmup_models()
    return mocr
