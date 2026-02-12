from ultralytics import YOLO
from manga_ocr import MangaOcr
import torch

from app.core.logger import logger
from app.core.paths import MODELS_DIR

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DET_MODEL_PATH = MODELS_DIR / "comic-text-segmenter.pt"
DET_MODEL = YOLO(str(DET_MODEL_PATH)).to(DEVICE)
logger.info(f"气泡检测模型加载成功，使用：{DET_MODEL.device}")

MOCR = MangaOcr(pretrained_model_name_or_path=str(MODELS_DIR / "manga-ocr-base"))
