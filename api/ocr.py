from ultralytics import YOLO
from manga_ocr import MangaOcr
from utils.logger import logger
import torch

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DET_MODEL_PATH = "assets/models/comic-text-segmenter.pt"
DET_MODEL = YOLO(DET_MODEL_PATH).to(DEVICE)
logger.info(f"气泡检测模型加载成功，使用：{DET_MODEL.device}")

MOCR = MangaOcr(pretrained_model_name_or_path="assets/models/manga-ocr-base")