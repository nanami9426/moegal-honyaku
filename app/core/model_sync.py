import os
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from filelock import FileLock
from huggingface_hub import hf_hub_download

from app.core.logger import logger
from app.core.paths import ASSETS_DIR, MODELS_DIR

HF_ENDPOINT = os.getenv("HF_ENDPOINT", os.getenv("HF_BASE_URL", "https://hf-mirror.com")).rstrip("/")
MANGA_OCR_REPO_ID = os.getenv("MANGA_OCR_REPO_ID", "kha-white/manga-ocr-base")
MANGA_OCR_MODEL_DIR = "manga-ocr-base"
COMIC_SEGMENTER_REPO_ID = os.getenv("COMIC_SEGMENTER_REPO_ID", "ogkalu/comic-text-segmenter-yolov8m")
COMIC_SEGMENTER_FILENAME = "comic-text-segmenter.pt"
COMIC_SEGMENTER_RELATIVE_PATH = "comic-text-segmenter.pt"
MODELS_MANIFEST_PATH = ASSETS_DIR / "models_manifest.txt"
SYNC_LOCK_PATH = MODELS_DIR / ".sync.lock"
SYNC_LOCK_TIMEOUT_SECONDS = 600

def _format_size(size_in_bytes: int) -> str:
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    units = ("KB", "MB", "GB", "TB")
    size = float(size_in_bytes)
    for unit in units:
        size /= 1024.0
        if size < 1024.0:
            return f"{size:.2f} {unit}"
    return f"{size:.2f} PB"


def _normalize_relative_path(raw_path: str) -> str:
    path = unquote(raw_path).strip().replace("\\", "/")
    path = path.split("?", 1)[0].split("#", 1)[0]
    path = path.strip("/")
    if not path:
        return ""
    normalized = PurePosixPath(path)
    if any(part in ("", ".", "..") for part in normalized.parts):
        raise ValueError(f"非法路径: {raw_path}")
    return normalized.as_posix()


def _load_models_manifest() -> list[str]:
    if not MODELS_MANIFEST_PATH.is_file():
        raise RuntimeError(f"模型清单不存在: {MODELS_MANIFEST_PATH}")

    required_files: list[str] = []
    for raw_line in MODELS_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        normalized = _normalize_relative_path(line)
        if normalized == ".gitkeep":
            continue
        required_files.append(normalized)

    required_files = sorted(set(required_files))
    if not required_files:
        raise RuntimeError(f"模型清单为空: {MODELS_MANIFEST_PATH}")
    return required_files


def _resolve_hf_download_target(relative_path: str) -> tuple[str, str, Path]:
    if relative_path == COMIC_SEGMENTER_RELATIVE_PATH:
        return COMIC_SEGMENTER_REPO_ID, COMIC_SEGMENTER_FILENAME, MODELS_DIR

    manga_prefix = f"{MANGA_OCR_MODEL_DIR}/"
    if relative_path.startswith(manga_prefix):
        repo_relative_path = relative_path.removeprefix(manga_prefix)
        if not repo_relative_path:
            raise RuntimeError(f"非法模型路径: {relative_path}")
        return MANGA_OCR_REPO_ID, repo_relative_path, MODELS_DIR / MANGA_OCR_MODEL_DIR

    raise RuntimeError(f"未配置下载地址的模型文件: {relative_path}")


def _download_single_file(relative_path: str) -> None:
    repo_id, filename, local_dir = _resolve_hf_download_target(relative_path)
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"开始下载 {relative_path} (repo={repo_id}, endpoint={HF_ENDPOINT})")

    download_kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": str(local_dir),
        "endpoint": HF_ENDPOINT,
    }

    hf_hub_download(**download_kwargs)

    local_path = MODELS_DIR / Path(relative_path)
    if not local_path.is_file():
        raise RuntimeError(f"下载完成但未找到模型文件: {local_path}")
    logger.info(f"下载完成 {relative_path}，大小 {_format_size(local_path.stat().st_size)}")


def ensure_models_ready() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with FileLock(str(SYNC_LOCK_PATH), timeout=SYNC_LOCK_TIMEOUT_SECONDS):
        logger.info(f"开始检查模型文件完整性: {MODELS_DIR}")
        required_files = _load_models_manifest()
        missing_files = [p for p in required_files if not (MODELS_DIR / p).is_file()]
        if not missing_files:
            logger.info(f"本地模型完整")
            return

        logger.warning(
            f"本地模型缺失 {len(missing_files)}/{len(required_files)} 个文件，开始下载缺失文件"
        )

        required_dirs = sorted(
            {
                str(PurePosixPath(relative_path).parent)
                for relative_path in missing_files
                if str(PurePosixPath(relative_path).parent) not in ("", ".")
            }
        )
        for relative_dir in required_dirs:
            (MODELS_DIR / relative_dir).mkdir(parents=True, exist_ok=True)

        total = len(missing_files)
        for index, relative_path in enumerate(missing_files, start=1):
            _download_single_file(relative_path)
            logger.info(f"模型缺失文件下载 {index}/{total}: {relative_path}")

        logger.info(f"模型缺失文件下载完成，共 {len(missing_files)} 个文件")
