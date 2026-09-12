import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from threading import Event
from urllib.parse import unquote

from dotenv import load_dotenv
from filelock import FileLock
from huggingface_hub import configure_http_backend, hf_hub_download
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.logger import logger
from app.core.paths import ASSETS_DIR, MODELS_DIR, PROJECT_ROOT

# 此模块在 OCR 配置之前导入，必须先读取 .env 才能应用下载源设置。
load_dotenv(PROJECT_ROOT / ".env")

OFFICIAL_HF_ENDPOINT = "https://huggingface.co"
HF_ENDPOINT = os.getenv("HF_ENDPOINT", os.getenv("HF_BASE_URL", "https://hf-mirror.com")).rstrip("/")
HF_FALLBACK_ENDPOINT = os.getenv("HF_FALLBACK_ENDPOINT", OFFICIAL_HF_ENDPOINT).rstrip("/")
MANGA_OCR_REPO_ID = os.getenv("MANGA_OCR_REPO_ID", "kha-white/manga-ocr-base")
MANGA_OCR_MODEL_DIR = "manga-ocr-base"
TEXT_BUBBLE_DETECTOR_REPO_ID = os.getenv(
    "TEXT_BUBBLE_DETECTOR_REPO_ID",
    "ogkalu/comic-text-and-bubble-detector",
)
TEXT_BUBBLE_DETECTOR_MODEL_DIR = "comic-text-and-bubble-detector"
MODELS_MANIFEST_PATH = ASSETS_DIR / "models_manifest.txt"
SYNC_LOCK_PATH = MODELS_DIR / ".sync.lock"
SYNC_LOCK_TIMEOUT_SECONDS = 600


def _download_session() -> requests.Session:
    # Hub 的 HEAD 元数据请求默认不重试连接异常；用已有 HTTP 库处理瞬时 TLS 中断。
    # 保留证书校验和系统/环境代理设置，不使用 verify=False 绕过 TLS。
    session = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3, other=3, status=3,
        backoff_factor=0.5, allowed_methods=frozenset({"GET", "HEAD"}),
        status_forcelist=(429, 500, 502, 503, 504),
    )
    for scheme in ("https://", "http://"):
        session.mount(scheme, HTTPAdapter(max_retries=retry))
    return session

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
    detector_prefix = f"{TEXT_BUBBLE_DETECTOR_MODEL_DIR}/"
    if relative_path.startswith(detector_prefix):
        repo_relative_path = relative_path.removeprefix(detector_prefix)
        if not repo_relative_path:
            raise RuntimeError(f"1145141919810: {relative_path}")
        return (
            TEXT_BUBBLE_DETECTOR_REPO_ID,
            repo_relative_path,
            MODELS_DIR / TEXT_BUBBLE_DETECTOR_MODEL_DIR,
        )

    manga_prefix = f"{MANGA_OCR_MODEL_DIR}/"
    if relative_path.startswith(manga_prefix):
        repo_relative_path = relative_path.removeprefix(manga_prefix)
        if not repo_relative_path:
            raise RuntimeError(f"非法模型路径: {relative_path}")
        return MANGA_OCR_REPO_ID, repo_relative_path, MODELS_DIR / MANGA_OCR_MODEL_DIR

    raise RuntimeError(f"未配置下载地址的模型文件: {relative_path}")


def _download_single_file(relative_path: str, fallback_active: Event) -> None:
    repo_id, filename, local_dir = _resolve_hf_download_target(relative_path)
    local_dir.mkdir(parents=True, exist_ok=True)
    endpoint = HF_FALLBACK_ENDPOINT if fallback_active.is_set() else HF_ENDPOINT
    logger.info(f"开始下载 {relative_path} (repo={repo_id}, endpoint={endpoint})")

    download_kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "local_dir": str(local_dir),
        "endpoint": endpoint,
    }

    try:
        hf_hub_download(**download_kwargs)
    except Exception as exc:
        if not HF_FALLBACK_ENDPOINT or HF_FALLBACK_ENDPOINT == endpoint:
            raise
        # 仅在本轮下载内记住失败源；后续任务不再逐文件重试同一个不可用镜像。
        fallback_active.set()
        logger.warning(
            f"Download failed from {endpoint}, switching to {HF_FALLBACK_ENDPOINT}: {exc}"
        )
        download_kwargs["endpoint"] = HF_FALLBACK_ENDPOINT
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
        try:
            workers = max(1, min(8, int(os.getenv("MODEL_DOWNLOAD_WORKERS", "2"))))
        except ValueError:
            logger.warning("MODEL_DOWNLOAD_WORKERS 无效，使用默认并发数 2")
            workers = 2
        workers = min(workers, total)
        logger.info(f"使用 {workers} 个并发任务下载模型；已有文件和下载缓存将继续复用")
        configure_http_backend(backend_factory=_download_session)
        fallback_active = Event()
        # 复用 Hugging Face 的下载、缓存与断点恢复逻辑，只并行处理不同文件。
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_download_single_file, path, fallback_active): path
                for path in missing_files
            }
            try:
                for index, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    logger.info(f"模型缺失文件下载 {index}/{total}: {futures[future]}")
            except Exception:
                # 保留已完成文件；取消尚未开始的任务，下一次启动继续补齐。
                for future in futures:
                    future.cancel()
                raise

        logger.info(f"模型缺失文件下载完成，共 {len(missing_files)} 个文件")
