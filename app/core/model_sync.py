import os
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote, urljoin

import httpx
from filelock import FileLock
from tqdm import tqdm

from app.core.logger import logger
from app.core.paths import ASSETS_DIR, MODELS_DIR

REMOTE_MODELS_URL = os.getenv(
    "REMOTE_MODELS_URL",
    "https://moegal.top:5552/moegal_honyaku/models/",
).rstrip("/") + "/"
MODELS_MANIFEST_PATH = ASSETS_DIR / "models_manifest.txt"
SYNC_LOCK_PATH = MODELS_DIR / ".sync.lock"
SYNC_LOCK_TIMEOUT_SECONDS = 600
SYNC_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=120.0, pool=10.0)
DOWNLOAD_CHUNK_SIZE = 1024 * 256
UNKNOWN_SIZE_PROGRESS_STEP = 8 * 1024 * 1024

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


def _relative_path_to_remote_url(relative_path: str) -> str:
    if not relative_path:
        return REMOTE_MODELS_URL
    encoded = "/".join(quote(part) for part in PurePosixPath(relative_path).parts)
    return urljoin(REMOTE_MODELS_URL, encoded)


def _download_single_file(client: httpx.Client, relative_path: str) -> None:
    remote_url = _relative_path_to_remote_url(relative_path)
    local_path = MODELS_DIR / Path(relative_path)
    tmp_path = local_path.with_name(f"{local_path.name}.download")

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with client.stream("GET", remote_url) as response:
            response.raise_for_status()
            total_bytes_header = response.headers.get("content-length")
            total_bytes: int | None = None
            if total_bytes_header:
                try:
                    parsed_total = int(total_bytes_header)
                    if parsed_total > 0:
                        total_bytes = parsed_total
                except ValueError:
                    total_bytes = None

            downloaded = 0
            show_progress = sys.stdout.isatty()
            if show_progress:
                progress_desc = f"下载 {relative_path}"
                with tqdm(
                    total=total_bytes,
                    desc=progress_desc,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    dynamic_ncols=True,
                    ascii=True,
                    leave=True,
                    file=sys.stdout,
                ) as progress_bar:
                    with tmp_path.open("wb") as f:
                        for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)
                                chunk_len = len(chunk)
                                downloaded += chunk_len
                                progress_bar.update(chunk_len)
            else:
                if total_bytes is not None:
                    logger.info(f"开始下载 {relative_path}，大小 {_format_size(total_bytes)}")
                else:
                    logger.info(f"开始下载 {relative_path}，大小未知")

                next_percent = 10
                next_unknown_threshold = UNKNOWN_SIZE_PROGRESS_STEP
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total_bytes is not None:
                                percent = min(int(downloaded * 100 / total_bytes), 100)
                                if percent >= next_percent or downloaded >= total_bytes:
                                    logger.info(
                                        f"下载进度 {relative_path}: {percent}% "
                                        f"({_format_size(downloaded)}/{_format_size(total_bytes)})"
                                    )
                                    while next_percent <= percent:
                                        next_percent += 10
                            elif downloaded >= next_unknown_threshold:
                                logger.info(f"下载进度 {relative_path}: {_format_size(downloaded)}")
                                next_unknown_threshold += UNKNOWN_SIZE_PROGRESS_STEP
        tmp_path.replace(local_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


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

        with httpx.Client(timeout=SYNC_TIMEOUT, follow_redirects=True) as client:
            total = len(missing_files)
            for index, relative_path in enumerate(missing_files, start=1):
                _download_single_file(client, relative_path)
                logger.info(f"模型缺失文件下载 {index}/{total}: {relative_path}")

        logger.info(f"模型缺失文件下载完成，共 {len(missing_files)} 个文件")
