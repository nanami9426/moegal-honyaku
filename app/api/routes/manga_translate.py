import asyncio
import base64
import json
import random
import time
from typing import Literal, cast

import httpx
from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from app.core.logger import logger
from app.services.translate_api import MissingTranslateProviderConfigError
from app.services.image_translation import TranslationBusyError, translate_image
from app.services.web_image_input import (
    TranslateWebInputError,
    decode_image_base64_data_url,
    ensure_body_size_within_limit,
)

manga_translate_router = APIRouter()

DOWNLOAD_RETRY_COUNT = 2
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)
TEXT_DIRECTION_OPTIONS = ("horizontal", "vertical")
TextDirection = Literal["horizontal", "vertical"]


async def _download_image_bytes(image_url: str, referer: str) -> bytes:
    headers = {
        "Referer": referer,
        "User-Agent": "Mozilla/5.0",
    }
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
        last_error: Exception | None = None
        for attempt in range(DOWNLOAD_RETRY_COUNT + 1):
            try:
                response = await client.get(image_url, headers=headers)
                response.raise_for_status()
                return response.content
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if 400 <= status_code < 500 and status_code != 429:
                    raise RuntimeError(f"图片下载失败，状态码: {status_code}") from exc
                last_error = RuntimeError(f"图片下载失败，状态码: {status_code}")
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPError) as exc:
                last_error = exc
            if attempt < DOWNLOAD_RETRY_COUNT:
                await asyncio.sleep(0.25 * (attempt + 1))
    raise RuntimeError(f"图片下载失败：{last_error}")


def _normalize_text_direction(value) -> TextDirection:
    if value is None:
        return "horizontal"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return "horizontal"
        if normalized in TEXT_DIRECTION_OPTIONS:
            return cast(TextDirection, normalized)
    raise ValueError(f"text_direction 必须是 {TEXT_DIRECTION_OPTIONS}")


async def _translate_image_bytes(
    file_bytes: bytes,
    include_res_img: bool,
    background_tasks: BackgroundTasks,
    text_direction: TextDirection = "horizontal",
    force_refresh: bool = False,
):
    result = await translate_image(file_bytes, text_direction, force_refresh=force_refresh)
    metadata = {key: result[key] for key in ("timings", "cache_hit", "coalesced")}
    if result.get("code") == "NO_TEXT_BUBBLES":
        return {"status": "skipped", "code": "NO_TEXT_BUBBLES", "info": "未检测出文字气泡", **metadata}

    from app.services.pic_process import save_img

    cn_file_bytes = result["image_bytes"]
    if not result["cache_hit"] and not result["coalesced"]:
        file_name = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}.png"
        background_tasks.add_task(save_img, cn_file_bytes, "cn", file_name)
        background_tasks.add_task(save_img, file_bytes, "raw", file_name)
    b64_img = (await asyncio.to_thread(base64.b64encode, cn_file_bytes)).decode("ascii") if include_res_img else None
    return {"status": "success", "raw_text": result["raw_text"], "cn_text": result["cn_text"],
            "price": round(result["price"], 8), "res_img": b64_img, **metadata}


def _error_response(info: str, status_code: int, code: str = "TRANSLATION_FAILED") -> JSONResponse:
    return JSONResponse(
        content={
            "status": "error",
            "info": info,
            "code": code,
        },
        status_code=status_code,
    )


def _validation_error_message(exc: ValidationError) -> str:
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    if not errors:
        return "请求参数不合法"
    message = str(errors[0].get("msg", "请求参数不合法")).strip()
    if message.startswith("Value error, "):
        message = message.removeprefix("Value error, ").strip()
    return message or "请求参数不合法"


@manga_translate_router.post("/api/v1/translate/upload")
async def translate_upload(
    background_tasks: BackgroundTasks,
    img: UploadFile = File(...),
    include_res_img: bool = True,
    text_direction: str = "horizontal",
    force_refresh: bool = False,
):
    start = time.time()
    try:
        text_direction_value = _normalize_text_direction(text_direction)
        file_bytes = await img.read()
        result = await _translate_image_bytes(
            file_bytes=file_bytes,
            include_res_img=include_res_img,
            background_tasks=background_tasks,
            text_direction=text_direction_value,
            force_refresh=force_refresh,
        )
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except MissingTranslateProviderConfigError as exc:
        return _error_response(str(exc), 400, "MISSING_TRANSLATE_CONFIG")
    except TranslationBusyError as exc:
        return _error_response(str(exc), 429, "QUEUE_FULL")
    except Exception as e:
        logger.error(f"翻译失败：{e}")
        return JSONResponse(content={
            "status": "error",
            "info": f"{e}",
        })
    duration = round(time.time() - start, 2)
    logger.info(f"图片处理 {result['status']}，耗时 {duration} 秒，分阶段 {result['timings']}")
    return JSONResponse(content={**result, "duration": duration})


class TranslateWebRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    referer: str
    source_type: Literal["img", "canvas"] | None = None
    include_res_img: bool = True
    text_direction: TextDirection = "horizontal"
    force_refresh: bool = False

    @field_validator("image_url", "image_base64", mode="before")
    @classmethod
    def _normalize_image_source(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("source_type", mode="before")
    @classmethod
    def _normalize_source_type(cls, value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("text_direction", mode="before")
    @classmethod
    def _normalize_text_direction_field(cls, value):
        return _normalize_text_direction(value)

    @model_validator(mode="after")
    def validate_image_source(self):
        has_url = bool(self.image_url)
        has_base64 = bool(self.image_base64)
        if not has_url and not has_base64:
            raise ValueError("image_url 和 image_base64 不能同时为空")
        if has_url and has_base64:
            raise ValueError("image_url 和 image_base64 不能同时存在")
        return self


@manga_translate_router.post("/api/v1/translate/web")
async def translate_web(request: Request, background_tasks: BackgroundTasks):
    start = time.time()
    try:
        ensure_body_size_within_limit(content_length=request.headers.get("content-length"))
        body = await request.body()
        ensure_body_size_within_limit(actual_size=len(body))
        if not body:
            raise TranslateWebInputError(400, "请求体不能为空")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise TranslateWebInputError(400, "请求体必须是 JSON 对象")

        req = TranslateWebRequest.model_validate(payload)
        download_start = time.perf_counter()
        if req.image_url is not None:
            file_bytes = await _download_image_bytes(req.image_url, req.referer)
        else:
            file_bytes = decode_image_base64_data_url(req.image_base64)
        download_duration = round(time.perf_counter() - download_start, 3)
        result = await _translate_image_bytes(
            file_bytes=file_bytes,
            include_res_img=req.include_res_img,
            background_tasks=background_tasks,
            text_direction=req.text_direction,
            force_refresh=req.force_refresh,
        )
        result["timings"]["download"] = download_duration
        duration = round(time.time() - start, 2)
        logger.info(f"图片处理 {result['status']}，耗时 {duration} 秒，分阶段 {result['timings']}")
        return JSONResponse(content={**result, "duration": duration})
    except json.JSONDecodeError:
        return _error_response("请求体不是合法 JSON", 400)
    except ValidationError as exc:
        return _error_response(_validation_error_message(exc), 400)
    except TranslateWebInputError as exc:
        return _error_response(exc.message, exc.status_code)
    except MissingTranslateProviderConfigError as exc:
        return _error_response(str(exc), 400, "MISSING_TRANSLATE_CONFIG")
    except TranslationBusyError as exc:
        return _error_response(str(exc), 429, "QUEUE_FULL")
    except Exception as e:
        logger.error(f"翻译失败：{e}")
        return _error_response(str(e), 500)
