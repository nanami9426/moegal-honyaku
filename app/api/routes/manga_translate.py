import asyncio
import base64
import json
import random
import time
from typing import Literal, cast

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from app.core.custom_conf import custom_conf
from app.core.logger import logger
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


def _decode_image(file_bytes: bytes):
    if not file_bytes:
        raise TranslateWebInputError(400, "图片为空")
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr_cv is None:
        raise TranslateWebInputError(400, "图片解码失败，请确认输入为有效图片")
    return img_bgr_cv, Image.fromarray(img_bgr_cv)


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
):
    from app.services.ocr import get_det_model
    from app.services.pic_process import draw_text_on_boxes, get_text_masked_pic, save_img
    from app.services.translate_api import translate_req

    price = -0.0001
    img_bgr_cv, img_pil = _decode_image(file_bytes)
    det_model = get_det_model()
    res = det_model(img_bgr_cv, verbose=False)
    bboxes = res[0].boxes.xyxy.cpu().numpy()
    all_text, inpaint = await get_text_masked_pic(img_pil, img_bgr_cv, bboxes, True)
    if len(all_text) == 0:
        logger.warning("未检测出文字")
        return None, None, None, None

    cn_text, price = await translate_req(
        all_text,
        api_type=custom_conf.translate_api_type,
        translate_mode=custom_conf.translate_mode,
    )
    img_res = draw_text_on_boxes(inpaint, bboxes, cn_text, text_direction=text_direction)
    ok, buffer = cv2.imencode(".png", img_res)
    if not ok:
        raise RuntimeError("结果图片编码失败")

    cn_file_bytes = buffer.tobytes()
    file_name = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}.png"
    background_tasks.add_task(save_img, cn_file_bytes, "cn", file_name)
    background_tasks.add_task(save_img, file_bytes, "raw", file_name)
    b64_img = base64.b64encode(cn_file_bytes).decode("utf8") if include_res_img else None
    return all_text, cn_text, price, (b64_img, file_name)


def _error_response(info: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        content={
            "status": "error",
            "info": info,
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
):
    start = time.time()
    try:
        text_direction_value = _normalize_text_direction(text_direction)
        file_bytes = await img.read()
        all_text, cn_text, price, img_result = await _translate_image_bytes(
            file_bytes=file_bytes,
            include_res_img=include_res_img,
            background_tasks=background_tasks,
            text_direction=text_direction_value,
        )
        if all_text is None:
            return JSONResponse(content={
                "status": "error",
                "info": "未检测出文字",
            })
    except ValueError as exc:
        return _error_response(str(exc), 400)
    except Exception as e:
        logger.error(f"翻译失败：{e}")
        return JSONResponse(content={
            "status": "error",
            "info": f"{e}",
        })
    b64_img, file_name = img_result
    duration = round(time.time() - start, 2)
    logger.info(f"翻译图片成功，耗时 {duration} 秒，保存为{file_name}")
    return JSONResponse(content={
        "status": "success",
        "duration": duration,
        "price": round(price, 8),
        "cn_text": cn_text,
        "raw_text": all_text,
        "res_img": b64_img,
    })


class TranslateWebRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    referer: str
    source_type: Literal["img", "canvas"] | None = None
    include_res_img: bool = True
    text_direction: TextDirection = "horizontal"

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
        if req.image_url is not None:
            file_bytes = await _download_image_bytes(req.image_url, req.referer)
        else:
            file_bytes = decode_image_base64_data_url(req.image_base64)
        all_text, cn_text, price, img_result = await _translate_image_bytes(
            file_bytes=file_bytes,
            include_res_img=req.include_res_img,
            background_tasks=background_tasks,
            text_direction=req.text_direction,
        )
        if all_text is None:
            return JSONResponse(content={
                "status": "error",
                "info": "未检测出文字",
            })
        b64_img, file_name = img_result
        duration = round(time.time() - start, 2)
        logger.info(f"翻译图片成功，耗时 {duration} 秒，保存为{file_name}")
        return JSONResponse(content={
            "status": "success",
            "duration": duration,
            "price": round(price, 8),
            "cn_text": cn_text,
            "raw_text": all_text,
            "res_img": b64_img,
        })
    except json.JSONDecodeError:
        return _error_response("请求体不是合法 JSON", 400)
    except ValidationError as exc:
        return _error_response(_validation_error_message(exc), 400)
    except TranslateWebInputError as exc:
        return _error_response(exc.message, exc.status_code)
    except Exception as e:
        logger.error(f"翻译失败：{e}")
        return _error_response(str(e), 500)
