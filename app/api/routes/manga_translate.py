import asyncio
import base64
import random
import time

import cv2
import httpx
import numpy as np
from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from app.core.custom_conf import custom_conf
from app.core.logger import logger
from app.services.ocr import DET_MODEL
from app.services.pic_process import get_text_masked_pic, draw_text_on_boxes, save_img
from app.services.translate_api import translate_req

manga_translate_router = APIRouter()

DOWNLOAD_RETRY_COUNT = 2
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=10.0, pool=5.0)


def _decode_image(file_bytes: bytes):
    if not file_bytes:
        raise RuntimeError("图片为空")
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr_cv is None:
        raise RuntimeError("图片解码失败，请确认输入为有效图片")
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


async def _translate_image_bytes(
    file_bytes: bytes,
    include_res_img: bool,
    background_tasks: BackgroundTasks,
):
    price = -0.0001
    img_bgr_cv, img_pil = _decode_image(file_bytes)
    res = DET_MODEL(img_bgr_cv, verbose=False)
    bboxes = res[0].boxes.xyxy.cpu().numpy()
    all_text, inpaint = await get_text_masked_pic(img_pil, img_bgr_cv, bboxes, False)
    if len(all_text) == 0:
        logger.warning("未检测出文字")
        return None, None, None, None

    cn_text, price = await translate_req(
        all_text,
        api_type=custom_conf.translate_api_type,
        translate_mode=custom_conf.translate_mode,
    )
    img_res = draw_text_on_boxes(inpaint, bboxes, cn_text)
    ok, buffer = cv2.imencode(".png", img_res)
    if not ok:
        raise RuntimeError("结果图片编码失败")

    cn_file_bytes = buffer.tobytes()
    file_name = f"{int(time.time() * 1000)}_{random.randint(1000, 9999)}.png"
    background_tasks.add_task(save_img, cn_file_bytes, "cn", file_name)
    background_tasks.add_task(save_img, file_bytes, "raw", file_name)
    b64_img = base64.b64encode(cn_file_bytes).decode("utf8") if include_res_img else None
    return all_text, cn_text, price, (b64_img, file_name)


@manga_translate_router.post("/api/v1/translate/upload")
async def translate_upload(
    background_tasks: BackgroundTasks,
    img: UploadFile = File(...),
    include_res_img: bool = True,
):
    start = time.time()
    try:
        file_bytes = await img.read()
        all_text, cn_text, price, img_result = await _translate_image_bytes(
            file_bytes=file_bytes,
            include_res_img=include_res_img,
            background_tasks=background_tasks,
        )
        if all_text is None:
            return JSONResponse(content={
                "status": "error",
                "info": "未检测出文字",
            })
    except Exception as e:
        logger.error(f"翻译失败：{e}")
        return JSONResponse(content={
            "status": "error",
            "info": f"{e}",
        })
    b64_img, file_name = img_result
    duration = round(time.time() - start, 2)
    logger.info(f"翻译图片成功，耗时 {duration} 秒，花费 {round(price, 8)} 元，保存为{file_name}")
    return JSONResponse(content={
        "status": "success",
        "duration": duration,
        "price": round(price, 8),
        "cn_text": cn_text,
        "raw_text": all_text,
        "res_img": b64_img,
    })

class ImageUrl(BaseModel):
    image_url: str
    referer: str
    include_res_img: bool = True

@manga_translate_router.post("/api/v1/translate/web")
async def translate_web(req: ImageUrl, background_tasks: BackgroundTasks):
    start = time.time()
    try:
        file_bytes = await _download_image_bytes(req.image_url, req.referer)
        all_text, cn_text, price, img_result = await _translate_image_bytes(
            file_bytes=file_bytes,
            include_res_img=req.include_res_img,
            background_tasks=background_tasks,
        )
        if all_text is None:
            return JSONResponse(content={
                "status": "error",
                "info": "未检测出文字",
            })
        b64_img, file_name = img_result
        duration = round(time.time() - start, 2)
        logger.info(f"翻译图片成功，耗时 {duration} 秒，花费 {round(price, 8)} 元，保存为{file_name}")
        return JSONResponse(content={
            "status": "success",
            "duration": duration,
            "price": round(price, 8),
            "cn_text": cn_text,
            "raw_text": all_text,
            "res_img": b64_img,
        })
    except Exception as e:
        logger.error(f"翻译失败：{e}")
        return JSONResponse(content={
            "status": "error",
            "info": str(e)
        }, status_code=500)
