from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import base64
import requests
from api.translate_api import translate_req_openai, translate_req_ernie
from api.ocr import DET_MODEL
from PIL import Image
from utils.pic_process import get_text_masked_pic, draw_text_on_boxes
import time
from utils.logger import logger
from pydantic import BaseModel

manga_translate_router = APIRouter()

@manga_translate_router.post("/api/v1/translate/upload")
async def translate_upload(img: UploadFile = File(...)):
    start = time.time()
    price = -0.0001
    file_bytes = await img.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_pil = Image.fromarray(img_bgr_cv)
    res = DET_MODEL(img_bgr_cv)
    bboxes = res[0].boxes.xyxy.cpu().numpy()
    all_text, inpaint = await get_text_masked_pic(img_pil, img_bgr_cv, bboxes, False)
    if len(all_text) == 0:
        logger.warning("未检测出文字")
        return JSONResponse(content={
        "status": "error",
        "info": "未检测出文字"
    })
    try:
        # cn_text = await translate_req_openai(all_text)
        cn_text, price = await translate_req_ernie(all_text)
        img_res = draw_text_on_boxes(inpaint, bboxes, cn_text)
    except Exception as e:
        logger.error(f"{e}")
        return JSONResponse(content={
        "status": "error",
        "info": f"{e}"
    })
    _, buffer = cv2.imencode('.png', img_res)
    b64_img = base64.b64encode(buffer).decode("utf8")
    duration = round(time.time() - start, 2)
    logger.info(f"翻译图片成功，耗时 {duration} 秒，花费 {round(price, 8)} 元")
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

@manga_translate_router.post("/api/v1/translate/web")
async def translate_web(req: ImageUrl):
    start = time.time()
    price = -0.0001
    try:
        headers = {
            "Referer": req.referer,
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(req.image_url, headers=headers)
        if resp.status_code != 200:
            return JSONResponse(content={
                "status": "error",
                "info": f"图片下载失败，状态码: {resp.status_code}"
            })

        file_bytes = resp.content
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(img_bgr_cv)
        res = DET_MODEL(img_bgr_cv)
        bboxes = res[0].boxes.xyxy.cpu().numpy()
        all_text, inpaint = await get_text_masked_pic(img_pil, img_bgr_cv, bboxes, False)
        if len(all_text) == 0:
            logger.warning("未检测出文字")
            return JSONResponse(content={
                "status": "error",
                "info": "未检测出文字"
            })
        # cn_text = await translate_req_openai(all_text)
        cn_text, price = await translate_req_ernie(all_text)
        img_res = draw_text_on_boxes(inpaint, bboxes, cn_text)
        _, buffer = cv2.imencode('.png', img_res)
        b64_img = base64.b64encode(buffer).decode("utf8")
        duration = round(time.time() - start, 2)
        logger.info(f"翻译图片成功，耗时 {duration} 秒，花费 {round(price, 8)} 元")
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
