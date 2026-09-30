"""图片翻译流水线：重叠本地计算和接口等待，并合并重复任务。"""

import asyncio
from collections import OrderedDict
import hashlib
import os
import time

import cv2
import numpy as np
from PIL import Image

from app.core.custom_conf import custom_conf
from app.services.translate_api import translate_req, translation_config_key
from app.services.web_image_input import TranslateWebInputError


# 只限制相应阶段，不让等待网络的图片占用模型名额。
IMAGE_JOBS = asyncio.Semaphore(max(1, int(os.getenv("IMAGE_TRANSLATE_CONCURRENCY", "3"))))
MODEL_JOBS = asyncio.Semaphore(1)
IMAGE_WORKERS = asyncio.Semaphore(2)
MAX_PENDING = 16
CACHE_MAX_BYTES = 128 * 1024 * 1024
CACHE_MAX_ITEMS = 32
CACHE_TTL = 600
_cache = OrderedDict()
_inflight = {}
_cache_bytes = 0


class TranslationBusyError(RuntimeError):
    pass


def _decode_image(file_bytes):
    image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise TranslateWebInputError(400, "图片解码失败，请确认输入为有效图片")
    return image, Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _render_image(image, boxes, texts, direction):
    from app.services.pic_process import draw_text_on_boxes

    rendered = draw_text_on_boxes(image, boxes, texts, text_direction=direction)
    ok, buffer = cv2.imencode(".png", rendered)
    if not ok:
        raise RuntimeError("结果图片编码失败")
    return buffer.tobytes()


async def _image_step(name, timings, function, *args):
    async with IMAGE_WORKERS:
        start = time.perf_counter()
        value = await asyncio.to_thread(function, *args)
        timings[name] = round(time.perf_counter() - start, 3)
        return value


async def _translate(file_bytes, direction, provider, mode):
    from app.services.ocr import detect_text_bubbles
    from app.services.pic_process import recognize_text_regions
    from app.services.text_erasure import erase_text_regions

    start = time.perf_counter()
    timings = {}
    async with IMAGE_JOBS:
        timings["queue"] = round(time.perf_counter() - start, 3)
        image, pil = await _image_step("decode", timings, _decode_image, file_bytes)
        async with MODEL_JOBS:
            stage = time.perf_counter()
            boxes = await asyncio.to_thread(detect_text_bubbles, image)
            timings["detect"] = round(time.perf_counter() - stage, 3)
        if len(boxes) == 0:
            # 无气泡是正常跳过；不启动 OCR、擦除和付费翻译请求。
            timings["total"] = round(time.perf_counter() - start, 3)
            return {"code": "NO_TEXT_BUBBLES", "raw_text": [], "cn_text": [],
                    "image_bytes": b"", "price": 0, "timings": timings}

        erase_task = asyncio.create_task(_image_step("erase", timings, erase_text_regions, image, boxes))
        try:
            async with MODEL_JOBS:
                stage = time.perf_counter()
                raw_text = await recognize_text_regions(pil, image, boxes)
                timings["ocr"] = round(time.perf_counter() - stage, 3)
            stage = time.perf_counter()
            cn_text, price = await translate_req(raw_text, api_type=provider, translate_mode=mode)
            timings["translate"] = round(time.perf_counter() - stage, 3)
            cleaned, _ = await erase_task
            image_bytes = await _image_step("render", timings, _render_image, cleaned, boxes, cn_text, direction)
        finally:
            # 计算线程不能强制取消；等待已经开始的擦除结束，避免提前释放计算限额。
            await asyncio.shield(erase_task)
        timings["total"] = round(time.perf_counter() - start, 3)
        return {"raw_text": raw_text, "cn_text": cn_text, "image_bytes": image_bytes,
                "price": price, "timings": timings}


def _expire_cache():
    global _cache_bytes
    now = time.monotonic()
    for key, (expires, size, _) in list(_cache.items()):
        if expires <= now:
            _cache_bytes -= size
            del _cache[key]


async def _run_and_cache(key, file_bytes, direction, provider, mode):
    global _cache_bytes
    try:
        result = await _translate(file_bytes, direction, provider, mode)
        size = len(result["image_bytes"]) + sum(len(text.encode()) for text in result["raw_text"] + result["cn_text"])
        _expire_cache()
        if size <= CACHE_MAX_BYTES:
            while _cache and (len(_cache) >= CACHE_MAX_ITEMS or _cache_bytes + size > CACHE_MAX_BYTES):
                _, (_, removed_size, _) = _cache.popitem(last=False)
                _cache_bytes -= removed_size
            _cache[key] = (time.monotonic() + CACHE_TTL, size, result)
            _cache_bytes += size
        return result
    finally:
        _inflight.pop(key, None)


async def translate_image(file_bytes: bytes, direction: str, force_refresh: bool = False):
    global _cache_bytes
    if not file_bytes:
        raise TranslateWebInputError(400, "图片为空")
    # 入队时固定配置，防止等待期间切换设置导致缓存键和实际结果不一致。
    provider, mode = custom_conf.translate_api_type, custom_conf.translate_mode
    key = (hashlib.sha256(file_bytes).hexdigest(), direction, provider, mode,
           translation_config_key(provider), custom_conf.use_gpu)
    _expire_cache()
    if force_refresh and key in _cache:
        _, size, _ = _cache.pop(key)
        _cache_bytes -= size
    if key in _cache:
        _cache.move_to_end(key)
        return {**_cache[key][2], "cache_hit": True, "coalesced": False, "timings": {}}
    task = _inflight.get(key)
    coalesced = task is not None
    if task is None:
        if len(_inflight) >= MAX_PENDING:
            raise TranslationBusyError("翻译队列已满，请稍后重试")
        task = asyncio.create_task(_run_and_cache(key, file_bytes, direction, provider, mode))
        _inflight[key] = task
        # 即使所有浏览器请求都已断开，也消费异常，并由任务自身负责清理和缓存。
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
    result = await asyncio.shield(task)
    return {**result, "cache_hit": False, "coalesced": coalesced, "timings": dict(result["timings"])}
