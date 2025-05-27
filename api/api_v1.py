import json
import requests
from ultralytics import YOLO
from manga_ocr import MangaOcr
from utils.logger import logger
from PIL import ImageFont
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
import asyncio
import aiohttp

load_dotenv()

MOCR = MangaOcr()
DET_MODEL_PATH = "models/comic-text-segmenter.pt"
DET_MODEL = YOLO(DET_MODEL_PATH)
logger.info("模型加载成功")

FONT_PATH = "fonts/LXGWWenKai-Regular.ttf"
DEFAULT_FONT_SIZE = 15
FONT = ImageFont.truetype(FONT_PATH, DEFAULT_FONT_SIZE)
LINE_HEIGHT = FONT.getbbox("中")[3] - FONT.getbbox("中")[1]
logger.info("字体加载成功")

OPEN_AI_CLIENT = AsyncOpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key=os.getenv("API_KEY_HONYAKU_OPENAI"),
)

class HonyakuEvent(BaseModel):
    result: list[str]

ernie_headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.getenv("API_KEY_HONYAKU_ERNIE")}'
}

async def translate_req_openai(text):
    text = f'{text}'
    res = await OPEN_AI_CLIENT.responses.parse(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "将列表中的句子翻译成中文（句子中的音译词、人名等可以直接用罗马音表示）"},
            {
            "role": "user",
            "content": text,
            },
        ],
        text_format=HonyakuEvent
    )
    event = res.output_parsed
    return event.result


async def translate_req_ernie_single(session, sentence):
    url = "https://qianfan.baidubce.com/v2/chat/completions"
    payload = {
        "model": "ernie-4.5-turbo-vl-32k",
        "messages": [
            {
                "role": "system",
                "content": '将句子翻译成中文（句子中的音译词、人名等可以直接用罗马音表示，如果是符号就直接输出），不要加任何多余的说明，直接给出翻译结果'
            },
            {
                "role": "user",
                "content": sentence
            }
        ],
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("API_KEY_HONYAKU_ERNIE")}'
    }

    async with session.post(url, headers=headers, json=payload) as response:
        resp_json = await response.json()
        return resp_json["choices"][0]["message"]["content"]

async def translate_req_ernie(all_text):
    async with aiohttp.ClientSession() as session:
        tasks = [translate_req_ernie_single(session, sentence) for sentence in all_text]
        res_text = await asyncio.gather(*tasks)
        return res_text