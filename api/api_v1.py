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
import torch

load_dotenv()
FONT_PATH = "assets/fonts/LXGWWenKai-Regular.ttf"

class FontConfig:
    def __init__(self, max_height, max_width, text, font_path=FONT_PATH):
        self.font_path = font_path
        self.font_size = self._find_font_size(max_height, max_width, len(text))

    def _find_font_size(self, max_height, max_width, n):
        font_size = 10
        while True:
            font = ImageFont.truetype(self.font_path, font_size)
            bbox = font.getbbox("中")
            h = bbox[3] - bbox[1]
            w = bbox[2] - bbox[0]
            if h * w * n >= max_height * max_width * 0.55:
                break
            font_size +=1
        return font_size -1
    
    @property
    def font(self):
        return ImageFont.truetype(self.font_path, self.font_size)


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DET_MODEL_PATH = "assets/models/comic-text-segmenter.pt"
DET_MODEL = YOLO(DET_MODEL_PATH).to(DEVICE)
logger.info(f"气泡检测模型加载成功，使用：{DET_MODEL.device}")

MOCR = MangaOcr(pretrained_model_name_or_path="assets/models/manga-ocr-base")

OPEN_AI_CLIENT = AsyncOpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key=os.getenv("API_KEY_HONYAKU_OPENAI"),
)

class HonyakuEvent(BaseModel):
    result: list[str]

ERINE_HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {os.getenv("API_KEY_HONYAKU_ERNIE")}'
}

ERNIE_PROMPT_PRICE = 0.003 / 1000
ERNIE_COMPLETION_PRICE = 0.009 / 1000

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
                "content": '将句子翻译成中文（音译词、人名直接用罗马音表示，如果是符号就直接输出，不要加任何解释、注解或括号内容，仅保留自然对话或原声风格的翻译。）'
            },
            {
                "role": "user",
                "content": sentence
            }
        ],
    }

    async with session.post(url, headers=ERINE_HEADERS, json=payload) as response:
        resp_json = await response.json()
        prompt_tokens = resp_json["usage"]["prompt_tokens"]
        completion_tokens = resp_json["usage"]["completion_tokens"]
        return resp_json["choices"][0]["message"]["content"], prompt_tokens*ERNIE_PROMPT_PRICE + completion_tokens*ERNIE_COMPLETION_PRICE

async def translate_req_ernie(all_text):
    async with aiohttp.ClientSession() as session:
        tasks = [translate_req_ernie_single(session, sentence) for sentence in all_text]
        res = await asyncio.gather(*tasks)
        res_text, price = zip(*res)
        return res_text, sum(price)