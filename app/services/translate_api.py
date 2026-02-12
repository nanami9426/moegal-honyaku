import asyncio
import json
import os
import re

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# 统一翻译风格：仅输出翻译内容，不附加解释。
TRANSLATE_SYSTEM_PROMPT = "将句子翻译成中文（如果是符号就直接输出，不要加任何解释、注解或括号内容，仅保留自然对话或原声风格的翻译。）"
TRANSLATE_STRUCTURED_SYSTEM_PROMPT = (
    "你会收到一个 JSON 数组，每项是待翻译句子。"
    "请返回 JSON：{\"result\": [\"翻译1\", \"翻译2\", ...]}。"
    "只输出 JSON，不要输出任何多余文本。"
)


# DashScope OpenAI 兼容接口配置，可通过 .env 覆盖默认值。
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen3-max")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai-proxy.org/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 复用单个异步客户端，避免重复建立连接。
ALI_CLIENT = AsyncOpenAI(
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_BASE_URL,
)

OPENAI_CLIENT = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


def _normalize_content(content) -> str:
    # 兼容字符串或多段内容结构，统一为纯文本。
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content or "").strip()


def _provider_options(api_type: str):
    if api_type == "dashscope":
        if not DASHSCOPE_API_KEY:
            raise RuntimeError("DASHSCOPE_API_KEY 未配置")
        return ALI_CLIENT, DASHSCOPE_MODEL, {"extra_body": {"enable_thinking": False}}
    if api_type == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未配置")
        return OPENAI_CLIENT, OPENAI_MODEL, {}
    raise RuntimeError(f"不支持的 translate_api_type: {api_type}")


def _extract_json_payload(raw: str):
    text = raw.strip()
    candidates = [text]

    # 兼容 ```json ... ``` 包裹。
    if text.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        stripped = re.sub(r"\n?```$", "", stripped)
        candidates.append(stripped.strip())

    # 兼容模型前后夹杂说明文本的情况。
    obj_match = re.search(r"\{[\s\S]*\}", text)
    arr_match = re.search(r"\[[\s\S]*\]", text)
    if obj_match:
        candidates.append(obj_match.group(0))
    if arr_match:
        candidates.append(arr_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise RuntimeError("结构化输出解析失败，模型返回非 JSON")


def _parse_structured_result(raw: str, expected_count: int):
    payload = _extract_json_payload(raw)
    if isinstance(payload, dict):
        result = payload.get("result")
    elif isinstance(payload, list):
        result = payload
    else:
        result = None
    if not isinstance(result, list):
        raise RuntimeError("结构化输出格式错误，缺少 result 列表")
    normalized = [str(item).strip() for item in result]
    if len(normalized) != expected_count:
        raise RuntimeError(
            f"结构化输出数量不匹配，期望 {expected_count} 条，实际 {len(normalized)} 条"
        )
    return normalized


async def _translate_single(sentence: str, api_type: str):
    if not sentence:
        return "", 0.0
    client, model, extra_kwargs = _provider_options(api_type)
    res = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ],
        **extra_kwargs,
    )
    content = _normalize_content(res.choices[0].message.content)
    return content, 0.0


async def _translate_parallel(all_text, api_type: str):
    # 并行模式：每个句子独立请求，整体用 gather 并发。
    tasks = [_translate_single(sentence, api_type) for sentence in all_text]
    if not tasks:
        return [], 0.0
    res = await asyncio.gather(*tasks)
    res_text, prices = zip(*res)
    return list(res_text), sum(prices)


async def _translate_structured(all_text, api_type: str):
    # 结构化模式：一次性输入列表，要求模型返回翻译列表 JSON。
    if not all_text:
        return [], 0.0
    client, model, extra_kwargs = _provider_options(api_type)
    payload = json.dumps(all_text, ensure_ascii=False)
    res = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TRANSLATE_STRUCTURED_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        **extra_kwargs,
    )
    raw = _normalize_content(res.choices[0].message.content)
    return _parse_structured_result(raw, len(all_text)), 0.0


async def translate_req(all_text, api_type: str = "dashscope", translate_mode: str = "parallel"):
    if translate_mode == "parallel":
        return await _translate_parallel(all_text, api_type)
    if translate_mode == "structured":
        return await _translate_structured(all_text, api_type)
    raise RuntimeError(f"不支持的 translate_mode: {translate_mode}")
