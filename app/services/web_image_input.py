import base64
import binascii
import os
import re

TRANSLATE_WEB_MAX_BODY_BYTES = max(1, int(os.getenv("TRANSLATE_WEB_MAX_BODY_BYTES", "20971520")))
SUPPORTED_IMAGE_MIME_TYPES = {"image/png"}
DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[^;,]+);base64,(?P<data>.+)$", re.IGNORECASE)


class TranslateWebInputError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def ensure_body_size_within_limit(*, content_length: str | None = None, actual_size: int | None = None) -> None:
    if content_length is not None:
        try:
            parsed_content_length = int(content_length)
        except (TypeError, ValueError):
            parsed_content_length = None
        if parsed_content_length is not None and parsed_content_length > TRANSLATE_WEB_MAX_BODY_BYTES:
            raise TranslateWebInputError(413, "请求体过大")

    if actual_size is not None and actual_size > TRANSLATE_WEB_MAX_BODY_BYTES:
        raise TranslateWebInputError(413, "请求体过大")


def decode_image_base64_data_url(image_base64: str) -> bytes:
    if not isinstance(image_base64, str):
        raise TranslateWebInputError(400, "image_base64 不是合法的 data URL")

    match = DATA_URL_PATTERN.fullmatch(image_base64.strip())
    if match is None:
        raise TranslateWebInputError(400, "image_base64 不是合法的 data URL")

    mime_type = match.group("mime").strip().lower()
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise TranslateWebInputError(415, f"MIME 类型不支持: {mime_type}")

    encoded_payload = "".join(match.group("data").split())
    if not encoded_payload:
        raise TranslateWebInputError(400, "图片为空")

    try:
        file_bytes = base64.b64decode(encoded_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TranslateWebInputError(400, "base64 解码失败") from exc

    if not file_bytes:
        raise TranslateWebInputError(400, "图片为空")

    ensure_body_size_within_limit(actual_size=len(file_bytes))
    return file_bytes
