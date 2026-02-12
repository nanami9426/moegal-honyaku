from app.core.logger import logger

TRANSLATE_API_TYPE_OPTIONS = ("dashscope", "openai")
TRANSLATE_MODE_OPTIONS = ("parallel", "structured")


class CustomConf:
    def __init__(
            self,
            # 默认使用 OpenAI，前端可改为 dashscope。
            translate_api_type="openai",
            # parallel: 每句并发请求；structured: 单请求列表输入输出。
            translate_mode="parallel",
            ):
        self.translate_api_type = translate_api_type
        self.translate_mode = translate_mode

    def update_conf(self, attr, v):
        assert hasattr(self, attr), f"attr '{attr}' is not exists."
        if attr == "translate_api_type":
            assert v in TRANSLATE_API_TYPE_OPTIONS, (
                f"translate_api_type 必须是 {TRANSLATE_API_TYPE_OPTIONS}"
            )
        if attr == "translate_mode":
            assert v in TRANSLATE_MODE_OPTIONS, (
                f"translate_mode 必须是 {TRANSLATE_MODE_OPTIONS}"
            )
        setattr(self, attr, v)
        logger.info(f"将 {attr} 设置为 {v}")
        return {
            attr: getattr(self, attr, None),
            "status": "success"
        }

    def to_dict(self, exclude=None):
        exclude = exclude or []
        assert isinstance(exclude, list)
        return {
            k: v for k, v in self.__dict__.items() if k not in exclude
        }

custom_conf = CustomConf()
