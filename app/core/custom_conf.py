import os

from dotenv import load_dotenv

from app.core.logger import logger

load_dotenv()

TRANSLATE_API_TYPE_OPTIONS = ("dashscope", "custom")
TRANSLATE_MODE_OPTIONS = ("parallel", "structured")


def _is_true_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_USE_GPU = _is_true_env("MOEGAL_USE_GPU", default=False)


class CustomConf:
    def __init__(
            self,
            # 默认使用自定义兼容接口，前端可改为 dashscope。
            translate_api_type="custom",
            # parallel: 每句并发请求；structured: 单请求列表输入输出。
            translate_mode="parallel",
            # 可由前端在运行时切换；环境变量只决定服务启动时的默认值。
            use_gpu=DEFAULT_USE_GPU,
            ):
        self.translate_api_type = translate_api_type
        self.translate_mode = translate_mode
        self.use_gpu = use_gpu

    def update_conf(self, attr, v):
        if not hasattr(self, attr):
            raise ValueError(f"attr '{attr}' is not exists.")
        if attr == "translate_api_type" and v not in TRANSLATE_API_TYPE_OPTIONS:
            raise ValueError(
                f"translate_api_type 必须是 {TRANSLATE_API_TYPE_OPTIONS}"
            )
        if attr == "translate_mode" and v not in TRANSLATE_MODE_OPTIONS:
            raise ValueError(
                f"translate_mode 必须是 {TRANSLATE_MODE_OPTIONS}"
            )
        if attr == "use_gpu" and type(v) is not bool:
            raise ValueError("use_gpu 必须是布尔值")
        setattr(self, attr, v)
        logger.info(f"将 {attr} 设置为 {v}")
        return {
            attr: getattr(self, attr, None),
            "status": "success"
        }

    def to_dict(self, exclude=None):
        exclude = exclude or []
        if not isinstance(exclude, list):
            raise ValueError("exclude 必须是 list")
        return {
            k: v for k, v in self.__dict__.items() if k not in exclude
        }


custom_conf = CustomConf()
