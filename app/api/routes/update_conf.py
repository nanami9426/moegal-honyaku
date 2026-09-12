from fastapi import APIRouter, HTTPException
from app.core.custom_conf import (
    DEFAULT_INPAINT_BACKEND,
    DEFAULT_USE_GPU,
    INPAINT_BACKEND_OPTIONS,
    custom_conf,
    TRANSLATE_API_TYPE_OPTIONS,
    TRANSLATE_MODE_OPTIONS,
)
from app.services.translate_api import get_provider_status
from pydantic import BaseModel

update_conf_router = APIRouter()

class UpdateItem(BaseModel):
    attr: str
    v: str | float | bool | None = None


def _serialize_conf():
    from app.services.inpainting import get_inpaint_status
    from app.services.ocr import get_gpu_status

    payload = custom_conf.to_dict()
    payload["provider_status"] = get_provider_status()
    payload["gpu_status"] = get_gpu_status()
    payload["inpaint_status"] = get_inpaint_status()
    return payload


@update_conf_router.post("/conf/init")
def init_conf():
    # 初始化默认值，并恢复服务启动时的设备和擦除方式偏好。
    old_use_gpu = custom_conf.use_gpu
    custom_conf.update_conf("translate_api_type", "custom")
    custom_conf.update_conf("translate_mode", "parallel")
    custom_conf.update_conf("use_gpu", DEFAULT_USE_GPU)
    custom_conf.update_conf("inpaint_backend", DEFAULT_INPAINT_BACKEND)
    if old_use_gpu != custom_conf.use_gpu:
        from app.services.ocr import reset_models

        reset_models()
    return _serialize_conf()

@update_conf_router.post("/conf/update")
def update_conf(item: UpdateItem):
    try:
        old_value = getattr(custom_conf, item.attr, None)
        custom_conf.update_conf(item.attr, item.v)
        if item.attr == "use_gpu" and old_value != custom_conf.use_gpu:
            from app.services.ocr import reset_models

            reset_models()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _serialize_conf()

@update_conf_router.get("/conf/query")
def query_conf():
    return _serialize_conf()


@update_conf_router.get("/conf/options")
def query_conf_options():
    return {
        "translate_api_type": list(TRANSLATE_API_TYPE_OPTIONS),
        "translate_mode": list(TRANSLATE_MODE_OPTIONS),
        "use_gpu": [False, True],
        "inpaint_backend": list(INPAINT_BACKEND_OPTIONS),
    }
