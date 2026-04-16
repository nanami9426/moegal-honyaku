from fastapi import APIRouter, HTTPException
from app.core.custom_conf import (
    custom_conf,
    TRANSLATE_API_TYPE_OPTIONS,
    TRANSLATE_MODE_OPTIONS,
)
from app.services.translate_api import get_provider_status
from pydantic import BaseModel
from typing import Union

update_conf_router = APIRouter()

class UpdateItem(BaseModel):
    attr: str
    v: Union[str, float] = None


def _serialize_conf():
    payload = custom_conf.to_dict()
    payload["provider_status"] = get_provider_status()
    return payload


@update_conf_router.post("/conf/init")
def init_conf():
    # 初始化默认值：custom + 并行模式。
    custom_conf.update_conf("translate_api_type", "custom")
    custom_conf.update_conf("translate_mode", "parallel")
    return _serialize_conf()

@update_conf_router.post("/conf/update")
def update_conf(item: UpdateItem):
    try:
        custom_conf.update_conf(item.attr, item.v)
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
    }
