from fastapi import APIRouter
from utils.custom_conf import (
    custom_conf,
    TRANSLATE_API_TYPE_OPTIONS,
    TRANSLATE_MODE_OPTIONS,
)
from pydantic import BaseModel
from typing import Union

update_conf_router = APIRouter()

class UpdateItem(BaseModel):
    attr: str
    v: Union[str, float] = None

@update_conf_router.post("/conf/init")
def init_conf():
    # 初始化默认值：OpenAI + 并行模式。
    custom_conf.update_conf("translate_api_type", "openai")
    custom_conf.update_conf("translate_mode", "parallel")
    return custom_conf.to_dict()

@update_conf_router.post("/conf/update")
def update_conf(item: UpdateItem):
    custom_conf.update_conf(item.attr, item.v)
    return custom_conf.to_dict()

@update_conf_router.get("/conf/query")
def query_conf():
    return custom_conf.to_dict()


@update_conf_router.get("/conf/options")
def query_conf_options():
    return {
        "translate_api_type": list(TRANSLATE_API_TYPE_OPTIONS),
        "translate_mode": list(TRANSLATE_MODE_OPTIONS),
    }
