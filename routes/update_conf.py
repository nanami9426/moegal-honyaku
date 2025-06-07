from fastapi import APIRouter
from utils.custom_conf import custom_conf
from pydantic import BaseModel
from typing import Union

update_conf_router = APIRouter()

class InitItem(BaseModel):
    translate_api_type: str

class UpdateItem(BaseModel):
    attr: str
    v: Union[str, float] = None

@update_conf_router.post("/conf/init")
def init_conf(item: InitItem):
    custom_conf.update_conf("translate_api_type", item.translate_api_type)
    return custom_conf.to_dict()

@update_conf_router.post("/conf/update")
def update_conf(item: UpdateItem):
    custom_conf.update_conf(item.attr, item.v)
    return custom_conf.to_dict()

@update_conf_router.get("/conf/query")
def query_conf():
    return custom_conf.to_dict()
