from fastapi import FastAPI

from app.api.routes.manga_translate import manga_translate_router
from app.api.routes.update_conf import update_conf_router


def register_routers(app: FastAPI) -> None:
    app.include_router(manga_translate_router)
    app.include_router(update_conf_router)
