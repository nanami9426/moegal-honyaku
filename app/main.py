from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.model_sync import ensure_models_ready


def create_app() -> FastAPI:
    ensure_models_ready()
    from app.api.routes import register_routers

    application = FastAPI()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routers(application)
    return application


app = create_app()
