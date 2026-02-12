from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import register_routers


def create_app() -> FastAPI:
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
