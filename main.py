from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils.pic_process import wrap_text_by_width


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.manga_translate import manga_translate_router
from routes.update_conf import update_conf_router
app.include_router(manga_translate_router)
app.include_router(update_conf_router)