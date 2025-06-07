import argparse
import os

from load_dotenv import load_dotenv
import uvicorn

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

app.include_router(manga_translate_router)

if __name__ == '__main__':
    # Load .env file
    load_dotenv()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run FastAPI server")
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host to run the server on")
    parser.add_argument("--port", type=int, default=os.getenv("PORT", "8000"), help="Port to run the server on")
    args = parser.parse_args()

    # Start FastAPI application
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
