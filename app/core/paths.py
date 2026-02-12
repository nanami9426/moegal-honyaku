from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"
FONTS_DIR = ASSETS_DIR / "fonts"
LOGS_DIR = PROJECT_ROOT / "logs"
SAVED_DIR = PROJECT_ROOT / "saved"
