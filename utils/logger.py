import logging
import sys

from colorlog import ColoredFormatter

logger = logging.getLogger("moegal")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

console_formatter = ColoredFormatter(
    "%(log_color)s%(levelname)s%(reset)s:     %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
console_handler.setFormatter(console_formatter)

file_handler = logging.FileHandler("logs/app.log", encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

file_formatter = logging.Formatter(
    "%(name)s, %(levelname)s, %(asctime)s, %(filename)s:%(lineno)d, %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(file_formatter)

logger.handlers.clear()
logger.addHandler(console_handler)
logger.addHandler(file_handler)