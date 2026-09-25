"""Project path helpers.

All paths are resolved relative to the repository root, so scripts can be launched
from any working directory.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAVED_DIR = PROJECT_ROOT / "saved"


def checkpoint_dir(model_name: str) -> Path:
    path = SAVED_DIR / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path
