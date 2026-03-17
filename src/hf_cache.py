from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_CONFIG_FILE = REPO_ROOT / ".model_cache_dir"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache"


def get_model_cache_dir() -> str:
    configured_path = os.environ.get("COMPRESSION_ECONOMICS_MODEL_CACHE")
    if configured_path:
        return _normalize_path(configured_path)

    if CACHE_CONFIG_FILE.exists():
        saved_path = CACHE_CONFIG_FILE.read_text(encoding="utf-8").strip()
        if saved_path:
            return _normalize_path(saved_path)

    return str(DEFAULT_CACHE_DIR)


def _normalize_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())