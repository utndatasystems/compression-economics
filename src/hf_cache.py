from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_CONFIG_FILE = REPO_ROOT / ".model_cache_dir"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache"
LOCAL_MODELS_DIR = REPO_ROOT / "models"
_REQUIRED_MODEL_FILES = ("config.json",)
_TOKENIZER_HINT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "spiece.model",
    "tokenizer.model",
    "vocab.json",
)


def get_model_cache_dir() -> str:
    configured_path = os.environ.get("COMPRESSION_ECONOMICS_MODEL_CACHE")
    if configured_path:
        return _normalize_path(configured_path)

    if CACHE_CONFIG_FILE.exists():
        saved_path = CACHE_CONFIG_FILE.read_text(encoding="utf-8").strip()
        if saved_path:
            return _normalize_path(saved_path)

    return str(DEFAULT_CACHE_DIR)


def resolve_pretrained_model_source(model_name: str) -> str:
    direct_path = _resolve_existing_dir(model_name)
    if direct_path is not None:
        return direct_path

    model_slug = model_name.rsplit("/", 1)[-1]
    local_snapshot = LOCAL_MODELS_DIR / model_slug
    if _looks_like_transformers_snapshot(local_snapshot):
        return str(local_snapshot.resolve())

    return model_name


def _normalize_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def _resolve_existing_dir(path_value: str) -> str | None:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_dir():
        return str(path.resolve())
    return None


def _looks_like_transformers_snapshot(path: Path) -> bool:
    if not path.is_dir():
        return False

    if not all((path / required_file).is_file() for required_file in _REQUIRED_MODEL_FILES):
        return False

    return any((path / tokenizer_file).is_file() for tokenizer_file in _TOKENIZER_HINT_FILES)