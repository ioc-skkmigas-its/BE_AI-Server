"""
model_loader.py
───────────────
Downloads the AutoGluon MSF ranking bundle from Hugging Face on first startup,
caches it locally under MODEL_CACHE_DIR, and exposes a singleton TabularPredictor.

Flow:
  1. Check if MODEL_CACHE_DIR already contains a valid AutoGluon predictor.
  2. If not, download the .zip bundle from HF using huggingface_hub.
  3. Unzip into MODEL_CACHE_DIR.
  4. Find the predictor root dir (contains 'predictor.pkl' or 'metadata.json').
  5. Load with TabularPredictor.load().
  6. Store as module-level singleton; subsequent calls return the cached instance.
"""

import logging
import zipfile
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download
from autogluon.tabular import TabularPredictor

from app.core.config import settings

logger = logging.getLogger(__name__)

_predictor: Optional[TabularPredictor] = None
_model_ready: bool = False
_model_error: Optional[str] = None


def _find_predictor_dir(base: Path) -> Path:
    """
    AutoGluon saves the predictor as a directory. After unzipping a bundle,
    the predictor dir is the one that contains 'predictor.pkl'.
    Walk the extracted tree to find it.
    """
    # Direct match
    if (base / "predictor.pkl").exists():
        return base

    # Search one level deep (zip may extract to a subdirectory)
    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "predictor.pkl").exists():
            return child

    # Fallback: return base and let AutoGluon handle the error
    logger.warning("predictor.pkl not found — attempting load from %s anyway", base)
    return base


def _download_and_extract() -> Path:
    """Download bundle from HF Hub and extract to MODEL_CACHE_DIR."""
    cache_dir = Path(settings.model_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / settings.model_bundle_filename

    if not zip_path.exists():
        logger.info(
            "Downloading model bundle '%s' from %s ...",
            settings.model_bundle_filename,
            settings.hf_model_repo,
        )
        downloaded = hf_hub_download(
            repo_id=settings.hf_model_repo,
            filename=settings.model_bundle_filename,
            token=settings.hf_token,
            local_dir=str(cache_dir),
        )
        logger.info("Download complete: %s", downloaded)
    else:
        logger.info("Bundle already cached at %s", zip_path)

    # Extract
    extract_dir = cache_dir / "extracted"
    if not extract_dir.exists():
        logger.info("Extracting bundle ...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("Extraction complete.")

    return _find_predictor_dir(extract_dir)


def load_model() -> TabularPredictor:
    """
    Public entry point called during app startup lifespan.
    Downloads (if needed) and loads the AutoGluon predictor into memory.
    """
    global _predictor, _model_ready, _model_error

    try:
        try:
            import lightgbm  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'lightgbm'. Install with: pip install -r requirements.txt"
            ) from exc

        predictor_dir = _download_and_extract()
        logger.info("Loading AutoGluon TabularPredictor from %s ...", predictor_dir)
        _predictor = TabularPredictor.load(str(predictor_dir), require_version_match=False)
        _model_ready = True
        logger.info("Model loaded successfully. Target: %s", _predictor.label)
        return _predictor

    except Exception as exc:
        _model_error = str(exc)
        _model_ready = False
        logger.exception("Failed to load model: %s", exc)
        raise


def get_predictor() -> TabularPredictor:
    """Return the loaded predictor singleton. Raises if model not ready."""
    if _predictor is None:
        raise RuntimeError(
            "Model is not loaded yet. Check /health/model for status."
        )
    return _predictor


def model_status() -> dict:
    """Return a status dict for the /health/model endpoint."""
    return {
        "ready": _model_ready,
        "error": _model_error,
        "cache_dir": settings.model_cache_dir,
        "hf_repo": settings.hf_model_repo,
    }
