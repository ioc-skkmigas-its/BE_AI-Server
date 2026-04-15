from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

try:
    import xgboost as xgb
except ModuleNotFoundError:  # pragma: no cover - handled at runtime by clear error.
    xgb = None

from app.core.config import Settings


def _install_numpy_pickle_compat() -> None:
    """Apply runtime compatibility for joblib artifacts produced on newer NumPy."""
    # NumPy 2.x artifacts may reference numpy._core.* modules while older runtimes
    # expose numpy.core.* only.
    import numpy.core as np_core

    sys.modules.setdefault("numpy._core", np_core)

    numeric = getattr(np_core, "numeric", None)
    if numeric is not None:
        sys.modules.setdefault("numpy._core.numeric", numeric)

    multiarray = getattr(np_core, "multiarray", None)
    if multiarray is not None:
        sys.modules.setdefault("numpy._core.multiarray", multiarray)

    # Some artifacts persist BitGenerator as class objects. Older NumPy expects
    # string names, so we normalize class -> __name__ before constructing.
    import numpy.random._pickle as np_pickle

    if getattr(np_pickle, "_copilot_compat_patched", False):
        return

    original_ctor = np_pickle.__bit_generator_ctor

    def compat_ctor(bit_generator_name: str | type = "MT19937"):
        if isinstance(bit_generator_name, type):
            bit_generator_name = bit_generator_name.__name__
        return original_ctor(bit_generator_name)

    np_pickle.__bit_generator_ctor = compat_ctor
    np_pickle._copilot_compat_patched = True


def _load_joblib_artifact(path: Path, artifact_label: str):
    _install_numpy_pickle_compat()
    try:
        return joblib.load(path)
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "")
        if missing_name == "xgboost":
            raise ModuleNotFoundError(
                f"Failed to load {artifact_label}: missing dependency 'xgboost'. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        raise


class XGBoostModelLoader:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._preprocessor = None
        self._manifest_feature_cols: list[str] | None = None

    def resolve_feature_columns(self, input_columns: list[str]) -> list[str]:
        if self._manifest_feature_cols is None:
            self._manifest_feature_cols = self._load_feature_columns_from_manifest()

        if self._manifest_feature_cols:
            return list(self._manifest_feature_cols)

        model = self._get_model()
        model_feature_cols: list[str] = []

        if hasattr(model, "feature_names_in_"):
            model_feature_cols = [str(col) for col in model.feature_names_in_]
        elif xgb is not None and isinstance(model, xgb.Booster) and model.feature_names:
            model_feature_cols = [str(col) for col in model.feature_names]
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()
            if booster.feature_names:
                model_feature_cols = [str(col) for col in booster.feature_names]

        if model_feature_cols:
            return model_feature_cols

        return [col for col in input_columns if not self._looks_like_identifier(col)]

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if features_df.empty:
            raise ValueError("No feature rows available for prediction.")

        model = self._get_model()
        matrix = features_df

        preprocessor = self._get_preprocessor()
        if preprocessor is not None:
            matrix = preprocessor.transform(features_df)

        if xgb is not None and isinstance(model, xgb.Booster):
            dmatrix = xgb.DMatrix(matrix, feature_names=list(features_df.columns))
            preds = model.predict(dmatrix)
        else:
            preds = model.predict(matrix)

        return np.asarray(preds, dtype=float).reshape(-1)

    def _get_model(self):
        if self._model is not None:
            return self._model

        model_path = self._resolve_artifact_path(self._settings.xgboost_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"XGBoost model file not found at '{model_path}'. "
                "Set XGBOOST_MODEL_PATH in your .env file."
            )

        suffix = model_path.suffix.lower()

        if suffix in {".json", ".ubj", ".bst"}:
            if xgb is None:
                raise ModuleNotFoundError(
                    "xgboost is not installed. Install dependencies with `pip install -r requirements.txt`."
                )
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            self._model = booster
            return self._model

        self._model = _load_joblib_artifact(model_path, "model artifact")
        return self._model

    def _get_preprocessor(self):
        if self._preprocessor is not None:
            return self._preprocessor

        path = self._settings.xgboost_preprocessor_path
        if not path:
            return None

        preprocessor_path = self._resolve_artifact_path(path)
        if not preprocessor_path.exists():
            return None

        self._preprocessor = _load_joblib_artifact(
            preprocessor_path,
            "preprocessor artifact",
        )
        return self._preprocessor

    def _load_feature_columns_from_manifest(self) -> list[str]:
        path = self._settings.xgboost_feature_manifest_path
        if not path:
            return []

        manifest_path = self._resolve_artifact_path(path)
        if not manifest_path.exists():
            return []

        content = json.loads(manifest_path.read_text(encoding="utf-8"))
        feature_cols = [str(col) for col in content.get("feature_cols", [])]
        id_cols = {str(col).lower() for col in content.get("id_cols", [])}

        return [
            col
            for col in feature_cols
            if col.lower() not in id_cols
        ]

    @staticmethod
    def _looks_like_identifier(col_name: str) -> bool:
        normalized = col_name.strip().lower()
        explicit = {
            "id",
            "well_id",
            "fid",
            "catalogid",
            "catalog_id",
            "source_uuid",
            "uuid",
            "uwi",
            "well_name",
            "field_name",
            "area_id",
            "basin_cluster",
        }
        return (
            normalized in explicit
            or normalized.endswith("_id")
            or "uuid" in normalized
        )

    def _resolve_artifact_path(self, configured_path: str) -> Path:
        target_path = Path(configured_path)
        if target_path.exists():
            return target_path

        bundle_path = self._resolve_bundle_zip_path()
        if bundle_path is None or not bundle_path.exists():
            return target_path

        member_name = target_path.name
        extract_dir = Path(self._settings.artifact_extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        extracted_path = extract_dir / member_name

        if extracted_path.exists():
            return extracted_path

        with zipfile.ZipFile(bundle_path, mode="r") as zip_ref:
            if member_name not in zip_ref.namelist():
                return target_path
            with zip_ref.open(member_name, mode="r") as source_stream:
                with extracted_path.open(mode="wb") as target_stream:
                    shutil.copyfileobj(source_stream, target_stream)

        return extracted_path

    def _resolve_bundle_zip_path(self) -> Path | None:
        bundle_path = Path(self._settings.artifact_bundle_zip_path)
        if bundle_path.exists():
            return bundle_path

        repo_id = self._settings.artifact_source_repo
        filename = self._settings.artifact_source_filename
        if not repo_id or not filename:
            return None

        local_dir = bundle_path.parent
        local_dir.mkdir(parents=True, exist_ok=True)

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=self._settings.hf_token,
            local_dir=str(local_dir),
        )
        downloaded_path = Path(downloaded)

        if downloaded_path.resolve() == bundle_path.resolve():
            return downloaded_path

        if not bundle_path.exists():
            shutil.copyfile(downloaded_path, bundle_path)
        return bundle_path
