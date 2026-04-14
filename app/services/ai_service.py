"""
ai_service.py
─────────────
Wraps the AutoGluon TabularPredictor to produce ranked well results.

Input:  pandas DataFrame containing model features (plus optional ID/reference cols)
Output: same DataFrame + predicted_score, rank_overall, rank_in_basin, rank_label
"""

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from app.services.model_loader import get_predictor

logger = logging.getLogger(__name__)

# Label cut-off percentiles (ascending probability)
# e.g., bottom 25% → BELOW_AVERAGE, top 10% → TOP_10%
_LABEL_BINS = [0.0, 0.25, 0.50, 0.75, 0.90, 1.01]
_LABEL_NAMES = ["BELOW_AVERAGE", "AVERAGE", "GOOD", "TOP_25%", "TOP_10%"]

def _required_feature_cols(predictor) -> list[str]:
    """
    Resolve required input feature columns from predictor metadata.
    """
    if getattr(predictor, "original_features", None):
        return list(predictor.original_features)

    feature_metadata_in = getattr(predictor, "feature_metadata_in", None)
    if feature_metadata_in is not None:
        return list(feature_metadata_in.get_features())

    raise RuntimeError("Predictor metadata does not expose required feature columns.")


def _prepare_feature_frame(
    wells_df: pd.DataFrame,
    required_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Align incoming data to model-required features.
    Missing columns are created with NaN so inference can still run.
    """
    columns_ci = {col.lower(): col for col in wells_df.columns}
    feature_data: dict[str, pd.Series] = {}
    missing: list[str] = []

    for required_col in required_features:
        source_col = columns_ci.get(required_col.lower())
        if source_col is None:
            missing.append(required_col)
            feature_data[required_col] = pd.Series(np.nan, index=wells_df.index)
        else:
            feature_data[required_col] = wells_df[source_col]

    feature_df = pd.DataFrame(feature_data, index=wells_df.index)
    return feature_df, missing


def _predict_with_fallback_models(predictor, feature_df: pd.DataFrame) -> pd.Series:
    """
    Try default predictor first, then fallback to alternative models if optional
    backend dependencies are missing for the default model.
    """
    default_error: str | None = None
    try:
        return predictor.predict(feature_df)
    except Exception as exc:
        if isinstance(exc, ModuleNotFoundError):
            missing_module = getattr(exc, "name", "") or "<unknown>"
            default_error = f"missing module '{missing_module}'"
        else:
            default_error = f"{type(exc).__name__}: {exc}"

        logger.warning(
            "Default predictor model failed (%s). Trying fallback models.",
            default_error,
        )

    fallback_models: list[str] = []
    try:
        model_names = predictor.model_names(can_infer=True)
    except Exception:
        model_names = predictor.model_names()

    if isinstance(model_names, str):
        fallback_models = [model_names]
    elif isinstance(model_names, Iterable):
        fallback_models = list(model_names)

    fallback_errors: list[str] = []
    for model_name in fallback_models:
        try:
            scores = predictor.predict(feature_df, model=model_name)
            logger.info("Inference succeeded using fallback model '%s'.", model_name)
            return scores
        except ModuleNotFoundError as exc:
            fallback_errors.append(
                f"{model_name}: missing module '{getattr(exc, 'name', '<unknown>')}'"
            )
        except Exception as exc:
            fallback_errors.append(f"{model_name}: {exc}")

    raise RuntimeError(
        "Inference failed for all available models. "
        f"Default error: {default_error}. Fallback attempts: {fallback_errors}"
    )


def predict_batch(wells_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run AutoGluon inference on a batch of well records.

    Args:
        wells_df: DataFrame containing all feature columns (and optionally ID cols).

    Returns:
        DataFrame with original columns plus:
          - predicted_score   (float)
          - rank_overall      (int, 1 = best globally)
          - rank_in_basin     (int, 1 = best within basin_cluster)
          - rank_label        (str)
    """
    predictor = get_predictor()
    required_features = _required_feature_cols(predictor)
    feature_df, missing = _prepare_feature_frame(wells_df, required_features)
    if missing:
        logger.warning(
            "Input data is missing %d/%d required model features. "
            "Filled with NaN for inference. Missing columns: %s",
            len(missing),
            len(required_features),
            missing,
        )

    logger.info("Running inference on %d wells ...", len(feature_df))
    scores: pd.Series = _predict_with_fallback_models(predictor, feature_df)

    result_df = wells_df.copy()
    result_df["predicted_score"] = scores.values

    # Global rank (1 = highest score = best)
    result_df["rank_overall"] = (
        result_df["predicted_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Within-basin rank
    if "basin_cluster" in result_df.columns:
        result_df["rank_in_basin"] = (
            result_df.groupby("basin_cluster")["predicted_score"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
    else:
        result_df["rank_in_basin"] = result_df["rank_overall"]

    # Percentile-based label (based on global score distribution)
    percentiles = result_df["predicted_score"].rank(pct=True)
    result_df["rank_label"] = pd.cut(
        percentiles,
        bins=_LABEL_BINS,
        labels=_LABEL_NAMES,
        right=True,
    ).astype(str)

    logger.info(
        "Inference complete. Score range: %.2f – %.2f",
        result_df["predicted_score"].min(),
        result_df["predicted_score"].max(),
    )

    return result_df
