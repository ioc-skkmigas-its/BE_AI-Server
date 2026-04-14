import os
from unittest.mock import patch

import pandas as pd

os.environ.setdefault("DEBUG", "false")

from app.services import ai_service


class _DummyPredictor:
    original_features = ["feat_a", "feat_b"]

    def __init__(self) -> None:
        self.last_input: pd.DataFrame | None = None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self.last_input = X.copy()
        return pd.Series([0.2, 0.8], index=X.index)


class _FallbackPredictor:
    original_features = ["feat_a"]

    def __init__(self) -> None:
        self.default_called = False
        self.fallback_called = False

    def model_names(self, can_infer=True):
        return ["ModelA"]

    def predict(self, X: pd.DataFrame, model: str | None = None) -> pd.Series:
        if model is None:
            self.default_called = True
            raise ModuleNotFoundError("No module named 'fastai'")
        if model == "ModelA":
            self.fallback_called = True
            return pd.Series([1.0] * len(X), index=X.index)
        raise RuntimeError("Unexpected model")


class _FallbackPredictorGenericError(_FallbackPredictor):
    def predict(self, X: pd.DataFrame, model: str | None = None) -> pd.Series:
        if model is None:
            self.default_called = True
            raise NotImplementedError("cannot instantiate PosixPath on your system")
        return super().predict(X, model=model)


def test_predict_batch_aligns_with_required_features_and_fills_missing():
    predictor = _DummyPredictor()
    wells_df = pd.DataFrame(
        {
            "feat_a": [10.0, 20.0],
            "UWI": ["W-1", "W-2"],
            "basin_cluster": ["A", "A"],
            "extra_col": [1, 2],
        }
    )

    with patch("app.services.ai_service.get_predictor", return_value=predictor):
        result_df = ai_service.predict_batch(wells_df)

    assert predictor.last_input is not None
    assert list(predictor.last_input.columns) == ["feat_a", "feat_b"]
    assert predictor.last_input["feat_b"].isna().all()

    assert "predicted_score" in result_df.columns
    assert "rank_overall" in result_df.columns
    assert "rank_in_basin" in result_df.columns
    assert result_df["rank_overall"].tolist() == [2, 1]
    assert result_df["rank_in_basin"].tolist() == [2, 1]


def test_predict_batch_maps_columns_case_insensitively():
    predictor = _DummyPredictor()
    wells_df = pd.DataFrame(
        {
            "FEAT_A": [1.0, 2.0],
            "basin_cluster": ["A", "A"],
        }
    )

    with patch("app.services.ai_service.get_predictor", return_value=predictor):
        ai_service.predict_batch(wells_df)

    assert predictor.last_input is not None
    assert predictor.last_input["feat_a"].tolist() == [1.0, 2.0]


def test_predict_batch_uses_fallback_model_when_default_backend_missing():
    predictor = _FallbackPredictor()
    wells_df = pd.DataFrame({"feat_a": [1.0, 2.0], "basin_cluster": ["A", "A"]})

    with patch("app.services.ai_service.get_predictor", return_value=predictor):
        out = ai_service.predict_batch(wells_df)

    assert predictor.default_called is True
    assert predictor.fallback_called is True
    assert out["predicted_score"].tolist() == [1.0, 1.0]


def test_predict_batch_uses_fallback_when_default_model_other_error():
    predictor = _FallbackPredictorGenericError()
    wells_df = pd.DataFrame({"feat_a": [1.0], "basin_cluster": ["A"]})

    with patch("app.services.ai_service.get_predictor", return_value=predictor):
        out = ai_service.predict_batch(wells_df)

    assert predictor.default_called is True
    assert predictor.fallback_called is True
    assert out["predicted_score"].tolist() == [1.0]
