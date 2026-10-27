"""Tests for spatial microclimate model targets, schemas, and persistence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import models
from config import get_config
from exceptions import DataIntegrityError, SchemaError
from models import XGBMicroclimate, load_model, make_target, save_model
from temporal_features import Fold


def _cell_frame(periods: int = 12, cells: int = 4) -> pd.DataFrame:
	"""Build a minimal multi-cell monthly model frame."""
	dates = pd.date_range("2000-01-01", periods=periods, freq="MS").as_unit("ns")
	records: list[dict[str, object]] = []
	for cell in range(cells):
		for month, date in enumerate(dates):
			records.append(
				{
					"city": "dehradun",
					"cell_id": cell,
					"date": date,
					"x": float(cell * 100),
					"y": float(cell * 100),
					"spatial_block": cell,
					"dem": float(cell * 100),
					"urban_fraction": float(cell / max(cells - 1, 1)),
					"t2m_downscaled": float(10 + cell + month / periods),
					"DTR_cell": float(8 + cell + month / periods),
					"PRECIP": float(month + cell),
					"slope": float(cell),
					"aspect_sin": 0.0,
					"aspect_cos": 1.0,
					"tpi_small": 0.0,
					"tpi_large": 0.0,
					"twi": 0.0,
					"solar_rad": 0.0,
					"ndvi": 0.0,
					"ndbi": 0.0,
					"bsi": 0.0,
					"RH2M": 50.0,
					"is_monsoon": int(date.month in {6, 7, 8, 9}),
					"sin_m": np.sin(2 * np.pi * date.month / 12),
					"cos_m": np.cos(2 * np.pi * date.month / 12),
				}
			)
	return pd.DataFrame(records)


def test_make_regression_target_returns_source_series():
	frame = _cell_frame()

	target, thresholds = make_target(frame, "DTR_cell")

	assert target.equals(frame["DTR_cell"].rename("DTR_cell"))
	assert thresholds == {}


def test_binary_thresholds_are_city_specific_and_reusable_for_test():
	frame = _cell_frame()
	training = frame.loc[frame["cell_id"] < 2]
	test = frame.loc[frame["cell_id"] >= 2]

	train_target, thresholds = make_target(training, "heat_snap_risk")
	test_target, reused = make_target(test, "heat_snap_risk", thresholds)

	assert thresholds == reused
	assert set(train_target.dropna().unique()) <= {0, 1}
	assert set(test_target.dropna().unique()) <= {0, 1}
	assert test_target.index.equals(test.index)


def test_make_target_rejects_missing_month_start_schema():
	frame = _cell_frame()
	frame["date"] = frame["date"].astype(str)

	with pytest.raises(SchemaError, match="datetime64\\[ns\\]"):
		make_target(frame, "DTR_cell")


def test_make_target_rejects_duplicate_cell_month_keys():
	frame = _cell_frame()
	frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

	with pytest.raises(SchemaError, match="duplicate"):
		make_target(frame, "DTR_cell")


def test_xgb_fit_predict_and_joblib_round_trip(tmp_path, monkeypatch):
	config = get_config()
	ml = replace(config.ml, prediction_chunk_size=7, scale_pos_weight=False)
	monkeypatch.setattr(models, "get_config", lambda: replace(config, ml=ml))
	frame = _cell_frame(periods=8)
	features = frame[["dem", "urban_fraction"]].astype("float32")
	target = frame["DTR_cell"]
	model = XGBMicroclimate(params={"n_estimators": 20, "max_depth": 2})
	model.fit(features, target)
	path = tmp_path / "model.joblib"

	save_model(model, path, thresholds={"dehradun": 1.0})
	loaded, metadata = load_model(path)

	assert loaded.feature_names_ == ["dem", "urban_fraction"]
	assert metadata["thresholds"] == {"dehradun": 1.0}
	np.testing.assert_allclose(loaded.predict(features), model.predict(features))


def test_predict_rejects_changed_feature_schema():
	features = pd.DataFrame({"dem": [1.0], "urban_fraction": [0.2]})
	model = XGBMicroclimate()
	model.fit(features, pd.Series([1.0]))

	with pytest.raises(SchemaError, match="feature schema"):
		model.predict(features[["urban_fraction", "dem"]])


def test_binary_fit_rejects_single_class_training_data():
	features = pd.DataFrame({"dem": [1.0, 2.0], "urban_fraction": [0.2, 0.3]})
	model = XGBMicroclimate(task="binary")

	with pytest.raises(DataIntegrityError, match="no positive"):
		model.fit(features, pd.Series([0, 0]))


def test_train_and_evaluate_returns_later_fold_predictions(monkeypatch, tmp_path):
	from sklearn.model_selection import GroupKFold

	config = get_config()
	ml = replace(
		config.ml,
		validation_months=2,
		early_stopping_rounds=3,
		save_fold_models=False,
		model_params={
			**config.ml.model_params,
			"xgb": {"n_estimators": 15, "max_depth": 2},
		},
		model_directory=tmp_path,
	)
	monkeypatch.setattr(models, "get_config", lambda: replace(config, ml=ml))
	frame = _cell_frame(periods=12, cells=4)
	dates = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
	train_positions = np.flatnonzero(frame["date"].isin(dates[:8]))
	test_positions = np.flatnonzero(frame["date"].isin(dates[8:]))
	fold = Fold(
		train_idx=train_positions,
		test_idx=test_positions,
		train_end=dates[7],
		test_start=dates[8],
	)

	result = models.train_and_evaluate(
		frame,
		"DTR_cell",
		"xgb",
		[fold],
		GroupKFold(n_splits=2),
	)

	assert len(result.oof_predictions) == len(test_positions)
	assert result.oof_predictions["date"].min() == dates[8]
	assert set(result.oof_predictions.columns) == {
		"city",
		"cell_id",
		"date",
		"y_true",
		"y_pred",
		"fold",
		"spatial_block",
	}


def test_predict_to_xarray_preserves_city_date_cell_coordinates(monkeypatch):
	config = get_config()
	ml = replace(config.ml, prediction_chunk_size=5)
	monkeypatch.setattr(models, "get_config", lambda: replace(config, ml=ml))
	frame = _cell_frame(periods=3, cells=3)
	features = models._prepare_features(frame, ml.feature_columns, frame)
	model = XGBMicroclimate(params={"n_estimators": 10, "max_depth": 2})
	model.fit(features, frame["DTR_cell"])

	arrays = models.predict_to_xarray(model, frame)

	assert set(arrays) == {"dehradun"}
	assert arrays["dehradun"].dims == ("date", "cell_id")
	assert arrays["dehradun"].shape == (3, 3)
	assert arrays["dehradun"].coords["x"].to_numpy().tolist() == [0.0, 100.0, 200.0]
