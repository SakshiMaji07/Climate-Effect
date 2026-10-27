"""Tests for elevation-stratified OOF evaluation and skill reporting."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

import validation.evaluation as evaluation
from config import get_config
from exceptions import DataIntegrityError, SchemaError
from validation.evaluation import (
	bootstrap_ci,
	classification_metrics,
	compare_to_split_strategy,
	elevation_bands,
	make_skill_references,
	regression_metrics,
	skill_score,
	stratified_report,
	write_report,
)


def _oof_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Make complete OOF data with known elevation-dependent errors."""
	config = get_config().validation
	low, high = config.elevation_bands["dehradun"]
	dem = np.array([low - 1, low, (low + high) / 2, high, high + 1, high + 2])
	errors = np.where(dem < low, 1.0, np.where(dem < high, 2.0, 4.0))
	dates = pd.date_range("2020-01-01", periods=4, freq="MS").as_unit("ns")
	rows: list[dict[str, object]] = []
	for cell_id, elevation in enumerate(dem):
		for date in dates:
			rows.append(
				{
					"city": "dehradun",
					"cell_id": cell_id,
					"date": date,
					"y_true": 10.0,
					"y_pred": 10.0 + errors[cell_id],
					"fold": 0,
					"spatial_block": cell_id // 2,
					"is_monsoon": int(date.month in config_months()),
				}
			)
	oof = pd.DataFrame(rows)
	oof["cell_id"] = oof["cell_id"].astype("int64")
	cells = pd.DataFrame(
		{
			"city": "dehradun",
			"cell_id": np.arange(len(dem), dtype=np.int64),
			"dem": dem,
		}
	)
	return oof, cells


def config_months() -> tuple[int, ...]:
	"""Return configured monsoon months for test frame construction."""
	return get_config().dataset.monsoon_months


def test_elevation_bands_use_config_and_quantile_schemes(monkeypatch):
	config = get_config()
	validation_config = replace(config.validation, min_elevation_band_cells=2)
	monkeypatch.setattr(
		evaluation,
		"get_config",
		lambda: replace(config, validation=validation_config),
	)
	low, high = validation_config.elevation_bands["default"]
	dem = pd.Series([low - 5, low - 2, low + 1, low + 2, high + 1, high + 2])

	configured = elevation_bands(dem)
	quantile = elevation_bands(pd.Series(np.arange(1.0, 10.0)), scheme="quantile")

	assert configured.ordered
	assert list(configured.categories) == list(validation_config.elevation_band_labels)
	assert len(quantile.categories) == 3


def test_elevation_bands_reject_invalid_scheme_and_small_stratum():
	with pytest.raises(SchemaError, match="Unsupported elevation scheme"):
		elevation_bands(pd.Series(np.arange(1.0, 10.0)), scheme="invalid")
	with pytest.raises(DataIntegrityError, match="fewer than"):
		elevation_bands(pd.Series([0.0, 10.0, 20.0]))


def test_regression_metrics_and_skill_scores_match_known_values():
	y = np.array([1.0, 2.0, 3.0])
	yhat = np.array([2.0, 2.0, 5.0])
	references = {
		"lapse_only": np.array([1.0, 1.0, 1.0]),
		"monthly_climatology": np.array([0.0, 0.0, 0.0]),
		"persistence": np.array([0.0, 1.0, 2.0]),
	}

	metrics = regression_metrics(y, yhat)
	skills = skill_score(y, yhat, references)

	assert metrics["MAE"] == pytest.approx(1.0)
	assert metrics["RMSE"] == pytest.approx(np.sqrt(5 / 3))
	assert metrics["bias"] == pytest.approx(1.0)
	assert skills.keys() == references.keys()
	assert skills["lapse_only"] == pytest.approx(
		1 - mean_squared_error_numpy(y, yhat) / mean_squared_error_numpy(y, references["lapse_only"])
	)


def mean_squared_error_numpy(y: np.ndarray, prediction: np.ndarray) -> float:
	"""Small deterministic MSE helper for expected test values."""
	return float(np.mean((y - prediction) ** 2))


def test_classification_metrics_include_extreme_and_reliability_scores():
	metrics = classification_metrics(
		np.array([0, 0, 1, 1]),
		np.array([0.1, 0.4, 0.6, 0.9]),
		threshold=0.5,
	)

	assert metrics["PR_AUC"] == pytest.approx(1.0)
	assert metrics["POD"] == pytest.approx(1.0)
	assert metrics["FAR"] == pytest.approx(0.0)
	assert metrics["CSI"] == pytest.approx(1.0)
	assert len(metrics["reliability_bins"]) == 4
	assert metrics["Brier_skill"] > 0


def test_stratified_report_reproduces_elevation_dependent_error():
	oof, cells = _oof_frame()

	report = stratified_report(oof, cells, by=("city", "elev_band"))

	config = get_config().validation
	results = report.set_index("elev_band")["RMSE"].to_dict()
	assert results[config.elevation_band_labels[0]] == pytest.approx(1.0)
	assert results[config.elevation_band_labels[1]] == pytest.approx(2.0)
	assert results[config.elevation_band_labels[2]] == pytest.approx(4.0)


def test_oof_rejects_duplicate_keys_and_missing_predictions():
	oof, _ = _oof_frame()
	duplicated = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
	with pytest.raises(DataIntegrityError, match="duplicated"):
		stratified_report(duplicated, _oof_frame()[1], by=("city", "fold"))
	missing = oof.copy()
	missing.loc[0, "y_pred"] = np.nan
	with pytest.raises(DataIntegrityError, match="cannot contain NaNs"):
		compare_to_split_strategy(missing, oof)


def test_two_way_block_bootstrap_is_deterministic_and_writes_reports(tmp_path):
	oof, _ = _oof_frame()
	statistic: Callable[[pd.DataFrame], float] = lambda frame: float(
		np.sqrt(np.mean((frame["y_true"] - frame["y_pred"]) ** 2))
	)
	first = bootstrap_ci(oof, statistic, n=30)
	second = bootstrap_ci(oof, statistic, n=30)
	report_path = tmp_path / "evaluation"

	write_report(first, report_path)

	pd.testing.assert_frame_equal(first, second)
	assert first.loc[0, "by_block"]
	assert first.loc[0, "ci_lower"] <= first.loc[0, "estimate"] <= first.loc[0, "ci_upper"]
	assert report_path.with_suffix(".csv").exists()
	assert report_path.with_suffix(".md").exists()
	assert report_path.with_suffix(".json").exists()


def test_compare_split_strategy_quantifies_random_split_optimism():
	random_oof, _ = _oof_frame()
	spatial_oof = random_oof.copy()
	spatial_oof["y_pred"] += np.where(spatial_oof["cell_id"] < 2, 1.0, 3.0)

	comparison = compare_to_split_strategy(random_oof, spatial_oof)

	assert comparison.loc[0, "optimism_RMSE"] > 0
	assert comparison.loc[0, "random_RMSE"] < comparison.loc[0, "spatial_RMSE"]


def test_skill_references_use_train_climatology_and_prior_month_only():
	dates = pd.date_range("2020-01-01", periods=14, freq="MS").as_unit("ns")
	train_rows = [
		{
			"city": "dehradun",
			"cell_id": cell_id,
			"date": date,
			"target": float(date.month + cell_id),
		}
		for cell_id in range(2)
		for date in dates[:12]
	]
	evaluation_rows = [
		{
			"city": "dehradun",
			"cell_id": cell_id,
			"date": date,
			"target": float(date.month + cell_id),
			"lapse_prediction": float(date.month + cell_id + 0.5),
		}
		for cell_id in range(2)
		for date in dates[12:]
	]

	references = make_skill_references(
		pd.DataFrame(train_rows),
		pd.DataFrame(evaluation_rows),
		"target",
		"lapse_prediction",
	)

	assert references["lapse_only"].shape == (4,)
	assert references["monthly_climatology"].shape == (4,)
	assert np.array_equal(references["persistence"], np.array([12, 1, 13, 2]))


def test_report_write_rejects_empty_data():
	with pytest.raises(SchemaError, match="non-empty"):
		write_report(pd.DataFrame(), "unused")
