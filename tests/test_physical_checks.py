"""Tests for physical plausibility and residual Moran diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import validation.physical_checks as physical_checks
from config import get_config
from exceptions import ConfigError, DataIntegrityError, SchemaError
from validation.physical_checks import (
	build_weights,
	extreme_physical_bounds,
	lapse_check,
	predicted_lapse_rate,
	residual_morans_i,
	run,
)


def _temperature_frame(
	slope: float = -6.5,
	month_number: int = 6,
	cell_count: int = 24,
) -> pd.DataFrame:
	"""Make predicted temperatures with a known elevation slope."""
	elevation = np.linspace(0.0, 3_000.0, cell_count)
	urban = np.tile([0.0, 0.25, 0.5, 0.75, 1.0, 0.1], 4)[:cell_count]
	x, y = np.meshgrid(np.arange(cell_count), np.zeros(1))
	frame = pd.DataFrame(
		{
			"city": "dehradun",
			"cell_id": np.arange(cell_count, dtype=np.int64),
			"x": x.ravel().astype(np.float64) * 500.0,
			"y": y.ravel().astype(np.float64),
			"date": pd.Timestamp(year=2020, month=month_number, day=1),
			"dem": elevation,
			"urban_fraction": urban,
		}
	)
	frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
	frame["y_pred"] = (
		20.0 + slope * elevation / 1_000.0 + 1.5 * urban
	)
	return frame


def _spatial_weights_config(monkeypatch, *, permutations: int = 99):
	config = get_config()
	validation = replace(
		config.validation,
		spatial_weights_k=4,
		moran_permutations=permutations,
	)
	monkeypatch.setattr(
		physical_checks, "get_config", lambda: replace(config, validation=validation)
	)


def test_predicted_lapse_rate_recovers_expected_controlled_slope():
	frame = _temperature_frame()

	rates = predicted_lapse_rate(frame)

	assert rates.loc[0, "slope_c_per_km"] == pytest.approx(-6.5)
	assert rates.loc[0, "ci_lower"] <= -6.5 <= rates.loc[0, "ci_upper"]
	assert rates.loc[0, "month"] == 6


def test_lapse_check_fails_zero_slope_and_allows_configured_inversion_month():
	summer_rates = predicted_lapse_rate(
		_temperature_frame(slope=0.0, month_number=6)
	)
	winter_rates = predicted_lapse_rate(
		_temperature_frame(slope=0.0, month_number=12)
	)

	assert not lapse_check(summer_rates).loc[0, "passed"]
	winter = lapse_check(winter_rates)
	assert winter.loc[0, "outside_tolerance"]
	assert winter.loc[0, "inversion_month"]
	assert winter.loc[0, "passed"]


def test_lapse_rate_rejects_missing_control_and_bad_date_schema():
	frame = _temperature_frame().drop(columns="urban_fraction")
	with pytest.raises(SchemaError, match="control columns"):
		predicted_lapse_rate(frame)
	frame = _temperature_frame()
	frame["date"] = frame["date"].astype(str)
	with pytest.raises(SchemaError, match="datetime64"):
		predicted_lapse_rate(frame, control=())


def test_distance_band_weights_connect_islands_and_are_row_standardized():
	xy = np.array([[0.0, 0.0], [1.0, 0.0], [10_000.0, 0.0]])

	weights = build_weights(xy, k=None, distance_band=5.0)

	assert weights.transform == "R"
	assert not weights.islands
	assert all(sum(row_weights) == pytest.approx(1.0) for row_weights in weights.weights.values())


def test_knn_weights_reject_invalid_neighbor_count():
	with pytest.raises(ConfigError, match="k must be an integer"):
		build_weights(np.array([[0.0, 0.0], [1.0, 0.0]]), k=2)
	with pytest.raises(SchemaError, match="unique"):
		build_weights(np.array([[0.0, 0.0], [0.0, 0.0]]), k=1)


def test_structured_residuals_are_flagged_by_moran(monkeypatch):
	_spatial_weights_config(monkeypatch)
	cell_count = 30
	coordinates = np.column_stack(
		(np.arange(cell_count, dtype=np.float64) * 100.0, np.zeros(cell_count))
	)
	weights = build_weights(coordinates, k=4)
	dates = [pd.Timestamp("2020-06-01")] * cell_count
	residuals = pd.Series(
		np.sin(np.arange(cell_count) * np.pi / 8),
		index=pd.MultiIndex.from_arrays(
			[dates, np.arange(cell_count)], names=["date", "cell_id"]
		),
	)

	result = residual_morans_i(residuals, weights, permutations=99)

	assert len(result["per_date"]) == 1
	assert result["mean_I"] > get_config().validation.max_morans_i
	assert result["residual_autocorrelation"]
	assert not result["passed"]


def test_moran_rejects_weight_length_mismatch(monkeypatch):
	_spatial_weights_config(monkeypatch)
	weights = build_weights(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), k=1)

	with pytest.raises(SchemaError, match="weights contain"):
		residual_morans_i(pd.Series([1.0, 2.0]), weights, permutations=9)


def test_extreme_bounds_reports_each_physical_failure():
	frame = pd.DataFrame(
		{
			"pred_DTR": [5.0, -1.0, 60.0],
			"pred_TMAX": [20.0, 10.0, 20.0],
			"pred_TMIN": [10.0, 15.0, 5.0],
			"pred_RH2M": [50.0, 101.0, -1.0],
		}
	)

	result = extreme_physical_bounds(frame)

	assert result["passed"].tolist() == [True, False, False]
	assert not result.loc[1, "temperature_ordered"]
	assert not result.loc[2, "dtr_below_cap"]
	assert not result.loc[2, "humidity_in_bounds"]


def test_run_writes_report_and_returns_physical_report(tmp_path, monkeypatch):
	config = get_config()
	validation = replace(
		config.validation,
		spatial_weights_k=4,
		moran_permutations=19,
		strict=False,
		physical_report_directory=tmp_path / "reports",
	)
	monkeypatch.setattr(
		physical_checks, "get_config", lambda: replace(config, validation=validation)
	)
	rows: list[dict[str, object]] = []
	for month in (6, 7):
		frame = _temperature_frame(month_number=month)
		for row in frame.to_dict(orient="records"):
			row.update(
				{
					"y_true": row["y_pred"]
					+ (0.01 if int(row["cell_id"]) % 2 == 0 else -0.01),
					"pred_DTR": 10.0,
					"pred_TMAX": row["y_pred"] + 5.0,
					"pred_TMIN": row["y_pred"] - 5.0,
					"pred_RH2M": 50.0,
					"spatial_block": row["cell_id"] // 2,
				}
			)
			rows.append(row)
	oof = pd.DataFrame(rows)
	oof["cell_id"] = oof["cell_id"].astype("int64")
	oof_path = tmp_path / "oof.csv"
	oof.to_csv(oof_path, index=False)

	report = run(oof_path)

	assert report.passed
	assert (validation.physical_report_directory / "physical_checks.json").exists()
	assert (validation.physical_report_directory / "physical_checks.md").exists()


def test_run_rejects_duplicate_oof_keys(tmp_path):
	frame = _temperature_frame()
	frame["pred_DTR"] = 10.0
	frame["pred_TMAX"] = 25.0
	frame["pred_TMIN"] = 15.0
	frame["pred_RH2M"] = 50.0
	frame["y_true"] = frame["y_pred"]
	duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
	path = tmp_path / "duplicated.csv"
	duplicated.to_csv(path, index=False)

	with pytest.raises(DataIntegrityError, match="duplicate"):
		run(path)
