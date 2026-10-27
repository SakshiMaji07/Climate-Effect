"""Tests for the ARDL/NARDL collinearity and stationarity gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import get_config
from econometric_diagnostics import (
	condition_index,
	compute_vif,
	integration_order,
	run_gate,
	save_report,
)
from exceptions import ConfigError, DataIntegrityError, SchemaError


def _stationary_frame(n: int = 180, seed: int | None = None) -> pd.DataFrame:
	"""Create stationary monthly regressors and a noisy stationary target."""
	random_state = get_config().seed if seed is None else seed
	rng = np.random.default_rng(random_state)
	dates = pd.date_range("2000-01-01", periods=n, freq="MS").as_unit("ns")
	urban = 0.3 + rng.normal(0.0, 0.03, n)
	rh2m = 50 + rng.normal(0, 10, n)
	monsoon = (dates.month >= 6) & (dates.month <= 9)
	sin_m = np.sin(2 * np.pi * dates.month / 12)
	cos_m = np.cos(2 * np.pi * dates.month / 12)
	dtr = 10 + 0.5 * urban + 0.02 * rh2m + rng.normal(0.0, 0.8, n)
	return pd.DataFrame(
		{
			"date": dates,
			"DTR": dtr,
			"urban_fraction": urban,
			"RH2M": rh2m,
			"is_monsoon": monsoon.astype(np.int8),
			"sin_m": sin_m,
			"cos_m": cos_m,
		}
	)


def test_compute_vif_returns_only_regressors():
	frame = _stationary_frame()
	regressors = frame[["urban_fraction", "RH2M", "is_monsoon", "sin_m", "cos_m"]]

	vif = compute_vif(regressors)

	assert set(vif.index) == set(regressors.columns)
	assert np.isfinite(vif.to_numpy()).all()


def test_compute_vif_rejects_target_as_config_error():
	with pytest.raises(ConfigError, match="target"):
		compute_vif(pd.DataFrame({"DTR": np.arange(60.0), "x": np.arange(60.0) ** 2}))


def test_condition_index_reports_component_proportions():
	rng = np.random.default_rng(get_config().seed)
	x = rng.normal(size=300)
	regressors = pd.DataFrame(
		{
			"x": x,
			"y": rng.normal(size=300),
			"z": rng.normal(size=300),
		}
	)

	result = condition_index(regressors)

	assert "condition_index" in result.columns
	assert {"x", "y", "z"}.issubset(result.columns)
	np.testing.assert_allclose(result[["x", "y", "z"]].sum(axis=0), 1.0)


def test_integration_order_identifies_stationary_series_as_i0():
	rng = np.random.default_rng(get_config().seed)
	series = pd.Series(rng.normal(size=240))

	assert integration_order(series, seasonal_adjust=False) == 0


def test_integration_order_identifies_random_walk_as_i1():
	rng = np.random.default_rng(get_config().seed + 1)
	series = pd.Series(np.cumsum(rng.normal(size=300)))

	assert integration_order(series, seasonal_adjust=False) == 1


def test_integration_order_hard_fails_i2_series():
	rng = np.random.default_rng(get_config().seed + 2)
	first_integral = np.cumsum(rng.normal(size=400))
	second_integral = np.cumsum(first_integral)

	with pytest.raises(DataIntegrityError, match="I\\(2\\)"):
		integration_order(pd.Series(second_integral), seasonal_adjust=False)


def test_run_gate_passes_clean_regressors_and_includes_target_correlation():
	frame = _stationary_frame()
	regressors = ["urban_fraction", "RH2M", "is_monsoon", "sin_m", "cos_m"]

	report = run_gate(frame, "DTR", regressors)

	assert report.passed
	assert "DTR" in report.corr_matrix.columns
	assert "DTR" not in report.vif.index


def test_run_gate_rejects_injected_collinear_regressor():
	frame = _stationary_frame()
	frame["urban_fraction_duplicate"] = frame["urban_fraction"] * 2.0
	regressors = [
		"urban_fraction",
		"urban_fraction_duplicate",
		"RH2M",
		"is_monsoon",
		"sin_m",
		"cos_m",
	]

	with pytest.raises(DataIntegrityError, match="VIF threshold"):
		run_gate(frame, "DTR", regressors)


def test_run_gate_rejects_target_in_regressors():
	frame = _stationary_frame()

	with pytest.raises(ConfigError, match="must not"):
		run_gate(frame, "DTR", ["DTR", "RH2M"])


def test_run_gate_schema_violation_raises():
	frame = _stationary_frame().drop(columns="RH2M")

	with pytest.raises(SchemaError, match="missing columns"):
		run_gate(frame, "DTR", ["RH2M"])


def test_save_report_writes_json_and_csv_sidecars(tmp_path):
	frame = _stationary_frame()
	report = run_gate(
		frame,
		"DTR",
		["urban_fraction", "RH2M", "is_monsoon", "sin_m", "cos_m"],
	)

	save_report(report, tmp_path / "diagnostics.json")

	assert (tmp_path / "diagnostics.json").is_file()
	assert (tmp_path / "diagnostics_vif.csv").is_file()
	assert (tmp_path / "diagnostics_condition_index.csv").is_file()
	assert (tmp_path / "diagnostics_correlation.csv").is_file()
