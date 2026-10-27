"""Tests for the ARDL/NARDL estimation pipeline."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import econometrics
from config import get_config
from econometrics import (
	ARDLSpec,
	bounds_test,
	diagnostics,
	fit_ardl,
	fit_nardl,
	long_run_multipliers,
	partial_sums,
	select_lags,
)
from exceptions import DataIntegrityError, SchemaError


def _cointegrated_data(n: int = 180, seed: int | None = None) -> tuple[pd.DataFrame, float]:
	"""Generate a stable error-correction system with known long-run slope."""
	config = get_config()
	rng = np.random.default_rng(config.seed if seed is None else seed)
	x = np.cumsum(rng.normal(0.0, 0.4, n))
	y = np.zeros(n, dtype=np.float64)
	long_run = 1.6
	for index in range(1, n):
		y[index] = (
			y[index - 1]
			+ 0.25 * (long_run * x[index - 1] - y[index - 1])
			+ rng.normal(0.0, 0.15)
		)
	dates = pd.date_range("2000-01-01", periods=n, freq="MS").as_unit("ns")
	frame = pd.DataFrame(
		{
			"date": dates,
			"DTR": y,
			"urban_fraction": x,
			"is_monsoon": ((dates.month >= 6) & (dates.month <= 9)).astype(np.int8),
			"sin_m": np.sin(2 * np.pi * dates.month / 12),
			"cos_m": np.cos(2 * np.pi * dates.month / 12),
			"RH2M": rng.normal(50.0, 5.0, n),
		}
	)
	return frame, long_run


def _spec(**overrides) -> ARDLSpec:
	"""Build a short synthetic-test model specification."""
	values = {
		"long_run": ["urban_fraction"],
		"fixed": ["is_monsoon", "sin_m", "cos_m", "RH2M"],
		"max_lag_dep": 2,
		"max_lag_x": 2,
		"ic": "bic",
		"case": 3,
	}
	values.update(overrides)
	return ARDLSpec(**values)


def test_partial_sums_decompose_increases_and_decreases():
	series = pd.Series([7.0, 9.0, 6.0, 7.0], name="urban_fraction")

	positive, negative = partial_sums(series)

	assert positive.tolist() == [0.0, 2.0, 2.0, 3.0]
	assert negative.tolist() == [0.0, 0.0, -3.0, -3.0]


def test_partial_sums_reject_nonfinite_input():
	series = pd.Series([1.0, np.nan, 2.0])

	with pytest.raises(SchemaError, match="finite"):
		partial_sums(series)


def test_lag_selection_respects_parameter_count_constraint():
	frame, _ = _cointegrated_data()
	spec = _spec(max_lag_dep=3, max_lag_x=3)

	p, q = select_lags(frame, spec)

	assert 1 <= p <= 3
	assert set(q) == {"urban_fraction"}
	assert 1 <= q["urban_fraction"] <= 3
	assert 1 + p + q["urban_fraction"] + 1 + len(spec.fixed) < len(frame) / 3


def test_fit_and_bounds_test_detect_cointegration_and_recover_multiplier():
	frame, expected = _cointegrated_data()

	fitted = fit_ardl(frame, _spec())
	bounds = bounds_test(fitted)
	multiplier = long_run_multipliers(fitted).loc[
		"urban_fraction", "multiplier"
	]

	assert bounds.verdict == "cointegrated"
	assert bounds.pesaran_critical_values["5%"]["lower"] < bounds.f_statistic
	assert bounds.small_sample_simulated_critical_values["5%"]["upper"] > 0
	assert multiplier == pytest.approx(expected, abs=0.5)


def test_fit_rejects_missing_model_variable():
	frame, _ = _cointegrated_data()

	with pytest.raises(SchemaError, match="missing columns"):
		fit_ardl(frame.drop(columns="RH2M"), _spec())


def test_fit_rejects_non_month_start_dates():
	frame, _ = _cointegrated_data()
	frame["date"] = frame["date"] + pd.Timedelta(days=1)

	with pytest.raises(SchemaError, match="month-start"):
		fit_ardl(frame, _spec())


def test_fit_rejects_missing_values_without_silent_row_dropping():
	frame, _ = _cointegrated_data()
	frame.loc[10, "RH2M"] = np.nan

	with pytest.raises(SchemaError, match="will not impute"):
		fit_ardl(frame, _spec())


def test_long_run_multiplier_refuses_non_cointegrated_result(monkeypatch):
	frame, _ = _cointegrated_data()
	fitted = fit_ardl(frame, _spec())
	monkeypatch.setattr(
		econometrics,
		"bounds_test",
		lambda _result: replace(
			bounds_test(fitted),
			verdict="no_cointegration",
			long_run_available=False,
		),
	)

	with pytest.raises(DataIntegrityError, match="unavailable"):
		long_run_multipliers(fitted)


def test_residual_diagnostics_returns_pvalues_and_pass_flags():
	frame, _ = _cointegrated_data()
	fitted = fit_ardl(frame, _spec())

	result = diagnostics(fitted)

	assert "breusch_godfrey" in result.results
	assert "breusch_pagan" in result.results
	assert "jarque_bera" in result.results
	assert "reset" in result.results
	assert all("pvalue" in result.results[key] for key in ("breusch_godfrey", "breusch_pagan", "jarque_bera", "reset"))
	assert result.passed


def test_nardl_reports_long_and_short_run_asymmetry_and_multipliers():
	frame, _ = _cointegrated_data()

	result = fit_nardl(frame, ["urban_fraction"])

	assert set(result.dynamic_multipliers.columns) == {
		"urban_fraction_pos",
		"urban_fraction_neg",
	}
	assert len(result.dynamic_multipliers) == get_config().econometrics.dynamic_horizon + 1
	assert np.isfinite(result.asymmetry_tests["urban_fraction"]["short_run_wald"])
	assert np.isfinite(result.asymmetry_tests["urban_fraction"]["short_run_pvalue"])


def test_narayan_values_require_supported_sample_and_interpolate_configured_tables():
	table = get_config().econometrics.narayan_critical_values
	configured = table["3"]["1"]["30"]

	result = econometrics._narayan_lookup(3, 1, 30)
	unsupported = econometrics._narayan_lookup(3, 1, 31)

	assert result is not None
	assert result["5"]["upper"] == configured["5_upper"]
	assert unsupported is None


def test_extreme_count_target_logs_gaussian_warning(caplog):
	frame, _ = _cointegrated_data()
	frame["heat_snap_count"] = np.maximum(0, np.rint(frame["DTR"]).astype(int))
	spec = _spec(dep="heat_snap_count", dep_is_count=True)
	spec.long_run = ["urban_fraction"]

	fit_ardl(frame, spec)

	assert "Gaussian ARDL assumptions" in caplog.text
