"""Tests for monthly forcing-series sanity checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from config import get_config
from data_sanity import (
	SanityReport,
	audit_series,
	detect_forcing_signature,
	run_gate,
	write_report,
)
from exceptions import DataIntegrityError, SchemaError


def _natural_series(
	name: str = "TMAX", periods: int = 72
) -> pd.Series:
	"""Build a deterministic seasonal signal with independent measurement noise."""
	rng = np.random.default_rng(get_config().seed)
	dates = pd.date_range(
		"2000-01-01", periods=periods, freq="MS"
	).as_unit("ns")
	seasonal = 18 + 7 * np.sin(2 * np.pi * np.arange(periods) / 12)
	return pd.Series(seasonal + rng.normal(0, 1.5, periods), index=dates, name=name)


def test_natural_seasonal_series_passes_gate():
	report = audit_series(_natural_series(), "dehradun", "TMAX")

	assert report.passed
	assert report.n_obs == 72
	assert report.seasonal_strength > 0.2
	assert report.noise_cv > 0
	assert report.failures == []


def test_cumulative_max_series_is_rejected():
	series = _natural_series()
	forced = pd.Series(
		np.maximum.accumulate(series.to_numpy()),
		index=series.index,
		name="TMAX",
	)

	report = audit_series(forced, "jammu", "TMAX")

	assert not report.passed
	assert {"monotonic_fraction", "max_run_nondecreasing"} & set(report.failures)


def test_schema_violation_raises_schema_error():
	frame = pd.DataFrame(
		{"city": ["dehradun"], "date": pd.to_datetime(["2000-01-01"])}
	)

	with pytest.raises(SchemaError, match="Missing required columns"):
		run_gate(frame, ["TMAX"])


def test_short_series_raises_data_integrity_error():
	short = _natural_series(periods=24)

	with pytest.raises(DataIntegrityError, match="at least"):
		audit_series(short, "dehradun", "TMAX")


def test_limited_interpolation_fills_small_internal_gap():
	series = _natural_series()
	series.iloc[20] = np.nan

	report = audit_series(series, "dehradun", "TMAX")

	assert "unfilled_gaps" not in report.failures
	assert np.isfinite(report.seasonal_strength)


def test_detect_forcing_signature_marks_repeated_increment_steps():
	dates = pd.date_range("2000-01-01", periods=60, freq="MS")
	series = pd.Series(np.arange(60, dtype=float) * 2.5, index=dates, name="LST")

	mask = detect_forcing_signature(series)

	assert not mask.iloc[0]
	assert mask.iloc[1:].all()


def test_run_gate_and_report_writes_json_and_markdown(tmp_path):
	series = _natural_series(periods=72)
	frame = pd.DataFrame(
		{
			"city": "dehradun",
			"date": series.index,
			"TMAX": series.to_numpy(),
		}
	)
	reports = run_gate(frame, ["TMAX"])
	report_path = tmp_path / "sanity"

	write_report(reports, report_path)

	assert (tmp_path / "sanity.json").is_file()
	assert (tmp_path / "sanity.md").is_file()
	contents = json.loads((tmp_path / "sanity.json").read_text(encoding="utf-8"))
	assert contents["dehradun:TMAX"]["passed"] is True
	assert isinstance(reports["dehradun:TMAX"], SanityReport)


def test_run_gate_raises_with_failed_variable_names():
	series = _natural_series(periods=72)
	forced = np.maximum.accumulate(series.to_numpy())
	frame = pd.DataFrame(
		{
			"city": "jammu",
			"date": series.index,
			"TMAX": series.to_numpy(),
			"LST": forced,
		}
	)

	with pytest.raises(DataIntegrityError, match="jammu:LST"):
		run_gate(frame, ["TMAX", "LST"])
