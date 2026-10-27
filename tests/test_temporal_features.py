"""Tests for seasonality features, temporal splits, and leakage safeguards."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import temporal_features
from config import get_config
from exceptions import DataIntegrityError, SchemaError
from temporal_features import (
	LeakageSafePreprocessor,
	add_seasonality,
	assert_no_leakage,
	rolling_origin_splits,
)


def _monthly_frame(periods: int = 72) -> pd.DataFrame:
	"""Make a configured-column monthly panel."""
	dates = pd.date_range("2000-01-01", periods=periods, freq="MS").as_unit("ns")
	values = np.arange(periods, dtype=np.float64)
	return pd.DataFrame(
		{
			"date": dates,
			"city": "dehradun",
			"TMAX": values + 20,
			"TMIN": values + 10,
			"DTR": np.full(periods, 10.0),
			"LST": values + 15,
			"RH2M": values + 50,
			"PRECIP": values + 1,
		}
	)


def test_add_seasonality_has_expected_monsoon_and_cyclic_schema():
	frame = _monthly_frame(12)

	result = add_seasonality(frame)

	assert result["is_monsoon"].dtype == np.dtype("int8")
	assert result["is_monsoon"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
	assert result.loc[0, "sin_m"] == pytest.approx(0.5)
	assert result.loc[0, "cos_m"] == pytest.approx(np.sqrt(3) / 2)
	assert result.loc[0, "date"] == frame.loc[0, "date"]


def test_add_seasonality_respects_configured_harmonic_count(monkeypatch):
	config = get_config()
	monkeypatch.setattr(
		temporal_features,
		"get_config",
		lambda: replace(
			config,
			dataset=replace(config.dataset, n_harmonics=3),
		),
	)

	result = add_seasonality(_monthly_frame(12))

	assert {"sin_m_2", "cos_m_2", "sin_m_3", "cos_m_3"}.issubset(result.columns)


def test_add_seasonality_rejects_non_datetime_schema():
	frame = _monthly_frame(12)
	frame["date"] = frame["date"].astype(str)

	with pytest.raises(SchemaError, match="datetime64\\[ns\\]"):
		add_seasonality(frame)


def test_rolling_origin_splits_panel_rows_by_unique_dates():
	frame = _monthly_frame(48)
	panel_dates = pd.concat([frame["date"], frame["date"]], ignore_index=True)

	folds = rolling_origin_splits(
		panel_dates,
		initial_train_months=24,
		horizon=6,
		step=6,
		gap=2,
	)

	assert len(folds) >= get_config().validation.min_folds
	for fold in folds:
		assert fold.train_end < fold.test_start
		assert len(fold.train_idx) % 2 == 0
		assert len(fold.test_idx) == 12
		assert not set(fold.train_idx) & set(fold.test_idx)


def test_rolling_origin_rejects_insufficient_folds():
	dates = _monthly_frame(12)["date"]

	with pytest.raises(SchemaError, match="folds possible"):
		rolling_origin_splits(dates, initial_train_months=10, horizon=2, step=2)


def test_preprocessor_fits_medians_and_scaler_on_training_only():
	frame = _monthly_frame(36)
	frame.loc[2, "TMAX"] = np.nan
	train = frame.iloc[:24].copy()
	test = frame.iloc[24:].copy()
	test.loc[test.index[0], "TMAX"] = 100_000.0
	preprocessor = LeakageSafePreprocessor().fit(train)
	mean_before = preprocessor.scaler.mean_.copy()
	medians_before = preprocessor.medians_.copy()

	train_out = preprocessor.transform(train)
	test_out = preprocessor.transform(test)
	assert_no_leakage(train_out, test_out)

	np.testing.assert_array_equal(preprocessor.scaler.mean_, mean_before)
	assert preprocessor.medians_ == medians_before
	assert preprocessor.medians_["TMAX"] == pytest.approx(train["TMAX"].median())
	assert test_out["TMAX"].max() > 100.0


def test_preprocessor_fit_on_full_period_is_detected_as_leakage():
	frame = _monthly_frame(36)
	train = frame.iloc[:24].copy()
	test = frame.iloc[24:].copy()
	preprocessor = LeakageSafePreprocessor().fit(pd.concat([train, test]))

	with pytest.raises(DataIntegrityError, match="beyond the training period"):
		assert_no_leakage(
			preprocessor.transform(train),
			preprocessor.transform(test),
		)


def test_lags_use_only_past_rows_and_drop_prepended_context():
	frame = _monthly_frame(36)
	train = frame.iloc[:24]
	test = frame.iloc[24:]
	context = train.tail(12)
	input_with_context = pd.concat([context, test], ignore_index=True)
	preprocessor = LeakageSafePreprocessor()

	result = preprocessor.make_lags(
		input_with_context,
		cols=("TMAX",),
		lags=(1, 12),
		context_rows=len(context),
	)

	assert len(result) == len(test)
	assert result["TMAX_lag_1"].iloc[0] == pytest.approx(train["TMAX"].iloc[-1])
	assert result["TMAX_lag_12"].iloc[0] == pytest.approx(
		train["TMAX"].iloc[-12]
	)


def test_assert_no_leakage_rejects_overlapping_dates():
	frame = _monthly_frame(30)

	with pytest.raises(DataIntegrityError, match="overlap"):
		assert_no_leakage(frame.iloc[:20], frame.iloc[19:])
