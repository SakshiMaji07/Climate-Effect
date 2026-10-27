"""Calendar features, temporal validation splits, and leakage-safe transforms."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import get_config
from exceptions import DataIntegrityError, SchemaError
from logger import get_logger

logger = get_logger(__name__)
_DATE_COLUMN = "date"
_FIT_END_ATTR = "temporal_preprocessor_fit_end"
_FIT_STATS_ATTR = "temporal_preprocessor_stats"


@dataclass(frozen=True)
class Fold:
	"""A leakage-safe rolling-origin train/test partition.

	Attributes:
		train_idx: Row positions assigned to training.
		test_idx: Row positions assigned to testing.
		train_end: Latest timestamp represented in the training partition.
		test_start: Earliest timestamp represented in the test partition.
	"""

	train_idx: np.ndarray
	test_idx: np.ndarray
	train_end: pd.Timestamp
	test_start: pd.Timestamp


def _validate_dates(
	frame: pd.DataFrame,
	date_col: str,
	*,
	require_month_start: bool = True,
) -> pd.Series:
	"""Validate and return a non-null datetime64[ns] date column."""
	if date_col not in frame.columns:
		message = f"Missing required date column {date_col!r}."
		logger.error(message)
		raise SchemaError(message)
	dates = frame[date_col]
	if not pd.api.types.is_datetime64_ns_dtype(dates.dtype):
		message = f"{date_col!r} must have dtype datetime64[ns]."
		logger.error(message)
		raise SchemaError(message)
	if dates.isna().any():
		message = f"{date_col!r} cannot contain NaT values."
		logger.error(message)
		raise SchemaError(message)
	if require_month_start and (
		not (dates.dt.day == 1).all()
		or not (dates == dates.dt.normalize()).all()
	):
		message = f"{date_col!r} must contain month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	return dates


def add_seasonality(
	df: pd.DataFrame,
	date_col: str = "date",
	monsoon_months: tuple[int, ...] = (6, 7, 8, 9),
) -> pd.DataFrame:
	"""Add monsoon, annual cyclic, and configured harmonic features.

	Args:
		df: Input dataframe with monthly timestamps.
		date_col: Datetime column name.
		monsoon_months: Month numbers marked as monsoon months.

	Returns:
		A copied dataframe with ``is_monsoon``, ``sin_m``, ``cos_m``, and
		configured higher-harmonic columns named ``sin_m_k``/``cos_m_k``.

	Raises:
		SchemaError: If dates, monsoon months, or harmonic settings are invalid.
	"""
	started = time.perf_counter()
	dates = _validate_dates(df, date_col)
	if any(month < 1 or month > 12 for month in monsoon_months):
		message = "Monsoon months must be integers between 1 and 12."
		logger.error(message)
		raise SchemaError(message)
	harmonics = get_config().dataset.n_harmonics
	if harmonics < 1:
		message = "dataset.n_harmonics must be at least one."
		logger.error(message)
		raise SchemaError(message)
	logger.info("Starting seasonality generation (rows=%d).", len(df))
	out = df.copy()
	month = dates.dt.month
	out["is_monsoon"] = month.isin(monsoon_months).astype("int8")
	angle = 2.0 * np.pi * month.to_numpy(dtype=np.float64) / 12.0
	for harmonic in range(1, harmonics + 1):
		sin_name = "sin_m" if harmonic == 1 else f"sin_m_{harmonic}"
		cos_name = "cos_m" if harmonic == 1 else f"cos_m_{harmonic}"
		out[sin_name] = np.sin(harmonic * angle)
		out[cos_name] = np.cos(harmonic * angle)
	logger.info(
		"Finished seasonality generation (rows=%d, harmonics=%d) in %.3fs.",
		len(out),
		harmonics,
		time.perf_counter() - started,
	)
	return out


def rolling_origin_splits(
	dates: pd.Series,
	initial_train_months: int,
	horizon: int,
	step: int,
	gap: int = 0,
	expanding: bool = True,
) -> list[Fold]:
	"""Create non-overlapping rolling-origin folds on unique monthly dates.

	Panel rows sharing a date are assigned to the same fold. The ``gap`` is
	the number of excluded monthly dates between each training end and test
	start. Test windows from adjacent folds may not overlap.

	Args:
		dates: Series of month-start ``datetime64[ns]`` timestamps.
		initial_train_months: Number of unique dates in the first training fold.
		horizon: Number of unique dates in each test fold.
		step: Number of unique dates by which the origin advances.
		gap: Number of months excluded between train and test.
		expanding: If false, use a fixed-size training window.

	Returns:
		Chronologically ordered fold definitions with original row positions.

	Raises:
		SchemaError: If parameters/dates are invalid or too few folds are possible.
		DataIntegrityError: If constructed folds overlap.
	"""
	started = time.perf_counter()
	if not isinstance(dates, pd.Series):
		message = "dates must be a pandas Series."
		logger.error(message)
		raise SchemaError(message)
	if not pd.api.types.is_datetime64_ns_dtype(dates.dtype) or dates.isna().any():
		message = "dates must be non-null datetime64[ns] values."
		logger.error(message)
		raise SchemaError(message)
	if initial_train_months < 1 or horizon < 1 or step < 1 or gap < 0:
		message = "initial_train_months, horizon, and step must be positive; gap nonnegative."
		logger.error(message)
		raise SchemaError(message)
	if step < horizon:
		message = "step must be at least horizon so test folds do not overlap."
		logger.error(message)
		raise SchemaError(message)
	unique_dates = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
	if (
		unique_dates.tz is not None
		or not (unique_dates.day == 1).all()
		or not (unique_dates == unique_dates.normalize()).all()
	):
		message = "All dates must be timezone-naive month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	if len(unique_dates) > 1:
		periods = unique_dates.to_period("M").asi8
		if not np.all(np.diff(periods) == 1):
			message = "rolling_origin_splits requires a complete monthly date axis."
			logger.error(message)
			raise SchemaError(message)
	row_dates = pd.DatetimeIndex(dates)
	folds: list[Fold] = []
	train_end_position = initial_train_months - 1
	previous_test_positions: set[int] = set()
	while train_end_position + gap + horizon < len(unique_dates):
		test_start_position = train_end_position + gap + 1
		test_end_position = test_start_position + horizon
		if expanding:
			train_start_position = 0
		else:
			train_start_position = train_end_position - initial_train_months + 1
		train_dates = unique_dates[train_start_position : train_end_position + 1]
		test_dates = unique_dates[test_start_position:test_end_position]
		train_mask = row_dates.isin(train_dates)
		test_mask = row_dates.isin(test_dates)
		train_idx = np.flatnonzero(train_mask).astype(np.int64)
		test_idx = np.flatnonzero(test_mask).astype(np.int64)
		if np.intersect1d(train_idx, test_idx).size or set(test_idx) & previous_test_positions:
			message = "Rolling-origin folds contain overlapping train/test rows."
			logger.error(message)
			raise DataIntegrityError(message)
		if (
			not len(train_idx)
			or not len(test_idx)
			or unique_dates[test_start_position]
			<= unique_dates[train_end_position] + pd.DateOffset(months=gap)
		):
			message = "Rolling-origin split violates the configured temporal gap."
			logger.error(message)
			raise DataIntegrityError(message)
		folds.append(
			Fold(
				train_idx=train_idx,
				test_idx=test_idx,
				train_end=pd.Timestamp(unique_dates[train_end_position]),
				test_start=pd.Timestamp(unique_dates[test_start_position]),
			)
		)
		previous_test_positions.update(test_idx.tolist())
		train_end_position += step
	minimum_folds = get_config().validation.min_folds
	if len(folds) < minimum_folds:
		message = (
			f"Only {len(folds)} rolling-origin folds possible; "
			f"at least {minimum_folds} are required."
		)
		logger.error(message)
		raise SchemaError(message)
	logger.info(
		"Created rolling-origin folds (rows=%d, unique_dates=%d, folds=%d) in %.3fs.",
		len(dates),
		len(unique_dates),
		len(folds),
		time.perf_counter() - started,
	)
	return folds


class LeakageSafePreprocessor:
	"""Training-fold-only median imputation and standard scaling.

	Feature names are selected through ``config.dataset.impute_cols`` and
	``config.dataset.scale_cols``. Fit provenance is retained in transformed
	frame attributes so :func:`assert_no_leakage` can detect a preprocessor
	fit using future test rows.
	"""

	def __init__(self) -> None:
		self.scaler: StandardScaler | None = None
		self.medians_: dict[str, float] = {}
		self.scale_cols_: tuple[str, ...] = ()
		self.impute_cols_: tuple[str, ...] = ()
		self.fit_end_: pd.Timestamp | None = None
		self.fit_stats_signature_: str | None = None
		self._is_fitted = False

	def fit(self, train: pd.DataFrame) -> Self:
		"""Fit medians and a scaler using training data only.

		Args:
			train: Chronologically bounded training dataframe.

		Returns:
			This fitted preprocessor.

		Raises:
			SchemaError: If configured columns are absent or non-numeric.
		"""
		started = time.perf_counter()
		logger.info("Fitting temporal preprocessor (rows=%d).", len(train))
		if train.empty:
			message = "Cannot fit temporal preprocessor on an empty dataframe."
			logger.error(message)
			raise SchemaError(message)
		dates = _validate_dates(train, _DATE_COLUMN, require_month_start=False)
		dataset_config = get_config().dataset
		self.scale_cols_ = tuple(dataset_config.scale_cols)
		self.impute_cols_ = tuple(dataset_config.impute_cols)
		configured_columns = tuple(dict.fromkeys((*self.scale_cols_, *self.impute_cols_)))
		missing = [column for column in configured_columns if column not in train.columns]
		if missing:
			message = f"Training data lacks configured preprocessing columns: {missing}."
			logger.error(message)
			raise SchemaError(message)
		non_numeric = [
			column for column in configured_columns
			if not pd.api.types.is_numeric_dtype(train[column].dtype)
		]
		if non_numeric:
			message = f"Preprocessing columns must be numeric: {non_numeric}."
			logger.error(message)
			raise SchemaError(message)
		if any(np.isinf(train[column].to_numpy(dtype=np.float64)).any() for column in configured_columns):
			message = "Preprocessing columns cannot contain infinite values."
			logger.error(message)
			raise SchemaError(message)
		for column in self.impute_cols_:
			median = train[column].median(skipna=True)
			if pd.isna(median):
				message = f"Cannot compute training median for all-missing column {column!r}."
				logger.error(message)
				raise SchemaError(message)
			self.medians_[column] = float(median)
		if self.scale_cols_:
			scale_frame = train.loc[:, self.scale_cols_].copy()
			for column in self.scale_cols_:
				if column in self.medians_:
					scale_frame[column] = scale_frame[column].fillna(
						self.medians_[column]
					)
				elif scale_frame[column].isna().any():
					message = (
						f"Scaling column {column!r} has NaNs but is not configured "
						"for imputation."
					)
					logger.error(message)
					raise SchemaError(message)
			self.scaler = StandardScaler()
			self.scaler.fit(scale_frame)
		else:
			self.scaler = None
		self.fit_end_ = pd.Timestamp(dates.max())
		stats_payload = {
			"fit_end": self.fit_end_.isoformat(),
			"medians": self.medians_,
			"mean": self.scaler.mean_.tolist() if self.scaler is not None else [],
			"scale": self.scaler.scale_.tolist() if self.scaler is not None else [],
			"columns": self.scale_cols_,
		}
		self.fit_stats_signature_ = hashlib.sha256(
			json.dumps(stats_payload, sort_keys=True).encode("utf-8")
		).hexdigest()
		self._is_fitted = True
		logger.info(
			"Finished temporal preprocessor fit (rows=%d, scale_cols=%d) in %.3fs.",
			len(train),
			len(self.scale_cols_),
			time.perf_counter() - started,
		)
		return self

	def transform(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Apply fitted training-fold imputation/scaling to new rows.

		Args:
			df: Dataframe with columns used during :meth:`fit`.

		Returns:
			Transformed copy; source frame and fitted statistics are unchanged.

		Raises:
			SchemaError: If called before fitting or required columns are absent.
		"""
		if not self._is_fitted or self.fit_end_ is None:
			message = "LeakageSafePreprocessor must be fit before transform."
			logger.error(message)
			raise SchemaError(message)
		missing = [
			column for column in set((*self.impute_cols_, *self.scale_cols_))
			if column not in df.columns
		]
		if missing:
			message = f"Transform data lacks fitted preprocessing columns: {sorted(missing)}."
			logger.error(message)
			raise SchemaError(message)
		for column in set((*self.impute_cols_, *self.scale_cols_)):
			if not pd.api.types.is_numeric_dtype(df[column].dtype):
				message = f"Transform column {column!r} must be numeric."
				logger.error(message)
				raise SchemaError(message)
			if np.isinf(df[column].to_numpy(dtype=np.float64)).any():
				message = f"Transform column {column!r} contains infinite values."
				logger.error(message)
				raise SchemaError(message)
		out = df.copy()
		for column, median in self.medians_.items():
			out[column] = out[column].fillna(median)
		if self.scaler is not None:
			out.loc[:, self.scale_cols_] = self.scaler.transform(
				out.loc[:, self.scale_cols_]
			)
		out.attrs.update(df.attrs)
		out.attrs[_FIT_END_ATTR] = self.fit_end_.isoformat()
		out.attrs[_FIT_STATS_ATTR] = self.fit_stats_signature_
		return out

	def make_lags(
		self,
		df: pd.DataFrame,
		cols: tuple[str, ...] | list[str],
		lags: tuple[int, ...] | list[int],
		group_cols: tuple[str, ...] = ("city",),
		context_rows: int = 0,
	) -> pd.DataFrame:
		"""Create past-only group shifts, optionally removing prepended context.

		For test folds, concatenate the training tail before test rows and pass
		the number of prepended context rows. The returned dataframe excludes
		those context rows and retains test-row lag values.

		Args:
			df: Chronologically ordered rows with a date column.
			cols: Variables to lag.
			lags: Positive month/row offsets, e.g. ``(1, 2, 12)``.
			group_cols: Entity keys whose lags must be independent.
			context_rows: Number of train-context rows prepended to ``df``.

		Returns:
			Copy with additional ``<column>_lag_<offset>`` columns.

		Raises:
			SchemaError: If columns/lags/order/context are invalid.
			DataIntegrityError: If a lag would draw from a future-dated row.
		"""
		if not isinstance(cols, (tuple, list)) or not cols:
			message = "cols must be a non-empty list or tuple."
			logger.error(message)
			raise SchemaError(message)
		if not isinstance(lags, (tuple, list)) or not lags or any(lag < 1 for lag in lags):
			message = "lags must contain positive integer offsets."
			logger.error(message)
			raise SchemaError(message)
		_validate_dates(df, _DATE_COLUMN, require_month_start=False)
		required = (*cols, *group_cols)
		missing = [column for column in required if column not in df.columns]
		if missing:
			message = f"Lag input is missing required columns: {missing}."
			logger.error(message)
			raise SchemaError(message)
		if context_rows < 0 or context_rows >= len(df):
			if context_rows != 0:
				message = "context_rows must be nonnegative and smaller than input length."
				logger.error(message)
				raise SchemaError(message)
		for _, group in df.groupby(list(group_cols), sort=False, dropna=False):
			group_dates = pd.DatetimeIndex(group[_DATE_COLUMN])
			if (
				not group_dates.is_monotonic_increasing
				or group_dates.has_duplicates
			):
				message = (
					"Lag input dates must be strictly increasing within each group."
				)
				logger.error(message)
				raise SchemaError(message)
		out = df.copy()
		for column in cols:
			grouped = out.groupby(list(group_cols), sort=False, dropna=False)[column]
			for lag in lags:
				name = f"{column}_lag_{lag}"
				out[name] = grouped.shift(lag)
		if context_rows:
			out = out.iloc[context_rows:].copy()
		return out


def assert_no_leakage(
	train: pd.DataFrame,
	test: pd.DataFrame,
	date_col: str = "date",
) -> None:
	"""Verify temporal ordering and that transforms were fit before test dates.

	Transformed outputs carry a fit-end timestamp and a scaler/median signature
	in ``DataFrame.attrs``. If the same preprocessor was fit on the full
	train+test period, its fit-end timestamp will be at or after the test start
	and this check fails.

	Args:
		train: Training partition, chronologically before test.
		test: Held-out future partition.
		date_col: Timestamp column.

	Raises:
		SchemaError: If date columns are missing or invalid.
		DataIntegrityError: If partitions overlap, are reversed, or metadata shows
			that fitted transforms used future test rows.
	"""
	train_dates = _validate_dates(train, date_col, require_month_start=False)
	test_dates = _validate_dates(test, date_col, require_month_start=False)
	if train.empty or test.empty:
		message = "Leakage checks require non-empty train and test partitions."
		logger.error(message)
		raise SchemaError(message)
	train_end = pd.Timestamp(train_dates.max())
	test_start = pd.Timestamp(test_dates.min())
	if test_start <= train_end:
		message = "Train/test dates overlap or test does not follow training."
		logger.error(message)
		raise DataIntegrityError(message)
	fit_end_values = {
		frame.attrs.get(_FIT_END_ATTR) for frame in (train, test)
		if frame.attrs.get(_FIT_END_ATTR) is not None
	}
	if len(fit_end_values) > 1:
		message = "Train and test were transformed with different fitted statistics."
		logger.error(message)
		raise DataIntegrityError(message)
	fit_end_value = next(iter(fit_end_values), None)
	if fit_end_value is not None and pd.Timestamp(fit_end_value) > train_end:
		message = (
			"Preprocessor fit metadata extends beyond the training period; "
			"test information may have leaked into fitted statistics."
		)
		logger.error(message)
		raise DataIntegrityError(message)
	fit_stats = {
		frame.attrs.get(_FIT_STATS_ATTR) for frame in (train, test)
		if frame.attrs.get(_FIT_STATS_ATTR) is not None
	}
	if len(fit_stats) > 1:
		message = "Scaler/imputation statistics changed between train and test transforms."
		logger.error(message)
		raise DataIntegrityError(message)


if __name__ == "__main__":
	"""Demonstrate temporal splitting, training-only scaling, and lags."""
	_config = get_config()
	_dates = pd.date_range("2000-01-01", periods=48, freq="MS").as_unit("ns")
	_frame = pd.DataFrame(
		{
			"date": _dates,
			"city": "dehradun",
			"TMAX": np.arange(len(_dates), dtype=np.float64),
			"TMIN": np.arange(len(_dates), dtype=np.float64) / 2,
			"DTR": np.arange(len(_dates), dtype=np.float64) / 2,
			"LST": np.arange(len(_dates), dtype=np.float64) * 2,
			"RH2M": 50 + np.arange(len(_dates), dtype=np.float64),
			"PRECIP": np.arange(len(_dates), dtype=np.float64) + 1,
		}
	)
	_frame = add_seasonality(_frame)
	_fold = rolling_origin_splits(
		_frame["date"], initial_train_months=24, horizon=6, step=6
	)[0]
	_train = _frame.iloc[_fold.train_idx].copy()
	_test = _frame.iloc[_fold.test_idx].copy()
	_preprocessor = LeakageSafePreprocessor().fit(_train)
	_train_processed = _preprocessor.transform(_train)
	_test_processed = _preprocessor.transform(_test)
	assert_no_leakage(_train_processed, _test_processed)
	_context = _train.tail(12)
	_test_with_context = pd.concat([_context, _test], ignore_index=True)
	_lagged = _preprocessor.make_lags(
		_test_with_context,
		cols=("TMAX",),
		lags=(1, 12),
		context_rows=len(_context),
	)
	assert len(_lagged) == len(_test)
	assert {"is_monsoon", "sin_m", "cos_m"}.issubset(_frame.columns)
	assert {"TMAX_lag_1", "TMAX_lag_12"}.issubset(_lagged.columns)
