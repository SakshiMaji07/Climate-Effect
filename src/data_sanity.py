"""Integrity gate for monthly remote-sensing and climate forcing series."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from statsmodels.tsa.seasonal import STL

from config import get_config
from data_loader import validate_schema
from exceptions import DataIntegrityError, SchemaError
from logger import get_logger


logger = get_logger(__name__)
_SUPPORTED_CITIES = frozenset({"dehradun", "jammu"})


@dataclass(frozen=True)
class SanityReport:
	"""Summary of integrity and seasonality checks for one monthly series.

	Attributes:
		city: Canonical city key.
		n_obs: Number of input observations.
		monotonic_fraction: Fraction of finite month-to-month differences >= 0.
		max_run_nondecreasing: Longest consecutive run of nonnegative differences.
		seasonal_strength: STL seasonal strength, or NaN if it cannot be computed.
		noise_cv: Residual coefficient of variation, or NaN if unavailable.
		mann_kendall_tau: Kendall rank correlation against time.
		passed: Whether all configured sanity checks passed.
		failures: Names of failed checks.
	"""

	city: str
	n_obs: int
	monotonic_fraction: float
	max_run_nondecreasing: int
	seasonal_strength: float
	noise_cv: float
	mann_kendall_tau: float
	passed: bool
	failures: list[str]


def _validate_monthly_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
	"""Validate and return a sorted month-start datetime index."""
	if not isinstance(index, pd.DatetimeIndex):
		message = f"{name!r} must be indexed by a DatetimeIndex."
		logger.error(message)
		raise SchemaError(message)
	if index.tz is not None or not index.is_monotonic_increasing or index.has_duplicates:
		message = f"{name!r} index must be unique, timezone-naive, and increasing."
		logger.error(message)
		raise SchemaError(message)
	if not (index.day == 1).all() or not (index == index.normalize()).all():
		message = f"{name!r} dates must be month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	if len(index) > 1:
		periods = index.to_period("M").asi8
		if not np.all(np.diff(periods) == 1):
			message = f"{name!r} index must contain consecutive months."
			logger.error(message)
			raise SchemaError(message)
	return index


def _longest_nonnegative_run(differences: np.ndarray) -> int:
	"""Return the longest run of nonnegative finite differences."""
	longest = current = 0
	for difference in differences:
		if np.isfinite(difference) and difference >= 0:
			current += 1
			longest = max(longest, current)
		else:
			current = 0
	return longest


def _modal_difference_mask(
	differences: np.ndarray, tolerance: float
) -> np.ndarray:
	"""Mark differences matching their most frequent tolerance-sized bin."""
	mask = np.zeros(differences.shape, dtype=bool)
	finite_positions = np.flatnonzero(np.isfinite(differences))
	if finite_positions.size == 0:
		return mask
	finite_differences = differences[finite_positions]
	bins = np.rint(finite_differences / tolerance).astype(np.int64)
	unique_bins, counts = np.unique(bins, return_counts=True)
	mode = unique_bins[int(np.argmax(counts))]
	mask[finite_positions] = bins == mode
	return mask


def detect_forcing_signature(s: pd.Series) -> pd.Series:
	"""Identify steps matching the series' most common constant increment.

	Args:
		s: Monthly series indexed by month-start timestamps.

	Returns:
		A boolean mask aligned to ``s``. The first value is always false.

	Raises:
		SchemaError: If the series is not numeric or its index is malformed.
	"""
	index = _validate_monthly_index(s.index, s.name or "series")
	config = get_config().sanity
	try:
		values = s.to_numpy(dtype=np.float64, na_value=np.nan)
	except (TypeError, ValueError) as error:
		message = f"{s.name or 'series'!r} must contain numeric values."
		logger.error(message)
		raise SchemaError(message) from error
	if np.isinf(values).any():
		message = f"{s.name or 'series'!r} contains infinite values."
		logger.error(message)
		raise DataIntegrityError(message)
	differences = np.diff(values)
	mask = _modal_difference_mask(differences, config.signature_tolerance)
	result = np.concatenate(([False], mask))
	return pd.Series(result, index=index, name=s.name, dtype=bool)


def audit_series(s: pd.Series, city: str, name: str) -> SanityReport:
	"""Audit monotonicity, seasonality, residual noise, and fixed increments.

	Args:
		s: Monthly series indexed by consecutive month-start timestamps.
		city: Canonical city key.
		name: Canonical variable name.

	Returns:
		A report containing all calculated metrics and failed checks.

	Raises:
		SchemaError: If the series schema, city, or values are invalid.
		DataIntegrityError: If there are too few input observations.
	"""
	started = time.perf_counter()
	config = get_config().sanity
	index = _validate_monthly_index(s.index, name)
	logger.info(
		"Starting sanity audit %s:%s (n=%d).",
		city,
		name,
		len(s),
	)
	if city not in _SUPPORTED_CITIES:
		message = f"Unsupported city {city!r}."
		logger.error(message)
		raise SchemaError(message)
	try:
		values = s.to_numpy(dtype=np.float64, na_value=np.nan)
	except (TypeError, ValueError) as error:
		message = f"{name!r} must contain numeric values."
		logger.error(message)
		raise SchemaError(message) from error
	if np.isinf(values).any():
		message = f"{name!r} contains infinite values."
		logger.error(message)
		raise DataIntegrityError(message)
	if len(values) < config.min_obs:
		message = (
			f"{city}:{name} has {len(values)} observations; "
			f"at least {config.min_obs} are required."
		)
		logger.error(message)
		raise DataIntegrityError(message)

	gap_count = int(np.isnan(values).sum())
	if gap_count:
		logger.warning(
			"Interpolating %d missing observations for %s:%s (limit=%d).",
			gap_count,
			city,
			name,
			config.max_gap,
		)
	interpolated = s.astype(float).interpolate(
		method="time",
		limit=config.max_gap,
		limit_area="inside",
	)
	usable = interpolated.to_numpy(dtype=np.float64, na_value=np.nan)
	unfilled_gaps = bool(np.isnan(usable).any())
	failures: list[str] = []
	if unfilled_gaps:
		failures.append("unfilled_gaps")
		logger.warning(
			"%s:%s has %d observations still missing after interpolation.",
			city,
			name,
			int(np.isnan(usable).sum()),
		)

	differences = np.diff(usable)
	finite_differences = differences[np.isfinite(differences)]
	monotonic_fraction = (
		float(np.mean(finite_differences >= 0))
		if finite_differences.size
		else float("nan")
	)
	max_run = _longest_nonnegative_run(differences)
	if (
		np.isfinite(monotonic_fraction)
		and monotonic_fraction > config.max_monotonic_fraction
	):
		failures.append("monotonic_fraction")
	if max_run > config.max_run:
		failures.append("max_run_nondecreasing")

	valid_positions = np.flatnonzero(np.isfinite(usable))
	tau = (
		float(kendalltau(valid_positions, usable[valid_positions]).statistic)
		if valid_positions.size > 1
		else float("nan")
	)
	seasonal_strength = float("nan")
	noise_cv = float("nan")
	if not unfilled_gaps and len(usable) >= config.min_obs:
		try:
			decomposition = STL(
				usable, period=config.seasonal_period, robust=True
			).fit()
		except (ValueError, np.linalg.LinAlgError) as error:
			message = f"Could not decompose {city}:{name} with STL."
			logger.error(message)
			raise DataIntegrityError(message) from error
		denominator = float(np.var(decomposition.resid + decomposition.seasonal))
		seasonal_strength = (
			max(0.0, 1.0 - float(np.var(decomposition.resid)) / denominator)
			if denominator > 0
			else 0.0
		)
		noise_scale = float(np.mean(np.abs(decomposition.trend + decomposition.seasonal)))
		residual_std = float(np.std(decomposition.resid))
		noise_cv = residual_std / noise_scale if noise_scale > 0 else 0.0
		if name in config.seasonal_variables:
			if seasonal_strength < config.min_seasonal_strength:
				failures.append("seasonal_strength")
		if noise_cv < config.min_noise_cv:
			failures.append("noise_cv")
	else:
		failures.append("seasonal_strength_unavailable")
		failures.append("noise_cv_unavailable")

	signature_mask = _modal_difference_mask(
		differences, config.signature_tolerance
	)
	signature_fraction = (
		float(np.mean(signature_mask[np.isfinite(differences)]))
		if finite_differences.size
		else 0.0
	)
	if signature_fraction > config.max_constant_diff_fraction:
		failures.append("constant_increment_signature")

	report = SanityReport(
		city=city,
		n_obs=len(s),
		monotonic_fraction=monotonic_fraction,
		max_run_nondecreasing=max_run,
		seasonal_strength=seasonal_strength,
		noise_cv=noise_cv,
		mann_kendall_tau=tau,
		passed=not failures,
		failures=failures,
	)
	logger.info(
		"Finished sanity audit %s:%s (n=%d, passed=%s) in %.3fs.",
		city,
		name,
		len(s),
		report.passed,
		time.perf_counter() - started,
	)
	return report


def run_gate(
	df: pd.DataFrame, columns: Sequence[str]
) -> dict[str, SanityReport]:
	"""Audit every requested variable for every city in a tidy dataframe.

	Args:
		df: Tidy dataframe with ``city``, month-start ``date``, and variables.
		columns: Variable columns to audit.

	Returns:
		Reports keyed by ``"<city>:<variable>"``.

	Raises:
		SchemaError: If the input table violates its required schema.
		DataIntegrityError: If any report fails and fail-fast is configured.
	"""
	started = time.perf_counter()
	config = get_config()
	logger.info(
		"Starting forcing sanity gate (rows=%d, variables=%d).",
		len(df),
		len(columns),
	)
	if not columns:
		message = "At least one variable column must be requested."
		logger.error(message)
		raise SchemaError(message)
	if len(set(columns)) != len(columns):
		message = "Requested variable columns must be unique."
		logger.error(message)
		raise SchemaError(message)
	validate_schema(df, ("city", "date", *columns))
	if not pd.api.types.is_datetime64_ns_dtype(df["date"].dtype):
		message = "The date column must have dtype datetime64[ns]."
		logger.error(message)
		raise SchemaError(message)
	if df["date"].isna().any():
		message = "The date column cannot contain missing values."
		logger.error(message)
		raise SchemaError(message)
	if df["city"].isna().any() or not df["city"].isin(_SUPPORTED_CITIES).all():
		message = "The city column contains missing or unsupported city keys."
		logger.error(message)
		raise SchemaError(message)
	if not (df["date"].dt.day == 1).all() or not (
		df["date"] == df["date"].dt.normalize()
	).all():
		message = "The date column must contain month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	if df.duplicated(["city", "date"]).any():
		message = "The dataframe contains duplicate city/date rows."
		logger.error(message)
		raise SchemaError(message)
	ordered = df.sort_values(["city", "date"])
	reports: dict[str, SanityReport] = {}
	for city, city_frame in ordered.groupby("city", sort=True):
		series_frame = city_frame.set_index("date")
		for column in columns:
			key = f"{city}:{column}"
			try:
				report = audit_series(series_frame[column], str(city), column)
			except DataIntegrityError as error:
				report = SanityReport(
					city=str(city),
					n_obs=len(series_frame),
					monotonic_fraction=float("nan"),
					max_run_nondecreasing=0,
					seasonal_strength=float("nan"),
					noise_cv=float("nan"),
					mann_kendall_tau=float("nan"),
					passed=False,
					failures=[str(error)],
				)
			reports[key] = report

	failed = [
		key for key, report in reports.items() if not report.passed
	]
	elapsed = time.perf_counter() - started
	if failed:
		message = f"Forcing sanity gate failed for: {', '.join(failed)}."
		logger.error(
			"%s (rows=%d, reports=%d) in %.3fs.",
			message,
			len(df),
			len(reports),
			elapsed,
		)
		if config.sanity.fail_fast:
			raise DataIntegrityError(message)
	else:
		logger.info(
			"Forcing sanity gate passed (rows=%d, reports=%d) in %.3fs.",
			len(df),
			len(reports),
			elapsed,
		)
	return reports


def _json_safe_report(report: SanityReport) -> dict[str, object]:
	"""Convert non-finite numeric metrics to JSON null values."""
	content: dict[str, object] = asdict(report)
	for key, value in content.items():
		if isinstance(value, float) and not np.isfinite(value):
			content[key] = None
	return content


def write_report(
	reports: Mapping[str, SanityReport], path: Path
) -> None:
	"""Write JSON and Markdown summaries for sanity reports.

	A filename without a parent directory is placed under the configured
	``reports/sanity`` directory. Explicit paths are honored for test and
	deployment output locations.

	Args:
		reports: Reports returned by :func:`run_gate`.
		path: Output filename stem or explicit destination path.
	"""
	_started = time.perf_counter()
	logger.info("Starting sanity report write (reports=%d).", len(reports))
	paths = get_config().paths
	configured_directory = paths.reports / paths.sanity_subdirectory
	output_stem = Path(path)
	if not output_stem.is_absolute() and output_stem.parent == Path("."):
		output_stem = configured_directory / output_stem
	json_path = output_stem.with_suffix(".json")
	markdown_path = output_stem.with_suffix(".md")

	payload = {key: _json_safe_report(report) for key, report in reports.items()}
	try:
		output_stem.parent.mkdir(parents=True, exist_ok=True)
		json_path.write_text(
			json.dumps(payload, indent=2, allow_nan=False) + "\n",
			encoding="utf-8",
		)
	except OSError as error:
		message = f"Could not create or write sanity report JSON at {json_path}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	lines = [
		"# Forcing-series sanity report",
		"",
		"| City | Variable | Observations | Monotonic fraction | "
		"Longest run | Seasonal strength | Noise CV | Kendall tau | Passed | Failures |",
		"|---|---|---:|---:|---:|---:|---:|---:|:---:|---|",
	]
	for key, report in reports.items():
		city, variable = key.split(":", maxsplit=1)
		failures = ", ".join(report.failures) if report.failures else "None"
		lines.append(
			f"| {city} | {variable} | {report.n_obs} | "
			f"{report.monotonic_fraction:.4g} | {report.max_run_nondecreasing} | "
			f"{report.seasonal_strength:.4g} | {report.noise_cv:.4g} | "
			f"{report.mann_kendall_tau:.4g} | {report.passed} | {failures} |"
		)
	try:
		markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	except OSError as error:
		message = f"Could not write sanity report Markdown to {markdown_path}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	logger.info(
		"Finished sanity report write (reports=%d) in %.3fs.",
		len(reports),
		time.perf_counter() - _started,
	)


if __name__ == "__main__":
	"""Exercise a natural seasonal series and a cumulative-max forcing series."""
	from numpy.random import default_rng

	_config = get_config()
	_rng = default_rng(_config.seed)
	_dates = pd.date_range("2000-01-01", periods=_config.sanity.min_obs, freq="MS")
	_seasonal = 20 + 8 * np.sin(2 * np.pi * np.arange(len(_dates)) / 12)
	_natural = pd.Series(
		_seasonal + _rng.normal(0, 1.5, len(_dates)),
		index=_dates,
		name="TMAX",
	)
	_natural_report = audit_series(_natural, "dehradun", "TMAX")
	assert _natural_report.passed, _natural_report
	_forced_values = np.maximum.accumulate(
		_seasonal + _rng.normal(0, 1.5, len(_dates))
	)
	_forced = pd.Series(_forced_values, index=_dates, name="TMAX")
	_forced_report = audit_series(_forced, "dehradun", "TMAX")
	assert not _forced_report.passed, _forced_report
