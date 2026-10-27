"""Physical plausibility and residual spatial-autocorrelation checks."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial import cKDTree
from scipy.stats import combine_pvalues, t as student_t

from config import get_config
from exceptions import ConfigError, DataIntegrityError, SchemaError
from logger import get_logger

try:
	from libpysal.weights import W
	from esda.moran import Moran
except ImportError as error:
	W = None  # type: ignore[assignment,misc]
	Moran = None  # type: ignore[assignment,misc]
	_SPATIAL_STATISTICS_IMPORT_ERROR: ImportError | None = error
else:
	_SPATIAL_STATISTICS_IMPORT_ERROR = None

logger = get_logger(__name__)


@dataclass(frozen=True)
class PhysicalReport:
	"""Final physical plausibility gate results."""

	passed: bool
	lapse_rates: list[dict[str, Any]]
	lapse_checks: list[dict[str, Any]]
	moran_results: list[dict[str, Any]]
	physical_bounds: list[dict[str, Any]]
	failures: tuple[str, ...]


def _require_spatial_statistics() -> tuple[Any, Any]:
	"""Return libpysal/esda classes or raise an explicit dependency error."""
	if W is None or Moran is None:
		message = (
			"Physical spatial-autocorrelation checks require libpysal and esda; "
			"install the dependencies declared in requirements.txt."
		)
		logger.error(message)
		raise DataIntegrityError(message) from _SPATIAL_STATISTICS_IMPORT_ERROR
	return W, Moran


def _column(frame: pd.DataFrame, preferred: str, aliases: Sequence[str]) -> str:
	"""Find a configured column or its explicit supported aliases."""
	if preferred in frame.columns:
		return preferred
	for alias in aliases:
		if alias in frame.columns:
			return alias
	message = f"Required physical-check column {preferred!r} is missing."
	logger.error(message)
	raise SchemaError(message)


def _validate_dates(frame: pd.DataFrame) -> None:
	"""Validate canonical monthly dates."""
	if "date" not in frame:
		message = "Physical checks require a date column."
		logger.error(message)
		raise SchemaError(message)
	dates = frame["date"]
	if (
		not pd.api.types.is_datetime64_ns_dtype(dates.dtype)
		or dates.isna().any()
		or not (dates.dt.day == 1).all()
		or not (dates == dates.dt.normalize()).all()
	):
		message = "Physical-check dates must be datetime64[ns] month-start timestamps."
		logger.error(message)
		raise SchemaError(message)


def predicted_lapse_rate(
	pred: pd.DataFrame,
	month: bool = True,
	control: Sequence[str] = ("urban_fraction",),
) -> pd.DataFrame:
	"""Estimate predicted-temperature lapse rate by city and month.

	The OLS slope is converted from Celsius per metre to Celsius per kilometre.
	Confidence intervals use the configured confidence level and residual
	degrees of freedom. If ``month=False``, one slope per city is returned.

	Args:
		pred: Predictions and features containing city, date, elevation,
			predicted temperature, and optional control variables.
		month: Include each timestamp as a separate monthly group.
		control: Numeric covariates to adjust for, such as urban fraction.

	Returns:
		Dataframe with city, date/month, sample count, lapse slope and confidence
		interval, p-value, and model R-squared.

	Raises:
		SchemaError: If required columns or date/elevation values are invalid.
		DataIntegrityError: If any city-month cannot support the configured model.
	"""
	started = time.perf_counter()
	logger.info("Starting predicted lapse-rate estimation (rows=%d).", len(pred))
	if pred.empty or "city" not in pred:
		message = "pred must be non-empty and contain city."
		logger.error(message)
		raise SchemaError(message)
	if pred["city"].isna().any() or not pred["city"].isin({"dehradun", "jammu"}).all():
		message = "Predictions contain missing or unsupported city keys."
		logger.error(message)
		raise SchemaError(message)
	_validate_dates(pred)
	config = get_config().validation
	elevation_column = _column(
		pred, config.physical_elevation_column, ("dem", "elevation")
	)
	temperature_column = _column(
		pred,
		config.physical_temperature_column,
		("pred_temperature", "t2m_downscaled_pred", "t2m_downscaled", "y_pred"),
	)
	missing_controls = sorted(set(control) - set(pred.columns))
	if missing_controls:
		message = f"Lapse-rate control columns are missing: {missing_controls}."
		logger.error(message)
		raise SchemaError(message)
	for column in (elevation_column, temperature_column, *control):
		if not pd.api.types.is_numeric_dtype(pred[column].dtype):
			message = f"Lapse-rate column {column!r} must be numeric."
			logger.error(message)
			raise SchemaError(message)
	group_columns = ["city", "date"] if month else ["city"]
	rows: list[dict[str, Any]] = []
	for group_key, group in pred.groupby(group_columns, sort=True, observed=True):
		key_values = group_key if isinstance(group_key, tuple) else (group_key,)
		row: dict[str, Any] = dict(zip(group_columns, key_values, strict=True))
		columns = [elevation_column, temperature_column, *control]
		clean = group.loc[:, columns].replace([np.inf, -np.inf], np.nan).dropna()
		if len(clean) != len(group):
			logger.warning(
				"Dropped %d incomplete lapse-rate rows for %s.",
				len(group) - len(clean),
				row,
			)
		if len(clean) < len(control) + 3:
			message = (
				f"Lapse regression for {row} has {len(clean)} rows; at least "
				f"{len(control) + 3} are required."
			)
			logger.error(message)
			raise DataIntegrityError(message)
		elevation_km = clean[elevation_column].to_numpy(dtype=np.float64) / 1_000.0
		design = pd.DataFrame({"elevation_km": elevation_km}, index=clean.index)
		for control_name in control:
			design[control_name] = clean[control_name].astype("float64")
		design = sm.add_constant(design, has_constant="add")
		fit = sm.OLS(
			clean[temperature_column].astype("float64"), design, missing="raise"
		).fit()
		confidence = config.bootstrap_confidence
		if not 0 < confidence < 1:
			message = "validation.bootstrap_confidence must lie strictly between zero and one."
			logger.error(message)
			raise ConfigError(message)
		degrees = float(fit.df_resid)
		if degrees <= 0:
			message = f"Lapse regression for {row} has no residual degrees of freedom."
			logger.error(message)
			raise DataIntegrityError(message)
		critical = float(student_t.ppf((1 + confidence) / 2, degrees))
		slope = float(fit.params["elevation_km"])
		standard_error = float(fit.bse["elevation_km"])
		total_sum_squares = float(
			np.sum(
				(
					clean[temperature_column].to_numpy(dtype=np.float64)
					- clean[temperature_column].mean()
				)
				** 2
			)
		)
		r_squared = (
			float(1.0 - fit.ssr / total_sum_squares)
			if total_sum_squares > np.finfo(np.float64).eps
			else (1.0 if float(fit.ssr) <= np.finfo(np.float64).eps else 0.0)
		)
		row.update(
			{
				"n": len(clean),
				"month": int(pd.Timestamp(row["date"]).month) if month else pd.NA,
				"slope_c_per_km": slope,
				"ci_lower": slope - critical * standard_error,
				"ci_upper": slope + critical * standard_error,
				"p_value": float(fit.pvalues["elevation_km"]),
				"r_squared": r_squared,
			}
		)
		rows.append(row)
	result = pd.DataFrame(rows)
	logger.info(
		"Finished predicted lapse-rate estimation (rows=%d, groups=%d) in %.3fs.",
		len(pred),
		len(result),
		time.perf_counter() - started,
	)
	return result


def _permutation_pvalue(
	values: np.ndarray,
	weights: Any,
	observed_i: float,
	permutations: int,
) -> float:
	"""Compute deterministic one-sided randomization p-value for positive I."""
	if permutations == 0:
		return float("nan")
	centered = values - values.mean()
	variance = float(centered @ centered)
	if variance <= np.finfo(np.float64).eps:
		message = "Moran permutation test requires residual variance."
		logger.error(message)
		raise DataIntegrityError(message)
	sparse_weights = weights.sparse
	scale = len(values) / float(weights.s0)
	rng = np.random.default_rng(get_config().seed)
	more_extreme = 0
	for _ in range(permutations):
		permuted = rng.permutation(centered)
		permuted_i = scale * float(permuted @ (sparse_weights @ permuted)) / variance
		more_extreme += int(permuted_i >= observed_i)
	return float((more_extreme + 1) / (permutations + 1))


def lapse_check(
	rates: pd.DataFrame,
	expected: float | None = None,
	tol: float | None = None,
) -> pd.DataFrame:
	"""Flag lapse rates outside tolerance while reporting permitted inversions.

	Args:
		rates: Output from :func:`predicted_lapse_rate`.
		expected: Expected lapse rate in Celsius per kilometre.
		tol: Absolute tolerance in Celsius per kilometre.

	Returns:
		Copy of rates with inversion-month, tolerance, pass, and reason flags.

	Raises:
		SchemaError: If lapse-rate fields are missing or invalid.
		ConfigError: If expected rate or tolerance is invalid.
	"""
	started = time.perf_counter()
	logger.info("Starting lapse-rate check (rows=%d).", len(rates))
	config = get_config().validation
	expected_rate = config.expected_lapse_rate_c_per_km if expected is None else expected
	tolerance = config.lapse_tol if tol is None else tol
	try:
		expected_rate = float(expected_rate)
		tolerance = float(tolerance)
	except (TypeError, ValueError) as error:
		message = "Expected lapse rate and tolerance must be numeric."
		logger.error(message)
		raise ConfigError(message) from error
	if not np.isfinite(expected_rate) or not np.isfinite(tolerance) or tolerance < 0:
		message = "Expected lapse rate must be finite and tolerance must be finite and nonnegative."
		logger.error(message)
		raise ConfigError(message)
	required = {"slope_c_per_km", "month"}
	if not required.issubset(rates.columns):
		message = f"Lapse rates are missing required columns: {sorted(required - set(rates.columns))}."
		logger.error(message)
		raise SchemaError(message)
	if rates.empty or not np.isfinite(rates["slope_c_per_km"].to_numpy(dtype=np.float64)).all():
		message = "Lapse rates must be non-empty and finite."
		logger.error(message)
		raise SchemaError(message)
	month_values = pd.to_numeric(rates["month"], errors="coerce")
	invalid_months = month_values.notna() & ~month_values.between(1, 12)
	if invalid_months.any():
		message = "Lapse rate month values must lie from 1 through 12."
		logger.error(message)
		raise SchemaError(message)
	result = rates.copy()
	result["month"] = month_values
	result["outside_tolerance"] = (
		(result["slope_c_per_km"] - expected_rate).abs() > tolerance
	)
	result["inversion_allowed"] = result["month"].isin(
		config.allowed_inversion_months
	)
	result["inversion_month"] = result["inversion_allowed"]
	result["passed"] = ~result["outside_tolerance"] | result["inversion_allowed"]
	result["failure_reason"] = np.where(
		result["outside_tolerance"] & ~result["inversion_allowed"],
		"lapse_rate_outside_tolerance",
		"",
	)
	logger.info(
		"Finished lapse-rate check (rows=%d, failed=%d) in %.3fs.",
		len(result),
		int((~result["passed"]).sum()),
		time.perf_counter() - started,
	)
	return result


def build_weights(
	xy: np.ndarray,
	k: int | None = None,
	distance_band: float | None = None,
) -> Any:
	"""Construct row-standardized KNN or distance-band spatial weights.

	Distance-band islands are linked to their nearest neighbour and reported
	as warnings so the returned weights remain usable by Moran's I.

	Args:
		xy: Unique projected coordinates in metres.
		k: Neighbours per observation. Set to ``None`` to use a distance band.
		distance_band: Neighbour radius in metres. If supplied, distance-band
			weights are used regardless of ``k``.

	Returns:
		A row-standardized ``libpysal.weights.W`` instance.

	Raises:
		SchemaError: If coordinates or neighbour settings are invalid.
		ConfigError: If settings request an impossible neighbour structure.
		DataIntegrityError: If libpysal/esda are unavailable.
	"""
	weights_class, _ = _require_spatial_statistics()
	started = time.perf_counter()
	logger.info("Starting spatial weights construction (rows=%d).", len(xy))
	try:
		coordinates = np.asarray(xy, dtype=np.float64)
	except (TypeError, ValueError) as error:
		message = "Spatial weights coordinates must be numeric."
		logger.error(message)
		raise SchemaError(message) from error
	if (
		coordinates.ndim != 2
		or coordinates.shape[1] != 2
		or not np.isfinite(coordinates).all()
		or len(coordinates) < 2
	):
		message = "xy must contain at least two finite projected coordinate pairs."
		logger.error(message)
		raise SchemaError(message)
	if pd.DataFrame(coordinates).duplicated().any():
		message = "Spatial weights require unique cell coordinates."
		logger.error(message)
		raise SchemaError(message)
	if distance_band is not None:
		try:
			distance_band = float(distance_band)
		except (TypeError, ValueError) as error:
			message = "distance_band must be a positive finite distance in metres."
			logger.error(message)
			raise ConfigError(message) from error
		if not np.isfinite(distance_band) or distance_band <= 0:
			message = "distance_band must be a positive finite distance in metres."
			logger.error(message)
			raise ConfigError(message)
		tree = cKDTree(coordinates)
		neighbors: dict[int, list[int]] = {}
		for index, coordinate in enumerate(coordinates):
			nearby = tree.query_ball_point(coordinate, r=distance_band)
			neighbors[index] = [candidate for candidate in nearby if candidate != index]
		islands = [index for index, neighbors_for_cell in neighbors.items() if not neighbors_for_cell]
		if islands:
			logger.warning(
				"Distance-band weights found %d islands; connecting each to its nearest neighbour.",
				len(islands),
			)
			_, nearest = tree.query(coordinates[islands], k=2)
			for index, candidates in zip(islands, nearest, strict=True):
				neighbor = int(candidates[1])
				neighbors[index].append(neighbor)
				neighbors[neighbor].append(index)
	else:
		neighbor_count = (
			get_config().validation.spatial_weights_k if k is None else k
		)
		if not isinstance(neighbor_count, (int, np.integer)) or not 1 <= neighbor_count < len(coordinates):
			message = "k must be an integer from one to the number of cells minus one."
			logger.error(message)
			raise ConfigError(message)
		tree = cKDTree(coordinates)
		_, nearest = tree.query(coordinates, k=int(neighbor_count) + 1)
		neighbors = {
			index: [int(candidate) for candidate in nearest[index, 1:]]
			for index in range(len(coordinates))
		}
	weights = weights_class(
		neighbors,
		{index: [1.0] * len(adjacent) for index, adjacent in neighbors.items()},
		silence_warnings=True,
	)
	weights.transform = "R"
	logger.info(
		"Finished spatial weights construction (rows=%d, islands=%d) in %.3fs.",
		len(coordinates),
		len(weights.islands),
		time.perf_counter() - started,
	)
	return weights


def residual_morans_i(
	resid: pd.Series,
	w: Any,
	permutations: int | None = None,
) -> dict[str, Any]:
	"""Compute Moran's I per date plus pooled mean and Fisher-combined p-value.

	For dated residuals, the Series index must be a DatetimeIndex or a
	MultiIndex with a ``date`` level, and each date must have the same number
	of residuals as spatial weights.

	Args:
		resid: Finite residuals, optionally indexed by date/cell.
		w: Row-standardized libpysal spatial weights matching each date's order.
		permutations: Number of permutation draws per date.

	Returns:
		Dictionary with per-date statistics, pooled mean I, Fisher p-value, and
		a residual-autocorrelation misspecification flag.

	Raises:
		SchemaError: If residuals, weights, dates, or permutation count are invalid.
		DataIntegrityError: If Moran's I cannot be computed.
	"""
	_, moran_class = _require_spatial_statistics()
	started = time.perf_counter()
	logger.info("Starting residual Moran test (rows=%d).", len(resid))
	if not isinstance(resid, pd.Series) or resid.empty:
		message = "resid must be a non-empty pandas Series."
		logger.error(message)
		raise SchemaError(message)
	permutation_count = (
		get_config().validation.moran_permutations
		if permutations is None
		else permutations
	)
	if not isinstance(permutation_count, int) or permutation_count < 1:
		message = "permutations must be a positive integer."
		logger.error(message)
		raise ConfigError(message)
	try:
		values = resid.to_numpy(dtype=np.float64)
	except (TypeError, ValueError) as error:
		message = "Residual series must be numeric."
		logger.error(message)
		raise SchemaError(message) from error
	if not np.isfinite(values).all():
		message = "Residual series must contain only finite values."
		logger.error(message)
		raise DataIntegrityError(message)
	n_weights = int(getattr(w, "n", -1))
	if n_weights < 2:
		message = "Spatial weights must contain at least two observations."
		logger.error(message)
		raise SchemaError(message)
	if isinstance(resid.index, pd.MultiIndex) and "date" in resid.index.names:
		date_values = pd.Series(
			resid.index.get_level_values("date"), index=resid.index
		)
		groups = [(str(date), group) for date, group in resid.groupby(date_values, sort=True)]
	elif isinstance(resid.index, pd.DatetimeIndex):
		groups = [(str(date), group) for date, group in resid.groupby(resid.index, sort=True)]
	else:
		groups = [("pooled", resid)]
	per_date: list[dict[str, float | int | str]] = []
	for date_label, group in groups:
		if len(group) != n_weights:
			message = (
				f"Residual group {date_label!r} has {len(group)} rows, but weights "
				f"contain {n_weights} observations."
			)
			logger.error(message)
			raise SchemaError(message)
		try:
			result = moran_class(
				group.to_numpy(dtype=np.float64),
				w,
				transformation="R",
				permutations=0,
			)
		except (ValueError, ZeroDivisionError, FloatingPointError) as error:
			message = f"Could not compute Moran's I for residual group {date_label!r}."
			logger.error(message)
			raise DataIntegrityError(message) from error
		per_date.append(
			{
				"date": date_label,
				"n": len(group),
				"I": float(result.I),
				"expected_I": float(result.EI),
				"z": float(result.z_norm),
				"p": _permutation_pvalue(
					group.to_numpy(dtype=np.float64),
					w,
					float(result.I),
					permutation_count,
				),
			}
		)
	if not per_date:
		message = "No residual groups were available for Moran's I."
		logger.error(message)
		raise DataIntegrityError(message)
	p_values = np.clip(
		np.asarray([row["p"] for row in per_date], dtype=np.float64),
		np.finfo(np.float64).tiny,
		1.0,
	)
	pooled_p = (
		float(p_values[0])
		if len(p_values) == 1
		else float(combine_pvalues(p_values, method="fisher").pvalue)
	)
	mean_i = float(np.mean([row["I"] for row in per_date]))
	config = get_config().validation
	autocorrelation = pooled_p < config.morans_alpha and mean_i > config.max_morans_i
	report = {
		"per_date": per_date,
		"mean_I": mean_i,
		"fisher_p": pooled_p,
		"max_morans_i": config.max_morans_i,
		"alpha": config.morans_alpha,
		"residual_autocorrelation": bool(autocorrelation),
		"passed": not autocorrelation,
	}
	logger.info(
		"Finished residual Moran test (groups=%d, mean_I=%.4f, p=%.4g) in %.3fs.",
		len(per_date),
		mean_i,
		pooled_p,
		time.perf_counter() - started,
	)
	return report


def _physical_column(
	frame: pd.DataFrame, predicted: str, observed: str
) -> str:
	"""Resolve the predicted physical variable or its canonical observed name."""
	variable = observed
	aliases = (observed, f"{observed}_pred")
	if variable == "DTR":
		aliases = (*aliases, "DTR_cell")
	return _column(frame, predicted, aliases)


def extreme_physical_bounds(pred: pd.DataFrame) -> pd.DataFrame:
	"""Check DTR, temperature ordering, and relative-humidity physical bounds.

	Accepted prediction column names are ``pred_<variable>``, ``<variable>_pred``,
	or the canonical observed variable name. DTR accepts ``DTR_cell`` as well.

	Args:
		pred: Prediction dataframe containing the four physical variables.

	Returns:
		Row-level checks with one boolean per physical constraint and an overall
		``passed`` flag.

	Raises:
		SchemaError: If a variable is missing or non-numeric.
		DataIntegrityError: If physical input values contain non-finite data.
	"""
	started = time.perf_counter()
	logger.info("Starting physical-bound checks (rows=%d).", len(pred))
	if pred.empty:
		message = "Physical-bound input cannot be empty."
		logger.error(message)
		raise SchemaError(message)
	config = get_config().validation
	column_map = {
		"DTR": _physical_column(pred, "pred_DTR", "DTR"),
		"TMAX": _physical_column(pred, "pred_TMAX", "TMAX"),
		"TMIN": _physical_column(pred, "pred_TMIN", "TMIN"),
		"RH2M": _physical_column(pred, "pred_RH2M", "RH2M"),
	}
	if column_map["DTR"] not in pred:
		column_map["DTR"] = _column(pred, "pred_DTR", ("DTR_cell",))
	for variable, column in column_map.items():
		if not pd.api.types.is_numeric_dtype(pred[column].dtype):
			message = f"Physical variable {variable!r} column {column!r} must be numeric."
			logger.error(message)
			raise SchemaError(message)
	values = pred.loc[:, list(column_map.values())].to_numpy(dtype=np.float64)
	if not np.isfinite(values).all():
		message = "Physical-bound values cannot contain NaNs or infinities."
		logger.error(message)
		raise DataIntegrityError(message)
	dtr = pred[column_map["DTR"]].to_numpy(dtype=np.float64)
	tmax = pred[column_map["TMAX"]].to_numpy(dtype=np.float64)
	tmin = pred[column_map["TMIN"]].to_numpy(dtype=np.float64)
	rh = pred[column_map["RH2M"]].to_numpy(dtype=np.float64)
	result = pred.copy()
	result["dtr_positive"] = dtr > 0
	result["temperature_ordered"] = tmax > tmin
	result["dtr_below_cap"] = dtr <= config.dtr_max_c
	result["humidity_in_bounds"] = (rh >= config.rh_min_percent) & (
		rh <= config.rh_max_percent
	)
	check_columns = (
		"dtr_positive",
		"temperature_ordered",
		"dtr_below_cap",
		"humidity_in_bounds",
	)
	result["passed"] = result.loc[:, check_columns].all(axis=1)
	logger.info(
		"Finished physical-bound checks (rows=%d, failed=%d) in %.3fs.",
		len(result),
		int((~result["passed"]).sum()),
		time.perf_counter() - started,
	)
	return result


def _load_oof(path: Path) -> pd.DataFrame:
	"""Read parquet or CSV OOF input explicitly by file format."""
	try:
		if path.suffix.lower() in {".parquet", ".pq"}:
			return pd.read_parquet(path)
		if path.suffix.lower() == ".csv":
			frame = pd.read_csv(path)
			if "date" in frame:
				frame["date"] = pd.to_datetime(frame["date"], errors="raise").astype(
					"datetime64[ns]"
				)
			return frame
	except (OSError, ValueError, ImportError) as error:
		message = f"Could not load physical-check OOF data from {path}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	message = f"Unsupported OOF file extension {path.suffix!r}; use parquet or CSV."
	logger.error(message)
	raise SchemaError(message)


def run(oof_path: Path) -> PhysicalReport:
	"""Run the final physical gate and write its JSON/Markdown report.

	The OOF table must include spatial coordinates, elevation, temperature
	predictions, and either predicted or canonical DTR/TMAX/TMIN/RH2M columns.
	Residual Moran tests are computed separately by city, using a distinct
	weight matrix per city and the same cell order for each date.

	Args:
		oof_path: Parquet or CSV path to the physical-check prediction panel.

	Returns:
		Physical report with ``passed`` false when any strict physical gate fails.

	Raises:
		SchemaError: If the input lacks required physical fields or key schemas.
		DataIntegrityError: If physical checks cannot be evaluated or report
			files cannot be written.
	"""
	started = time.perf_counter()
	logger.info("Starting physical checks (path=%s).", oof_path)
	frame = _load_oof(Path(oof_path))
	if frame.empty or not {"city", "cell_id", "date"}.issubset(frame.columns):
		message = "Physical-check OOF needs city, cell_id, date, and non-empty rows."
		logger.error(message)
		raise SchemaError(message)
	required_spatial = {"x", "y"}
	if not required_spatial.issubset(frame.columns):
		message = f"Physical-check OOF lacks projected coordinates: {sorted(required_spatial - set(frame.columns))}."
		logger.error(message)
		raise SchemaError(message)
	if frame["cell_id"].dtype != np.dtype("int64"):
		message = "Physical-check cell_id must have dtype int64."
		logger.error(message)
		raise SchemaError(message)
	if frame.duplicated(["city", "cell_id", "date"]).any():
		message = "Physical-check OOF contains duplicate city-cell-date keys."
		logger.error(message)
		raise DataIntegrityError(message)
	_validate_dates(frame)
	lapse_rates = predicted_lapse_rate(
		frame,
		month=True,
		control=get_config().validation.physical_control_columns,
	)
	lapse_results = lapse_check(lapse_rates)
	bounds = extreme_physical_bounds(frame)
	temperature_column = _column(
		frame,
		get_config().validation.physical_temperature_column,
		("pred_temperature", "t2m_downscaled_pred", "t2m_downscaled", "y_pred"),
	)
	residual_column = _column(frame, "residual", ("y_true",))
	if residual_column == "y_true":
		try:
			residuals = frame["y_true"].to_numpy(
				dtype=np.float64
			) - frame[temperature_column].to_numpy(dtype=np.float64)
		except (TypeError, ValueError) as error:
			message = "OOF y_true and predicted temperature must be numeric."
			logger.error(message)
			raise SchemaError(message) from error
	else:
		try:
			residuals = frame[residual_column].to_numpy(dtype=np.float64)
		except (TypeError, ValueError) as error:
			message = f"Residual column {residual_column!r} must be numeric."
			logger.error(message)
			raise SchemaError(message) from error
	if not np.isfinite(residuals).all():
		message = "Physical-check residuals contain non-finite values."
		logger.error(message)
		raise DataIntegrityError(message)
	coordinates = frame.loc[:, ["x", "y"]]
	if coordinates.isna().any().any() or not np.isfinite(
		coordinates.to_numpy(dtype=np.float64)
	).all():
		message = "Physical-check projected coordinates must be finite."
		logger.error(message)
		raise SchemaError(message)
	moran_results: list[dict[str, Any]] = []
	for city, city_frame in frame.assign(_residual=residuals).groupby("city", sort=True):
		cell_table = city_frame[["cell_id", "x", "y"]].drop_duplicates()
		if cell_table["cell_id"].duplicated().any():
			message = f"Projected cell coordinates change over time in {city!r}."
			logger.error(message)
			raise SchemaError(message)
		cell_table = cell_table.sort_values("cell_id", kind="stable")
		weight_config = get_config().validation
		weights = build_weights(
			cell_table[["x", "y"]].to_numpy(dtype=np.float64),
			k=weight_config.spatial_weights_k,
			distance_band=weight_config.spatial_distance_band_m,
		)
		ordered = city_frame.merge(
			cell_table[["cell_id"]],
			on="cell_id",
			how="inner",
			validate="many_to_one",
		).sort_values(["date", "cell_id"], kind="stable")
		residual_series = pd.Series(
			ordered["_residual"].to_numpy(dtype=np.float64),
			index=pd.MultiIndex.from_arrays(
				[ordered["date"], ordered["cell_id"]],
				names=["date", "cell_id"],
			),
		)
		moran = residual_morans_i(
			residual_series,
			weights,
			permutations=weight_config.moran_permutations,
		)
		moran["city"] = str(city)
		moran_results.append(moran)
	failures: list[str] = []
	if not lapse_results["passed"].all():
		failures.append("lapse_rate")
	if not bounds["passed"].all():
		failures.append("physical_bounds")
	if any(result["residual_autocorrelation"] for result in moran_results):
		failures.append("residual_spatial_autocorrelation")
	report = PhysicalReport(
		passed=not failures,
		lapse_rates=lapse_rates.to_dict(orient="records"),
		lapse_checks=lapse_results.to_dict(orient="records"),
		moran_results=moran_results,
		physical_bounds=bounds.to_dict(orient="records"),
		failures=tuple(failures),
	)
	output_dir = get_config().validation.physical_report_directory
	_write_physical_report(report, Path(output_dir))
	logger.info(
		"Finished physical checks (rows=%d, passed=%s) in %.3fs.",
		len(frame),
		report.passed,
		time.perf_counter() - started,
	)
	if get_config().validation.strict and not report.passed:
		message = f"Physical validation gate failed: {', '.join(report.failures)}."
		logger.error(message)
		raise DataIntegrityError(message)
	return report


def _write_physical_report(report: PhysicalReport, output_dir: Path) -> None:
	"""Write machine-readable JSON and concise Markdown gate summaries."""
	started = time.perf_counter()
	logger.info("Starting physical report write (rows=%d).", len(report.physical_bounds))
	payload = asdict(report)
	json_path = output_dir / "physical_checks.json"
	markdown_path = output_dir / "physical_checks.md"
	lines = [
		"# Physical validation report",
		"",
		f"**Passed:** {report.passed}",
		"",
		"## Gate summary",
		"",
		"| Check | Rows | Failures |",
		"|---|---:|---:|",
		f"| Lapse rate | {len(report.lapse_checks)} | "
		f"{sum(not bool(row['passed']) for row in report.lapse_checks)} |",
		f"| Physical bounds | {len(report.physical_bounds)} | "
		f"{sum(not bool(row['passed']) for row in report.physical_bounds)} |",
		f"| Residual Moran | {len(report.moran_results)} | "
		f"{sum(bool(row['residual_autocorrelation']) for row in report.moran_results)} |",
		"",
		"## Failures",
		"",
		", ".join(report.failures) if report.failures else "None",
	]
	try:
		output_dir.mkdir(parents=True, exist_ok=True)
		json_path.write_text(
			json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
			encoding="utf-8",
		)
		markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	except (OSError, TypeError, ValueError) as error:
		message = f"Could not write physical validation report to {output_dir}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	logger.info(
		"Finished physical report write (rows=%d) in %.3fs.",
		len(report.physical_bounds),
		time.perf_counter() - started,
	)


def _synthetic_demo() -> None:
	"""Assert exact lapse-rate pass/fail and spatial Moran detection."""
	config = get_config().validation
	x = np.arange(30, dtype=np.float64) * config.synthetic_range_m
	xy = np.column_stack((x, np.zeros_like(x)))
	elevation = np.linspace(0.0, 3_000.0, len(x))
	base = pd.DataFrame(
		{
			"city": "dehradun",
			"cell_id": np.arange(len(x), dtype=np.int64),
			"date": pd.Timestamp("2020-06-01"),
			"dem": elevation,
			"urban_fraction": np.linspace(0.0, 1.0, len(x)),
		}
	)
	base["date"] = pd.to_datetime(base["date"]).astype("datetime64[ns]")
	exact = base.assign(y_pred=20.0 + config.expected_lapse_rate_c_per_km * elevation / 1_000.0)
	exact_rates = predicted_lapse_rate(exact, control=())
	assert np.isclose(exact_rates["slope_c_per_km"].iloc[0], config.expected_lapse_rate_c_per_km)
	assert lapse_check(exact_rates)["passed"].all()
	inversion = base.assign(y_pred=20.0)
	inversion_rates = predicted_lapse_rate(inversion, control=())
	assert not lapse_check(inversion_rates)["passed"].all()
	weights = build_weights(xy, k=config.spatial_weights_k)
	structured = pd.Series(
		np.sin(np.arange(len(x)) * np.pi / 8),
		index=pd.MultiIndex.from_arrays(
			[[pd.Timestamp("2020-01-01")] * len(x), np.arange(len(x))],
			names=["date", "cell_id"],
		),
	)
	moran = residual_morans_i(
		structured, weights, permutations=config.moran_permutations
	)
	assert tuple(exact_rates.columns) == (
		"city",
		"date",
		"n",
		"month",
		"slope_c_per_km",
		"ci_lower",
		"ci_upper",
		"p_value",
		"r_squared",
	)
	assert moran["mean_I"] > config.max_morans_i
	assert moran["fisher_p"] < config.morans_alpha
	assert moran["residual_autocorrelation"]


if __name__ == "__main__":
	_synthetic_demo()
