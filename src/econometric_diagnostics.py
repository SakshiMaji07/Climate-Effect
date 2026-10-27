"""Collinearity and integration-order gate for ARDL/NARDL inputs."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews

from config import get_config
from exceptions import ConfigError, DataIntegrityError, SchemaError
from logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DiagnosticsReport:
	"""Collinearity, correlation, and stationarity diagnostics.

	Attributes:
		vif: Variance inflation factor indexed by regressor.
		condition_index: Belsley condition-index and variance-proportion table.
		corr_matrix: Pearson correlation matrix, including target for information.
		integration_orders: Estimated differencing order per target/regressor.
		flagged: Human-readable warnings or failed diagnostic names.
		passed: Whether configured VIF and I(2) gates passed.
	"""

	vif: pd.Series
	condition_index: pd.DataFrame
	corr_matrix: pd.DataFrame
	integration_orders: dict[str, int]
	flagged: list[str]
	passed: bool


def _numeric_matrix(X: pd.DataFrame, *, add_const: bool = False) -> pd.DataFrame:
	"""Validate and return a complete finite numeric design matrix."""
	if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
		message = "Regressors must be a non-empty DataFrame with unique columns."
		logger.error(message)
		raise SchemaError(message)
	if "DTR" in X.columns:
		message = "DTR is the target and must not be included in regressors."
		logger.error(message)
		raise ConfigError(message)
	non_numeric = [name for name in X if not pd.api.types.is_numeric_dtype(X[name])]
	if non_numeric:
		message = f"Regressors must be numeric: {non_numeric}."
		logger.error(message)
		raise SchemaError(message)
	values = X.astype(np.float64)
	if not np.isfinite(values.to_numpy()).all():
		message = "Regressor matrix contains NaN or infinite values."
		logger.error(message)
		raise SchemaError(message)
	if (values.nunique(dropna=False) <= 1).any():
		constant = values.columns[values.nunique(dropna=False) <= 1].tolist()
		message = f"Constant regressors cannot be diagnosed: {constant}."
		logger.error(message)
		raise SchemaError(message)
	if add_const:
		from statsmodels.tools.tools import add_constant

		return add_constant(values, has_constant="add")
	return values


def compute_vif(X: pd.DataFrame, add_const: bool = True) -> pd.Series:
	"""Compute VIF for regressors only, never for a target column.

	Args:
		X: Regressor dataframe. Do not include target DTR.
		add_const: Include a constant in the auxiliary regression design.

	Returns:
		VIF values indexed by original regressor names.

	Raises:
		ConfigError: If ``DTR`` is passed as a regressor.
		SchemaError: If matrix columns are invalid or degenerate.
	"""
	started = time.perf_counter()
	logger.info("Starting VIF computation (rows=%d, regressors=%d).", len(X), len(X.columns))
	matrix = _numeric_matrix(X, add_const=add_const)
	values: dict[str, float] = {}
	for column in X.columns:
		column_position = matrix.columns.get_loc(column)
		try:
			values[column] = float(
				variance_inflation_factor(matrix.to_numpy(), column_position)
			)
		except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as error:
			message = f"Could not compute VIF for regressor {column!r}."
			logger.error(message)
			raise DataIntegrityError(message) from error
	result = pd.Series(values, dtype="float64", name="VIF")
	logger.info(
		"Finished VIF computation (regressors=%d) in %.3fs.",
		len(result),
		time.perf_counter() - started,
	)
	return result


def condition_index(
	X: pd.DataFrame,
	center: bool = True,
	scale: bool = True,
) -> pd.DataFrame:
	"""Compute Belsley-Kuh-Welsch condition indices and variance proportions.

	Args:
		X: Numeric regressors without the target.
		center: Subtract each column mean before SVD.
		scale: Scale columns to unit Euclidean norm after centering.

	Returns:
		A table with one row per singular component, its condition index, and
		variance-decomposition proportion per regressor. High-condition rows
		with multiple large proportions are listed in ``DataFrame.attrs``.

	Raises:
		ConfigError: If the target DTR is included.
		SchemaError: If the matrix is malformed or degenerate.
	"""
	matrix = _numeric_matrix(X)
	values = matrix.to_numpy(dtype=np.float64)
	if center:
		values = values - values.mean(axis=0, keepdims=True)
	column_norms = np.linalg.norm(values, axis=0)
	if np.any(column_norms <= np.finfo(np.float64).eps):
		message = "A regressor has zero variance after centering."
		logger.error(message)
		raise SchemaError(message)
	if scale:
		values = values / column_norms
	try:
		_, singular_values, right_vectors = np.linalg.svd(values, full_matrices=False)
	except np.linalg.LinAlgError as error:
		message = "SVD failed while computing condition indices."
		logger.error(message)
		raise DataIntegrityError(message) from error
	eigenvalues = singular_values**2
	maximum = float(eigenvalues.max())
	safe_eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).tiny)
	indices = np.sqrt(maximum / safe_eigenvalues)
	unscaled_proportions = (right_vectors.T**2) / safe_eigenvalues[np.newaxis, :]
	denominators = unscaled_proportions.sum(axis=1, keepdims=True)
	proportions = unscaled_proportions / np.maximum(
		denominators, np.finfo(np.float64).tiny
	)
	result = pd.DataFrame(
		proportions.T,
		columns=matrix.columns,
		index=pd.Index(np.arange(1, len(indices) + 1), name="component"),
	)
	result.insert(0, "condition_index", indices)
	threshold = get_config().econometrics
	flagged_components: list[int] = []
	for component, row in result.iterrows():
		if row["condition_index"] > threshold.condition_index_max:
			large_props = [
				column for column in matrix.columns
				if row[column] > threshold.variance_proportion_min
			]
			if len(large_props) >= 2:
				flagged_components.append(int(component))
	result.attrs["flagged_components"] = flagged_components
	return result


def _stationarity_tests(values: np.ndarray) -> tuple[bool, bool, float, float]:
	"""Run ADF, KPSS, and Zivot-Andrews tests on one finite vector."""
	econometrics = get_config().econometrics
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		try:
			adf_result = adfuller(values, autolag="AIC")
			kpss_result = kpss(values, regression="c", nlags="auto")
		except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as error:
			message = "Stationarity tests failed for a series."
			logger.error(message)
			raise DataIntegrityError(message) from error
		try:
			za_p = float(zivot_andrews(values, regression="c", autolag="AIC")[1])
		except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as error:
			logger.warning(
				"Zivot-Andrews test could not be estimated for this series: %s",
				error,
			)
			za_p = float("nan")
	return (
		bool(adf_result[1] < econometrics.unit_root_alpha),
		bool(kpss_result[1] > econometrics.unit_root_alpha),
		float(adf_result[1]),
		za_p,
	)


def _seasonally_adjust(values: np.ndarray, period: int) -> np.ndarray:
	"""Remove STL seasonal component when enough observations are available."""
	if len(values) < 2 * period:
		logger.warning(
			"Skipping STL seasonal adjustment: %d observations are fewer than %d.",
			len(values),
			2 * period,
		)
		return values
	from statsmodels.tsa.seasonal import STL

	try:
		decomposition = STL(values, period=period, robust=True).fit()
	except (ValueError, np.linalg.LinAlgError) as error:
		message = "STL seasonal adjustment failed."
		logger.error(message)
		raise DataIntegrityError(message) from error
	adjusted = values - decomposition.seasonal
	if np.var(adjusted) <= np.finfo(np.float64).eps * max(np.var(values), 1.0):
		logger.warning(
			"STL adjustment removed nearly all variance; retaining original "
			"deterministic seasonal series for unit-root tests."
		)
		return values
	return adjusted


def integration_order(
	s: pd.Series,
	max_d: int = 2,
	seasonal_adjust: bool = True,
) -> int:
	"""Estimate integration order using agreeing ADF and KPSS tests.

	ADF must reject the unit-root null and KPSS must fail to reject stationarity
	for a series to be considered stationary. When both tests disagree, the
	ambiguity is logged and tested differences are considered. Structural-break
	ADF p-values from Zivot-Andrews are also calculated and logged as a
	robustness diagnostic; they do not override the required ADF/KPSS agreement.

	Args:
		s: Numeric, finite series, preferably in chronological order.
		max_d: Maximum differencing order to test.
		seasonal_adjust: Whether to remove a fitted STL seasonal component.

	Returns:
		The smallest differencing order for which ADF and KPSS agree on stationarity.

	Raises:
		SchemaError: If input values or max_d are invalid.
		DataIntegrityError: If no order is established or the series is I(2).
	"""
	started = time.perf_counter()
	logger.info(
		"Starting integration-order tests (series=%s, n=%d, max_d=%d).",
		s.name,
		len(s),
		max_d,
	)
	if not isinstance(s, pd.Series) or not pd.api.types.is_numeric_dtype(s.dtype):
		message = "integration_order requires a numeric pandas Series."
		logger.error(message)
		raise SchemaError(message)
	if max_d < 0:
		message = "max_d must be nonnegative."
		logger.error(message)
		raise SchemaError(message)
	values = s.to_numpy(dtype=np.float64, na_value=np.nan)
	values = values[np.isfinite(values)]
	minimum_obs = get_config().econometrics.min_stationarity_obs
	if len(values) < minimum_obs:
		message = (
			f"Stationarity series has {len(values)} finite observations; "
			f"at least {minimum_obs} are required."
		)
		logger.error(message)
		raise SchemaError(message)
	if np.allclose(values, values[0]):
		message = "Stationarity series is constant and has no estimable integration order."
		logger.error(message)
		raise SchemaError(message)
	econometrics = get_config().econometrics
	if seasonal_adjust:
		values = _seasonally_adjust(values, econometrics.seasonal_period)
	for order in range(max_d + 1):
		if len(values) < minimum_obs:
			break
		adf_stationary, kpss_stationary, adf_p, za_p = _stationarity_tests(values)
		logger.info(
			"Stationarity tests d=%d: ADF p=%.4g, Zivot-Andrews p=%.4g, "
			"KPSS stationary=%s.",
			order,
			adf_p,
			za_p,
			kpss_stationary,
		)
		if adf_stationary and kpss_stationary:
			if order >= 2:
				message = "Series is I(2) or higher; ARDL bounds testing is invalid."
				logger.error(message)
				raise DataIntegrityError(message)
			logger.info(
				"Estimated integration order d=%d (n=%d) in %.3fs.",
				order,
				len(s),
				time.perf_counter() - started,
			)
			return order
		if adf_stationary != kpss_stationary:
			logger.warning(
				"ADF and KPSS disagree at differencing order %d (ADF p=%.4g, "
				"KPSS stationary=%s).",
				order,
				adf_p,
				kpss_stationary,
			)
		if order < max_d:
			values = np.diff(values)
	message = (
		f"Integration order is ambiguous through maximum differencing order {max_d}."
	)
	logger.error(message)
	raise DataIntegrityError(message)


def run_gate(
	df: pd.DataFrame,
	target: str,
	regressors: Sequence[str],
) -> DiagnosticsReport:
	"""Run collinearity and integration-order checks before ARDL/NARDL.

	The correlation matrix includes the target for information, but VIF and
	condition-index checks use regressors only. ``urban_fraction`` is retained
	in levels when trending; this gate reports its I(1) status but does not
	detrend it.

	Args:
		df: Dataframe with target and candidate regressors, ordered by time.
		target: Target series name.
		regressors: Regressor names only; may not include the target.

	Returns:
		Diagnostics report; a failed report raises if VIF or I(2) checks fail.

	Raises:
		ConfigError: If target is included in regressors.
		SchemaError: If required columns or data values are invalid.
		DataIntegrityError: If any series is I(2) or diagnostics fail.
	"""
	started = time.perf_counter()
	logger.info(
		"Starting econometric diagnostics gate (rows=%d, regressors=%d).",
		len(df),
		len(regressors),
	)
	if target in regressors:
		message = f"Target {target!r} must not be included in regressors."
		logger.error(message)
		raise ConfigError(message)
	if not regressors or len(set(regressors)) != len(regressors):
		message = "Regressors must be non-empty and unique."
		logger.error(message)
		raise SchemaError(message)
	required = [target, *regressors]
	missing = [column for column in required if column not in df.columns]
	if missing:
		message = f"Diagnostics input is missing columns: {missing}."
		logger.error(message)
		raise SchemaError(message)
	diagnostic_frame = df.loc[:, required]
	for column in required:
		if not pd.api.types.is_numeric_dtype(diagnostic_frame[column]):
			message = f"Diagnostics column {column!r} must be numeric."
			logger.error(message)
			raise SchemaError(message)
	if not np.isfinite(diagnostic_frame.to_numpy(dtype=np.float64)).all():
		message = "Diagnostics input contains NaN or infinite values."
		logger.error(message)
		raise SchemaError(message)
	vif = compute_vif(diagnostic_frame.loc[:, list(regressors)])
	ci = condition_index(diagnostic_frame.loc[:, list(regressors)])
	correlations = diagnostic_frame.corr(method="pearson")
	flagged: list[str] = []
	econometrics = get_config().econometrics
	for name, value in vif.items():
		if value > econometrics.vif_warn:
			logger.warning("VIF %.3f for %s exceeds warning threshold %.3f.", value, name, econometrics.vif_warn)
		if value > econometrics.vif_max:
			flagged.append(f"high_vif:{name}")
	if ci.attrs.get("flagged_components"):
		for component in ci.attrs["flagged_components"]:
			row = ci.loc[component]
			associated = [
				name for name in regressors
				if row[name] > econometrics.variance_proportion_min
			]
			flagged.append(
				f"condition_index:{component} ({', '.join(associated)}); "
				"consider centering/residualising interaction terms"
			)
	orders: dict[str, int] = {}
	for column in required:
		orders[column] = integration_order(diagnostic_frame[column])
	if orders.get("urban_fraction") == 1:
		flagged.append(
			"urban_fraction_is_I1: trending urban_fraction reported in levels; "
			"do not detrend"
		)
	target_correlations = correlations[target].drop(labels=[target]).abs()
	high_correlations = target_correlations[
		target_correlations > econometrics.correlation_warn
	]
	for name, value in high_correlations.items():
		logger.warning(
			"Correlation screen: |corr(%s, %s)|=%.3f; target correlation is informational only.",
			target,
			name,
			value,
		)
	passed = not any(item.startswith("high_vif:") for item in flagged)
	report = DiagnosticsReport(
		vif=vif,
		condition_index=ci,
		corr_matrix=correlations,
		integration_orders=orders,
		flagged=flagged,
		passed=passed,
	)
	if not passed:
		drop_suggestions = [
			item.partition(":")[2]
			for item in flagged
			if item.startswith("high_vif:")
		]
		logger.error(
			"Econometric diagnostic gate failed (rows=%d); consider dropping or "
			"residualising: %s.",
			len(df),
			", ".join(drop_suggestions),
		)
		raise DataIntegrityError(
			"VIF threshold exceeded for: " + ", ".join(drop_suggestions)
		)
	logger.info(
		"Finished econometric diagnostics gate (rows=%d, passed=%s) in %.3fs.",
		len(df),
		report.passed,
		time.perf_counter() - started,
	)
	return report


def save_report(rep: DiagnosticsReport, path: Path) -> None:
	"""Save diagnostics as JSON and CSV sidecars.

	Args:
		rep: Diagnostic report from :func:`run_gate`.
		path: Output path stem or JSON filename.

	Raises:
		DataIntegrityError: If any report file cannot be written.
	"""
	started = time.perf_counter()
	logger.info("Starting diagnostics report save.")
	configured_path = Path(path)
	if not configured_path.is_absolute() and configured_path.parent == Path("."):
		configured_path = (
			get_config().paths.reports / "econometrics" / configured_path
		)
	json_path = configured_path.with_suffix(".json")
	try:
		json_path.parent.mkdir(parents=True, exist_ok=True)
		payload: dict[str, Any] = {
			"vif": {str(key): float(value) for key, value in rep.vif.items()},
			"condition_index": rep.condition_index.to_dict(orient="index"),
			"condition_index_flagged_components": rep.condition_index.attrs.get(
				"flagged_components", []
			),
			"correlation_matrix": rep.corr_matrix.to_dict(orient="index"),
			"integration_orders": rep.integration_orders,
			"flagged": rep.flagged,
			"passed": rep.passed,
		}
		json_path.write_text(
			json.dumps(payload, indent=2, allow_nan=False) + "\n",
			encoding="utf-8",
		)
		rep.vif.rename_axis("regressor").to_csv(
			json_path.with_name(f"{json_path.stem}_vif.csv")
		)
		rep.condition_index.to_csv(
			json_path.with_name(f"{json_path.stem}_condition_index.csv")
		)
		rep.corr_matrix.to_csv(
			json_path.with_name(f"{json_path.stem}_correlation.csv")
		)
	except (OSError, TypeError, ValueError) as error:
		message = f"Could not write econometric diagnostic report to {json_path}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	logger.info(
		"Finished diagnostics report save in %.3fs.",
		time.perf_counter() - started,
	)


if __name__ == "__main__":
	"""Show clean regressors passing and injected collinearity being refused."""
	from numpy.random import default_rng

	_rng = default_rng(get_config().seed)
	_count = 240
	_date = pd.date_range("2000-01-01", periods=_count, freq="MS").as_unit("ns")
	urban_fraction = 0.3 + _rng.normal(0.0, 0.03, _count)
	rh2m = 50 + _rng.normal(0.0, 10.0, _count)
	sin_m = np.sin(np.arange(_count) * 2 * np.pi / 12)
	cos_m = np.cos(np.arange(_count) * 2 * np.pi / 12)
	clean = pd.DataFrame(
		{
			"date": _date,
			"DTR": 10 + 0.4 * urban_fraction + 0.02 * rh2m + _rng.normal(0, 0.5, _count),
			"urban_fraction": urban_fraction,
			"RH2M": rh2m,
			"is_monsoon": (_date.month >= 6) & (_date.month <= 9),
			"sin_m": sin_m,
			"cos_m": cos_m,
		}
	)
	_regressors = ["urban_fraction", "RH2M", "is_monsoon", "sin_m", "cos_m"]
	_clean_report = run_gate(clean, "DTR", _regressors)
	assert _clean_report.passed
	_collinear = clean.copy()
	_collinear["urban_fraction_copy"] = _collinear["urban_fraction"] * 2.0
	try:
		run_gate(_collinear, "DTR", [*_regressors, "urban_fraction_copy"])
	except DataIntegrityError:
		pass
	else:
		raise AssertionError("Injected collinearity must fail the diagnostics gate.")
