"""ARDL/NARDL estimation, bounds testing, and residual diagnostics."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import (
	het_breuschpagan,
	linear_reset,
)
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.ardl import ARDL, UECM

from config import get_config
from dataset_builder import CITY_MONTH_SCHEMA
from econometric_diagnostics import run_gate as run_diagnostics_gate
from exceptions import ConfigError, DataIntegrityError, SchemaError
from logger import get_logger

logger = get_logger(__name__)
ICName = Literal["aic", "bic"]


@dataclass
class ARDLSpec:
	"""ARDL model specification.

	Attributes:
		dep: Dependent variable name, normally ``DTR``.
		long_run: Regressors with lagged-level terms.
		fixed: Regressors that enter only as fixed deterministic controls.
		max_lag_dep: Maximum dependent-variable lag.
		max_lag_x: Maximum distributed lag for each long-run regressor.
		ic: Information criterion for lag selection.
		case: Pesaran bounds case (2, 3, or 4).
		dep_is_count: Explicitly identify a count-valued target.
	"""

	dep: str = "DTR"
	long_run: list[str] = field(default_factory=lambda: ["urban_fraction"])
	fixed: list[str] = field(
		default_factory=lambda: ["is_monsoon", "sin_m", "cos_m", "RH2M"]
	)
	max_lag_dep: int | None = None
	max_lag_x: int | None = None
	ic: ICName | None = None
	case: int | None = None
	dep_is_count: bool = False


@dataclass
class ARDLResult:
	"""Fitted ARDL with its selected orders and UECM representation."""

	result: Any
	spec: ARDLSpec
	dep_lag: int
	x_lags: dict[str, int]
	uecm_result: Any
	n_effective: int


@dataclass(frozen=True)
class BoundsResult:
	"""Bounds-test statistics and critical-value comparisons."""

	f_statistic: float
	t_statistic: float
	t_pvalue: float
	pesaran_critical_values: dict[str, dict[str, float]]
	narayan_critical_values: dict[str, dict[str, float] | None]
	small_sample_simulated_critical_values: dict[str, dict[str, float]]
	verdict: Literal["cointegrated", "inconclusive", "no_cointegration"]
	long_run_available: bool


@dataclass(frozen=True)
class DiagnosticsBundle:
	"""Residual and parameter-stability diagnostic outcomes."""

	results: dict[str, Any]
	passed: dict[str, bool]


@dataclass
class NARDLResult:
	"""Fitted asymmetric ARDL and dynamic multiplier outputs."""

	result: ARDLResult
	asymmetry_tests: dict[str, dict[str, float]]
	dynamic_multipliers: pd.DataFrame


def _effective_spec(spec: ARDLSpec) -> ARDLSpec:
	"""Return a copy resolved against project defaults and validate settings."""
	config = get_config().econometrics
	resolved = ARDLSpec(
		dep=spec.dep,
		long_run=list(spec.long_run),
		fixed=list(spec.fixed),
		max_lag_dep=spec.max_lag_dep or config.max_lag_dep,
		max_lag_x=spec.max_lag_x or config.max_lag_x,
		ic=spec.ic or config.ic,  # type: ignore[arg-type]
		case=spec.case or config.bounds_case,
		dep_is_count=spec.dep_is_count,
	)
	if resolved.dep in resolved.long_run or resolved.dep in resolved.fixed:
		message = "The dependent variable cannot also be a regressor."
		logger.error(message)
		raise ConfigError(message)
	if not resolved.long_run:
		message = "ARDL requires at least one long-run regressor."
		logger.error(message)
		raise SchemaError(message)
	if len(set(resolved.long_run)) != len(resolved.long_run):
		message = "long_run regressors must be unique."
		logger.error(message)
		raise SchemaError(message)
	if len(set(resolved.fixed)) != len(resolved.fixed):
		message = "fixed regressors must be unique."
		logger.error(message)
		raise SchemaError(message)
	if resolved.ic not in {"aic", "bic"}:
		message = "Information criterion must be 'aic' or 'bic'."
		logger.error(message)
		raise ConfigError(message)
	if resolved.case not in {2, 3, 4}:
		message = "ARDL bounds case must be 2, 3, or 4."
		logger.error(message)
		raise ConfigError(message)
	if resolved.max_lag_dep is None or resolved.max_lag_dep < 1:
		message = "max_lag_dep must be at least one."
		logger.error(message)
		raise ConfigError(message)
	if resolved.max_lag_x is None or resolved.max_lag_x < 1:
		message = "max_lag_x must be at least one."
		logger.error(message)
		raise ConfigError(message)
	if resolved.dep_is_count or resolved.dep in config.count_targets:
		logger.warning(
			"Dependent variable %s is count-valued. Gaussian ARDL assumptions may "
			"be inappropriate; compare with log-link or Poisson models.",
			resolved.dep,
		)
	return resolved


def _validate_model_frame(df: pd.DataFrame, spec: ARDLSpec) -> pd.DataFrame:
	"""Select complete numeric data in chronological order."""
	required = list(dict.fromkeys([spec.dep, *spec.long_run, *spec.fixed]))
	missing = [column for column in required if column not in df.columns]
	if "date" not in df.columns:
		missing.append("date")
	if missing:
		message = f"ARDL data is missing columns: {missing}."
		logger.error(message)
		raise SchemaError(message)
	frame = df.loc[:, required].copy()
	if not pd.api.types.is_datetime64_ns_dtype(df["date"].dtype):
		message = "ARDL date must have dtype datetime64[ns]."
		logger.error(message)
		raise SchemaError(message)
	frame.index = pd.DatetimeIndex(df["date"])
	if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
		message = "ARDL dates must be unique and chronologically sorted."
		logger.error(message)
		raise SchemaError(message)
	if not frame.index.is_month_start.all():
		message = "ARDL dates must use canonical month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	month_number = frame.index.year * 12 + frame.index.month
	if len(month_number) > 1 and not np.all(np.diff(month_number) == 1):
		message = "ARDL dates must be consecutive monthly observations."
		logger.error(message)
		raise DataIntegrityError(message)
	for column in required:
		if not pd.api.types.is_numeric_dtype(frame[column]):
			message = f"ARDL variable {column!r} must be numeric."
			logger.error(message)
			raise SchemaError(message)
	frame = frame.astype(np.float64)
	if not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
		message = "ARDL inputs contain NaN or infinite values; model fitting will not impute."
		logger.error(message)
		raise SchemaError(message)
	if len(frame) < get_config().econometrics.min_obs:
		message = (
			f"ARDL effective sample has {len(frame)} rows; at least "
			f"{get_config().econometrics.min_obs} are required."
		)
		logger.error(message)
		raise DataIntegrityError(message)
	return frame


def partial_sums(
	x: pd.Series,
	threshold: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
	"""Decompose a level series into positive and negative partial sums.

	The first observation is used as the common initial level; subsequent
	positive and negative changes are accumulated separately.

	Args:
		x: Numeric series ordered in time.
		threshold: Change threshold separating positive and negative movements.

	Returns:
		Positive and negative partial-sum level series.

	Raises:
		SchemaError: If x is non-numeric, non-finite, or too short.
	"""
	if not isinstance(x, pd.Series) or not pd.api.types.is_numeric_dtype(x.dtype):
		message = "partial_sums requires a numeric pandas Series."
		logger.error(message)
		raise SchemaError(message)
	values = x.to_numpy(dtype=np.float64, na_value=np.nan)
	if len(values) < 2 or not np.isfinite(values).all():
		message = "partial_sums requires at least two finite observations."
		logger.error(message)
		raise SchemaError(message)
	differences = np.diff(values, prepend=values[0])
	positive = np.maximum(differences - threshold, 0.0)
	negative = np.minimum(differences - threshold, 0.0)
	positive[0] = 0.0
	negative[0] = 0.0
	x_positive = pd.Series(
		np.cumsum(positive), index=x.index, name=f"{x.name or 'x'}_pos"
	)
	x_negative = pd.Series(
		np.cumsum(negative), index=x.index, name=f"{x.name or 'x'}_neg"
	)
	return x_positive, x_negative


def _parameter_count(p: int, q: dict[str, int], fixed_count: int) -> int:
	"""Return the ARDL coefficient count including intercept."""
	return 1 + p + sum(lag + 1 for lag in q.values()) + fixed_count


def select_lags(df: pd.DataFrame, spec: ARDLSpec) -> tuple[int, dict[str, int]]:
	"""Select ARDL orders by constrained AIC/BIC grid search.

	Args:
		df: Monthly dataset containing dependent and explanatory variables.
		spec: ARDL specification with maximum lags and information criterion.

	Returns:
		Selected dependent order and a lag order for each long-run regressor.

	Raises:
		SchemaError: If model inputs/specification are invalid.
		DataIntegrityError: If no feasible lag combination can be estimated.
	"""
	resolved = _effective_spec(spec)
	frame = _validate_model_frame(df, resolved)
	maximum_parameters = len(frame) / 3.0
	started = time.perf_counter()
	logger.info(
		"Starting ARDL lag search (rows=%d, max_p=%d, max_q=%d).",
		len(frame),
		resolved.max_lag_dep,
		resolved.max_lag_x,
	)
	best_ic = float("inf")
	best_orders: tuple[int, dict[str, int]] | None = None
	q_values = range(1, int(resolved.max_lag_x) + 1)
	for p in range(1, int(resolved.max_lag_dep) + 1):
		for q_combo in itertools.product(q_values, repeat=len(resolved.long_run)):
			q = dict(zip(resolved.long_run, q_combo, strict=True))
			if _parameter_count(p, q, len(resolved.fixed)) > maximum_parameters:
				continue
			try:
				model = ARDL(
					frame[resolved.dep],
					lags=p,
					exog=frame[resolved.long_run],
					order=q,
					fixed=frame[resolved.fixed] if resolved.fixed else None,
					trend="c",
					missing="raise",
				)
				result = model.fit()
			except (ValueError, np.linalg.LinAlgError) as error:
				logger.warning(
					"Skipping infeasible ARDL order p=%d q=%s: %s", p, q, error
				)
				continue
			criterion = float(getattr(result, resolved.ic))
			if np.isfinite(criterion) and criterion < best_ic:
				best_ic = criterion
				best_orders = (p, q)
	if best_orders is None:
		message = "No feasible ARDL lag combination meets the sample-size constraint."
		logger.error(message)
		raise DataIntegrityError(message)
	logger.info(
		"Finished ARDL lag search (p=%d, q=%s) in %.3fs.",
		best_orders[0],
		best_orders[1],
		time.perf_counter() - started,
	)
	return best_orders


def _trend_for_case(case: int) -> str:
	"""Map Pesaran cases to statsmodels deterministic trend options."""
	return {2: "c", 3: "c", 4: "ct"}[case]


def fit_ardl(df: pd.DataFrame, spec: ARDLSpec) -> ARDLResult:
	"""Select and fit ARDL and corresponding UECM representations.

	Args:
		df: Monthly dependent and explanatory variables.
		spec: Model specification.

	Returns:
		ARDL result wrapper with selected lag orders and UECM result.

	Raises:
		SchemaError: If data/specification is invalid.
		DataIntegrityError: If effective sample is too small or fitting fails.
	"""
	started = time.perf_counter()
	resolved = _effective_spec(spec)
	frame = _validate_model_frame(df, resolved)
	p, q = select_lags(df, resolved)
	if _parameter_count(p, q, len(resolved.fixed)) > len(frame) / 3:
		message = "Selected ARDL parameter count exceeds one third of effective n."
		logger.error(message)
		raise DataIntegrityError(message)
	logger.info("Starting ARDL fit (rows=%d, p=%d, q=%s).", len(frame), p, q)
	try:
		model = ARDL(
			frame[resolved.dep],
			lags=p,
			exog=frame[resolved.long_run],
			order=q,
			fixed=frame[resolved.fixed] if resolved.fixed else None,
			trend="c" if resolved.case != 4 else "ct",
			missing="raise",
		)
		result = model.fit()
		uecm_model = UECM(
			frame[resolved.dep],
			lags=p,
			exog=frame[resolved.long_run],
			order=q,
			fixed=frame[resolved.fixed] if resolved.fixed else None,
			trend=_trend_for_case(int(resolved.case)),
			missing="raise",
		)
		uecm_result = uecm_model.fit()
	except (ValueError, np.linalg.LinAlgError) as error:
		message = (
			f"ARDL fit failed for spec dep={resolved.dep}, long_run="
			f"{resolved.long_run}, fixed={resolved.fixed}, p={p}, q={q}."
		)
		logger.error("%s %s", message, error)
		raise DataIntegrityError(message) from error
	logger.info(
		"Finished ARDL fit (n_eff=%d, p=%d, q=%s) in %.3fs.",
		len(frame),
		p,
		q,
		time.perf_counter() - started,
	)
	return ARDLResult(result, resolved, p, q, uecm_result, len(frame))


def _critical_frame_to_dict(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
	"""Convert statsmodels bounds critical-value table to a portable mapping."""
	levels = {90.0: "10%", 95.0: "5%", 99.0: "1%", 99.9: "0.1%"}
	return {
		levels.get(float(level), f"{100.0 - float(level):g}%"): {
			"lower": float(row["lower"]),
			"upper": float(row["upper"]),
		}
		for level, row in frame.iterrows()
	}


def _narayan_lookup(
	case: int,
	regressor_count: int,
	sample_size: int,
) -> dict[str, dict[str, float]] | None:
	"""Find nearest configured Narayan critical-value sample table."""
	table = get_config().econometrics.narayan_critical_values
	case_table = table.get(str(case), {})
	regressor_table = case_table.get(str(regressor_count), {})
	if not regressor_table:
		return None
	sample_sizes = sorted(int(value) for value in regressor_table)
	if sample_size < sample_sizes[0] or sample_size > sample_sizes[-1]:
		return None
	lower_n = max(value for value in sample_sizes if value <= sample_size)
	upper_n = min(value for value in sample_sizes if value >= sample_size)
	lower_table = regressor_table[str(lower_n)]
	upper_table = regressor_table[str(upper_n)]
	weight = (
		0.0
		if lower_n == upper_n
		else (sample_size - lower_n) / (upper_n - lower_n)
	)
	return {
		level: {
			bound: float(
				lower_table[f"{level}_{bound}"]
				+ weight
				* (
					upper_table[f"{level}_{bound}"]
					- lower_table[f"{level}_{bound}"]
				)
			)
			for bound in ("lower", "upper")
		}
		for level in ("10", "5", "1")
	}


def bounds_test(res: ARDLResult) -> BoundsResult:
	"""Run Pesaran bounds testing and compare configured Narayan small-sample values.

	Args:
		res: Fitted ARDL/UECM wrapper.

	Returns:
		F and lagged-level t statistics, critical tables, and cointegration verdict.

	Raises:
		DataIntegrityError: If the bounds test cannot be calculated.
	"""
	config = get_config().econometrics
	case = int(res.spec.case)
	started = time.perf_counter()
	try:
		asymptotic = res.uecm_result.bounds_test(
			case=case,
			asymptotic=True,
			rng=config.bounds_seed,
		)
		simulated = res.uecm_result.bounds_test(
			case=case,
			asymptotic=False,
			nsim=config.bounds_simulations,
			rng=config.bounds_seed,
		)
	except (ValueError, np.linalg.LinAlgError) as error:
		message = f"Bounds test failed for Pesaran case {case}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	f_stat = float(asymptotic.statistic)
	pesaran_values = _critical_frame_to_dict(asymptotic.critical_values)
	simulated_values = _critical_frame_to_dict(simulated.critical_values)
	narayan_values = _narayan_lookup(
		case,
		len(res.spec.long_run),
		res.n_effective,
	)
	if narayan_values is None:
		logger.warning(
			"No applicable Narayan (2005) critical-value table configured for "
			"case=%d, k=%d, n=%d; report includes statsmodels finite-sample "
			"simulation separately.",
			case,
			len(res.spec.long_run),
			res.n_effective,
		)
	lower_5 = pesaran_values["5%"]["lower"]
	upper_5 = pesaran_values["5%"]["upper"]
	if narayan_values is not None:
		lower_5 = min(lower_5, narayan_values["5"]["lower"])
		upper_5 = max(upper_5, narayan_values["5"]["upper"])
	t_name = f"{res.spec.dep}.L1"
	t_stat = float(res.uecm_result.tvalues.get(t_name, np.nan))
	t_pvalue = float(res.uecm_result.pvalues.get(t_name, np.nan))
	if f_stat > upper_5:
		verdict: Literal["cointegrated", "inconclusive", "no_cointegration"] = (
			"cointegrated"
		)
	elif f_stat < lower_5:
		verdict = "no_cointegration"
	else:
		verdict = "inconclusive"
	long_run_available = verdict == "cointegrated"
	if not long_run_available:
		logger.warning(
			"Bounds verdict is %s; long-run multipliers will not be reported.",
			verdict,
		)
	logger.info(
		"Finished bounds test (F=%.4f, t=%.4f, verdict=%s) in %.3fs.",
		f_stat,
		t_stat,
		verdict,
		time.perf_counter() - started,
	)
	return BoundsResult(
		f_statistic=f_stat,
		t_statistic=t_stat,
		t_pvalue=t_pvalue,
		pesaran_critical_values=pesaran_values,
		narayan_critical_values=narayan_values,
		small_sample_simulated_critical_values=simulated_values,
		verdict=verdict,
		long_run_available=long_run_available,
	)


def long_run_multipliers(res: ARDLResult) -> pd.DataFrame:
	"""Calculate delta-method long-run multipliers and confidence intervals.

	Args:
		res: Fitted ARDL wrapper with a cointegrated bounds verdict.

	Returns:
		Table of multiplier, delta-method standard error, and 95% confidence bounds.

	Raises:
		DataIntegrityError: If bounds testing does not support cointegration.
	"""
	bounds = bounds_test(res)
	if not bounds.long_run_available:
		logger.warning(
			"Refusing long-run multiplier calculation: bounds verdict=%s.",
			bounds.verdict,
		)
		message = "Long-run multipliers are unavailable without cointegration."
		logger.error(message)
		raise DataIntegrityError(message)
	params = res.result.params
	covariance = res.result.cov_params()
	dep_lag_names = [
		name for name in params.index
		if name.startswith(f"{res.spec.dep}.L")
	]
	dep_sum = float(params[dep_lag_names].sum())
	denominator = 1.0 - dep_sum
	if abs(denominator) < np.finfo(np.float64).eps:
		message = "ARDL long-run denominator is numerically zero."
		logger.error(message)
		raise DataIntegrityError(message)
	rows: list[dict[str, float | str]] = []
	for variable in res.spec.long_run:
		coefficient_names = [
			name for name in params.index
			if name.startswith(f"{variable}.L")
		]
		if not coefficient_names:
			message = f"No estimated lag coefficients for {variable!r}."
			logger.error(message)
			raise DataIntegrityError(message)
		numerator = float(params[coefficient_names].sum())
		multiplier = numerator / denominator
		gradient = pd.Series(0.0, index=params.index)
		gradient[coefficient_names] = 1.0 / denominator
		gradient[dep_lag_names] = numerator / denominator**2
		variance = float(gradient.to_numpy() @ covariance.to_numpy() @ gradient.to_numpy())
		standard_error = float(np.sqrt(max(variance, 0.0)))
		rows.append(
			{
				"variable": variable,
				"multiplier": multiplier,
				"std_error": standard_error,
				"ci_lower": multiplier - 1.96 * standard_error,
				"ci_upper": multiplier + 1.96 * standard_error,
			}
		)
	return pd.DataFrame(rows).set_index("variable")


def fit_nardl(
	df: pd.DataFrame,
	spec_with_asym: Sequence[str],
) -> NARDLResult:
	"""Fit an asymmetric ARDL by positive/negative partial-sum regressors.

	Args:
		df: Monthly data containing DTR and variables.
		spec_with_asym: Long-run variables to decompose into positive/negative sums.

	Returns:
		Fitted asymmetric model, long-/short-run Wald tests, and dynamic multipliers.

	Raises:
		SchemaError: If requested asymmetric variables are missing.
		DataIntegrityError: If model fitting or sample constraints fail.
	"""
	started = time.perf_counter()
	if not spec_with_asym:
		message = "At least one asymmetric regressor must be specified."
		logger.error(message)
		raise SchemaError(message)
	working = df.copy()
	spec = ARDLSpec()
	for variable in spec_with_asym:
		if variable not in working:
			message = f"Asymmetric regressor {variable!r} is missing."
			logger.error(message)
			raise SchemaError(message)
		positive, negative = partial_sums(working[variable])
		working[positive.name] = positive
		working[negative.name] = negative
	spec.long_run = [
		name for variable in spec_with_asym
		for name in (f"{variable}_pos", f"{variable}_neg")
	]
	estimation = fit_ardl(working, spec)
	params = estimation.result.params
	covariance = estimation.result.cov_params()
	uecm_params = estimation.uecm_result.params
	uecm_covariance = estimation.uecm_result.cov_params()
	asymmetry: dict[str, dict[str, float]] = {}
	for variable in spec_with_asym:
		positive_names = [
			name for name in params.index if name.startswith(f"{variable}_pos.L")
		]
		negative_names = [
			name for name in params.index if name.startswith(f"{variable}_neg.L")
		]
		if not positive_names or not negative_names:
			continue
		contrast = pd.Series(0.0, index=params.index)
		contrast[positive_names] = 1.0
		contrast[negative_names] = -1.0
		contrast_value = float(contrast @ params)
		variance = float(
			contrast.to_numpy() @ covariance.to_numpy() @ contrast.to_numpy()
		)
		wald_stat = contrast_value**2 / max(variance, np.finfo(np.float64).tiny)
		p_value = float(stats.chi2.sf(wald_stat, df=1))
		short_positive = {
			name.rsplit(".L", maxsplit=1)[-1]: name
			for name in uecm_params.index
			if name.startswith(f"D.{variable}_pos.L")
		}
		short_negative = {
			name.rsplit(".L", maxsplit=1)[-1]: name
			for name in uecm_params.index
			if name.startswith(f"D.{variable}_neg.L")
		}
		shared_lags = sorted(set(short_positive) & set(short_negative))
		short_stat = float("nan")
		short_p = float("nan")
		if shared_lags:
			restriction = np.zeros((len(shared_lags), len(uecm_params)))
			for row_index, lag in enumerate(shared_lags):
				restriction[row_index, uecm_params.index.get_loc(short_positive[lag])] = 1.0
				restriction[row_index, uecm_params.index.get_loc(short_negative[lag])] = -1.0
			difference = restriction @ uecm_params.to_numpy()
			difference_covariance = (
				restriction @ uecm_covariance.to_numpy() @ restriction.T
			)
			short_stat = float(
				difference
				@ np.linalg.pinv(difference_covariance)
				@ difference
			)
			short_p = float(stats.chi2.sf(short_stat, df=len(shared_lags)))
		asymmetry[variable] = {
			"long_run_wald": float(wald_stat),
			"long_run_pvalue": p_value,
			"short_run_wald": short_stat,
			"short_run_pvalue": short_p,
		}
	multipliers = _dynamic_multipliers(estimation)
	logger.info(
		"Finished NARDL fit (rows=%d, asymmetric=%d) in %.3fs.",
		len(working),
		len(spec_with_asym),
		time.perf_counter() - started,
	)
	return NARDLResult(estimation, asymmetry, multipliers)


def _dynamic_multipliers(res: ARDLResult) -> pd.DataFrame:
	"""Calculate recursive cumulative responses from estimated ARDL coefficients."""
	parameters = res.result.params
	horizon = get_config().econometrics.dynamic_horizon
	output: dict[str, np.ndarray] = {}
	dependent_lags = {
		lag: float(parameters.get(f"{res.spec.dep}.L{lag}", 0.0))
		for lag in range(1, res.dep_lag + 1)
	}
	for variable in res.spec.long_run:
		input_lags = {
			lag: float(parameters.get(f"{variable}.L{lag}", 0.0))
			for lag in range(int(res.x_lags[variable]) + 1)
		}
		impulse = np.zeros(horizon + 1, dtype=np.float64)
		response = np.zeros(horizon + 1, dtype=np.float64)
		for time_index in range(horizon + 1):
			impulse[time_index] = input_lags.get(time_index, 0.0)
			response[time_index] = impulse[time_index] + sum(
				coefficient * response[time_index - lag]
				for lag, coefficient in dependent_lags.items()
				if time_index >= lag
			)
		output[variable] = np.cumsum(response)
	return pd.DataFrame(output, index=pd.RangeIndex(horizon + 1, name="horizon"))


def monsoon_interaction_model(
	df: pd.DataFrame,
	spec: ARDLSpec,
	regime_split: bool = False,
) -> dict[str, Any]:
	"""Estimate urban-fraction/monsoon interaction and test regime equality.

	Args:
		df: Monthly table containing configured model columns.
		spec: Base ARDL specification.
		regime_split: Also estimate separate models for monsoon and non-monsoon rows.

	Returns:
		Interaction fit, Wald test, and optional regime-specific fits.
	"""
	working = df.copy()
	if "urban_fraction" not in working or "is_monsoon" not in working:
		message = "Monsoon interaction requires urban_fraction and is_monsoon."
		logger.error(message)
		raise SchemaError(message)
	interaction_name = "urban_fraction_x_monsoon"
	working[interaction_name] = (
		working["urban_fraction"] * working["is_monsoon"]
	)
	interaction_spec = ARDLSpec(
		dep=spec.dep,
		long_run=[*spec.long_run, interaction_name],
		fixed=list(spec.fixed),
		max_lag_dep=spec.max_lag_dep,
		max_lag_x=spec.max_lag_x,
		ic=spec.ic,
		case=spec.case,
		dep_is_count=spec.dep_is_count,
	)
	fitted = fit_ardl(working, interaction_spec)
	params = fitted.result.params
	urban_names = [
		name for name in params.index if name.startswith("urban_fraction.L")
	]
	interaction_names = [
		name for name in params.index if name.startswith(f"{interaction_name}.L")
	]
	contrast = pd.Series(0.0, index=params.index)
	contrast[interaction_names] = 1.0
	covariance = fitted.result.cov_params()
	value = float(contrast @ params)
	variance = float(
		contrast.to_numpy() @ covariance.to_numpy() @ contrast.to_numpy()
	)
	wald = value**2 / max(variance, np.finfo(np.float64).tiny)
	output: dict[str, Any] = {
		"fit": fitted,
		"long_run_equality_wald": float(wald),
		"long_run_equality_pvalue": float(stats.chi2.sf(wald, df=1)),
		"urban_coefficients": urban_names,
		"interaction_coefficients": interaction_names,
	}
	if regime_split:
		output["regime_fits"] = {
			"monsoon": fit_ardl(
				working.loc[working["is_monsoon"].astype(bool)], spec
			),
			"non_monsoon": fit_ardl(
				working.loc[~working["is_monsoon"].astype(bool)], spec
			),
		}
	return output


def diagnostics(res: ARDLResult) -> DiagnosticsBundle:
	"""Calculate residual serial-correlation, variance, normality and stability tests.

	Args:
		res: Fitted ARDL wrapper.

	Returns:
		Diagnostic statistics, p-values, and threshold-derived pass flags.
	"""
	started = time.perf_counter()
	fitted = res.result
	residuals = np.asarray(fitted.resid, dtype=np.float64)
	exog = np.asarray(fitted.model._x, dtype=np.float64)
	alpha = get_config().econometrics.unit_root_alpha
	results: dict[str, Any] = {}
	passed: dict[str, bool] = {}
	try:
		nlags = min(
			get_config().econometrics.residual_lags,
			max(1, len(residuals) // 5),
		)
		lagged_residuals = np.column_stack(
			[residuals[nlags - lag - 1 : -lag - 1] for lag in range(nlags)]
		)
		auxiliary_x = np.column_stack(
			[exog[nlags:], lagged_residuals]
		)
		auxiliary = OLS(
			residuals[nlags:],
			add_constant(auxiliary_x, has_constant="add"),
		).fit()
		bg_statistic = float(auxiliary.nobs * auxiliary.rsquared)
		bg_pvalue = float(stats.chi2.sf(bg_statistic, nlags))
		bg_f_test = auxiliary.f_test(
			np.column_stack(
				[
					np.zeros((nlags, auxiliary.model.exog.shape[1] - nlags)),
					np.eye(nlags),
				]
			)
		)
		bp = het_breuschpagan(residuals, exog)
		jb = stats.jarque_bera(residuals)
		reset_model = OLS(
			np.asarray(fitted.model.endog, dtype=np.float64)[-len(residuals) :],
			add_constant(exog, has_constant="add"),
		).fit()
		reset = linear_reset(reset_model, power=3, use_f=True)
	except (ValueError, np.linalg.LinAlgError) as error:
		message = "Residual diagnostic test failed."
		logger.error(message)
		raise DataIntegrityError(message) from error
	results["breusch_godfrey"] = {
		"statistic": bg_statistic,
		"pvalue": bg_pvalue,
		"f_statistic": float(np.asarray(bg_f_test.fvalue).item()),
		"f_pvalue": float(np.asarray(bg_f_test.pvalue).item()),
	}
	results["breusch_pagan"] = {"statistic": float(bp[0]), "pvalue": float(bp[1])}
	results["jarque_bera"] = {
		"statistic": float(jb.statistic),
		"pvalue": float(jb.pvalue),
	}
	results["reset"] = {
		"statistic": float(np.asarray(reset.fvalue).item()),
		"pvalue": float(np.asarray(reset.pvalue).item()),
	}
	passed["no_serial_correlation"] = bg_pvalue >= alpha
	passed["homoskedasticity"] = float(bp[1]) >= alpha
	passed["normality"] = float(jb.pvalue) >= alpha
	passed["reset"] = float(np.asarray(reset.pvalue).item()) >= alpha

	try:
		from statsmodels.stats.diagnostic import breaks_cusumolsresid, recursive_olsresiduals

		cusum = breaks_cusumolsresid(residuals, ddof=int(exog.shape[1]))
		results["cusum"] = {"statistic": float(cusum[0]), "pvalue": float(cusum[1])}
		passed["cusum_stability"] = float(cusum[1]) >= alpha
		recursive = recursive_olsresiduals(
			reset_model,
			skip=max(int(reset_model.model.exog.shape[1]) + 1, 5),
		)
		recursive_residuals = np.asarray(recursive[4], dtype=np.float64)
		cusumsq = np.cumsum(recursive_residuals**2)
		if len(cusumsq) and cusumsq[-1] > 0:
			cusumsq /= cusumsq[-1]
			max_deviation = float(np.max(np.abs(cusumsq - np.linspace(0, 1, len(cusumsq)))))
			cusumsq_p = float(stats.kstest(cusumsq, "uniform").pvalue)
		else:
			max_deviation, cusumsq_p = float("nan"), float("nan")
		results["cusum_sq"] = {
			"statistic": max_deviation,
			"pvalue": cusumsq_p,
		}
		passed["cusum_sq_stability"] = bool(cusumsq_p >= alpha)
	except (ValueError, np.linalg.LinAlgError, TypeError) as error:
		logger.warning("CUSUM/CUSUMSQ stability tests unavailable: %s", error)
		results["cusum"] = {"statistic": float("nan"), "pvalue": float("nan")}
		results["cusum_sq"] = {"statistic": float("nan"), "pvalue": float("nan")}
		passed["cusum_stability"] = False
		passed["cusum_sq_stability"] = False

	values = np.asarray(fitted.model.endog, dtype=np.float64)
	try:
		from statsmodels.tsa.stattools import zivot_andrews

		za = zivot_andrews(values, regression="c", autolag="AIC")
		break_index = int(za[4])
		results["zivot_andrews"] = {
			"statistic": float(za[0]),
			"pvalue": float(za[1]),
			"break_index": break_index,
		}
		if break_index > 0 and break_index < len(values) - 1:
			break_date: str | None = None
			if isinstance(fitted.model.data.row_labels, pd.DatetimeIndex):
				break_date = str(fitted.model.data.row_labels[break_index].date())
			residual_offset = len(values) - len(residuals)
			residual_break = max(1, min(len(residuals) - 1, break_index - residual_offset))
			pre = residuals[:residual_break]
			post = residuals[residual_break:]
			pooled_variance = (
				(len(pre) - 1) * np.var(pre, ddof=1)
				+ (len(post) - 1) * np.var(post, ddof=1)
			) / max(len(pre) + len(post) - 2, 1)
			chow = (
				(np.mean(pre) - np.mean(post)) ** 2
				/ max(pooled_variance * (1 / len(pre) + 1 / len(post)), 1e-12)
			)
			chow_p = float(stats.f.sf(chow, 1, len(residuals) - 2))
			results["chow_at_break"] = {
				"statistic": float(chow),
				"pvalue": chow_p,
				"break_date": break_date,
			}
			passed["chow_stability"] = chow_p >= alpha
		else:
			results["chow_at_break"] = {
				"statistic": float("nan"),
				"pvalue": float("nan"),
				"break_date": None,
			}
			passed["chow_stability"] = False
		passed["no_structural_break"] = float(za[1]) >= alpha
	except (ValueError, np.linalg.LinAlgError) as error:
		logger.warning("Zivot-Andrews/Chow tests unavailable: %s", error)
		results["zivot_andrews"] = {
			"statistic": float("nan"),
			"pvalue": float("nan"),
			"break_index": None,
		}
		results["chow_at_break"] = {
			"statistic": float("nan"),
			"pvalue": float("nan"),
			"break_date": None,
		}
		passed["no_structural_break"] = False
		passed["chow_stability"] = False
	logger.info(
		"Finished residual diagnostics (rows=%d) in %.3fs.",
		len(residuals),
		time.perf_counter() - started,
	)
	return DiagnosticsBundle(results, passed)


def _jsonable(value: Any) -> Any:
	"""Convert numpy/pandas values recursively to JSON-safe Python values."""
	if isinstance(value, dict):
		return {str(key): _jsonable(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_jsonable(item) for item in value]
	if isinstance(value, (np.integer,)):
		return int(value)
	if isinstance(value, (np.floating, float)):
		return float(value) if np.isfinite(value) else None
	if isinstance(value, (np.bool_,)):
		return bool(value)
	return value


def _save_results(
	city: str,
	spec: ARDLSpec,
	estimation: ARDLResult,
	bounds: BoundsResult,
	model_diagnostics: DiagnosticsBundle,
	multipliers: pd.DataFrame | None,
) -> Path:
	"""Persist model metadata and tables under configured report directory."""
	output = (
		get_config().paths.reports
		/ get_config().econometrics.output_directory
		/ city
	)
	try:
		output.mkdir(parents=True, exist_ok=True)
		payload: dict[str, Any] = {
			"city": city,
			"spec": asdict(spec),
			"dep_lag": estimation.dep_lag,
			"x_lags": estimation.x_lags,
			"n_effective": estimation.n_effective,
			"bounds": asdict(bounds),
			"diagnostics": model_diagnostics.results,
			"diagnostic_pass": model_diagnostics.passed,
			"coefficients": {
				name: float(value)
				for name, value in estimation.result.params.items()
			},
		}
		(output / "report.json").write_text(
			json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n",
			encoding="utf-8",
		)
		estimation.result.params.rename("coefficient").to_csv(
			output / "coefficients.csv"
		)
		if multipliers is not None:
			multipliers.to_csv(output / "long_run_multipliers.csv")
		pd.DataFrame(model_diagnostics.results).to_csv(
			output / "diagnostics.csv"
		)
	except (OSError, ValueError, TypeError) as error:
		message = f"Could not persist econometric results for {city}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	return output


def run(city: str) -> Path:
	"""Run the city-specific diagnostic gate, ARDL and reports.

	Args:
		city: Canonical city key.

	Returns:
		Directory containing model reports and CSV tables.

	Raises:
		SchemaError: If city/input schema is invalid.
		DataIntegrityError: If data are insufficient, I(2), or model estimation fails.
	"""
	started = time.perf_counter()
	if city not in {"dehradun", "jammu"}:
		message = f"Unsupported city {city!r}."
		logger.error(message)
		raise SchemaError(message)
	dataset_path = (
		get_config().paths.interim / get_config().dataset.city_month_filename
	)
	if not dataset_path.is_file():
		message = f"City-month dataset not found: {dataset_path}."
		logger.error(message)
		raise SchemaError(message)
	try:
		frame = pd.read_parquet(dataset_path)
	except (OSError, ValueError, ImportError) as error:
		message = f"Could not load city-month dataset from {dataset_path}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	required_schema = list(CITY_MONTH_SCHEMA)
	missing = [column for column in required_schema if column not in frame.columns]
	if missing:
		message = f"City-month dataset violates schema; missing {missing}."
		logger.error(message)
		raise SchemaError(message)
	frame = frame.loc[frame["city"].astype(str) == city].sort_values("date").reset_index(drop=True)
	if len(frame) < get_config().econometrics.min_obs:
		message = (
			f"{city} has {len(frame)} city-month rows; at least "
			f"{get_config().econometrics.min_obs} are required."
		)
		logger.error(message)
		raise DataIntegrityError(message)
	spec = ARDLSpec()
	regressors = list(dict.fromkeys([*spec.long_run, *spec.fixed]))
	logger.info(
		"Starting Block D model run (city=%s, rows=%d).",
		city,
		len(frame),
	)
	run_diagnostics_gate(frame, spec.dep, regressors)
	estimation = fit_ardl(frame, spec)
	bounds = bounds_test(estimation)
	multipliers = (
		long_run_multipliers(estimation)
		if bounds.long_run_available
		else None
	)
	model_diagnostics = diagnostics(estimation)
	output = _save_results(
		city, spec, estimation, bounds, model_diagnostics, multipliers
	)
	logger.info(
		"Finished Block D model run (city=%s, rows=%d) in %.3fs.",
		city,
		len(frame),
		time.perf_counter() - started,
	)
	return output


if __name__ == "__main__":
	"""Recover a known long-run multiplier from a synthetic cointegrated system."""
	rng = np.random.default_rng(get_config().seed)
	n_obs = 240
	x = np.cumsum(rng.normal(0.0, 0.4, n_obs))
	y = np.zeros(n_obs, dtype=np.float64)
	true_long_run = 1.75
	for index in range(1, n_obs):
		y[index] = (
			y[index - 1]
			+ 0.18 * (true_long_run * x[index - 1] - y[index - 1])
			+ rng.normal(0.0, 0.25)
		)
	dates = pd.date_range("2000-01-01", periods=n_obs, freq="MS").as_unit("ns")
	synthetic = pd.DataFrame(
		{
			"date": dates,
			"DTR": y,
			"urban_fraction": x,
			"is_monsoon": ((dates.month >= 6) & (dates.month <= 9)).astype(float),
			"sin_m": np.sin(2 * np.pi * dates.month / 12),
			"cos_m": np.cos(2 * np.pi * dates.month / 12),
			"RH2M": 50 + rng.normal(0.0, 8.0, n_obs),
		}
	)
	specification = ARDLSpec(
		long_run=["urban_fraction"],
		fixed=["is_monsoon", "sin_m", "cos_m", "RH2M"],
		max_lag_dep=2,
		max_lag_x=2,
		ic="bic",
		case=3,
	)
	fitted = fit_ardl(synthetic, specification)
	bounds = bounds_test(fitted)
	assert bounds.verdict == "cointegrated", bounds
	multiplier_table = long_run_multipliers(fitted)
	multiplier = multiplier_table.loc["urban_fraction", "multiplier"]
	assert abs(multiplier - true_long_run) < 0.5, (multiplier, true_long_run)
	assert set(synthetic.columns) == {
		"date",
		"DTR",
		"urban_fraction",
		"is_monsoon",
		"sin_m",
		"cos_m",
		"RH2M",
	}
	assert {"multiplier", "std_error", "ci_lower", "ci_upper"} <= set(
		multiplier_table.columns
	)
