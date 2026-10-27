"""Elevation-stratified metrics, skill scores, and OOF evaluation reports."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
	average_precision_score,
	brier_score_loss,
	mean_absolute_error,
	mean_squared_error,
	r2_score,
)

from config import get_config
from exceptions import ConfigError, DataIntegrityError, SchemaError
from logger import get_logger

logger = get_logger(__name__)
_OOF_KEYS = ("city", "cell_id", "date")
_PREDICTION_COLUMNS = ("y_true", "y_pred")
_SKILL_REFERENCES = ("lapse_only", "monthly_climatology", "persistence")


def _finite_pair(
	y: Sequence[float] | np.ndarray | pd.Series,
	yhat: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
	"""Return finite, aligned one-dimensional metric arrays."""
	try:
		actual = np.asarray(y, dtype=np.float64)
		predicted = np.asarray(yhat, dtype=np.float64)
	except (TypeError, ValueError) as error:
		message = "Metric inputs must be numeric."
		logger.error(message)
		raise SchemaError(message) from error
	if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted):
		message = "Metric inputs must be aligned one-dimensional arrays."
		logger.error(message)
		raise SchemaError(message)
	if len(actual) == 0:
		message = "Metric inputs cannot be empty."
		logger.error(message)
		raise SchemaError(message)
	if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
		message = "Metric inputs must contain only finite values."
		logger.error(message)
		raise DataIntegrityError(message)
	return actual, predicted


def _validate_oof(oof: pd.DataFrame) -> pd.DataFrame:
	"""Validate canonical OOF keys, predictions, and temporal schema."""
	required = {*_OOF_KEYS, *_PREDICTION_COLUMNS}
	missing = sorted(required - set(oof.columns))
	if missing:
		message = f"OOF predictions are missing required columns: {missing}."
		logger.error(message)
		raise SchemaError(message)
	if oof.empty:
		message = "OOF predictions cannot be empty."
		logger.error(message)
		raise SchemaError(message)
	if oof[list(_OOF_KEYS)].isna().any().any():
		message = "OOF keys cannot contain missing values."
		logger.error(message)
		raise DataIntegrityError(message)
	if not pd.api.types.is_datetime64_ns_dtype(oof["date"].dtype):
		message = "OOF date must have dtype datetime64[ns]."
		logger.error(message)
		raise SchemaError(message)
	if (
		oof["date"].isna().any()
		or not (oof["date"].dt.day == 1).all()
		or not (oof["date"] == oof["date"].dt.normalize()).all()
	):
		message = "OOF date values must be non-null month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	if not oof["city"].isin({"dehradun", "jammu"}).all():
		message = "OOF predictions contain unsupported city keys."
		logger.error(message)
		raise SchemaError(message)
	if oof["cell_id"].dtype != np.dtype("int64"):
		message = "OOF cell_id must have dtype int64."
		logger.error(message)
		raise SchemaError(message)
	try:
		predicted = oof["y_pred"].to_numpy(dtype=np.float64)
		actual = oof["y_true"].to_numpy(dtype=np.float64)
	except (TypeError, ValueError) as error:
		message = "OOF y_true and y_pred columns must be numeric."
		logger.error(message)
		raise SchemaError(message) from error
	if not np.isfinite(predicted).all() or not np.isfinite(actual).all():
		message = "OOF predictions and observations cannot contain NaNs or infinities."
		logger.error(message)
		raise DataIntegrityError(message)
	if oof.duplicated(list(_OOF_KEYS)).any():
		message = "OOF predictions contain duplicated (city, cell_id, date) keys."
		logger.error(message)
		raise DataIntegrityError(message)
	return oof.copy()


def elevation_bands(
	dem: pd.Series,
	scheme: Literal["config", "quantile"] = "config",
) -> pd.Categorical:
	"""Assign three ordered elevation classes and enforce minimum band size.

	For ``scheme="config"``, use city-specific cutoffs selected from
	``dem.attrs["city"]`` or the configured ``default`` thresholds.

	Args:
		dem: Numeric elevation series in metres.
		scheme: Configured elevation cutoffs or one-third/two-third quantiles.

	Returns:
		Ordered categorical values using configured band labels.

	Raises:
		SchemaError: If elevations, scheme, city, or configured thresholds are invalid.
		DataIntegrityError: If any resulting band has too few observed cells.
	"""
	started = time.perf_counter()
	logger.info("Starting elevation band assignment (rows=%d).", len(dem))
	if not isinstance(dem, pd.Series) or dem.empty:
		message = "dem must be a non-empty pandas Series."
		logger.error(message)
		raise SchemaError(message)
	try:
		values = pd.to_numeric(dem, errors="raise").to_numpy(dtype=np.float64)
	except (TypeError, ValueError) as error:
		message = "dem must contain numeric elevations."
		logger.error(message)
		raise SchemaError(message) from error
	if not np.isfinite(values).all():
		message = "dem elevations must be finite and non-null."
		logger.error(message)
		raise SchemaError(message)
	config = get_config().validation
	labels = config.elevation_band_labels
	if len(labels) != 3 or len(set(labels)) != 3:
		message = "validation.elevation_band_labels must contain three unique labels."
		logger.error(message)
		raise ConfigError(message)
	if scheme == "config":
		city = dem.attrs.get("city", "default")
		cutoffs = config.elevation_bands.get(str(city))
		if cutoffs is None:
			message = f"No elevation band thresholds configured for city {city!r}."
			logger.error(message)
			raise ConfigError(message)
		if (
			len(cutoffs) != 2
			or not np.isfinite(np.asarray(cutoffs, dtype=np.float64)).all()
			or cutoffs[0] >= cutoffs[1]
		):
			message = f"Configured elevation cutoffs for {city!r} must be finite and increasing."
			logger.error(message)
			raise ConfigError(message)
		edges = [-np.inf, float(cutoffs[0]), float(cutoffs[1]), np.inf]
	elif scheme == "quantile":
		cutoffs = np.quantile(values, [1 / 3, 2 / 3])
		if not np.isfinite(cutoffs).all() or cutoffs[0] >= cutoffs[1]:
			message = "Quantile elevation bands require at least three distinguishable levels."
			logger.error(message)
			raise DataIntegrityError(message)
		edges = [-np.inf, float(cutoffs[0]), float(cutoffs[1]), np.inf]
	else:
		message = f"Unsupported elevation scheme {scheme!r}."
		logger.error(message)
		raise SchemaError(message)
	bands = pd.cut(
		values,
		bins=edges,
		labels=labels,
		right=False,
		include_lowest=True,
		ordered=True,
	)
	counts = pd.Series(bands).value_counts(sort=False)
	minimum = config.min_elevation_band_cells
	if minimum < 1:
		message = "validation.min_elevation_band_cells must be positive."
		logger.error(message)
		raise ConfigError(message)
	small_bands = counts[counts < minimum]
	if not small_bands.empty:
		message = (
			"Elevation bands have fewer than "
			f"{minimum} cells: {small_bands.to_dict()}."
		)
		logger.error(message)
		raise DataIntegrityError(message)
	logger.info(
		"Finished elevation band assignment (rows=%d, bands=%d) in %.3fs.",
		len(dem),
		len(labels),
		time.perf_counter() - started,
	)
	return bands


def regression_metrics(
	y: Sequence[float] | np.ndarray | pd.Series,
	yhat: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
	"""Compute standard regression metrics and Kling-Gupta efficiency.

	Args:
		y: Observed values.
		yhat: Model predictions.

	Returns:
		Mapping with MAE, RMSE, bias (prediction minus observation), R²,
		Pearson correlation, and KGE. Undefined correlation/KGE are NaN for
		constant series.

	Raises:
		SchemaError: If inputs are not aligned numeric vectors.
		DataIntegrityError: If inputs contain non-finite values.
	"""
	actual, predicted = _finite_pair(y, yhat)
	correlation = (
		float(stats.pearsonr(actual, predicted).statistic)
		if len(actual) > 1
		and np.std(actual) > 0
		and np.std(predicted) > 0
		else float("nan")
	)
	mean_actual = float(np.mean(actual))
	std_actual = float(np.std(actual, ddof=1)) if len(actual) > 1 else float("nan")
	std_predicted = float(np.std(predicted, ddof=1)) if len(predicted) > 1 else float("nan")
	if (
		not np.isfinite(correlation)
		or mean_actual == 0
		or std_actual == 0
		or not np.isfinite(std_predicted)
	):
		kge = float("nan")
	else:
		alpha = std_predicted / std_actual
		beta = float(np.mean(predicted)) / mean_actual
		kge = float(1.0 - np.sqrt((correlation - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))
	r_squared = (
		float(r2_score(actual, predicted, force_finite=True))
		if len(actual) >= 2
		else float("nan")
	)
	return {
		"MAE": float(mean_absolute_error(actual, predicted)),
		"RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
		"bias": float(np.mean(predicted - actual)),
		"R2": r_squared,
		"pearson_r": correlation,
		"KGE": kge,
	}


def skill_score(
	y: Sequence[float] | np.ndarray | pd.Series,
	yhat: Sequence[float] | np.ndarray | pd.Series,
	y_ref: (
		Sequence[float]
		| np.ndarray
		| pd.Series
		| Mapping[str, Sequence[float] | np.ndarray | pd.Series]
	),
	metric: Literal["mse", "mae", "rmse"] = "mse",
) -> float | dict[str, float]:
	"""Compute model skill versus one or several reference predictions.

	Passing a reference vector returns its scalar score. Passing a mapping with
	``lapse_only``, ``monthly_climatology``, and ``persistence`` returns all
	three named scores. The training-only provenance of references is the
	caller's responsibility; use :func:`make_skill_references` to construct them
	from a training period.

	Args:
		y: Observed values.
		yhat: Model predictions.
		y_ref: One reference prediction vector or mapping of reference names to
			prediction vectors.
		metric: Error metric used for the skill denominator.

	Returns:
		Scalar ``1 - model_error / reference_error`` or named reference scores.

	Raises:
		SchemaError: If metric or input shapes are invalid.
		DataIntegrityError: If the reference has zero error.
	"""
	actual, predicted = _finite_pair(y, yhat)
	if metric not in {"mse", "mae", "rmse"}:
		message = f"Unsupported skill metric {metric!r}."
		logger.error(message)
		raise SchemaError(message)

	def score(reference: Sequence[float] | np.ndarray | pd.Series) -> float:
		ref_actual, reference_prediction = _finite_pair(actual, reference)
		if not np.array_equal(ref_actual, actual):
			message = "Reference observations do not align with model observations."
			logger.error(message)
			raise SchemaError(message)
		if metric == "mae":
			model_error = mean_absolute_error(actual, predicted)
			reference_error = mean_absolute_error(actual, reference_prediction)
		else:
			model_error = mean_squared_error(actual, predicted)
			reference_error = mean_squared_error(actual, reference_prediction)
			if metric == "rmse":
				model_error = float(np.sqrt(model_error))
				reference_error = float(np.sqrt(reference_error))
		if reference_error == 0:
			message = "Skill score is undefined because the reference error is zero."
			logger.error(message)
			raise DataIntegrityError(message)
		return float(1.0 - model_error / reference_error)

	if isinstance(y_ref, Mapping):
		missing = sorted(set(_SKILL_REFERENCES) - set(y_ref))
		if missing:
			message = f"Skill reference mapping lacks required references: {missing}."
			logger.error(message)
			raise SchemaError(message)
		return {name: score(y_ref[name]) for name in _SKILL_REFERENCES}
	return score(y_ref)


def make_skill_references(
	train: pd.DataFrame,
	evaluation: pd.DataFrame,
	target_column: str,
	lapse_only_column: str,
) -> dict[str, np.ndarray]:
	"""Build lapse-only, training climatology, and persistence references.

	Monthly climatology means are fit exclusively on ``train`` and mapped to
	evaluation rows by city and calendar month. Persistence uses the immediately
	preceding monthly observation for each city/cell key, taking training
	context into account.

	Args:
		train: Training-period dataframe containing target and reference columns.
		evaluation: Later period to evaluate.
		target_column: Observed response column.
		lapse_only_column: Lapse-only reference prediction column.

	Returns:
		Named numpy predictions for the three reference approaches.

	Raises:
		SchemaError: If required columns or month-start dates are invalid.
		DataIntegrityError: If references cannot cover every evaluation row.
	"""
	started = time.perf_counter()
	logger.info(
		"Starting skill reference construction (train_rows=%d, evaluation_rows=%d).",
		len(train),
		len(evaluation),
	)
	required = {"city", "cell_id", "date", target_column}
	if not required.issubset(train) or not required.issubset(evaluation):
		message = "Training/evaluation data lacks city, cell_id, date, or target columns."
		logger.error(message)
		raise SchemaError(message)
	if lapse_only_column not in evaluation:
		message = f"Evaluation data lacks lapse-only reference {lapse_only_column!r}."
		logger.error(message)
		raise SchemaError(message)
	for label, frame in (("train", train), ("evaluation", evaluation)):
		dates = frame["date"]
		if (
			not pd.api.types.is_datetime64_ns_dtype(dates.dtype)
			or dates.isna().any()
			or not (dates.dt.day == 1).all()
			or not (dates == dates.dt.normalize()).all()
		):
			message = f"{label} dates must be datetime64[ns] month-start timestamps."
			logger.error(message)
			raise SchemaError(message)
	if train.empty or evaluation.empty:
		message = "Training and evaluation frames must be non-empty."
		logger.error(message)
		raise SchemaError(message)
	if (
		not pd.api.types.is_numeric_dtype(train[target_column].dtype)
		or not pd.api.types.is_numeric_dtype(evaluation[target_column].dtype)
		or not pd.api.types.is_numeric_dtype(evaluation[lapse_only_column].dtype)
	):
		message = "Reference target and lapse-only prediction columns must be numeric."
		logger.error(message)
		raise SchemaError(message)
	if train["date"].max() >= evaluation["date"].min():
		message = "Reference climatology and persistence require a strictly later evaluation period."
		logger.error(message)
		raise DataIntegrityError(message)
	duplicate_keys = ["city", "cell_id", "date"]
	if train.duplicated(duplicate_keys).any() or evaluation.duplicated(duplicate_keys).any():
		message = "Training/evaluation references contain duplicate city-cell-date keys."
		logger.error(message)
		raise DataIntegrityError(message)
	climatology = train.assign(_month=train["date"].dt.month).groupby(
		["city", "_month"], sort=True
	)[target_column].mean()
	lookup = pd.MultiIndex.from_arrays(
		[evaluation["city"], evaluation["date"].dt.month]
	)
	climatology_prediction = climatology.reindex(lookup).to_numpy(dtype=np.float64)
	prior = pd.concat([train, evaluation], ignore_index=True).sort_values(
		["city", "cell_id", "date"], kind="stable"
	)
	prior["_persistence"] = prior.groupby(["city", "cell_id"], sort=False)[
		target_column
	].shift(1)
	prior["_previous_date"] = prior.groupby(["city", "cell_id"], sort=False)[
		"date"
	].shift(1)
	expected_previous = prior["date"] - pd.offsets.MonthBegin(1)
	prior.loc[prior["_previous_date"] != expected_previous, "_persistence"] = np.nan
	evaluation_keys = evaluation[["city", "cell_id", "date"]].merge(
		prior[["city", "cell_id", "date", "_persistence"]],
		on=["city", "cell_id", "date"],
		how="left",
		validate="one_to_one",
		sort=False,
	)
	references = {
		"lapse_only": evaluation[lapse_only_column].to_numpy(dtype=np.float64),
		"monthly_climatology": climatology_prediction,
		"persistence": evaluation_keys["_persistence"].to_numpy(dtype=np.float64),
	}
	if any(not np.isfinite(prediction).all() for prediction in references.values()):
		message = "At least one reference prediction is missing or non-finite."
		logger.error(message)
		raise DataIntegrityError(message)
	logger.info(
		"Finished skill reference construction (rows=%d) in %.3fs.",
		len(evaluation),
		time.perf_counter() - started,
	)
	return references


def classification_metrics(
	y: Sequence[int] | np.ndarray | pd.Series,
	p: Sequence[float] | np.ndarray | pd.Series,
	threshold: float,
) -> dict[str, object]:
	"""Compute probabilistic classification, reliability, and extreme-event metrics.

	Args:
		y: Binary observed labels.
		p: Predicted event probabilities.
		threshold: Probability cutoff used for POD, FAR, and CSI.

	Returns:
		PR-AUC, Brier score and skill, confusion-derived extreme metrics, and
		reliability-bin summaries.

	Raises:
		SchemaError: If labels, probabilities, or threshold are malformed.
		DataIntegrityError: If values are non-finite.
	"""
	try:
		observed = np.asarray(y, dtype=np.float64)
		probability = np.asarray(p, dtype=np.float64)
	except (TypeError, ValueError) as error:
		message = "Classification labels and probabilities must be numeric."
		logger.error(message)
		raise SchemaError(message) from error
	if (
		observed.ndim != 1
		or probability.ndim != 1
		or len(observed) != len(probability)
		or len(observed) == 0
	):
		message = "Classification inputs must be non-empty aligned vectors."
		logger.error(message)
		raise SchemaError(message)
	if not np.isfinite(observed).all() or not np.isfinite(probability).all():
		message = "Classification inputs must be finite."
		logger.error(message)
		raise DataIntegrityError(message)
	if not np.isin(observed, [0.0, 1.0]).all():
		message = "Classification labels must contain only 0 and 1."
		logger.error(message)
		raise SchemaError(message)
	if (probability < 0).any() or (probability > 1).any():
		message = "Classification probabilities must lie in [0, 1]."
		logger.error(message)
		raise SchemaError(message)
	if not np.isfinite(threshold) or not 0 <= threshold <= 1:
		message = "Classification threshold must lie in [0, 1]."
		logger.error(message)
		raise ConfigError(message)
	bins = get_config().validation.reliability_bins
	if bins < 1:
		message = "validation.reliability_bins must be positive."
		logger.error(message)
		raise ConfigError(message)
	positive_rate = float(observed.mean())
	base_brier = positive_rate * (1.0 - positive_rate)
	brier = float(brier_score_loss(observed, probability))
	predicted_class = probability >= threshold
	positive = observed == 1
	true_positive = int(np.sum(predicted_class & positive))
	false_positive = int(np.sum(predicted_class & ~positive))
	false_negative = int(np.sum(~predicted_class & positive))
	pod = true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
	far = false_positive / (true_positive + false_positive) if true_positive + false_positive else float("nan")
	csi_denominator = true_positive + false_positive + false_negative
	csi = true_positive / csi_denominator if csi_denominator else float("nan")
	bin_ids = np.minimum((probability * bins).astype(np.int64), bins - 1)
	reliability_bins: list[dict[str, float | int]] = []
	for bin_index in range(bins):
		mask = bin_ids == bin_index
		if not mask.any():
			continue
		reliability_bins.append(
			{
				"bin": bin_index,
				"count": int(mask.sum()),
				"mean_probability": float(probability[mask].mean()),
				"observed_frequency": float(observed[mask].mean()),
			}
		)
	return {
		"PR_AUC": float(average_precision_score(observed, probability))
		if positive.any()
		else float("nan"),
		"Brier": brier,
		"Brier_skill": float(1.0 - brier / base_brier) if base_brier > 0 else float("nan"),
		"POD": float(pod),
		"FAR": float(far),
		"CSI": float(csi),
		"reliability_bins": reliability_bins,
	}


def stratified_report(
	oof: pd.DataFrame,
	cells: pd.DataFrame,
	by: Sequence[str] = ("city", "elev_band", "is_monsoon", "fold"),
) -> pd.DataFrame:
	"""Report OOF metrics by city, elevation, monsoon, and fold strata.

	Args:
		oof: Canonical OOF prediction dataframe.
		cells: Cell metadata with ``city``, ``cell_id``, and ``dem``; optional
			``date`` and ``is_monsoon`` columns are used when present.
		by: Grouping columns to include in the report.

	Returns:
		One row per observed stratum, with regression or classification metrics.

	Raises:
		SchemaError: If cell metadata or grouping columns are malformed.
		DataIntegrityError: If OOF keys are duplicated or predictions missing.
	"""
	started = time.perf_counter()
	logger.info("Starting stratified OOF report (rows=%d).", len(oof))
	predictions = _validate_oof(oof)
	required_cells = {"city", "cell_id", "dem"}
	if not required_cells.issubset(cells.columns):
		message = f"Cell metadata is missing required columns: {sorted(required_cells - set(cells.columns))}."
		logger.error(message)
		raise SchemaError(message)
	if cells.empty or cells["city"].isna().any():
		message = "Cell metadata must be non-empty with valid city keys."
		logger.error(message)
		raise SchemaError(message)
	if not cells["city"].isin({"dehradun", "jammu"}).all():
		message = "Cell metadata contains unsupported city keys."
		logger.error(message)
		raise SchemaError(message)
	if cells["cell_id"].dtype != np.dtype("int64"):
		message = "Cell metadata cell_id must have dtype int64."
		logger.error(message)
		raise SchemaError(message)
	if not pd.api.types.is_numeric_dtype(cells["dem"].dtype):
		message = "Cell metadata dem elevations must be numeric."
		logger.error(message)
		raise SchemaError(message)
	join_keys = ["city", "cell_id"]
	if "date" in cells:
		cell_dates = cells["date"]
		if (
			not pd.api.types.is_datetime64_ns_dtype(cell_dates.dtype)
			or cell_dates.isna().any()
			or not (cell_dates.dt.day == 1).all()
			or not (cell_dates == cell_dates.dt.normalize()).all()
		):
			message = "Cell metadata date must be datetime64[ns] month-start timestamps."
			logger.error(message)
			raise SchemaError(message)
		join_keys.append("date")
	metadata_cols = [*join_keys, "dem"]
	if "is_monsoon" in cells:
		metadata_cols.append("is_monsoon")
	metadata = cells.loc[:, list(dict.fromkeys(metadata_cols))].copy()
	if metadata.duplicated(join_keys).any():
		message = f"Cell metadata contains duplicate {join_keys} keys."
		logger.error(message)
		raise SchemaError(message)
	if "dem" not in predictions:
		predictions = predictions.merge(
			metadata,
			on=join_keys,
			how="left",
			validate="many_to_one",
			suffixes=("", "_cell"),
		)
	elif "is_monsoon" in metadata and "is_monsoon" not in predictions:
		predictions = predictions.merge(
			metadata[join_keys + ["is_monsoon"]],
			on=join_keys,
			how="left",
			validate="many_to_one",
		)
	if predictions["dem"].isna().any():
		message = "Cell elevations are missing for one or more OOF rows."
		logger.error(message)
		raise DataIntegrityError(message)
	predictions["elev_band"] = pd.Series(index=predictions.index, dtype="object")
	for city, index in predictions.groupby("city", sort=False).groups.items():
		city_rows = predictions.loc[index, ["cell_id", "dem"]]
		cell_elevations = city_rows.drop_duplicates("cell_id")
		if cell_elevations["cell_id"].duplicated().any():
			message = f"Elevation dem is not static within cells for city {city!r}."
			logger.error(message)
			raise SchemaError(message)
		city_dem = cell_elevations.set_index("cell_id")["dem"].astype("float64")
		city_dem.attrs["city"] = str(city)
		cell_bands = elevation_bands(city_dem).astype(object)
		band_by_cell = pd.Series(cell_bands, index=city_dem.index)
		predictions.loc[index, "elev_band"] = predictions.loc[index, "cell_id"].map(
			band_by_cell
		).to_numpy()
	if "is_monsoon" not in predictions:
		predictions["is_monsoon"] = predictions["date"].dt.month.isin(
			get_config().dataset.monsoon_months
		).astype("int8")
	missing_groups = sorted(set(by) - set(predictions.columns))
	if missing_groups:
		message = f"Stratified report is missing grouping columns: {missing_groups}."
		logger.error(message)
		raise SchemaError(message)
	results: list[dict[str, object]] = []
	group_columns = list(by)
	grouped = predictions.groupby(group_columns, observed=True, dropna=False, sort=True)
	for group_key, group in grouped:
		key_values = group_key if isinstance(group_key, tuple) else (group_key,)
		row: dict[str, object] = dict(zip(group_columns, key_values, strict=True))
		y = group["y_true"].to_numpy(dtype=np.float64)
		yhat = group["y_pred"].to_numpy(dtype=np.float64)
		unique_labels = np.unique(y)
		is_binary = (
			len(unique_labels) <= 2
			and np.isin(unique_labels, [0.0, 1.0]).all()
			and group["y_pred"].between(0.0, 1.0).all()
		)
		if is_binary:
			metrics = classification_metrics(
				y, yhat, get_config().validation.classification_threshold
			)
		else:
			metrics = regression_metrics(y, yhat)
		row["n"] = len(group)
		row.update(metrics)
		results.append(row)
	if not results:
		message = "No non-empty strata were available for evaluation."
		logger.error(message)
		raise DataIntegrityError(message)
	report = pd.DataFrame(results)
	logger.info(
		"Finished stratified OOF report (rows=%d, strata=%d) in %.3fs.",
		len(predictions),
		len(report),
		time.perf_counter() - started,
	)
	return report


def bootstrap_ci(
	oof: pd.DataFrame,
	stat_fn: Callable[[pd.DataFrame], float],
	by_block: bool = True,
	n: int | None = None,
) -> pd.DataFrame:
	"""Estimate percentile confidence intervals using spatial/date block bootstrap.

	When ``by_block`` is true, spatial block IDs and unique dates are sampled
	independently with replacement. Row multiplicities use the product of the
	two sampled cluster counts, preserving the two-way block structure.

	Args:
		oof: Canonical OOF rows with ``spatial_block`` or ``cell_id`` and date.
		stat_fn: Scalar statistic function accepting one sampled dataframe.
		by_block: Use two-way spatial/date resampling instead of row resampling.
		n: Number of bootstrap replications.

	Returns:
		Single-row dataframe with estimate, percentile bounds, replication count,
		and the requested clustering mode.

	Raises:
		SchemaError: If the OOF or bootstrap settings are invalid.
		DataIntegrityError: If any bootstrap statistic is non-finite.
	"""
	started = time.perf_counter()
	replicates = get_config().validation.bootstrap_replicates if n is None else n
	logger.info(
		"Starting OOF bootstrap (rows=%d, replicates=%d).", len(oof), replicates
	)
	frame = _validate_oof(oof)
	if replicates < 1:
		message = "Bootstrap replicate count n must be positive."
		logger.error(message)
		raise ConfigError(message)
	if not callable(stat_fn):
		message = "stat_fn must be callable."
		logger.error(message)
		raise SchemaError(message)
	def evaluate_statistic(sample: pd.DataFrame) -> float:
		try:
			return float(stat_fn(sample))
		except (TypeError, ValueError, KeyError) as error:
			message = "stat_fn failed for an OOF bootstrap sample."
			logger.error(message)
			raise SchemaError(message) from error

	estimate = evaluate_statistic(frame.copy())
	if not np.isfinite(estimate):
		message = "stat_fn returned a non-finite estimate for OOF data."
		logger.error(message)
		raise DataIntegrityError(message)
	rng = np.random.default_rng(get_config().seed)
	statistics: list[float] = []
	if by_block:
		block_column = "spatial_block" if "spatial_block" in frame else "cell_id"
		if frame[block_column].isna().any():
			message = f"Bootstrap block column {block_column!r} contains missing values."
			logger.error(message)
			raise DataIntegrityError(message)
		block_codes, blocks = pd.factorize(frame[block_column], sort=True)
		date_codes, dates = pd.factorize(frame["date"], sort=True)
		if len(blocks) < 2 or len(dates) < 2:
			message = "Two-way block bootstrap requires at least two spatial blocks and dates."
			logger.error(message)
			raise DataIntegrityError(message)
		for _ in range(replicates):
			sampled_blocks = rng.integers(0, len(blocks), size=len(blocks))
			sampled_dates = rng.integers(0, len(dates), size=len(dates))
			block_counts = np.bincount(sampled_blocks, minlength=len(blocks))
			date_counts = np.bincount(sampled_dates, minlength=len(dates))
			row_weights = block_counts[block_codes] * date_counts[date_codes]
			row_positions = np.repeat(np.arange(len(frame)), row_weights)
			if len(row_positions) == 0:
				continue
			sample = frame.iloc[row_positions].reset_index(drop=True)
			value = evaluate_statistic(sample)
			if not np.isfinite(value):
				message = "stat_fn returned a non-finite block-bootstrap statistic."
				logger.error(message)
				raise DataIntegrityError(message)
			statistics.append(value)
	else:
		for _ in range(replicates):
			positions = rng.integers(0, len(frame), size=len(frame))
			value = evaluate_statistic(frame.iloc[positions].reset_index(drop=True))
			if not np.isfinite(value):
				message = "stat_fn returned a non-finite row-bootstrap statistic."
				logger.error(message)
				raise DataIntegrityError(message)
			statistics.append(value)
	if not statistics:
		message = "No valid bootstrap replications were produced."
		logger.error(message)
		raise DataIntegrityError(message)
	confidence = get_config().validation.bootstrap_confidence
	if not 0 < confidence < 1:
		message = "validation.bootstrap_confidence must lie strictly between zero and one."
		logger.error(message)
		raise ConfigError(message)
	alpha = (1.0 - confidence) / 2.0
	result = pd.DataFrame(
		[
			{
				"estimate": estimate,
				"ci_lower": float(np.quantile(statistics, alpha)),
				"ci_upper": float(np.quantile(statistics, 1.0 - alpha)),
				"n_bootstrap": len(statistics),
				"by_block": by_block,
			}
		]
	)
	logger.info(
		"Finished OOF bootstrap (rows=%d, replicates=%d) in %.3fs.",
		len(frame),
		len(statistics),
		time.perf_counter() - started,
	)
	return result


def compare_to_split_strategy(
	oof_random: pd.DataFrame,
	oof_spatial: pd.DataFrame,
) -> pd.DataFrame:
	"""Quantify random-split optimism relative to spatial OOF evaluation.

	Args:
		oof_random: OOF predictions from random/non-spatial validation.
		oof_spatial: OOF predictions from spatial block validation.

	Returns:
		One-row dataframe with both RMSE/MAE estimates and absolute/relative
		optimism. Positive optimism means the non-spatial score is better.

	Raises:
		SchemaError: If the OOF schemas are invalid.
		DataIntegrityError: If OOF keys are duplicated or predictions missing.
	"""
	started = time.perf_counter()
	logger.info(
		"Starting split-strategy comparison (random_rows=%d, spatial_rows=%d).",
		len(oof_random),
		len(oof_spatial),
	)
	random_frame = _validate_oof(oof_random)
	spatial_frame = _validate_oof(oof_spatial)
	random_metrics = regression_metrics(random_frame["y_true"], random_frame["y_pred"])
	spatial_metrics = regression_metrics(spatial_frame["y_true"], spatial_frame["y_pred"])
	absolute_optimism = spatial_metrics["RMSE"] - random_metrics["RMSE"]
	result = pd.DataFrame(
		[
			{
				"random_n": len(random_frame),
				"spatial_n": len(spatial_frame),
				"random_RMSE": random_metrics["RMSE"],
				"spatial_RMSE": spatial_metrics["RMSE"],
				"optimism_RMSE": absolute_optimism,
				"optimism_RMSE_pct": (
					100.0 * absolute_optimism / spatial_metrics["RMSE"]
					if spatial_metrics["RMSE"] != 0
					else float("nan")
				),
				"random_MAE": random_metrics["MAE"],
				"spatial_MAE": spatial_metrics["MAE"],
				"optimism_MAE": spatial_metrics["MAE"] - random_metrics["MAE"],
			}
		]
	)
	logger.info(
		"Finished split-strategy comparison (rows=%d) in %.3fs.",
		len(random_frame) + len(spatial_frame),
		time.perf_counter() - started,
	)
	return result


def write_report(df: pd.DataFrame, path: str | Path) -> None:
	"""Write evaluation report as CSV, Markdown, and optional figure-data JSON.

	Args:
		df: Evaluation report dataframe.
		path: Explicit destination filename stem or CSV path.

	Raises:
		SchemaError: If the report dataframe is empty or invalid.
		DataIntegrityError: If report files cannot be written.
	"""
	started = time.perf_counter()
	logger.info("Starting evaluation report write (rows=%d).", len(df))
	if df.empty or not df.columns.is_unique:
		message = "Evaluation report must be non-empty with unique columns."
		logger.error(message)
		raise SchemaError(message)
	destination = Path(path)
	csv_path = destination.with_suffix(".csv")
	markdown_path = destination.with_suffix(".md")

	def markdown_table(report: pd.DataFrame) -> str:
		headers = [str(column) for column in report.columns]
		rows = [
			[str(value).replace("|", "\\|") for value in row]
			for row in report.itertuples(index=False, name=None)
		]
		lines = [
			"| " + " | ".join(headers) + " |",
			"| " + " | ".join("---" for _ in headers) + " |",
		]
		lines.extend("| " + " | ".join(row) + " |" for row in rows)
		return "\n".join(lines) + "\n"

	try:
		csv_path.parent.mkdir(parents=True, exist_ok=True)
		df.to_csv(csv_path, index=False)
		markdown_path.write_text(markdown_table(df), encoding="utf-8")
		if get_config().validation.write_figure_data:
			json_path = destination.with_suffix(".json")
			payload = json.loads(df.to_json(orient="records"))
			json_path.write_text(
				json.dumps(payload, indent=2, allow_nan=False) + "\n",
				encoding="utf-8",
			)
	except (OSError, ValueError, TypeError) as error:
		message = f"Could not write evaluation report files using stem {destination}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	logger.info(
		"Finished evaluation report write (rows=%d) in %.3fs.",
		len(df),
		time.perf_counter() - started,
	)


def _synthetic_demo() -> None:
	"""Assert elevation-stratified errors match known synthetic values."""
	config = get_config().validation
	low, high = config.elevation_bands["default"]
	elevations = np.array([low - 1, low, (low + high) / 2, high, high + 1] * 4)
	errors = np.where(elevations < low, 1.0, np.where(elevations < high, 2.0, 4.0))
	frame = pd.DataFrame(
		{
			"city": "dehradun",
			"cell_id": np.arange(len(elevations), dtype=np.int64),
			"date": pd.Timestamp("2020-01-01"),
			"y_true": np.full(len(elevations), 10.0),
			"y_pred": 10.0 + errors,
			"fold": 0,
			"dem": elevations,
			"is_monsoon": 0,
		}
	)
	frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
	bands = pd.Series(elevation_bands(pd.Series(elevations, name="dem")))
	expected = {
		config.elevation_band_labels[0]: 1.0,
		config.elevation_band_labels[1]: 2.0,
		config.elevation_band_labels[2]: 4.0,
	}
	for band, expected_rmse in expected.items():
		mask = bands.astype(object) == band
		actual = regression_metrics(
			frame.loc[mask, "y_true"], frame.loc[mask, "y_pred"]
		)["RMSE"]
		assert actual == expected_rmse
	report = stratified_report(
		frame.drop(columns="dem"),
		pd.DataFrame(
			{
				"city": "dehradun",
				"cell_id": frame["cell_id"],
				"dem": elevations,
				"is_monsoon": 0,
			}
		),
		by=("city", "elev_band"),
	)
	assert set(report["RMSE"].to_numpy()) == {1.0, 2.0, 4.0}


if __name__ == "__main__":
	_synthetic_demo()
