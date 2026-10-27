"""Spatial microclimate tree-ensemble training and evaluation."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, mean_squared_error

from config import get_config
from dataset_builder import CELL_MONTH_SCHEMA
from exceptions import ConfigError, DataIntegrityError, SchemaError
from logger import get_logger
from temporal_features import Fold, LeakageSafePreprocessor

logger = get_logger(__name__)
_REGRESSION_TARGET = "DTR_cell"
_BINARY_TARGETS = frozenset(
	{"heat_snap_risk", "cold_snap_risk", "precip_extreme"}
)
_KEY_COLUMNS = ("city", "cell_id", "date", "x", "y", "spatial_block")


@runtime_checkable
class MicroclimateModel(Protocol):
	"""Common interface for fitted microclimate regressors and classifiers."""

	feature_names_: list[str]

	def fit(
		self,
		X: pd.DataFrame,
		y: pd.Series,
		sample_weight: pd.Series | np.ndarray | None = None,
		validation_data: tuple[pd.DataFrame, pd.Series] | None = None,
	) -> MicroclimateModel:
		"""Fit the model and return itself."""

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		"""Predict in input row order."""


class SpatialCrossValidator(Protocol):
	"""Sklearn-compatible spatial block splitter."""

	def split(
		self,
		X: pd.DataFrame,
		y: pd.Series | None = None,
		groups: pd.Series | None = None,
	) -> Iterable[tuple[np.ndarray, np.ndarray]]:
		"""Yield train/validation row positions that separate spatial blocks."""


@dataclass(frozen=True)
class TrainResult:
	"""Out-of-fold predictions and fitted fold models."""

	oof_predictions: pd.DataFrame
	models: tuple[MicroclimateModel, ...]
	thresholds: tuple[dict[str, float], ...]


class _TreeEnsemble:
	"""Shared XGBoost/LightGBM model wrapper."""

	def __init__(
		self,
		model_name: str,
		task: str,
		params: dict[str, Any] | None = None,
	) -> None:
		self.model_name = model_name
		self.task = task
		self.params = params or {}
		self.estimator: Any | None = None
		self.feature_names_: list[str] = []
		self.validation_target_: pd.Series | None = None
		self._is_fitted = False

	def fit(
		self,
		X: pd.DataFrame,
		y: pd.Series,
		sample_weight: pd.Series | np.ndarray | None = None,
		validation_data: tuple[pd.DataFrame, pd.Series] | None = None,
	) -> _TreeEnsemble:
		started = time.perf_counter()
		if X.empty or y.empty or len(X) != len(y):
			message = "Model training requires non-empty, row-aligned X and y."
			logger.error(message)
			raise SchemaError(message)
		if not X.index.equals(y.index):
			message = "Model training X and y must have identical indexes."
			logger.error(message)
			raise SchemaError(message)
		if not X.columns.is_unique or not all(
			pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes
		):
			message = "Model features must have unique, numeric columns."
			logger.error(message)
			raise SchemaError(message)
		if sample_weight is not None and len(sample_weight) != len(X):
			message = "sample_weight must contain one weight per training row."
			logger.error(message)
			raise SchemaError(message)

		config = get_config()
		params = dict(config.ml.model_params.get(self.model_name, {}))
		params.update(self.params)
		params.update({"random_state": config.seed, "n_jobs": config.ml.n_jobs})
		constraints = tuple(
			config.ml.monotonic_constraints.get(name, 0) for name in X.columns
		)
		if any(constraints):
			params["monotone_constraints"] = constraints
		if self.task == "binary":
			positive = int(y.astype("int64").sum())
			negative = len(y) - positive
			if positive == 0:
				message = "Binary training fold contains no positive samples."
				logger.error(message)
				raise DataIntegrityError(message)
			if negative == 0:
				message = "Binary training fold contains no negative samples."
				logger.error(message)
				raise DataIntegrityError(message)
			if config.ml.scale_pos_weight:
				params["scale_pos_weight"] = negative / positive
		if config.ml.early_stopping_rounds < 1:
			message = "ml.early_stopping_rounds must be positive."
			logger.error(message)
			raise ConfigError(message)

		evaluation_set: list[tuple[pd.DataFrame, pd.Series]] = []
		fit_kwargs: dict[str, Any] = {}
		if validation_data is not None:
			x_valid, y_valid = validation_data
			if x_valid.empty or len(x_valid) != len(y_valid):
				message = "Early-stopping validation data must be non-empty and aligned."
				logger.error(message)
				raise SchemaError(message)
			if list(x_valid.columns) != list(X.columns):
				message = "Validation features must match training feature order."
				logger.error(message)
				raise SchemaError(message)
			evaluation_set.append((x_valid.astype("float32"), y_valid))
			self.validation_target_ = y_valid.copy()
			params["early_stopping_rounds"] = config.ml.early_stopping_rounds

		if self.model_name == "xgb":
			try:
				from xgboost import XGBClassifier, XGBRegressor
			except ImportError as error:
				message = "xgboost is required to use the xgb microclimate model."
				logger.error(message)
				raise DataIntegrityError(message) from error
			estimator_type = XGBClassifier if self.task == "binary" else XGBRegressor
			if evaluation_set:
				fit_kwargs = {"eval_set": evaluation_set, "verbose": False}
		elif self.model_name == "lgbm":
			try:
				from lightgbm import LGBMClassifier, LGBMRegressor
			except ImportError as error:
				message = "lightgbm is required to use the lgbm microclimate model."
				logger.error(message)
				raise DataIntegrityError(message) from error
			estimator_type = LGBMClassifier if self.task == "binary" else LGBMRegressor
		else:
			message = f"Unsupported microclimate model {self.model_name!r}."
			logger.error(message)
			raise ConfigError(message)

		self.estimator = estimator_type(**params)
		if sample_weight is not None:
			fit_kwargs["sample_weight"] = np.asarray(sample_weight)
		if self.model_name == "lgbm" and evaluation_set:
			from lightgbm import early_stopping

			fit_kwargs["eval_set"] = evaluation_set
			fit_kwargs["callbacks"] = [
				early_stopping(config.ml.early_stopping_rounds, verbose=False)
			]
		self.estimator.fit(X.astype("float32"), y, **fit_kwargs)
		self.feature_names_ = list(X.columns)
		self._is_fitted = True
		logger.info(
			"Finished %s %s fit (rows=%d, features=%d) in %.3fs.",
			self.model_name,
			self.task,
			len(X),
			len(X.columns),
			time.perf_counter() - started,
		)
		return self

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		"""Predict rows in bounded chunks using float32 model inputs."""
		if not self._is_fitted or self.estimator is None:
			message = "Model must be fit before prediction."
			logger.error(message)
			raise DataIntegrityError(message)
		if list(X.columns) != self.feature_names_:
			message = "Prediction features do not match the fitted feature schema."
			logger.error(message)
			raise SchemaError(message)
		chunk_size = get_config().ml.prediction_chunk_size
		if chunk_size < 1:
			message = "ml.prediction_chunk_size must be positive."
			logger.error(message)
			raise ConfigError(message)
		predictions: list[np.ndarray] = []
		for start in range(0, len(X), chunk_size):
			chunk = X.iloc[start : start + chunk_size].astype("float32")
			values = (
				self.estimator.predict_proba(chunk)[:, 1]
				if self.task == "binary"
				else self.estimator.predict(chunk)
			)
			predictions.append(np.asarray(values))
		if not predictions:
			return np.empty(0, dtype=np.float64)
		return np.concatenate(predictions)


class XGBMicroclimate(_TreeEnsemble):
	"""XGBoost tree ensemble for regression or binary extreme-event tasks."""

	def __init__(
		self, task: str = "regression", params: dict[str, Any] | None = None
	) -> None:
		super().__init__("xgb", task, params)


class LGBMMicroclimate(_TreeEnsemble):
	"""LightGBM tree ensemble for regression or binary extreme-event tasks."""

	def __init__(
		self, task: str = "regression", params: dict[str, Any] | None = None
	) -> None:
		super().__init__("lgbm", task, params)


def _validate_model_frame(df: pd.DataFrame) -> None:
	"""Validate cell-month key and temporal requirements."""
	required = {"city", "cell_id", "date"}
	missing = sorted(required - set(df.columns))
	if missing:
		message = f"Model input is missing required cell-month columns: {missing}."
		logger.error(message)
		raise SchemaError(message)
	if df.empty:
		message = "Model input dataframe cannot be empty."
		logger.error(message)
		raise SchemaError(message)
	if not pd.api.types.is_datetime64_ns_dtype(df["date"].dtype):
		message = "Model input date must have dtype datetime64[ns]."
		logger.error(message)
		raise SchemaError(message)
	if (
		df["date"].isna().any()
		or not (df["date"].dt.day == 1).all()
		or not (df["date"] == df["date"].dt.normalize()).all()
	):
		message = "Model input dates must be non-null month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	if df["city"].isna().any() or not df["city"].isin({"dehradun", "jammu"}).all():
		message = "Model input contains missing or unsupported city values."
		logger.error(message)
		raise SchemaError(message)
	if df["cell_id"].dtype != np.dtype("int64"):
		message = "Model input cell_id must have dtype int64."
		logger.error(message)
		raise SchemaError(message)
	if df.duplicated(["city", "cell_id", "date"]).any():
		message = "Model input contains duplicate (city, cell_id, date) keys."
		logger.error(message)
		raise SchemaError(message)


def make_target(
	df: pd.DataFrame,
	kind: str,
	thresholds: dict[str, float] | None = None,
) -> tuple[pd.Series, dict[str, float]]:
	"""Create a model target and fit city thresholds on the supplied data.

	Args:
		df: Cell-month data. If thresholds are omitted, only this data is used
			to fit city-specific percentile thresholds.
		kind: Configured target key (``DTR_cell`` or an extreme-event target).
		thresholds: Optional per-city thresholds fitted on training data.

	Returns:
		A target series indexed like ``df`` and the thresholds used.

	Raises:
		SchemaError: If target data is missing or malformed.
	"""
	_validate_model_frame(df)
	config = get_config().ml
	if kind not in config.target_columns:
		message = f"Unsupported model target {kind!r}."
		logger.error(message)
		raise SchemaError(message)
	source = config.target_columns[kind]
	if source not in df.columns:
		message = f"Target {kind!r} requires missing column {source!r}."
		logger.error(message)
		raise SchemaError(message)
	if not pd.api.types.is_numeric_dtype(df[source].dtype):
		message = f"Target source {source!r} must be numeric."
		logger.error(message)
		raise SchemaError(message)
	values = df[source].astype("float64")
	if values.notna().sum() == 0 or np.isinf(values.dropna()).any():
		message = f"Target source {source!r} has no finite observations."
		logger.error(message)
		raise SchemaError(message)
	if kind == _REGRESSION_TARGET:
		return values.astype("float64").rename(kind), {}

	percentile = config.target_percentiles.get(kind)
	if percentile is None or not 0 <= percentile <= 1:
		message = f"No valid configured percentile exists for target {kind!r}."
		logger.error(message)
		raise ConfigError(message)
	fitted_thresholds = dict(thresholds or {})
	if thresholds is None:
		for city, group in df.assign(_target_value=values).groupby("city", sort=True):
			observed = group["_target_value"].dropna()
			if observed.empty:
				message = f"City {city!r} has no observations for target {kind!r}."
				logger.error(message)
				raise SchemaError(message)
			fitted_thresholds[str(city)] = float(observed.quantile(percentile))
	missing_cities = sorted(set(df["city"].astype(str)) - set(fitted_thresholds))
	if missing_cities:
		message = f"Thresholds are missing for cities: {missing_cities}."
		logger.error(message)
		raise SchemaError(message)
	comparisons = values.copy()
	target_values = pd.Series(index=df.index, dtype="float64", name=kind)
	for city, threshold in fitted_thresholds.items():
		mask = df["city"].eq(city)
		if kind == "cold_snap_risk":
			target_values.loc[mask] = (comparisons.loc[mask] <= threshold).astype("float64")
		else:
			target_values.loc[mask] = (comparisons.loc[mask] >= threshold).astype("float64")
	target_values.loc[values.isna()] = np.nan
	return target_values.astype("Int8"), fitted_thresholds


def _prepare_features(
	df: pd.DataFrame,
	feature_columns: Sequence[str],
	reference_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
	"""Select configured model features and create optional past-only lags."""
	working = df.copy()
	if (
		"t2m_downscaled_anom" in feature_columns
		and "t2m_downscaled_anom" not in working
		and "t2m_downscaled" in working
	):
		reference = reference_frame if reference_frame is not None else df
		if "t2m_downscaled" not in reference:
			message = "Training climatology requires t2m_downscaled."
			logger.error(message)
			raise SchemaError(message)
		climatology_data = reference.assign(_month=reference["date"].dt.month)
		climatology = climatology_data.groupby(["city", "_month"])[
			"t2m_downscaled"
		].mean()
		months = pd.MultiIndex.from_arrays(
			[working["city"], working["date"].dt.month]
		)
		working["t2m_downscaled_anom"] = (
			working["t2m_downscaled"].to_numpy(dtype=np.float64)
			- climatology.reindex(months).to_numpy(dtype=np.float64)
		)
	lag_columns = get_config().ml.lag_columns
	if lag_columns:
		available = [column for column in lag_columns if column in working]
		missing = sorted(set(lag_columns) - set(available))
		if missing:
			message = f"Configured lag columns are missing from model input: {missing}."
			logger.error(message)
			raise SchemaError(message)
		working["_model_row_order"] = np.arange(len(working))
		working = working.sort_values(["city", "cell_id", "date"], kind="stable")
		preprocessor = LeakageSafePreprocessor()
		working = preprocessor.make_lags(
			working,
			cols=tuple(available),
			lags=get_config().ml.lag_periods,
			group_cols=("city", "cell_id"),
		)
		working = working.sort_values("_model_row_order", kind="stable")
		working = working.drop(columns="_model_row_order")
		feature_columns = (*feature_columns, *(
			f"{column}_lag_{lag}"
			for column in available
			for lag in get_config().ml.lag_periods
		))
	missing_features = [name for name in feature_columns if name not in working]
	if missing_features:
		# Configured feature columns may be optional in an upstream dataset build.
		logger.warning("Configured model features absent from input: %s.", missing_features)
	feature_names = [name for name in feature_columns if name in working]
	if not feature_names:
		message = "No configured model features are present in the input."
		logger.error(message)
		raise SchemaError(message)
	features = working.loc[:, feature_names].copy()
	for column in features.columns:
		if not pd.api.types.is_numeric_dtype(features[column].dtype):
			message = f"Model feature {column!r} must be numeric."
			logger.error(message)
			raise SchemaError(message)
	if np.isinf(features.to_numpy(dtype=np.float64)).any():
		message = "Model features cannot contain infinite values."
		logger.error(message)
		raise SchemaError(message)
	features = features.astype("float32")
	return features


def _make_model(model_name: str, task: str) -> _TreeEnsemble:
	"""Construct a configured model by its public alias."""
	if model_name == "xgb":
		return XGBMicroclimate(task)
	if model_name == "lgbm":
		return LGBMMicroclimate(task)
	message = f"Unsupported model {model_name!r}; expected 'xgb' or 'lgbm'."
	logger.error(message)
	raise ConfigError(message)


def _spatial_splits(
	spatial_cv: SpatialCrossValidator,
	X: pd.DataFrame,
	y: pd.Series,
	groups: pd.Series,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
	"""Yield splits from an injected sklearn-compatible spatial splitter."""
	try:
		splits = spatial_cv.split(X, y, groups)
	except AttributeError as error:
		message = "spatial_cv must provide a sklearn-compatible split method."
		logger.error(message)
		raise ConfigError(message) from error
	for train_idx, valid_idx in splits:
		train_positions = np.asarray(train_idx, dtype=np.int64)
		valid_positions = np.asarray(valid_idx, dtype=np.int64)
		if not len(train_positions) or not len(valid_positions):
			continue
		yield train_positions, valid_positions


def train_and_evaluate(
	df: pd.DataFrame,
	target: str,
	model_name: str,
	folds: Iterable[Fold],
	spatial_cv: SpatialCrossValidator,
) -> TrainResult:
	"""Fit each temporal fold with spatially and temporally held-out stopping data.

	Args:
		df: Complete canonical cell-month panel.
		target: Target key accepted by :func:`make_target`.
		model_name: ``xgb`` or ``lgbm``.
		folds: Outer rolling-origin folds whose test dates are strictly later.
		spatial_cv: Object exposing sklearn's ``split(X, y, groups)`` interface.

	Returns:
		OOF predictions, fitted models, and the per-fold training thresholds.

	Raises:
		SchemaError: If folds, targets, or features violate their schemas.
		DataIntegrityError: If a safe inner validation split cannot be formed.
	"""
	started = time.perf_counter()
	logger.info("Starting microclimate training (rows=%d, target=%s).", len(df), target)
	_validate_model_frame(df)
	config = get_config().ml
	target_column = get_config().ml.target_columns.get(target)
	if target_column is None or target_column not in df.columns:
		message = f"Model target {target!r} is not available in the input."
		logger.error(message)
		raise SchemaError(message)
	if "spatial_block" not in df or df["spatial_block"].isna().any():
		message = "Spatial training requires non-null spatial_block values."
		logger.error(message)
		raise SchemaError(message)

	prediction_frames: list[pd.DataFrame] = []
	models: list[MicroclimateModel] = []
	fold_thresholds: list[dict[str, float]] = []
	for fold_number, fold in enumerate(folds):
		train_positions = np.asarray(fold.train_idx, dtype=np.int64)
		test_positions = np.asarray(fold.test_idx, dtype=np.int64)
		if (
			not len(train_positions)
			or not len(test_positions)
			or train_positions.min() < 0
			or test_positions.min() < 0
			or train_positions.max() >= len(df)
			or test_positions.max() >= len(df)
			or np.intersect1d(train_positions, test_positions).size
		):
			message = f"Temporal fold {fold_number} has invalid or overlapping positions."
			logger.error(message)
			raise SchemaError(message)
		train_dates = df.iloc[train_positions]["date"]
		test_dates = df.iloc[test_positions]["date"]
		if train_dates.max() >= test_dates.min():
			message = f"Temporal fold {fold_number} does not test strictly later dates."
			logger.error(message)
			raise DataIntegrityError(message)
		outer_train = df.iloc[train_positions]
		y_train, thresholds = make_target(outer_train, target)
		y_test, _ = make_target(df.iloc[test_positions], target, thresholds or None)
		usable_train = y_train.notna().to_numpy()
		outer_positions = np.flatnonzero(usable_train)
		x_outer_full = _prepare_features(
			outer_train, config.feature_columns, reference_frame=outer_train
		)
		x_outer = x_outer_full.iloc[outer_positions]
		y_outer = y_train.iloc[outer_positions]
		groups = outer_train.iloc[outer_positions]["spatial_block"]
		split = next(_spatial_splits(spatial_cv, x_outer, y_outer, groups), None)
		if split is None:
			message = f"Spatial CV produced no usable split for temporal fold {fold_number}."
			logger.error(message)
			raise DataIntegrityError(message)
		inner_train, inner_valid = split
		unique_dates = pd.DatetimeIndex(outer_train["date"].drop_duplicates().sort_values())
		validation_dates = unique_dates[-config.validation_months :]
		temporal_validation = outer_train.iloc[outer_positions]["date"].isin(validation_dates).to_numpy()
		valid_mask = np.isin(np.arange(len(x_outer)), inner_valid) & temporal_validation
		train_mask = np.isin(np.arange(len(x_outer)), inner_train) & ~temporal_validation
		if not valid_mask.any() or not train_mask.any():
			message = (
				f"Spatial/temporal early-stopping split is empty in temporal fold "
				f"{fold_number}."
			)
			logger.error(message)
			raise DataIntegrityError(message)
		fit_positions = np.flatnonzero(train_mask)
		valid_positions = np.flatnonzero(valid_mask)
		early_stop_train = outer_train.iloc[outer_positions[fit_positions]]
		x_outer_full = _prepare_features(
			outer_train,
			config.feature_columns,
			reference_frame=early_stop_train,
		)
		x_outer = x_outer_full.iloc[outer_positions]
		x_fit, y_fit = x_outer.iloc[fit_positions], y_outer.iloc[fit_positions]
		x_valid, y_valid = x_outer.iloc[valid_positions], y_outer.iloc[valid_positions]
		model = _make_model(model_name, "binary" if target in _BINARY_TARGETS else "regression")
		if config.tuning_enabled:
			model.params = tune(outer_train, target, model_name, spatial_cv)
		model.fit(x_fit, y_fit, validation_data=(x_valid, y_valid))
		x_test = _prepare_features(
			df.iloc[test_positions],
			config.feature_columns,
			reference_frame=early_stop_train,
		)
		y_test_values = y_test.to_numpy()
		valid_test = pd.notna(y_test_values)
		predictions = model.predict(x_test.loc[valid_test])
		test_rows = df.iloc[test_positions].loc[valid_test]
		prediction_frames.append(
			pd.DataFrame(
				{
					"city": test_rows["city"].astype("string").to_numpy(),
					"cell_id": test_rows["cell_id"].astype("int64").to_numpy(),
					"date": test_rows["date"].to_numpy(dtype="datetime64[ns]"),
					"y_true": y_test_values[valid_test].astype("float64"),
					"y_pred": predictions.astype("float64"),
					"fold": fold_number,
					"spatial_block": test_rows["spatial_block"].astype("int64").to_numpy(),
				}
			)
		)
		if config.save_fold_models:
			save_model(
				model,
				config.model_directory / f"{model_name}_{target}_fold_{fold_number}.joblib",
				thresholds=thresholds,
			)
		models.append(model)
		fold_thresholds.append(thresholds)
	if not prediction_frames:
		message = "No temporal folds produced out-of-fold predictions."
		logger.error(message)
		raise SchemaError(message)
	result = TrainResult(
		oof_predictions=pd.concat(prediction_frames, ignore_index=True),
		models=tuple(models),
		thresholds=tuple(fold_thresholds),
	)
	logger.info(
		"Finished microclimate training (rows=%d, folds=%d, predictions=%d) in %.3fs.",
		len(df),
		len(models),
		len(result.oof_predictions),
		time.perf_counter() - started,
	)
	return result


def tune(
	df: pd.DataFrame,
	target: str,
	model_name: str,
	spatial_cv: SpatialCrossValidator,
) -> dict[str, Any]:
	"""Tune configured parameters using spatial-CV training-only thresholds.

	Args:
		df: Training-period dataframe only; outer temporal test rows must not be
			passed to this function.
		target: Target key accepted by :func:`make_target`.
		model_name: ``xgb`` or ``lgbm``.
		spatial_cv: Spatial splitter exposing ``split(X, y, groups)``.

	Returns:
		The best parameter mapping selected by Optuna.

	Raises:
		DataIntegrityError: If Optuna is unavailable or no valid split exists.
	"""
	try:
		import optuna
	except ImportError as error:
		message = "Optuna is optional; install it to tune microclimate models."
		logger.error(message)
		raise DataIntegrityError(message) from error
	features = _prepare_features(df, get_config().ml.feature_columns, df)
	if "spatial_block" not in df or df["spatial_block"].isna().any():
		message = "Tuning requires non-null spatial_block values."
		logger.error(message)
		raise SchemaError(message)
	groups = df["spatial_block"]
	split_labels = pd.Series(np.zeros(len(features)), index=features.index)
	splits = list(_spatial_splits(spatial_cv, features, split_labels, groups))
	if not splits:
		message = "Spatial CV produced no valid tuning splits."
		logger.error(message)
		raise DataIntegrityError(message)
	search_space = get_config().ml.tuning_search_space
	task = "binary" if target in _BINARY_TARGETS else "regression"

	def objective(trial: Any) -> float:
		params: dict[str, Any] = {}
		for name, bounds in search_space.items():
			lower, upper = bounds
			if name == "max_depth":
				params[name] = trial.suggest_int(name, int(lower), int(upper))
			else:
				params[name] = trial.suggest_float(name, lower, upper, log=name == "learning_rate")
		scores: list[float] = []
		for train_idx, valid_idx in splits:
			train_frame = df.iloc[train_idx]
			valid_frame = df.iloc[valid_idx]
			y_train, thresholds = make_target(train_frame, target)
			y_valid, _ = make_target(valid_frame, target, thresholds or None)
			train_mask = y_train.notna().to_numpy()
			valid_mask = y_valid.notna().to_numpy()
			if not train_mask.any() or not valid_mask.any():
				continue
			if task == "binary" and y_valid.iloc[valid_mask].nunique() < 2:
				logger.warning(
					"Skipping spatial tuning split with one validation class."
				)
				continue
			x_train = _prepare_features(
				train_frame, get_config().ml.feature_columns, train_frame
			)
			x_valid = _prepare_features(
				valid_frame, get_config().ml.feature_columns, train_frame
			)
			model = _make_model(model_name, task)
			model.params = params
			model.fit(
				x_train.iloc[train_mask],
				y_train.iloc[train_mask],
			)
			prediction = model.predict(x_valid.iloc[valid_mask])
			if task == "regression":
				score = float(
					np.sqrt(mean_squared_error(y_valid.iloc[valid_mask], prediction))
				)
			else:
				score = float(
					average_precision_score(y_valid.iloc[valid_mask], prediction)
				)
			scores.append(score)
		if not scores:
			raise DataIntegrityError("No tuning split had usable train and validation targets.")
		return float(np.mean(scores)) if task == "binary" else float(np.mean(scores))

	study = optuna.create_study(
		direction="maximize" if task == "binary" else "minimize",
		sampler=optuna.samplers.TPESampler(seed=get_config().seed),
	)
	study.optimize(objective, n_trials=get_config().ml.tuning_trials)
	return study.best_params


def explain(model: MicroclimateModel, X: pd.DataFrame) -> pd.DataFrame:
	"""Compute held-out permutation importance and an elevation-dominance flag.

	Args:
		model: Fitted wrapper with a held-out validation target.
		X: The matching held-out spatial-block features.

	Returns:
		Importance table ordered by descending share. The ``elevation_dominates``
		column flags a configured lapse-rate leakage diagnostic.

	Raises:
		SchemaError: If held-out labels or feature schemas are unavailable.
	"""
	if not isinstance(model, _TreeEnsemble) or model.validation_target_ is None:
		message = "Permutation importance requires a fitted model validation set."
		logger.error(message)
		raise SchemaError(message)
	if len(X) != len(model.validation_target_) or list(X.columns) != model.feature_names_:
		message = "Held-out explanation data does not match the stored validation target."
		logger.error(message)
		raise SchemaError(message)
	scoring = "neg_root_mean_squared_error" if model.task == "regression" else "average_precision"
	result = permutation_importance(
		model.estimator,
		X.astype("float32"),
		model.validation_target_,
		n_repeats=get_config().ml.importance_repeats,
		random_state=get_config().seed,
		scoring=scoring,
		n_jobs=get_config().ml.n_jobs,
	)
	importance = np.maximum(result.importances_mean, 0.0)
	total = float(importance.sum())
	shares = importance / total if total else np.zeros_like(importance)
	table = pd.DataFrame(
		{
			"feature": model.feature_names_,
			"importance_mean": result.importances_mean,
			"importance_std": result.importances_std,
			"importance_share": shares,
		}
	).sort_values("importance_share", ascending=False, ignore_index=True)
	elevation_share = float(
		table.loc[table["feature"].isin({"dem", "elevation"}), "importance_share"].sum()
	)
	table["elevation_dominates"] = (
		elevation_share > get_config().ml.max_importance_share
	)
	if table["elevation_dominates"].iloc[0]:
		logger.warning(
			"Elevation accounts for %.3f of held-out feature importance; inspect for lapse-rate leakage.",
			elevation_share,
		)
	return table


def predict_to_xarray(
	model: MicroclimateModel,
	df: pd.DataFrame,
) -> dict[str, xr.DataArray]:
	"""Predict in chunks and assemble one date-by-cell raster table per city.

	Args:
		model: Fitted microclimate model.
		df: Model features with canonical city, cell, date, and projected x/y keys.

	Returns:
		City-keyed arrays with dimensions ``date`` and ``cell_id`` and projected
		``x``/``y`` coordinates attached to each cell.

	Raises:
		SchemaError: If spatial coordinates do not define stable cell locations.
	"""
	_validate_model_frame(df)
	if not {"x", "y"}.issubset(df.columns):
		message = "Map prediction requires projected x and y coordinates."
		logger.error(message)
		raise SchemaError(message)
	output: dict[str, xr.DataArray] = {}
	for city, city_frame in df.groupby("city", sort=True):
		if city_frame.duplicated(["cell_id", "date"]).any():
			message = f"Map prediction has duplicate cell-date rows for {city!r}."
			logger.error(message)
			raise SchemaError(message)
		cell_locations = city_frame[["cell_id", "x", "y"]].drop_duplicates()
		if cell_locations["cell_id"].duplicated().any():
			message = f"Map prediction coordinates change over time for {city!r}."
			logger.error(message)
			raise SchemaError(message)
		cells = np.sort(cell_locations["cell_id"].unique())
		dates = pd.DatetimeIndex(city_frame["date"].drop_duplicates().sort_values())
		features = _prepare_features(
			city_frame, get_config().ml.feature_columns, city_frame
		)
		prediction = model.predict(features)
		predicted = pd.Series(prediction, index=city_frame.index, name="prediction")
		matrix = pd.DataFrame(
			{"date": city_frame["date"], "cell_id": city_frame["cell_id"], "prediction": predicted}
		).pivot(index="date", columns="cell_id", values="prediction")
		matrix = matrix.reindex(index=dates, columns=cells)
		location_index = cell_locations.set_index("cell_id").reindex(cells)
		output[str(city)] = xr.DataArray(
			matrix.to_numpy(dtype=np.float32),
			dims=("date", "cell_id"),
			coords={
				"date": dates.to_numpy(dtype="datetime64[ns]"),
				"cell_id": cells.astype("int64"),
				"x": ("cell_id", location_index["x"].to_numpy(dtype=np.float64)),
				"y": ("cell_id", location_index["y"].to_numpy(dtype=np.float64)),
			},
			name="prediction",
			attrs={"city": str(city), "coordinate_units": "projected metres"},
		)
	return output


def _config_hash() -> str:
	"""Return a stable hash for the effective pipeline configuration."""
	payload = json.dumps(asdict(get_config()), sort_keys=True, default=str)
	return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_model(
	model: MicroclimateModel,
	path: str | Path,
	*,
	thresholds: dict[str, float] | None = None,
) -> None:
	"""Save a fitted estimator and its schema/config provenance.

	Args:
		model: Fitted microclimate model.
		path: Destination joblib path.
		thresholds: Per-city target thresholds used during training.

	Raises:
		DataIntegrityError: If persistence fails.
	"""
	if not isinstance(model, _TreeEnsemble) or not model._is_fitted:
		message = "Only a fitted microclimate model can be saved."
		logger.error(message)
		raise DataIntegrityError(message)
	destination = Path(path)
	metadata = {
		"feature_names": model.feature_names_,
		"thresholds": thresholds or {},
		"config_hash": _config_hash(),
		"cell_month_schema": CELL_MONTH_SCHEMA,
		"model_name": model.model_name,
		"task": model.task,
	}
	try:
		destination.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump({"model": model, "metadata": metadata}, destination)
	except (OSError, ValueError, TypeError) as error:
		message = f"Could not save model to {destination}."
		logger.error(message)
		raise DataIntegrityError(message) from error


def load_model(
	path: str | Path,
) -> tuple[MicroclimateModel, dict[str, Any]]:
	"""Load a saved model and validate its recorded feature schema.

	Args:
		path: Source joblib path.

	Returns:
		The fitted model and provenance metadata.

	Raises:
		DataIntegrityError: If loading fails or metadata is malformed.
	"""
	try:
		artifact = joblib.load(path)
	except (OSError, ValueError, TypeError, EOFError) as error:
		message = f"Could not load model artifact from {path}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	if (
		not isinstance(artifact, dict)
		or not isinstance(artifact.get("metadata"), dict)
		or not isinstance(artifact.get("model"), _TreeEnsemble)
	):
		message = f"Model artifact at {path} has an invalid structure."
		logger.error(message)
		raise DataIntegrityError(message)
	metadata = artifact["metadata"]
	model = artifact["model"]
	if metadata.get("feature_names") != model.feature_names_:
		message = f"Model artifact at {path} has inconsistent feature metadata."
		logger.error(message)
		raise DataIntegrityError(message)
	if metadata.get("config_hash") != _config_hash():
		logger.warning("Loaded model config hash differs from the current configuration.")
	return model, metadata


def _synthetic_demo() -> None:
	"""Verify the model improves on a lapse-only baseline over held-out cells."""
	from sklearn.linear_model import LinearRegression

	config = get_config()
	rng = np.random.default_rng(config.seed)
	cell_count = max(config.validation.min_folds * 4, 12)
	month_count = max(config.ml.validation_months + 5, 12)
	dates = pd.date_range("2000-01-01", periods=month_count, freq="MS").as_unit("ns")
	rows: list[dict[str, Any]] = []
	for cell in range(cell_count):
		dem = float(cell * 100)
		urban = float((cell % 4) / 3)
		for date in dates:
			rows.append(
				{
					"city": "dehradun",
					"cell_id": cell,
					"date": date,
					"spatial_block": cell,
					"dem": dem,
					"urban_fraction": urban,
					"slope": float(rng.normal(0.0, 0.1)),
					"aspect_sin": 0.0,
					"aspect_cos": 1.0,
					"tpi_small": 0.0,
					"tpi_large": 0.0,
					"twi": 0.0,
					"solar_rad": 1.0,
					"ndvi": urban,
					"ndbi": urban,
					"bsi": 0.0,
					"t2m_downscaled": 20.0 - 0.0065 * dem + 2.0 * urban,
					"RH2M": 50.0,
					"is_monsoon": int(date.month in {6, 7, 8, 9}),
					"sin_m": np.sin(2 * np.pi * date.month / 12),
					"cos_m": np.cos(2 * np.pi * date.month / 12),
					"DTR_cell": 12.0 - 0.001 * dem + 2.0 * urban + rng.normal(0.0, 0.1),
				}
			)
	frame = pd.DataFrame(rows)
	split_date = dates[-config.ml.validation_months]
	train_rows = frame["date"] < split_date
	test_rows = ~train_rows
	X_train = frame.loc[train_rows, ["dem", "urban_fraction"]].astype("float32")
	y_train = frame.loc[train_rows, "DTR_cell"]
	X_test = frame.loc[test_rows, ["dem", "urban_fraction"]].astype("float32")
	y_test = frame.loc[test_rows, "DTR_cell"]
	model = XGBMicroclimate()
	model.fit(X_train, y_train)
	model_error = float(np.sqrt(mean_squared_error(y_test, model.predict(X_test))))
	baseline = LinearRegression().fit(X_train[["dem"]], y_train)
	baseline_error = float(
		np.sqrt(mean_squared_error(y_test, baseline.predict(X_test[["dem"]])))
	)
	assert model_error < baseline_error, (
		f"Tree ensemble error {model_error:.4f} did not beat lapse-only "
		f"baseline {baseline_error:.4f}."
	)
	predictions = model.predict(X_test)
	assert model.feature_names_ == list(X_train.columns)
	assert predictions.shape == (len(X_test),)
	assert np.isfinite(predictions).all()


if __name__ == "__main__":
	_synthetic_demo()
