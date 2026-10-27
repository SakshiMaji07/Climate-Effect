"""Spatially cross-validated, terrain-aware land-cover classification."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine

from config import get_config
from exceptions import DataIntegrityError, SchemaError
from logger import get_logger
from raster_processing import load_terrain_stack

logger = get_logger(__name__)
CLASSES = list(get_config().classifier.classes)
_SPECTRAL_FEATURES = ("ndvi", "ndbi", "ndwi", "mndwi", "bsi")


@dataclass
class ClassifierArtifacts:
	"""Trained classifier and its validation metadata.

	Attributes:
		model: Fitted scikit-learn or XGBoost estimator/pipeline.
		feature_names: Numeric features used for training and prediction.
		class_names: Configured class order.
		metrics: Spatial cross-validation metrics.
		trained_on: SHA-256 digest identifying the training labels.
	"""

	model: Any
	feature_names: list[str]
	class_names: list[str]
	metrics: dict[str, Any]
	trained_on: str


def _validate_spatial_data(index_stack: xr.Dataset, terrain: xr.Dataset) -> None:
	"""Validate feature input dimensions and matching spatial coordinates."""
	if not isinstance(index_stack, xr.Dataset) or not isinstance(terrain, xr.Dataset):
		message = "Spectral and terrain inputs must be xarray Datasets."
		logger.error(message)
		raise SchemaError(message)
	for dataset_name, dataset in (
		("index stack", index_stack),
		("terrain stack", terrain),
	):
		if not {"y", "x"}.issubset(dataset.dims):
			message = f"The {dataset_name} must have y and x dimensions."
			logger.error(message)
			raise SchemaError(message)
	if index_stack.sizes["x"] != terrain.sizes["x"] or index_stack.sizes["y"] != terrain.sizes["y"]:
		message = "Spectral and terrain grids have different spatial shapes."
		logger.error(message)
		raise SchemaError(message)
	for coordinate in ("x", "y"):
		if coordinate in index_stack.coords and coordinate in terrain.coords:
			if not np.array_equal(index_stack[coordinate].values, terrain[coordinate].values):
				message = f"Spectral and terrain {coordinate} coordinates do not match."
				logger.error(message)
				raise SchemaError(message)
	if "time" in index_stack.dims and "time" in terrain.dims:
		if not index_stack["time"].equals(terrain["time"]):
			message = "Spectral and terrain time coordinates do not match."
			logger.error(message)
			raise SchemaError(message)


def _terrain_feature_names(terrain: xr.Dataset) -> list[str]:
	"""Resolve terrain feature variables into canonical column names."""
	names: list[str] = []
	for source, target in (
		("elevation", "dem"),
		("dem", "dem"),
		("slope", "slope"),
		("aspect_sin", "aspect_sin"),
		("aspect_cos", "aspect_cos"),
		("sin_m", "aspect_sin"),
		("cos_m", "aspect_cos"),
		("twi", "twi"),
	):
		if source in terrain.data_vars and target not in names:
			names.append(target)
	names.extend(
		sorted(
			name for name in terrain.data_vars
			if name.startswith("tpi_") and name not in names
		)
	)
	return names


def _to_canonical_feature_name(terrain: xr.Dataset, target: str) -> str:
	"""Return a terrain variable name matching a canonical feature name."""
	if target == "dem":
		return "elevation" if "elevation" in terrain.data_vars else "dem"
	if target == "aspect_sin":
		return "aspect_sin" if "aspect_sin" in terrain.data_vars else "sin_m"
	if target == "aspect_cos":
		return "aspect_cos" if "aspect_cos" in terrain.data_vars else "cos_m"
	return target


def build_feature_matrix(
	index_stack: xr.Dataset,
	terrain: xr.Dataset,
	spatial_context: bool = True,
) -> pd.DataFrame:
	"""Combine spectral and terrain rasters into a cell-indexed feature table.

	Three-by-three mean and standard deviation texture features are computed
	from NDBI and NDVI when spatial context is enabled. Multiple timestamps
	are represented with a ``(time, cell_id)`` MultiIndex; a single timestamp
	uses ``cell_id`` as the index.

	Args:
		index_stack: Spectral index dataset containing NDVI, NDBI, and related indices.
		terrain: Terrain dataset containing elevation, slope, aspect encodings,
			TPI, and/or TWI.
		spatial_context: Whether to include local 3-by-3 mean/std texture features.

	Returns:
		A numeric feature DataFrame with row-major int64 cell IDs.

	Raises:
		SchemaError: If variables or spatial-grid shapes do not match.
	"""
	started = time.perf_counter()
	logger.info(
		"Starting classifier feature assembly (spectral=%d, terrain=%d).",
		len(index_stack.data_vars),
		len(terrain.data_vars),
	)
	terrain = load_terrain_stack(terrain)
	_validate_spatial_data(index_stack, terrain)
	missing = [name for name in _SPECTRAL_FEATURES if name not in index_stack.data_vars]
	if missing:
		message = f"Index stack is missing required features: {missing}."
		logger.error(message)
		raise SchemaError(message)
	width = int(index_stack.sizes["x"])
	features = xr.Dataset(
		{name: index_stack[name] for name in _SPECTRAL_FEATURES}
	)
	for canonical in _terrain_feature_names(terrain):
		source = _to_canonical_feature_name(terrain, canonical)
		if source in terrain.data_vars and canonical not in features.data_vars:
			features[canonical] = terrain[source]

	if spatial_context:
		for variable in ("ndbi", "ndvi"):
			data = index_stack[variable]
			rolling = data.rolling(y=3, x=3, center=True, min_periods=1)
			features[f"{variable}_mean_3x3"] = rolling.mean()
			features[f"{variable}_std_3x3"] = rolling.std(ddof=0)

	if "time" not in features.dims:
		features = features.expand_dims(time=[pd.Timestamp("1970-01-01")])
	frame = features.to_dataframe().reset_index()
	frame["cell_id"] = (
		frame["y"].map({value: index for index, value in enumerate(features["y"].values)})
		.astype(np.int64)
		* width
		+ frame["x"].map({value: index for index, value in enumerate(features["x"].values)})
		.astype(np.int64)
	)
	feature_columns = [
		column for column in frame.columns
		if column not in {"time", "y", "x", "cell_id"}
	]
	frame = frame.set_index(["time", "cell_id"])
	if frame.index.get_level_values("time").nunique() == 1:
		frame.index = frame.index.droplevel("time")
		frame.index.name = "cell_id"
	if isinstance(frame.index, pd.MultiIndex):
		frame.index = frame.index.set_levels(
			frame.index.levels[1].astype(np.int64), level=1
		)
	else:
		frame.index = frame.index.astype(np.int64)
	frame = frame[feature_columns]
	frame.attrs["feature_columns"] = feature_columns
	frame.attrs["grid_shape"] = (
		int(index_stack.sizes["y"]),
		int(index_stack.sizes["x"]),
	)
	frame.attrs["city"] = index_stack.attrs.get("city", terrain.attrs.get("city"))
	frame.attrs["crs"] = str(index_stack.rio.crs) if hasattr(index_stack, "rio") else None
	frame.attrs["transform"] = (
		tuple(index_stack.rio.transform())
		if hasattr(index_stack, "rio") and index_stack.rio.crs is not None
		else None
	)
	logger.info(
		"Finished classifier feature assembly (rows=%d, features=%d) in %.3fs.",
		len(frame),
		len(frame.columns),
		time.perf_counter() - started,
	)
	return frame


def load_training_labels(path: Path) -> Any:
	"""Load and validate polygon/point land-cover labels from a vector file.

	Args:
		path: Path to a GeoPackage, shapefile, or other geopandas-readable file.

	Returns:
		A GeoDataFrame with validated class labels and source SHA-256 metadata.

	Raises:
		SchemaError: If the file, CRS, class values, or sample counts are invalid.
	"""
	started = time.perf_counter()
	logger.info("Starting training-label load from %s.", path)
	try:
		import geopandas as gpd
	except ImportError as error:
		message = "geopandas is required to load vector training labels."
		logger.error(message)
		raise SchemaError(message) from error
	if not path.is_file():
		message = f"Training label file does not exist: {path}."
		logger.error(message)
		raise SchemaError(message)
	try:
		labels = gpd.read_file(path)
	except (OSError, ValueError) as error:
		message = f"Could not read training labels from {path}."
		logger.error(message)
		raise SchemaError(message) from error
	if "class" not in labels.columns or labels.crs is None:
		message = "Training labels require a 'class' column and a defined CRS."
		logger.error(message)
		raise SchemaError(message)
	allowed = set(get_config().classifier.classes)
	observed = set(labels["class"].dropna().astype(str))
	unknown = observed - allowed
	if labels["class"].isna().any() or unknown:
		message = f"Training labels contain missing or unsupported classes: {sorted(unknown)}."
		logger.error(message)
		raise SchemaError(message)
	counts = labels["class"].astype(str).value_counts()
	insufficient = {
		class_name: int(counts.get(class_name, 0))
		for class_name in allowed
		if counts.get(class_name, 0) < get_config().classifier.min_samples
	}
	if insufficient:
		message = f"Insufficient training samples per class: {insufficient}."
		logger.error(message)
		raise SchemaError(message)
	labels.attrs["source_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
	logger.info(
		"Finished training-label load (rows=%d, classes=%d) in %.3fs.",
		len(labels),
		len(observed),
		time.perf_counter() - started,
	)
	return labels


def _balanced_sample_weights(labels: np.ndarray, class_names: list[str]) -> np.ndarray:
	"""Compute deterministic inverse-frequency class weights."""
	counts = pd.Series(labels).value_counts()
	total = len(labels)
	class_count = len(class_names)
	return np.asarray(
		[total / (class_count * int(counts[value])) for value in labels],
		dtype=np.float64,
	)


def _make_estimator(model_name: str, class_names: list[str]) -> Any:
	"""Construct a configured random-forest or XGBoost classifier."""
	cfg = get_config().classifier
	if model_name == "rf":
		from sklearn.ensemble import RandomForestClassifier

		return RandomForestClassifier(
			n_estimators=cfg.n_estimators,
			max_depth=cfg.max_depth,
			class_weight="balanced",
			random_state=get_config().seed,
			n_jobs=-1,
		)
	if model_name == "xgb":
		try:
			from xgboost import XGBClassifier
		except ImportError as error:
			message = "xgboost is required when classifier.model='xgb'."
			logger.error(message)
			raise SchemaError(message) from error
		return XGBClassifier(
			n_estimators=cfg.n_estimators,
			max_depth=cfg.max_depth or cfg.xgb_max_depth,
			objective="multi:softprob",
			num_class=len(class_names),
			eval_metric="mlogloss",
			random_state=get_config().seed,
			n_jobs=-1,
		)
	message = f"Unsupported classifier model {model_name!r}; expected 'rf' or 'xgb'."
	logger.error(message)
	raise SchemaError(message)


def _fit_fold_model(
	model_name: str,
	x_train: pd.DataFrame,
	y_train: pd.Series,
	class_names: list[str],
) -> Any:
	"""Fit one fold, imputing RF features from that fold's training rows only."""
	from sklearn.impute import SimpleImputer
	from sklearn.pipeline import make_pipeline

	estimator = _make_estimator(model_name, class_names)
	if model_name == "rf":
		return make_pipeline(
			SimpleImputer(strategy="median", keep_empty_features=True),
			estimator,
		).fit(x_train, y_train)
	estimator.fit(
		x_train,
		y_train,
		sample_weight=_balanced_sample_weights(y_train.to_numpy(), class_names),
	)
	return estimator


def _feature_hash(labels: pd.Series) -> str:
	"""Build a deterministic content hash for labels and their row identifiers."""
	digest = hashlib.sha256()
	digest.update(pd.util.hash_pandas_object(labels, index=True).values.tobytes())
	return digest.hexdigest()


def train(
	features: pd.DataFrame,
	labels: pd.Series,
	groups: pd.Series,
) -> ClassifierArtifacts:
	"""Train a configured model and evaluate it with spatial GroupKFold.

	Args:
		features: Numeric feature rows, indexed by cell ID or time/cell ID.
		labels: Class labels aligned to feature rows.
		groups: Spatial block IDs aligned to feature rows.

	Returns:
		Fitted model, ordered feature/class names, metrics, and label digest.

	Raises:
		SchemaError: If inputs or model configuration are invalid.
		DataIntegrityError: If CV metrics do not satisfy the urban F1 threshold.
	"""
	started = time.perf_counter()
	logger.info("Starting spatial classifier training (rows=%d).", len(features))
	if not features.index.equals(labels.index) or not features.index.equals(groups.index):
		message = "Features, labels, and spatial groups must have identical indexes."
		logger.error(message)
		raise SchemaError(message)
	cfg = get_config().classifier
	class_names = list(cfg.classes)
	if features.empty or not features.columns.is_unique:
		message = "Training features must be non-empty with unique column names."
		logger.error(message)
		raise SchemaError(message)
	if not set(labels.astype(str)).issubset(class_names):
		message = "Training labels include classes outside the configured class list."
		logger.error(message)
		raise SchemaError(message)
	if labels.isna().any() or groups.isna().any():
		message = "Training labels and spatial groups cannot contain missing values."
		logger.error(message)
		raise SchemaError(message)
	if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
		message = "All model features must be numeric."
		logger.error(message)
		raise SchemaError(message)
	model_name = cfg.model.lower()
	x_values = features.astype(np.float64)
	y_values = labels.astype(str)
	group_values = groups
	unique_groups = pd.unique(group_values)
	if len(unique_groups) < cfg.n_splits:
		message = (
			f"Spatial GroupKFold requires at least {cfg.n_splits} unique groups; "
			f"received {len(unique_groups)}."
		)
		logger.error(message)
		raise SchemaError(message)
	# Resolve the Block E helper lazily to avoid introducing an import cycle.
	try:
		from validation.spatial_cv import make_blocks  # type: ignore[import-not-found]
	except ImportError:
		make_blocks = None
	if make_blocks is not None and groups.attrs.get("coordinates") is not None:
		generated_groups = make_blocks(
			groups.attrs["coordinates"],
			block_size=get_config().grid.block_size,
		)
		group_values = pd.Series(generated_groups, index=features.index)

	from sklearn.metrics import (
		classification_report,
		confusion_matrix,
		f1_score,
	)
	from sklearn.model_selection import GroupKFold

	oof = np.empty(len(features), dtype=object)
	splitter = GroupKFold(n_splits=cfg.n_splits)
	for train_idx, valid_idx in splitter.split(x_values, y_values, group_values):
		fold_model = _fit_fold_model(
			model_name,
			x_values.iloc[train_idx],
			y_values.iloc[train_idx],
			class_names,
		)
		oof[valid_idx] = fold_model.predict(x_values.iloc[valid_idx])
	report = classification_report(
		y_values,
		oof,
		labels=class_names,
		output_dict=True,
		zero_division=0,
	)
	confusion = confusion_matrix(y_values, oof, labels=class_names)
	urban_f1 = float(f1_score(y_values, oof, labels=["urban"], average="micro", zero_division=0))
	metrics: dict[str, Any] = {
		"per_class": {
			class_name: {
				"precision": float(report[class_name]["precision"]),
				"recall": float(report[class_name]["recall"]),
				"f1": float(report[class_name]["f1-score"]),
				"support": int(report[class_name]["support"]),
			}
			for class_name in class_names
		},
		"macro_f1": float(report["macro avg"]["f1-score"]),
		"urban_f1": urban_f1,
		"confusion_matrix": confusion.tolist(),
		"class_order": class_names,
		"cv": "GroupKFold",
		"n_splits": cfg.n_splits,
	}
	if urban_f1 < cfg.min_urban_f1:
		message = (
			f"Spatial CV urban F1={urban_f1:.3f} is below required "
			f"{cfg.min_urban_f1:.3f}."
		)
		logger.error(message)
		raise DataIntegrityError(message)
	final_model = _fit_fold_model(model_name, x_values, y_values, class_names)
	trained_on = str(labels.attrs.get("source_sha256") or _feature_hash(labels))
	logger.info(
		"Finished classifier training (rows=%d, features=%d, urban_f1=%.3f) in %.3fs.",
		len(features),
		len(features.columns),
		urban_f1,
		time.perf_counter() - started,
	)
	return ClassifierArtifacts(
		model=final_model,
		feature_names=list(features.columns),
		class_names=class_names,
		metrics=metrics,
		trained_on=trained_on,
	)


def predict_class_map(
	art: ClassifierArtifacts,
	features: pd.DataFrame,
) -> xr.DataArray:
	"""Predict class names for each feature row and restore raster dimensions.

	Args:
		art: Fitted artifacts returned by :func:`train`.
		features: Features ordered as produced by :func:`build_feature_matrix`.

	Returns:
		Class-label DataArray with ``(time, y, x)`` dimensions when metadata exists.

	Raises:
		SchemaError: If feature columns do not match the fitted model.
	"""
	logger.info("Starting classifier prediction (rows=%d).", len(features))
	if list(features.columns) != art.feature_names:
		message = "Prediction feature columns do not match the trained feature order."
		logger.error(message)
		raise SchemaError(message)
	predictions = art.model.predict(features.astype(np.float64))
	shape = features.attrs.get("grid_shape")
	if shape is None:
		return xr.DataArray(
			predictions,
			dims=("sample",),
			coords={"sample": np.arange(len(predictions))},
			name="class",
		)
	nrows, ncols = shape
	index = features.index
	if isinstance(index, pd.MultiIndex) and "time" in index.names:
		times = pd.Index(index.get_level_values("time")).unique()
		values = np.asarray(predictions).reshape(len(times), nrows, ncols)
		result = xr.DataArray(
			values,
			dims=("time", "y", "x"),
			coords={
				"time": times.to_numpy(),
				"y": np.arange(nrows),
				"x": np.arange(ncols),
			},
			name="class",
		)
	else:
		values = np.asarray(predictions).reshape(nrows, ncols)
		result = xr.DataArray(
			values,
			dims=("y", "x"),
			coords={"y": np.arange(nrows), "x": np.arange(ncols)},
			name="class",
		).expand_dims(time=[pd.Timestamp("1970-01-01")])
	logger.info("Finished classifier prediction (rows=%d).", len(features))
	return result


def urban_fraction(
	class_map_fine: xr.DataArray,
	grid: xr.Dataset,
) -> xr.DataArray:
	"""Aggregate fine categorical maps to urban pixel shares on a target grid.

	Args:
		class_map_fine: Predicted class labels with y/x and optional time dims.
		grid: Target grid dataset with canonical y/x coordinates and rio metadata.

	Returns:
		Urban share in [0, 1], with dimensions ``(time, y, x)``.

	Raises:
		SchemaError: If inputs lack spatial dimensions or target metadata.
	"""
	if not {"y", "x"}.issubset(class_map_fine.dims) or not {"y", "x"}.issubset(grid.dims):
		message = "Class map and target grid must both have y and x dimensions."
		logger.error(message)
		raise SchemaError(message)
	if not grid.data_vars:
		message = "Target grid must contain a raster variable for its spatial template."
		logger.error(message)
		raise SchemaError(message)
	if getattr(grid.rio, "crs", None) is None:
		message = "Target grid must define its canonical CRS."
		logger.error(message)
		raise SchemaError(message)
	if "time" not in class_map_fine.dims:
		class_map_fine = class_map_fine.expand_dims(time=[pd.Timestamp("1970-01-01")])
	binary = xr.where(class_map_fine == "urban", 1.0, 0.0).astype(np.float32)
	try:
		from rasterio.enums import Resampling

		template = grid[next(iter(grid.data_vars))]
		outputs: list[xr.DataArray] = []
		for timestamp in binary.time.values:
			slice_data = binary.sel(time=timestamp).rio.write_crs(
				class_map_fine.rio.crs, inplace=False
			).rio.write_transform(class_map_fine.rio.transform(), inplace=False)
			aggregated = slice_data.rio.reproject_match(
				template, resampling=Resampling.average
			)
			outputs.append(aggregated)
	except (AttributeError, KeyError, ValueError) as error:
		message = "Could not aggregate class map onto the target grid."
		logger.error(message)
		raise SchemaError(message) from error
	result = xr.concat(outputs, dim=xr.IndexVariable("time", binary.time.values))
	result.name = "urban_fraction"
	result = result.clip(0.0, 1.0)
	return result.transpose("time", "y", "x")


def temporal_consistency_check(uf: xr.DataArray) -> pd.DataFrame:
	"""Report implausible annual jumps and urban-to-nonurban transitions.

	No temporal smoothing or monotonicity correction is applied.

	Args:
		uf: Urban fraction with dimensions ``time``, ``y``, and ``x``.

	Returns:
		One row per cell containing violation flags and maximum annual change.

	Raises:
		SchemaError: If required dimensions are missing or values are out of range.
	"""
	started = time.perf_counter()
	logger.info(
		"Starting urban-fraction temporal check (rows=%d).",
		uf.size,
	)
	if not {"time", "y", "x"}.issubset(uf.dims):
		message = "Urban fraction must have time, y, and x dimensions."
		logger.error(message)
		raise SchemaError(message)
	values = np.asarray(uf.transpose("time", "y", "x").values, dtype=float)
	finite = np.isfinite(values)
	if np.any((values[finite] < 0.0) | (values[finite] > 1.0)):
		message = "Urban fraction values must be within [0, 1]."
		logger.error(message)
		raise SchemaError(message)
	timestamps = pd.DatetimeIndex(uf.time.values)
	if len(timestamps) < 2:
		delta_years = np.asarray([], dtype=float)
		diffs = np.empty((0, *values.shape[1:]), dtype=float)
	else:
		delta_years = (
			np.diff(timestamps.values).astype("timedelta64[D]").astype(float)
			/ 365.2425
		)
		delta_years[delta_years <= 0] = np.nan
		diffs = np.diff(values, axis=0)
	annual_change = np.abs(diffs) / delta_years.reshape((-1, 1, 1))
	max_change = (
		np.nanmax(annual_change, axis=0)
		if annual_change.size
		else np.zeros(values.shape[1:], dtype=float)
	)
	jump_flag = max_change > get_config().classifier.max_jump
	urban_to_nonurban = np.any(
		(values[:-1] >= get_config().classifier.urban_presence_threshold)
		& (values[1:] < get_config().classifier.urban_presence_threshold),
		axis=0,
	) if len(values) > 1 else np.zeros(values.shape[1:], dtype=bool)
	nrows, ncols = values.shape[1:]
	cell_ids = np.arange(nrows * ncols, dtype=np.int64)
	report = pd.DataFrame(
		{
			"cell_id": cell_ids,
			"max_abs_jump_per_year": max_change.reshape(-1),
			"flag_jump": jump_flag.reshape(-1),
			"flag_urban_to_non_urban": urban_to_nonurban.reshape(-1),
		}
	).set_index("cell_id")
	logger.info(
		"Finished urban-fraction temporal check (cells=%d) in %.3fs.",
		len(report),
		time.perf_counter() - started,
	)
	return report


def save_outputs(
	urban_fraction_map: xr.DataArray,
	artifacts: ClassifierArtifacts,
	city: str,
) -> dict[str, Path]:
	"""Persist urban fraction, fitted model, and metrics under interim outputs.

	Args:
		urban_fraction_map: Monthly urban-fraction raster.
		artifacts: Fitted classifier artifacts.
		city: Canonical city key.

	Returns:
		Mapping of output kinds to their saved paths.

	Raises:
		DataIntegrityError: If persistence fails.
	"""
	started = time.perf_counter()
	if city not in get_config().grid.crs:
		message = f"Unsupported city {city!r}."
		logger.error(message)
		raise SchemaError(message)
	output_dir = (
		get_config().paths.interim
		/ get_config().classifier.output_directory
		/ city
	)
	stack_path = output_dir / "urban_fraction.zarr"
	model_path = output_dir / "model.joblib"
	metrics_path = output_dir / "metrics.json"
	logger.info("Starting classifier output save (rows=%d, city=%s).", urban_fraction_map.size, city)
	try:
		output_dir.mkdir(parents=True, exist_ok=True)
		urban_fraction_map.to_dataset(name="urban_fraction").to_zarr(
			stack_path, mode="w", consolidated=True
		)
		import joblib

		joblib.dump(artifacts, model_path)
		metrics_path.write_text(
			json.dumps(
				{
					**artifacts.metrics,
					"feature_names": artifacts.feature_names,
					"class_names": artifacts.class_names,
					"trained_on": artifacts.trained_on,
				},
				indent=2,
			)
			+ "\n",
			encoding="utf-8",
		)
	except (OSError, ValueError, ImportError) as error:
		message = f"Could not save classifier outputs for {city}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	logger.info(
		"Finished classifier output save (rows=%d) in %.3fs.",
		urban_fraction_map.size,
		time.perf_counter() - started,
	)
	return {"urban_fraction": stack_path, "model": model_path, "metrics": metrics_path}


if __name__ == "__main__":
	"""Run a small synthetic spatial training and prediction check."""
	_config = get_config().classifier
	_class_names = list(_config.classes)
	_rng = np.random.default_rng(get_config().seed)
	_rows: list[np.ndarray] = []
	_label_values: list[str] = []
	_group_values: list[int] = []
	for _group in range(_config.n_splits):
		for _class_index, _class_name in enumerate(_class_names):
			for _ in range(10):
				_row = np.zeros(len(_class_names), dtype=np.float64)
				_row[_class_index] = 1.0
				_rows.append(_row + _rng.normal(0.0, 0.01, len(_class_names)))
				_label_values.append(_class_name)
				_group_values.append(_group)
	_features = pd.DataFrame(
		_rows, columns=[f"feature_{i}" for i in range(len(_class_names))]
	)
	_features.index = pd.Index(
		np.arange(len(_features), dtype=np.int64), name="cell_id"
	)
	_labels = pd.Series(_label_values, index=_features.index)
	_groups = pd.Series(_group_values, index=_features.index)
	_artifacts = train(_features, _labels, _groups)
	_predictions = predict_class_map(_artifacts, _features)
	assert _predictions.size == len(_features)
	assert set(np.unique(_predictions.values)).issubset(set(_class_names))
	_fine_transform = get_config().grid.transform["dehradun"]
	_fine_values = np.asarray(_predictions.values[:16]).reshape(1, 4, 4)
	_fine_map = xr.DataArray(
		_fine_values,
		dims=("time", "y", "x"),
		coords={
			"time": [pd.Timestamp("2020-01-01")],
			"y": _fine_transform.f + _fine_transform.e * (np.arange(4) + 0.5),
			"x": _fine_transform.c + _fine_transform.a * (np.arange(4) + 0.5),
		},
	).rio.write_crs(get_config().grid.crs["dehradun"]).rio.write_transform(
		_fine_transform
	)
	_grid_transform = Affine(
		_fine_transform.a * 2,
		0,
		_fine_transform.c,
		0,
		_fine_transform.e * 2,
		_fine_transform.f,
	)
	_grid = xr.Dataset(
		{"template": (("y", "x"), np.zeros((2, 2)))},
		coords={
			"y": _grid_transform.f + _grid_transform.e * (np.arange(2) + 0.5),
			"x": _grid_transform.c + _grid_transform.a * (np.arange(2) + 0.5),
		},
	).rio.write_crs(get_config().grid.crs["dehradun"]).rio.write_transform(
		_grid_transform
	)
	_urban_fraction = urban_fraction(_fine_map, _grid)
	assert np.all(
		(_urban_fraction.values >= 0.0) & (_urban_fraction.values <= 1.0)
	)
