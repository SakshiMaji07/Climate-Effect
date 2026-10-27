"""Spatial block generation, variograms, and leakage-safe cross-validation."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree, distance
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection._split import BaseCrossValidator

from config import get_config
from exceptions import ConfigError, DataIntegrityError, SchemaError
from logger import get_logger
from temporal_features import Fold

logger = get_logger(__name__)
VARIOGRAM_COLUMNS = ("lag", "distance", "semivariance", "count")


def _validate_xy(xy: np.ndarray, *, minimum_rows: int = 2) -> np.ndarray:
	"""Validate projected coordinate pairs and return float64 coordinates."""
	try:
		coordinates = np.asarray(xy, dtype=np.float64)
	except (TypeError, ValueError) as error:
		message = "xy must contain numeric projected coordinates."
		logger.error(message)
		raise SchemaError(message) from error
	if (
		coordinates.ndim != 2
		or coordinates.shape[1] != 2
		or coordinates.shape[0] < minimum_rows
	):
		message = f"xy must have shape (n, 2) with at least {minimum_rows} rows."
		logger.error(message)
		raise SchemaError(message)
	if not np.isfinite(coordinates).all():
		message = "xy coordinates must contain only finite values."
		logger.error(message)
		raise SchemaError(message)
	return coordinates


def empirical_variogram(
	values: np.ndarray,
	xy: np.ndarray,
	n_lags: int = 20,
	max_dist: float | None = None,
	subsample: int = 5_000,
) -> pd.DataFrame:
	"""Compute an empirical semivariogram from projected residual locations.

	Args:
		values: Numeric baseline residuals, not raw response values.
		xy: Projected ``(x, y)`` coordinate pairs in metres.
		n_lags: Number of equally spaced distance bins.
		max_dist: Maximum included pair distance in metres. Defaults to the
			maximum observed distance.
		subsample: Maximum number of distinct point pairs used in the estimate.

	Returns:
		Dataframe containing lag index, mean pair distance, semivariance, and
		pair count. Residual and coordinate provenance is retained in attrs for
		the Moran correlogram fallback in :func:`estimate_range`.

	Raises:
		SchemaError: If observations, coordinates, or settings are invalid.
	"""
	started = time.perf_counter()
	logger.info("Starting empirical variogram (rows=%d).", len(values))
	coordinates = _validate_xy(xy)
	try:
		residuals = np.asarray(values, dtype=np.float64)
	except (TypeError, ValueError) as error:
		message = "values must contain numeric baseline residuals."
		logger.error(message)
		raise SchemaError(message) from error
	if residuals.ndim != 1 or len(residuals) != len(coordinates):
		message = "values must be one-dimensional and aligned with xy."
		logger.error(message)
		raise SchemaError(message)
	if not np.isfinite(residuals).all():
		message = "Variogram residual values must be finite."
		logger.error(message)
		raise SchemaError(message)
	if n_lags < 2 or subsample < 1:
		message = "n_lags must be at least two and subsample must be positive."
		logger.error(message)
		raise ConfigError(message)
	pair_i, pair_j = np.triu_indices(len(residuals), k=1)
	if len(pair_i) == 0:
		message = "At least two coordinate observations are required."
		logger.error(message)
		raise SchemaError(message)
	if len(pair_i) > subsample:
		rng = np.random.default_rng(get_config().seed)
		selected = np.sort(rng.choice(len(pair_i), size=subsample, replace=False))
		pair_i, pair_j = pair_i[selected], pair_j[selected]
	pair_distances = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
	positive_distances = pair_distances[pair_distances > 0]
	if not len(positive_distances):
		message = "Variogram coordinates must include at least one distinct location."
		logger.error(message)
		raise SchemaError(message)
	distance_limit = float(max_dist) if max_dist is not None else float(positive_distances.max())
	if not np.isfinite(distance_limit) or distance_limit <= 0:
		message = "max_dist must be a positive finite distance in metres."
		logger.error(message)
		raise ConfigError(message)
	semivariances = 0.5 * (residuals[pair_i] - residuals[pair_j]) ** 2
	edges = np.linspace(0.0, distance_limit, n_lags + 1)
	bin_index = np.digitize(pair_distances, edges[1:-1], right=False)
	rows: list[dict[str, float | int]] = []
	for index in range(n_lags):
		mask = (
			(bin_index == index)
			& (pair_distances > 0)
			& (pair_distances <= distance_limit)
		)
		if not mask.any():
			continue
		rows.append(
			{
				"lag": index + 1,
				"distance": float(pair_distances[mask].mean()),
				"semivariance": float(semivariances[mask].mean()),
				"count": int(mask.sum()),
			}
		)
	variogram = pd.DataFrame(rows, columns=VARIOGRAM_COLUMNS)
	if len(variogram) < 2:
		message = "Too few populated distance bins to estimate a variogram range."
		logger.error(message)
		raise DataIntegrityError(message)
	variogram.attrs["residuals"] = residuals.copy()
	variogram.attrs["xy"] = coordinates.copy()
	variogram.attrs["max_dist"] = distance_limit
	variogram.attrs["n_lags"] = n_lags
	logger.info(
		"Computed empirical variogram (rows=%d, pairs=%d, lags=%d) in %.3fs.",
		len(residuals),
		len(pair_i),
		len(variogram),
		time.perf_counter() - started,
	)
	return variogram


def _spherical(
	distance_values: np.ndarray, nugget: float, sill: float, range_m: float
) -> np.ndarray:
	"""Evaluate the spherical semivariogram model."""
	ratio = np.minimum(distance_values / range_m, 1.0)
	return nugget + sill * (1.5 * ratio - 0.5 * ratio**3)


def _exponential(
	distance_values: np.ndarray, nugget: float, sill: float, range_m: float
) -> np.ndarray:
	"""Evaluate an exponential model parameterized by its 95% effective range."""
	return nugget + sill * (1.0 - np.exp(-3.0 * distance_values / range_m))


def _moran_range(variogram: pd.DataFrame) -> float:
	"""Estimate the first positive-to-nonpositive residual Moran crossing."""
	residual_values = variogram.attrs.get("residuals")
	coordinate_values = variogram.attrs.get("xy")
	if residual_values is None or coordinate_values is None:
		message = "Variogram lacks residual coordinates for Moran fallback."
		logger.error(message)
		raise DataIntegrityError(message)
	residuals = np.asarray(residual_values, dtype=np.float64)
	coordinates = np.asarray(coordinate_values, dtype=np.float64)
	if residuals.ndim != 1 or coordinates.shape != (len(residuals), 2):
		message = "Variogram lacks residual coordinates for Moran fallback."
		logger.error(message)
		raise DataIntegrityError(message)
	distance_limit = float(variogram.attrs["max_dist"])
	i, j = np.triu_indices(len(residuals), k=1)
	pair_distances = np.linalg.norm(coordinates[i] - coordinates[j], axis=1)
	centered = residuals - residuals.mean()
	variance = float(np.mean(centered**2))
	if variance <= np.finfo(np.float64).eps:
		message = "Baseline residuals have no variance for a Moran correlogram."
		logger.error(message)
		raise DataIntegrityError(message)
	n_lags = int(
		variogram.attrs.get(
			"n_lags", get_config().validation.variogram_n_lags
		)
	)
	edges = np.linspace(0.0, distance_limit, n_lags + 1)
	bins = np.digitize(pair_distances, edges[1:-1], right=False)
	correlogram: list[tuple[float, float]] = []
	for index in range(len(edges) - 1):
		mask = (bins == index) & (pair_distances <= distance_limit)
		if mask.any():
			moran_i = float(np.mean(centered[i[mask]] * centered[j[mask]]) / variance)
			correlogram.append((float(pair_distances[mask].mean()), moran_i))
	for (left_distance, left_i), (right_distance, right_i) in zip(
		correlogram, correlogram[1:]
	):
		if left_i > 0 >= right_i:
			if left_i == right_i:
				return right_distance
			fraction = left_i / (left_i - right_i)
			return float(left_distance + fraction * (right_distance - left_distance))
	logger.warning(
		"Moran correlogram did not cross zero; using the maximum variogram distance."
	)
	return distance_limit


def estimate_range(variogram: pd.DataFrame) -> float:
	"""Fit spherical/exponential variograms and return effective range in metres.

	Args:
		variogram: Output from :func:`empirical_variogram`, computed from
			residuals of a baseline model.

	Returns:
		Effective spatial correlation range in metres. Exponential ranges denote
		the distance at which 95% of the sill is attained.

	Raises:
		SchemaError: If variogram columns or measurements are invalid.
		DataIntegrityError: If neither fitted models nor Moran fallback work.
	"""
	started = time.perf_counter()
	logger.info("Starting residual variogram range fit (lags=%d).", len(variogram))
	required = set(VARIOGRAM_COLUMNS)
	if not required.issubset(variogram.columns) or len(variogram) < 2:
		message = f"Variogram must contain at least two rows and columns {sorted(required)}."
		logger.error(message)
		raise SchemaError(message)
	x_values = variogram["distance"].to_numpy(dtype=np.float64)
	y_values = variogram["semivariance"].to_numpy(dtype=np.float64)
	if (
		not np.isfinite(x_values).all()
		or not np.isfinite(y_values).all()
		or (x_values <= 0).any()
		or (y_values < 0).any()
	):
		message = "Variogram distances and semivariances must be finite and nonnegative."
		logger.error(message)
		raise SchemaError(message)
	max_distance = float(x_values.max())
	max_semivariance = float(y_values.max())
	if max_semivariance <= np.finfo(np.float64).eps:
		message = "Variogram semivariance has no measurable sill."
		logger.error(message)
		raise DataIntegrityError(message)
	weights = np.sqrt(variogram["count"].to_numpy(dtype=np.float64))
	fitted: list[tuple[float, str, float]] = []
	for model_name, model_fn in (("spherical", _spherical), ("exponential", _exponential)):
		try:
			parameters, _ = curve_fit(
				model_fn,
				x_values,
				y_values,
				p0=(max(0.0, float(y_values.min())), max_semivariance, max_distance / 2.0),
				bounds=(
					(0.0, np.finfo(np.float64).eps, max_distance / 1_000.0),
					(max_semivariance, max_semivariance * 10.0, max_distance * 10.0),
				),
				sigma=1.0 / np.maximum(weights, 1.0),
				maxfev=10_000,
			)
			predicted = model_fn(x_values, *parameters)
			score = float(np.average((y_values - predicted) ** 2, weights=weights))
			fitted.append((score, model_name, float(parameters[2])))
		except (RuntimeError, ValueError, FloatingPointError):
			continue
	if fitted:
		_, model_name, effective_range = min(fitted, key=lambda result: result[0])
		if np.isfinite(effective_range) and effective_range > 0:
			logger.info(
				"Finished residual range estimate (method=%s, range_m=%.3f) in %.3fs.",
				model_name,
				effective_range,
				time.perf_counter() - started,
			)
			return effective_range
	fallback_range = _moran_range(variogram)
	logger.info(
		"Finished residual range estimate (method=Moran's I, range_m=%.3f) in %.3fs.",
		fallback_range,
		time.perf_counter() - started,
	)
	return fallback_range


def estimate_city_ranges(
	frame: pd.DataFrame,
	value_column: str,
	elevation_column: str = "dem",
	city_column: str = "city",
	xy_columns: tuple[str, str] = ("x", "y"),
) -> dict[str, float]:
	"""Estimate per-city correlation ranges from lapse-only residuals.

	For each city, a least-squares baseline containing elevation and, when
	available, date-specific intercepts is fit to that city's response. The
	empirical variogram is then computed from location-averaged residuals; raw
	observations are never used as the spatial response.

	Args:
		frame: Dataframe containing city, response, elevation, and projected xy.
		value_column: Numeric response column.
		elevation_column: Elevation feature used by the lapse-only baseline.
		city_column: City grouping key.
		xy_columns: Projected coordinate columns in metres.

	Returns:
		Mapping from city key to residual correlation range in metres.

	Raises:
		SchemaError: If required columns or city data are invalid.
		DataIntegrityError: If a city's baseline or variogram cannot be estimated.
	"""
	started = time.perf_counter()
	logger.info("Starting per-city residual range estimation (rows=%d).", len(frame))
	required = {city_column, value_column, elevation_column, *xy_columns}
	missing = sorted(required - set(frame.columns))
	if missing:
		message = f"Range-estimation data is missing columns: {missing}."
		logger.error(message)
		raise SchemaError(message)
	if (
		not pd.api.types.is_numeric_dtype(frame[value_column].dtype)
		or not pd.api.types.is_numeric_dtype(frame[elevation_column].dtype)
	):
		message = "Range-estimation response and elevation columns must be numeric."
		logger.error(message)
		raise SchemaError(message)
	results: dict[str, float] = {}
	for city, group in frame.groupby(city_column, sort=True):
		if city_column == "city" and city not in {"dehradun", "jammu"}:
			message = f"Unsupported city key {city!r} in range-estimation data."
			logger.error(message)
			raise SchemaError(message)
		clean = group.dropna(subset=[value_column, elevation_column, *xy_columns])
		if len(clean) < 3:
			message = f"City {city!r} has too few complete rows for lapse-only residuals."
			logger.error(message)
			raise DataIntegrityError(message)
		response = clean[value_column].to_numpy(dtype=np.float64)
		elevation = clean[elevation_column].to_numpy(dtype=np.float64)
		coordinates = _validate_xy(clean.loc[:, list(xy_columns)].to_numpy())
		if not np.isfinite(response).all() or not np.isfinite(elevation).all():
			message = f"City {city!r} contains non-finite response/elevation values."
			logger.error(message)
			raise SchemaError(message)
		design_columns = [np.ones(len(elevation))]
		if np.ptp(elevation) > np.finfo(np.float64).eps:
			design_columns.append(elevation)
		if "date" in clean:
			dates = clean["date"]
			if (
				not pd.api.types.is_datetime64_ns_dtype(dates.dtype)
				or dates.isna().any()
				or not (dates.dt.day == 1).all()
				or not (dates == dates.dt.normalize()).all()
			):
				message = "Range-estimation dates must be datetime64[ns] month-start values."
				logger.error(message)
				raise SchemaError(message)
			date_codes = pd.Categorical(dates).codes
			date_count = int(date_codes.max()) + 1
			if date_count > 1:
				date_indicators = np.eye(date_count, dtype=np.float64)[
					:, 1:
				][date_codes]
				design_columns.extend(date_indicators[:, index] for index in range(date_count - 1))
		baseline_design = np.column_stack(design_columns)
		coefficients, _, rank, _ = np.linalg.lstsq(baseline_design, response, rcond=None)
		if rank < baseline_design.shape[1]:
			message = f"City {city!r} elevation does not identify a lapse-only baseline."
			logger.error(message)
			raise DataIntegrityError(message)
		residuals = response - baseline_design @ coefficients
		residual_frame = pd.DataFrame(
			{
				xy_columns[0]: coordinates[:, 0],
				xy_columns[1]: coordinates[:, 1],
				"_residual": residuals,
			}
		)
		by_location = residual_frame.groupby(list(xy_columns), sort=True)["_residual"].mean()
		if len(by_location) < 3:
			message = f"City {city!r} has fewer than three unique residual locations."
			logger.error(message)
			raise DataIntegrityError(message)
		residual_coordinates = by_location.index.to_frame(index=False).to_numpy(
			dtype=np.float64
		)
		residual_values = by_location.to_numpy(dtype=np.float64)
		variogram = empirical_variogram(
			residual_values,
			residual_coordinates,
			n_lags=get_config().validation.variogram_n_lags,
			subsample=get_config().validation.variogram_subsample,
		)
		results[str(city)] = estimate_range(variogram)
	logger.info(
		"Finished per-city residual range estimation (cities=%d) in %.3fs.",
		len(results),
		time.perf_counter() - started,
	)
	return results


def make_blocks(
	xy: np.ndarray,
	block_size_m: float | None = None,
	n_blocks_min: int | None = None,
	shape: Literal["square", "hex"] = "square",
	*,
	block_size: float | None = None,
) -> np.ndarray:
	"""Assign projected locations to deterministic square or hexagonal blocks.

	Args:
		xy: Projected ``(x, y)`` coordinates in metres.
		block_size_m: Requested block width; clamped to configured minimum size.
		n_blocks_min: Minimum number of distinct blocks required.
		shape: ``square`` or ``hex`` tessellation.
		block_size: Backward-compatible alias for ``block_size_m``.

	Returns:
		Contiguous int64 block identifiers aligned to ``xy`` rows.

	Raises:
		SchemaError: If coordinates, shape, or size are invalid.
		DataIntegrityError: If the resulting block count is insufficient.
	"""
	started = time.perf_counter()
	coordinates = _validate_xy(xy)
	logger.info(
		"Starting spatial block creation (rows=%d, shape=%s).",
		len(coordinates),
		shape,
	)
	config = get_config().validation
	if block_size is not None and block_size_m is not None:
		message = "Pass only one of block_size_m or the legacy block_size alias."
		logger.error(message)
		raise SchemaError(message)
	requested_size = block_size_m if block_size_m is not None else block_size
	if requested_size is not None:
		try:
			requested_size = float(requested_size)
		except (TypeError, ValueError) as error:
			message = "Requested block size must be a positive finite distance."
			logger.error(message)
			raise ConfigError(message) from error
		if not np.isfinite(requested_size) or requested_size <= 0:
			message = "Requested block size must be a positive finite distance."
			logger.error(message)
			raise ConfigError(message)
	size = max(
		config.min_block_m,
		config.min_block_m if requested_size is None else float(requested_size),
	)
	minimum_blocks = config.min_blocks if n_blocks_min is None else n_blocks_min
	if not np.isfinite(size) or size <= 0 or minimum_blocks < 1:
		message = "Block size and minimum block count must be positive."
		logger.error(message)
		raise ConfigError(message)
	if shape == "square":
		indices = np.floor(coordinates / size).astype(np.int64)
	elif shape == "hex":
		q = (2.0 / 3.0 * coordinates[:, 0]) / size
		r = (
			-coordinates[:, 0] / 3.0 + np.sqrt(3.0) * coordinates[:, 1] / 3.0
		) / size
		cube_x, cube_z = q, r
		cube_y = -cube_x - cube_z
		rounded_x, rounded_y, rounded_z = (
			np.rint(cube_x),
			np.rint(cube_y),
			np.rint(cube_z),
		)
		x_diff, y_diff, z_diff = (
			np.abs(rounded_x - cube_x),
			np.abs(rounded_y - cube_y),
			np.abs(rounded_z - cube_z),
		)
		fix_x = (x_diff > y_diff) & (x_diff > z_diff)
		fix_y = (~fix_x) & (y_diff > z_diff)
		fix_z = ~(fix_x | fix_y)
		rounded_x[fix_x] = -rounded_y[fix_x] - rounded_z[fix_x]
		rounded_y[fix_y] = -rounded_x[fix_y] - rounded_z[fix_y]
		rounded_z[fix_z] = -rounded_x[fix_z] - rounded_y[fix_z]
		indices = np.column_stack((rounded_x, rounded_z)).astype(np.int64)
	else:
		message = f"Unsupported block shape {shape!r}; expected 'square' or 'hex'."
		logger.error(message)
		raise SchemaError(message)
	_, block_ids = np.unique(indices, axis=0, return_inverse=True)
	blocks = block_ids.astype(np.int64)
	block_count = int(np.unique(blocks).size)
	logger.info(
		"Finished spatial block creation (rows=%d, blocks=%d, size_m=%.3f, shape=%s) in %.3fs.",
		len(coordinates),
		block_count,
		size,
		shape,
		time.perf_counter() - started,
	)
	if block_count < minimum_blocks:
		message = (
			f"Block partition produced {block_count} blocks; at least "
			f"{minimum_blocks} are required. Increase spatial extent or reduce "
			"the configured block size."
		)
		logger.error(message)
		raise DataIntegrityError(message)
	return blocks


def _balanced_block_folds(
	blocks: np.ndarray,
	n_splits: int,
	stratify_by: np.ndarray | None,
	seed: int,
) -> list[np.ndarray]:
	"""Allocate whole blocks to folds while balancing rows and optional tertiles."""
	unique_blocks, inverse, block_sizes = np.unique(
		blocks, return_inverse=True, return_counts=True
	)
	if len(unique_blocks) < n_splits:
		message = (
			f"Spatial CV needs at least {n_splits} blocks; received {len(unique_blocks)}."
		)
		logger.error(message)
		raise DataIntegrityError(message)
	strata = np.zeros(len(blocks), dtype=np.int64)
	if stratify_by is not None:
		values = np.asarray(stratify_by)
		try:
			numeric_values = values.astype(np.float64)
		except (TypeError, ValueError) as error:
			message = "stratify_by must be a finite numeric vector aligned with blocks."
			logger.error(message)
			raise SchemaError(message) from error
		if (
			values.ndim != 1
			or len(values) != len(blocks)
			or not np.isfinite(numeric_values).all()
		):
			message = "stratify_by must be a finite numeric vector aligned with blocks."
			logger.error(message)
			raise SchemaError(message)
		edges = np.unique(np.quantile(numeric_values, [1 / 3, 2 / 3]))
		strata = np.digitize(numeric_values, edges, right=True)
	strata_count = int(strata.max()) + 1
	block_strata = np.zeros((len(unique_blocks), strata_count), dtype=np.int64)
	for block_idx in range(len(unique_blocks)):
		block_strata[block_idx] = np.bincount(
			strata[inverse == block_idx], minlength=strata_count
		)
	rng = np.random.default_rng(seed)
	random_tie_break = rng.random(len(unique_blocks))
	order = sorted(
		range(len(unique_blocks)),
		key=lambda index: (-int(block_sizes[index]), random_tie_break[index]),
	)
	fold_blocks: list[list[int]] = [[] for _ in range(n_splits)]
	fold_counts = np.zeros((n_splits, strata_count), dtype=np.float64)
	fold_sizes = np.zeros(n_splits, dtype=np.int64)
	target_strata = block_strata.sum(axis=0) / n_splits
	for block_index in order:
		candidates = np.flatnonzero(fold_sizes == fold_sizes.min())
		scores: list[tuple[float, int]] = []
		for fold_index in candidates:
			updated = fold_counts[fold_index] + block_strata[block_index]
			strata_imbalance = float(
				np.sum(((updated - target_strata) / np.maximum(target_strata, 1.0)) ** 2)
			)
			scores.append((strata_imbalance, int(fold_index)))
		_, chosen_fold = min(scores)
		fold_blocks[chosen_fold].append(int(unique_blocks[block_index]))
		fold_counts[chosen_fold] += block_strata[block_index]
		fold_sizes[chosen_fold] += block_sizes[block_index]
	if any(not assigned for assigned in fold_blocks):
		message = "Spatial blocks could not be balanced across all requested folds."
		logger.error(message)
		raise DataIntegrityError(message)
	if stratify_by is not None:
		available_strata = np.flatnonzero(block_strata.sum(axis=0) >= n_splits)
		for fold_idx in range(n_splits):
			if np.any(fold_counts[fold_idx, available_strata] == 0):
				logger.warning(
					"Spatial fold %d cannot cover every elevation tertile.",
					fold_idx,
				)
	return [np.asarray(fold, dtype=unique_blocks.dtype) for fold in fold_blocks]


class SpatialBlockCV(BaseCrossValidator):
	"""Cross-validator that holds out whole spatial blocks per fold."""

	def __init__(
		self,
		blocks: np.ndarray,
		n_splits: int,
		buffer_m: float = 0.0,
		stratify_by: np.ndarray | None = None,
		seed: int | None = None,
		xy: np.ndarray | None = None,
	) -> None:
		self.blocks = np.asarray(blocks)
		self.n_splits = n_splits
		self.buffer_m = float(buffer_m)
		self.stratify_by = None if stratify_by is None else np.asarray(stratify_by)
		self.seed = get_config().seed if seed is None else seed
		self.xy = None if xy is None else _validate_xy(xy)
		if self.blocks.ndim != 1 or len(self.blocks) == 0:
			message = "blocks must be a non-empty one-dimensional array."
			logger.error(message)
			raise SchemaError(message)
		if pd.isna(self.blocks).any():
			message = "blocks cannot contain missing values."
			logger.error(message)
			raise SchemaError(message)
		if n_splits < 2 or self.buffer_m < 0 or not np.isfinite(self.buffer_m):
			message = "n_splits must be at least two and buffer_m nonnegative."
			logger.error(message)
			raise ConfigError(message)
		if self.stratify_by is not None and (
			self.stratify_by.ndim != 1 or len(self.stratify_by) != len(self.blocks)
		):
			message = "stratify_by must align with blocks."
			logger.error(message)
			raise SchemaError(message)
		if self.xy is not None and len(self.xy) != len(self.blocks):
			message = "xy must align with blocks when buffering is enabled."
			logger.error(message)
			raise SchemaError(message)
		self._fold_blocks = _balanced_block_folds(
			self.blocks, self.n_splits, self.stratify_by, int(self.seed)
		)

	def _iter_test_indices(
		self,
		X: object = None,
		y: object = None,
		groups: object = None,
	) -> Iterator[np.ndarray]:
		"""Yield buffered-out spatial test indices."""
		for _, test_indices in self._iter_splits(X):
			yield test_indices

	def _iter_splits(
		self, X: object = None
	) -> Iterator[tuple[np.ndarray, np.ndarray]]:
		"""Yield the actual training/test arrays, preserving buffer exclusions."""
		started = time.perf_counter()
		logger.info("Starting spatial block CV (rows=%d, folds=%d).", len(self.blocks), self.n_splits)
		yielded = 0
		if X is not None and len(X) != len(self.blocks):  # type: ignore[arg-type]
			message = "X row count must match the configured spatial blocks."
			logger.error(message)
			raise SchemaError(message)
		for test_blocks in self._fold_blocks:
			test_mask = np.isin(self.blocks, test_blocks)
			train_mask = ~test_mask
			if self.buffer_m > 0:
				if self.xy is None:
					message = "A positive buffer_m requires xy coordinates."
					logger.error(message)
					raise ConfigError(message)
				tree = cKDTree(self.xy[test_mask])
				distances, _ = tree.query(self.xy[train_mask], k=1)
				train_candidates = np.flatnonzero(train_mask)
				train_mask[train_candidates[distances < self.buffer_m]] = False
			train_indices = np.flatnonzero(train_mask)
			test_indices = np.flatnonzero(test_mask)
			if not len(train_indices) or not len(test_indices):
				message = "Spatial fold has an empty train or test partition after buffering."
				logger.error(message)
				raise DataIntegrityError(message)
			if self.xy is not None and self.buffer_m > 0:
				assert_spatial_independence(
					self.xy[train_indices], self.xy[test_indices], self.buffer_m
				)
			yield train_indices, test_indices
			yielded += 1
		logger.info(
			"Finished spatial block CV (rows=%d, folds=%d) in %.3fs.",
			len(self.blocks),
			yielded,
			time.perf_counter() - started,
		)

	def split(
		self,
		X: object,
		y: object = None,
		groups: object = None,
	) -> Iterator[tuple[np.ndarray, np.ndarray]]:
		"""Yield row indices with entire spatial blocks held out.

		Args:
			X: Feature matrix with the same row count used to construct blocks.
			y: Ignored; present for sklearn compatibility.
			groups: Ignored; blocks are supplied at construction.

		Returns:
			Iterator of training and test row-index arrays.

		Raises:
			SchemaError: If input row counts disagree with configured blocks.
			DataIntegrityError: If buffering empties a fold partition.
		"""
		if len(X) != len(self.blocks):  # type: ignore[arg-type]
			message = "X row count must match the configured spatial blocks."
			logger.error(message)
			raise SchemaError(message)
		yield from self._iter_splits(X)

	def get_n_splits(
		self, X: object = None, y: object = None, groups: object = None
	) -> int:
		"""Return the configured number of spatial folds."""
		return self.n_splits


def spatiotemporal_splits(
	df: pd.DataFrame,
	blocks: np.ndarray,
	temporal_folds: Iterable[Fold],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
	"""Combine later-time evaluation with held-out spatial blocks.

	Args:
		df: Panel dataframe with month-start ``date`` values.
		blocks: Spatial block id per dataframe row.
		temporal_folds: Rolling-origin folds indexed against ``df``.

	Returns:
		Iterator of ``(train_idx, test_idx)`` arrays. Training consists of
		earlier dates in non-test blocks; test consists of later dates in held
		blocks.

	Raises:
		SchemaError: If dates or block assignments do not align with the panel.
		DataIntegrityError: If a generated split violates spatial/temporal order.
	"""
	started = time.perf_counter()
	logger.info("Starting spatiotemporal split generation (rows=%d).", len(df))
	if "date" not in df:
		message = "Spatiotemporal validation requires a date column."
		logger.error(message)
		raise SchemaError(message)
	dates = df["date"]
	if (
		not pd.api.types.is_datetime64_ns_dtype(dates.dtype)
		or dates.isna().any()
		or not (dates.dt.day == 1).all()
		or not (dates == dates.dt.normalize()).all()
	):
		message = "Spatiotemporal date must be non-null month-start datetime64[ns]."
		logger.error(message)
		raise SchemaError(message)
	block_values = np.asarray(blocks)
	if block_values.ndim != 1 or len(block_values) != len(df) or pd.isna(block_values).any():
		message = "blocks must be non-null and aligned with dataframe rows."
		logger.error(message)
		raise SchemaError(message)
	config = get_config().validation
	held_block_folds = _balanced_block_folds(
		block_values,
		config.block_n_splits,
		None,
		get_config().seed,
	)
	yielded = 0
	for temporal_fold in temporal_folds:
		train_positions = np.asarray(temporal_fold.train_idx, dtype=np.int64)
		test_positions = np.asarray(temporal_fold.test_idx, dtype=np.int64)
		if (
			not len(train_positions)
			or not len(test_positions)
			or train_positions.max() >= len(df)
			or test_positions.max() >= len(df)
			or train_positions.min() < 0
			or test_positions.min() < 0
			or dates.iloc[train_positions].max() >= dates.iloc[test_positions].min()
		):
			message = "Temporal fold positions must be valid and strictly rolling-origin."
			logger.error(message)
			raise DataIntegrityError(message)
		for held_blocks in held_block_folds:
			train = train_positions[~np.isin(block_values[train_positions], held_blocks)]
			test = test_positions[np.isin(block_values[test_positions], held_blocks)]
			if not len(train) or not len(test):
				logger.warning("Skipping empty spatiotemporal fold intersection.")
				continue
			if (
				dates.iloc[train].max() >= dates.iloc[test].min()
				or np.isin(block_values[train], held_blocks).any()
			):
				message = "Combined spatiotemporal split violates its holdout constraints."
				logger.error(message)
				raise DataIntegrityError(message)
			yield train, test
			yielded += 1
	logger.info(
		"Finished spatiotemporal split generation (rows=%d, splits=%d) in %.3fs.",
		len(df),
		yielded,
		time.perf_counter() - started,
	)


def assert_spatial_independence(
	train_xy: np.ndarray,
	test_xy: np.ndarray,
	min_dist: float,
) -> None:
	"""Assert every training location is at least ``min_dist`` from test sites.

	Args:
		train_xy: Projected training coordinates in metres.
		test_xy: Projected held-out coordinates in metres.
		min_dist: Minimum allowed nearest-neighbour distance in metres.

	Raises:
		SchemaError: If coordinates or distance are invalid.
		DataIntegrityError: If any train/test pair violates the separation.
	"""
	train = _validate_xy(train_xy, minimum_rows=1)
	test = _validate_xy(test_xy, minimum_rows=1)
	if not np.isfinite(min_dist) or min_dist < 0:
		message = "min_dist must be finite and nonnegative."
		logger.error(message)
		raise ConfigError(message)
	nearest, _ = cKDTree(test).query(train, k=1)
	if np.any(nearest < min_dist):
		message = (
			f"Train/test coordinates violate minimum spatial separation "
			f"({float(nearest.min()):.3f}m < {min_dist:.3f}m)."
		)
		logger.error(message)
		raise DataIntegrityError(message)


def _synthetic_demo() -> None:
	"""Show random-fold error is optimistic for a spatially smooth GP field."""
	config = get_config()
	rng = np.random.default_rng(config.seed)
	axis = np.linspace(0.0, 9.0, 10) * config.validation.synthetic_range_m
	grid_x, grid_y = np.meshgrid(axis, axis)
	coordinates = np.column_stack((grid_x.ravel(), grid_y.ravel()))
	squared_distance = distance.cdist(coordinates, coordinates, metric="sqeuclidean")
	covariance = np.exp(
		-squared_distance
		/ (2.0 * config.validation.synthetic_range_m**2)
	)
	field = rng.multivariate_normal(
		np.zeros(len(coordinates)),
		covariance + np.eye(len(coordinates)) * np.finfo(np.float64).eps,
	)
	blocks = make_blocks(
		coordinates,
		block_size_m=config.validation.synthetic_range_m * 1.5,
		n_blocks_min=config.validation.min_blocks,
	)
	block_cv = SpatialBlockCV(
		blocks,
		n_splits=config.validation.block_n_splits,
		seed=config.seed,
	)
	model = KNeighborsRegressor(n_neighbors=config.validation.demo_neighbors)
	random_cv = KFold(
		n_splits=config.validation.block_n_splits,
		shuffle=True,
		random_state=config.seed,
	)
	random_errors: list[float] = []
	block_errors: list[float] = []
	for train_idx, test_idx in random_cv.split(coordinates):
		model.fit(coordinates[train_idx], field[train_idx])
		random_errors.extend((model.predict(coordinates[test_idx]) - field[test_idx]) ** 2)
	for train_idx, test_idx in block_cv.split(coordinates):
		model.fit(coordinates[train_idx], field[train_idx])
		block_errors.extend((model.predict(coordinates[test_idx]) - field[test_idx]) ** 2)
	variogram = empirical_variogram(
		field,
		coordinates,
		n_lags=config.validation.variogram_n_lags,
		subsample=config.validation.variogram_subsample,
	)
	assert tuple(variogram.columns) == VARIOGRAM_COLUMNS
	assert blocks.dtype == np.dtype("int64")
	assert np.sqrt(np.mean(random_errors)) < np.sqrt(np.mean(block_errors)), (
		"Random-KFold RMSE should be lower than spatial block-CV RMSE for the "
		"configured smooth GP field."
	)


if __name__ == "__main__":
	_synthetic_demo()
