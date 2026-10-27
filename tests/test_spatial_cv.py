"""Tests for empirical spatial ranges, block construction, and spatial CV."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor

import validation.spatial_cv as spatial_cv
from config import get_config
from exceptions import DataIntegrityError, SchemaError
from validation.spatial_cv import (
	SpatialBlockCV,
	assert_spatial_independence,
	empirical_variogram,
	estimate_city_ranges,
	estimate_range,
	make_blocks,
	spatiotemporal_splits,
)
from temporal_features import Fold


def _grid(rows: int = 8, spacing: float = 100.0) -> np.ndarray:
	"""Create deterministic projected coordinate pairs."""
	x, y = np.meshgrid(np.arange(rows) * spacing, np.arange(rows) * spacing)
	return np.column_stack((x.ravel(), y.ravel())).astype(np.float64)


def test_empirical_variogram_schema_and_range_from_spatial_residuals():
	xy = _grid(10)
	values = np.sin(xy[:, 0] / 250.0) + np.cos(xy[:, 1] / 350.0)

	variogram = empirical_variogram(values, xy, n_lags=12, subsample=1000)
	range_m = estimate_range(variogram)

	assert tuple(variogram.columns) == spatial_cv.VARIOGRAM_COLUMNS
	assert (variogram["count"] > 0).all()
	assert np.isfinite(range_m)
	assert range_m > 0


def test_empirical_variogram_rejects_misaligned_or_nonfinite_values():
	xy = _grid(4)

	with pytest.raises(SchemaError, match="aligned"):
		empirical_variogram(np.ones(len(xy) - 1), xy)
	with pytest.raises(SchemaError, match="finite"):
		empirical_variogram(np.full(len(xy), np.nan), xy)


def test_range_estimation_falls_back_to_moran_correlogram(monkeypatch):
	xy = np.array(
		[[0.0, 0.0], [100.0, 0.0], [1_000.0, 0.0], [1_100.0, 0.0]]
	)
	values = np.array([1.0, 1.0, -1.0, -1.0])
	variogram = empirical_variogram(values, xy, n_lags=4)

	def fail_fit(*args, **kwargs):
		raise RuntimeError("forced fit failure")

	monkeypatch.setattr(spatial_cv, "curve_fit", fail_fit)

	range_m = estimate_range(variogram)

	assert np.isfinite(range_m)
	assert range_m > 0


def test_make_blocks_supports_square_and_hex_and_enforces_minimum():
	xy = _grid(8, spacing=500.0)
	config = get_config().validation

	square = make_blocks(xy, block_size_m=800.0, n_blocks_min=4)
	hexagonal = make_blocks(
		xy, block_size_m=800.0, n_blocks_min=4, shape="hex"
	)

	assert square.dtype == np.dtype("int64")
	assert np.unique(square).size >= 4
	assert np.unique(hexagonal).size >= 4
	assert np.unique(make_blocks(xy, block_size_m=config.min_block_m)).size > 0
	with pytest.raises(DataIntegrityError, match="at least 100 are required"):
		make_blocks(xy, block_size_m=10_000.0, n_blocks_min=100)


def test_spatial_block_cv_holds_out_blocks_and_removes_buffer_points():
	xy = _grid(8, spacing=1_000.0)
	blocks = make_blocks(xy, block_size_m=2_000.0, n_blocks_min=4)
	cv = SpatialBlockCV(
		blocks,
		n_splits=4,
		buffer_m=1_000.0,
		xy=xy,
		seed=get_config().seed,
	)

	for train_idx, test_idx in cv.split(np.zeros((len(xy), 1))):
		assert not set(blocks[train_idx]) & set(blocks[test_idx])
		assert np.linalg.norm(
			xy[train_idx, None, :] - xy[None, test_idx, :], axis=2
		).min() >= 1_000.0
	assert cv.get_n_splits() == 4


def test_spatial_cv_rejects_insufficient_blocks_and_independence_failure():
	with pytest.raises(DataIntegrityError, match="at least 3 blocks"):
		SpatialBlockCV(np.array([1, 1, 2, 2]), n_splits=3)
	with pytest.raises(DataIntegrityError, match="minimum spatial separation"):
		assert_spatial_independence(
			np.array([[0.0, 0.0], [1.0, 0.0]]),
			np.array([[2.0, 0.0]]),
			min_dist=2.0,
		)


def test_per_city_range_uses_residuals_after_elevation_baseline():
	xy = _grid(10, spacing=100.0)
	elevation = xy[:, 0] * 0.1 + xy[:, 1] * 0.05
	residual_pattern = np.sin(xy[:, 0] / 300.0) + np.cos(xy[:, 1] / 400.0)
	frame = pd.DataFrame(
		{
			"city": np.where(xy[:, 0] < xy[:, 0].mean(), "dehradun", "jammu"),
			"x": xy[:, 0],
			"y": xy[:, 1],
			"dem": elevation,
			"response": 4.0 * elevation + residual_pattern,
		}
	)

	ranges = estimate_city_ranges(frame, "response")

	assert set(ranges) == {"dehradun", "jammu"}
	assert all(np.isfinite(value) and value > 0 for value in ranges.values())


def test_spatiotemporal_splits_hold_out_later_dates_and_blocks(monkeypatch):
	config = get_config()
	validation = replace(config.validation, block_n_splits=2)
	monkeypatch.setattr(
		spatial_cv, "get_config", lambda: replace(config, validation=validation)
	)
	dates = pd.date_range("2020-01-01", periods=6, freq="MS").as_unit("ns")
	rows = [
		{"date": date, "cell_id": cell}
		for date in dates
		for cell in range(4)
	]
	frame = pd.DataFrame(rows)
	blocks = frame["cell_id"].to_numpy(dtype=np.int64)
	train_positions = np.flatnonzero(frame["date"].isin(dates[:4]))
	test_positions = np.flatnonzero(frame["date"].isin(dates[4:]))
	fold = Fold(
		train_idx=train_positions,
		test_idx=test_positions,
		train_end=dates[3],
		test_start=dates[4],
	)

	splits = list(spatiotemporal_splits(frame, blocks, [fold]))

	assert len(splits) == 2
	for train_idx, test_idx in splits:
		assert frame.iloc[train_idx]["date"].max() < frame.iloc[test_idx]["date"].min()
		assert not set(blocks[train_idx]) & set(blocks[test_idx])


def test_spatiotemporal_splits_reject_non_month_start_dates():
	frame = pd.DataFrame(
		{
			"date": pd.to_datetime(["2020-01-02", "2020-02-01"]),
			"cell_id": [1, 2],
		}
	)

	with pytest.raises(SchemaError, match="month-start"):
		list(spatiotemporal_splits(frame, np.array([1, 2]), []))


def test_random_kfold_is_optimistic_for_smooth_gp_field():
	config = get_config()
	rng = np.random.default_rng(config.seed)
	xy = _grid(10, spacing=config.validation.synthetic_range_m / 2.0)
	squared_distance = spatial_cv.distance.cdist(xy, xy, metric="sqeuclidean")
	covariance = np.exp(
		-squared_distance
		/ (2 * config.validation.synthetic_range_m**2)
	)
	values = rng.multivariate_normal(
		np.zeros(len(xy)), covariance + np.eye(len(xy)) * np.finfo(float).eps
	)
	blocks = make_blocks(
		xy,
		block_size_m=config.validation.synthetic_range_m * 1.5,
		n_blocks_min=config.validation.min_blocks,
	)
	random_cv = KFold(
		n_splits=config.validation.block_n_splits,
		shuffle=True,
		random_state=config.seed,
	)
	spatial_cv_splitter = SpatialBlockCV(
		blocks,
		n_splits=config.validation.block_n_splits,
		seed=config.seed,
	)
	model = KNeighborsRegressor(n_neighbors=config.validation.demo_neighbors)
	errors: list[list[float]] = [[], []]
	for error_bucket, splitter in zip(errors, (random_cv, spatial_cv_splitter)):
		for train_idx, test_idx in splitter.split(xy):
			model.fit(xy[train_idx], values[train_idx])
			error_bucket.extend(
				(model.predict(xy[test_idx]) - values[test_idx]) ** 2
			)

	assert np.sqrt(np.mean(errors[0])) < np.sqrt(np.mean(errors[1]))
