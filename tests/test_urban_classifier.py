"""Tests for feature assembly, spatial training, and urban-fraction outputs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from affine import Affine

import urban_classifier
from config import get_config
from exceptions import DataIntegrityError, SchemaError
from urban_classifier import (
	build_feature_matrix,
	predict_class_map,
	save_outputs,
	temporal_consistency_check,
	train,
	urban_fraction,
)


@pytest.fixture
def raster_inputs() -> tuple[xr.Dataset, xr.Dataset]:
	"""Create aligned three-month spectral and static terrain features."""
	time = pd.date_range("2020-01-01", periods=3, freq="MS").values
	y = np.arange(2, dtype=float)
	x = np.arange(2, dtype=float)
	base = np.arange(12, dtype=np.float32).reshape(3, 2, 2) / 20
	spectral = xr.Dataset(
		{
			"ndvi": (("time", "y", "x"), base),
			"ndbi": (("time", "y", "x"), base * 0.5),
			"ndwi": (("time", "y", "x"), base * 0.25),
			"mndwi": (("time", "y", "x"), base * 0.2),
			"bsi": (("time", "y", "x"), base * 0.1),
		},
		coords={"time": time, "y": y, "x": x},
		attrs={"city": "dehradun"},
	)
	terrain = xr.Dataset(
		{
			"elevation": (("y", "x"), np.array([[10, 20], [30, 40]], dtype=float)),
			"slope": (("y", "x"), np.array([[1, 2], [3, 4]], dtype=float)),
			"aspect_sin": (("y", "x"), np.zeros((2, 2))),
			"aspect_cos": (("y", "x"), np.ones((2, 2))),
			"tpi_300m": (("y", "x"), np.zeros((2, 2))),
			"twi": (("y", "x"), np.ones((2, 2))),
		},
		coords={"y": y, "x": x},
	)
	return spectral, terrain


def _training_rows(monkeypatch: pytest.MonkeyPatch):
	"""Make class-separable samples with five spatial CV groups."""
	cfg = get_config()
	classifier_cfg = replace(
		cfg.classifier,
		n_estimators=20,
		min_urban_f1=0.8,
		n_splits=5,
	)
	monkeypatch.setattr(
		urban_classifier,
		"get_config",
		lambda: replace(cfg, classifier=classifier_cfg),
	)
	class_names = list(classifier_cfg.classes)
	rows: list[np.ndarray] = []
	labels: list[str] = []
	groups: list[int] = []
	rng = np.random.default_rng(cfg.seed)
	for group in range(classifier_cfg.n_splits):
		for class_index, class_name in enumerate(class_names):
			for _ in range(8):
				row = np.zeros(len(class_names), dtype=float)
				row[class_index] = 1.0
				rows.append(row + rng.normal(0.0, 0.01, len(class_names)))
				labels.append(class_name)
				groups.append(group)
	index = pd.Index(np.arange(len(rows), dtype=np.int64), name="cell_id")
	features = pd.DataFrame(rows, index=index, columns=class_names)
	return (
		features,
		pd.Series(labels, index=index),
		pd.Series(groups, index=index),
	)


def test_build_feature_matrix_includes_spectral_terrain_and_texture(raster_inputs):
	spectral, terrain = raster_inputs

	features = build_feature_matrix(spectral, terrain)

	assert isinstance(features.index, pd.MultiIndex)
	assert features.index.names == ["time", "cell_id"]
	assert {"ndvi", "dem", "slope", "tpi_300m", "twi"}.issubset(features.columns)
	assert {"ndbi_mean_3x3", "ndvi_std_3x3"}.issubset(features.columns)
	assert features.index.get_level_values("cell_id").dtype == np.dtype("int64")


def test_missing_spectral_index_raises_schema_error(raster_inputs):
	spectral, terrain = raster_inputs

	with pytest.raises(SchemaError, match="missing required features"):
		build_feature_matrix(spectral.drop_vars("ndvi"), terrain)


def test_spatial_group_training_and_prediction(monkeypatch):
	features, labels, groups = _training_rows(monkeypatch)

	artifacts = train(features, labels, groups)
	prediction = predict_class_map(artifacts, features)

	assert artifacts.metrics["cv"] == "GroupKFold"
	assert artifacts.metrics["urban_f1"] >= 0.8
	assert artifacts.trained_on
	assert prediction.shape == (len(features),)
	assert set(np.unique(prediction.values)) == set(artifacts.class_names)


def test_train_rejects_misaligned_labels(monkeypatch):
	features, labels, groups = _training_rows(monkeypatch)
	labels.index = labels.index + 1

	with pytest.raises(SchemaError, match="identical indexes"):
		train(features, labels, groups)


def test_urban_fraction_aggregates_to_target_grid():
	fine_transform = Affine(10, 0, 0, 0, -10, 40)
	target_transform = Affine(20, 0, 0, 0, -20, 40)
	fine = xr.DataArray(
		np.array(
			[
				[
					["urban", "urban", "forest", "forest"],
					["urban", "forest", "forest", "forest"],
					["water", "water", "urban", "urban"],
					["water", "water", "forest", "urban"],
				]
			],
			dtype=object,
		),
		dims=("time", "y", "x"),
		coords={
			"time": pd.to_datetime(["2020-01-01"]),
			"y": np.array([35.0, 25.0, 15.0, 5.0]),
			"x": np.array([5.0, 15.0, 25.0, 35.0]),
		},
		name="class",
	).rio.write_crs("EPSG:32644").rio.write_transform(fine_transform)
	grid = xr.Dataset(
		{"template": (("y", "x"), np.zeros((2, 2)))},
		coords={"y": np.array([30.0, 10.0]), "x": np.array([10.0, 30.0])},
	).rio.write_crs("EPSG:32644").rio.write_transform(target_transform)

	result = urban_fraction(fine, grid)

	assert result.dims == ("time", "y", "x")
	assert np.all((result.values >= 0.0) & (result.values <= 1.0))
	assert result.values[0, 0, 0] == pytest.approx(0.75)


def test_temporal_check_flags_jumps_and_urban_reversal():
	urban = xr.DataArray(
		np.array(
			[
				[[0.8, 0.1]],
				[[0.2, 0.9]],
			]
		),
		dims=("time", "y", "x"),
		coords={"time": pd.to_datetime(["2020-01-01", "2021-01-01"])},
	)

	report = temporal_consistency_check(urban)

	assert report.loc[0, "flag_jump"]
	assert report.loc[0, "flag_urban_to_non_urban"]
	assert report.loc[1, "flag_jump"]
	assert not report.loc[1, "flag_urban_to_non_urban"]


def test_temporal_check_rejects_fraction_outside_valid_range():
	urban = xr.DataArray(
		np.array([[[1.2]]]),
		dims=("time", "y", "x"),
		coords={"time": pd.to_datetime(["2020-01-01"])},
	)

	with pytest.raises(SchemaError, match="within \\[0, 1\\]"):
		temporal_consistency_check(urban)


def test_low_urban_f1_fails_integrity_threshold(monkeypatch):
	features, labels, groups = _training_rows(monkeypatch)
	cfg = get_config()
	monkeypatch.setattr(
		urban_classifier,
		"get_config",
		lambda: replace(
			cfg,
			classifier=replace(
				cfg.classifier,
				n_estimators=1,
				min_urban_f1=1.1,
			),
		),
	)

	with pytest.raises(DataIntegrityError, match="urban F1"):
		train(features, labels, groups)


def test_save_outputs_persists_raster_model_and_metrics(monkeypatch, tmp_path):
	features, labels, groups = _training_rows(monkeypatch)
	artifacts = train(features, labels, groups)
	cfg = get_config()
	paths_cfg = replace(cfg.paths, interim=tmp_path)
	monkeypatch.setattr(
		urban_classifier,
		"get_config",
		lambda: replace(cfg, paths=paths_cfg),
	)
	urban = xr.DataArray(
		np.full((1, 2, 2), 0.25),
		dims=("time", "y", "x"),
		coords={
			"time": pd.to_datetime(["2020-01-01"]),
			"y": np.arange(2),
			"x": np.arange(2),
		},
		name="urban_fraction",
	)

	outputs = save_outputs(urban, artifacts, "dehradun")

	assert set(outputs) == {"urban_fraction", "model", "metrics"}
	assert all(path.exists() for path in outputs.values())
