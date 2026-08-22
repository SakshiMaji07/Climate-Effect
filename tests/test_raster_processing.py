import numpy as np
import pytest
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling

from raster_processing import load_and_align


def _raster(values, crs, transform, name="raster"):
	return xr.DataArray(
		values,
		dims=("y", "x"),
		name=name,
	).rio.write_crs(crs).rio.write_transform(transform)


def test_different_sources_align_to_identical_target_grid():
	target_crs = CRS.from_epsg(3857)
	target_transform = Affine(1000, 0, 0, 0, -1000, 20_000)
	target_shape = (20, 20)
	source_a = _raster(
		np.ones((20, 20), dtype=np.float32),
		"EPSG:3857",
		Affine(1000, 0, 0, 0, -1000, 20_000),
	)
	source_b = _raster(
		np.full((40, 40), 2, dtype=np.float32),
		"EPSG:4326",
		Affine(0.009, 0, -0.09, 0, -0.009, 0.18),
	)

	aligned_a = load_and_align(
		source_a, target_crs, target_transform, target_shape,
		Resampling.bilinear, "continuous",
	)
	aligned_b = load_and_align(
		source_b, target_crs, target_transform, target_shape,
		Resampling.bilinear, "continuous",
	)

	assert aligned_a.rio.crs == aligned_b.rio.crs
	assert aligned_a.rio.transform() == aligned_b.rio.transform()
	assert aligned_a.shape == aligned_b.shape == target_shape


def test_load_and_align_rejects_source_without_crs():
	source = xr.DataArray(np.ones((20, 20)), dims=("y", "x"))
	with pytest.raises(ValueError, match="no CRS"):
		load_and_align(
			source, "EPSG:3857", Affine.identity(), (20, 20),
			Resampling.bilinear, "continuous",
		)


def test_categorical_bilinear_resampling_is_rejected():
	source = _raster(
		np.zeros((20, 20), dtype=np.int16), "EPSG:3857", Affine.identity()
	)
	with pytest.raises(ValueError, match="nearest"):
		load_and_align(
			source, "EPSG:3857", Affine.identity(), (20, 20),
			Resampling.bilinear, "categorical",
		)


def test_categorical_nearest_resampling_does_not_fabricate_classes():
	values = np.tile(np.array([0, 1, 2, 1], dtype=np.int16), (20, 5))
	source = _raster(values, "EPSG:3857", Affine(1, 0, 0, 0, -1, 20))
	aligned = load_and_align(
		source, "EPSG:3857", Affine(0.5, 0, 0, 0, -0.5, 20), (40, 40),
		Resampling.nearest, "categorical",
	)

	assert set(np.unique(aligned.values)).issubset({0, 1, 2})
