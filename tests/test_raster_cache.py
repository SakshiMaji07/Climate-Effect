import numpy as np
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr
from affine import Affine

import config
from raster_cache import (
	assert_round_trip,
	get_or_compute,
	load_from_geotiff,
	load_from_zarr,
	save_to_geotiff,
	save_to_zarr,
)


def _raster():
	return xr.DataArray(
		np.arange(400, dtype=np.float32).reshape(20, 20),
		dims=("y", "x"),
		name="temperature",
		attrs={"units": "degC", "description": "synthetic"},
	).rio.write_crs("EPSG:32643").rio.write_transform(
		Affine(10, 0, 500_000, 0, -10, 3_000_000)
	).rio.write_nodata(-9999)


def test_geotiff_and_zarr_round_trip_metadata(tmp_path):
	raster = _raster()
	dataset = xr.Dataset({raster.name: raster}, attrs={"source": "synthetic"})

	geotiff_path = tmp_path / "temperature.tif"
	save_to_geotiff(raster, str(geotiff_path))
	reloaded_geotiff = load_from_geotiff(str(geotiff_path))
	assert_round_trip(raster, reloaded_geotiff)

	zarr_path = tmp_path / "temperature.zarr"
	save_to_zarr(dataset, str(zarr_path))
	reloaded_zarr = load_from_zarr(str(zarr_path))
	assert_round_trip(dataset, reloaded_zarr)


def test_get_or_compute_does_not_recompute_existing_cache(tmp_path, monkeypatch):
	config.CACHE_DIR = str(tmp_path)
	computed = _raster().to_dataset()
	call_count = 0

	def compute():
		nonlocal call_count
		call_count += 1
		return computed

	get_or_compute("synthetic_temperature_10m_static", compute)
	get_or_compute("synthetic_temperature_10m_static", compute)

	assert call_count == 1