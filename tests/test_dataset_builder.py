"""Tests for strict city-month and cell-month dataset construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr

from config import get_config
from dataset_builder import (
	CELL_MONTH_SCHEMA,
	CITY_MONTH_SCHEMA,
	_validate_city_month,
	build_cell_month,
	build_city_month,
	_write_cell_partitioned,
	_write_city_parquet,
)
from exceptions import SchemaError


@pytest.fixture
def raster_sources() -> tuple[pd.DataFrame, xr.DataArray, xr.Dataset, xr.Dataset, xr.Dataset]:
	"""Create climate and raster inputs on the configured Dehradun grid."""
	city = "dehradun"
	config = get_config()
	dates = pd.date_range(
		"2000-01-01", periods=24, freq="MS"
	).as_unit("ns")
	month = np.arange(len(dates))
	tmin = 4.0 + np.sin(2 * np.pi * month / 12)
	tmax = tmin + 10 + 0.3 * np.cos(2 * np.pi * month / 12)
	climate = pd.DataFrame(
		{
			"date": dates,
			"DTR": tmax - tmin,
			"TMAX": tmax,
			"TMIN": tmin,
			"LST": 15 + np.sin(2 * np.pi * month / 12),
			"RH2M": 55 + 8 * np.cos(2 * np.pi * month / 12),
			"PRECIP": 20 + 4 * np.sin(2 * np.pi * month / 12),
		}
	)
	transform = config.grid.transform[city]
	x = transform.c + transform.a * (np.arange(2) + 0.5)
	y = transform.f + transform.e * (np.arange(2) + 0.5)
	coords = {"time": dates, "y": y, "x": x}
	uf = xr.DataArray(
		np.full((24, 2, 2), 0.4, dtype=np.float32),
		dims=("time", "y", "x"),
		coords=coords,
		name="urban_fraction",
	).rio.write_crs(config.grid.crs[city]).rio.write_transform(transform)
	climate_cells = xr.Dataset(
		{
			"t2m_coarse": (("time", "y", "x"), np.full((24, 2, 2), 12.0)),
			"RH2M": (("time", "y", "x"), np.full((24, 2, 2), 55.0)),
			"PRECIP": (("time", "y", "x"), np.full((24, 2, 2), 20.0)),
			"LST": (("time", "y", "x"), np.full((24, 2, 2), 15.0)),
		},
		coords=coords,
	).rio.write_crs(config.grid.crs[city]).rio.write_transform(transform)
	indices = xr.Dataset(
		{
			"ndvi": (("time", "y", "x"), np.full((24, 2, 2), 0.4)),
			"ndbi": (("time", "y", "x"), np.full((24, 2, 2), 0.1)),
			"bsi": (("time", "y", "x"), np.full((24, 2, 2), 0.2)),
		},
		coords=coords,
	).rio.write_crs(config.grid.crs[city]).rio.write_transform(transform)
	terrain = xr.Dataset(
		{
			"elevation": (("y", "x"), np.full((2, 2), 400.0)),
			"slope": (("y", "x"), np.full((2, 2), 5.0)),
			"aspect_sin": (("y", "x"), np.zeros((2, 2))),
			"aspect_cos": (("y", "x"), np.ones((2, 2))),
			"tpi_300m": (("y", "x"), np.full((2, 2), 1.0)),
			"tpi_3000m": (("y", "x"), np.full((2, 2), 2.0)),
			"twi": (("y", "x"), np.full((2, 2), 3.0)),
			"solar_radiation_proxy": (("y", "x"), np.full((2, 2), 4.0)),
		},
		coords={"y": y, "x": x},
	).rio.write_crs(config.grid.crs[city]).rio.write_transform(transform)
	return climate, uf, climate_cells, indices, terrain


def test_build_city_month_has_exact_schema_and_precip_anomaly(raster_sources):
	climate, uf, *_ = raster_sources

	result = build_city_month(climate, uf, "dehradun")

	assert tuple(result.columns) == tuple(CITY_MONTH_SCHEMA)
	assert result["DTR"].equals(result["TMAX"] - result["TMIN"])
	assert result["n_valid_cells"].eq(4).all()
	assert result["precip_anom"].notna().all()
	assert result.attrs["climatology_years"] == get_config().dataset.climatology_years


def test_corrupt_dtr_is_rejected(raster_sources):
	climate, uf, *_ = raster_sources
	climate.loc[0, "DTR"] += 1

	with pytest.raises(SchemaError, match="DTR must equal"):
		build_city_month(climate, uf, "dehradun")


def test_missing_climate_column_is_rejected(raster_sources):
	climate, uf, *_ = raster_sources

	with pytest.raises(SchemaError, match="Missing required columns"):
		build_city_month(climate.drop(columns="RH2M"), uf, "dehradun")


def test_city_month_validator_rejects_duplicate_primary_key(raster_sources):
	climate, uf, *_ = raster_sources
	result = build_city_month(climate, uf, "dehradun")
	duplicated = pd.concat([result, result.iloc[[0]]], ignore_index=True)

	with pytest.raises(SchemaError, match="duplicate"):
		_validate_city_month(duplicated)


def test_build_cell_month_emits_long_schema_and_nullable_future_fields(raster_sources):
	_, uf, climate_cells, indices, terrain = raster_sources

	result = build_cell_month(
		climate_cells, uf, indices, terrain, "dehradun"
	)

	assert tuple(result.columns) == tuple(CELL_MONTH_SCHEMA)
	assert len(result) == 24 * 4
	assert result["cell_id"].dtype == np.dtype("int64")
	assert result["urban_fraction"].between(0, 1).all()
	assert result["t2m_downscaled"].isna().all()
	assert result["spatial_block"].isna().all()


def test_urban_fraction_out_of_range_is_rejected(raster_sources):
	climate, uf, *_ = raster_sources
	invalid = uf.copy()
	invalid.values[0, 0, 0] = 1.2

	with pytest.raises(SchemaError, match="outside \\[0, 1\\]"):
		build_city_month(climate, invalid, "dehradun")


def test_build_cell_month_rejects_grid_shape_mismatch(raster_sources):
	_, uf, climate_cells, indices, terrain = raster_sources
	bad_terrain = terrain.isel(x=[0])

	with pytest.raises(SchemaError, match="shape does not match"):
		build_cell_month(climate_cells, uf, indices, bad_terrain, "dehradun")


def test_parquet_writers_preserve_schema_and_partition_by_city_year(
	raster_sources, tmp_path
):
	climate, uf, climate_cells, indices, terrain = raster_sources
	city_frame = build_city_month(climate, uf, "dehradun")
	cell_frame = build_cell_month(
		climate_cells, uf, indices, terrain, "dehradun"
	)
	city_path = tmp_path / "city_month.parquet"
	cell_path = tmp_path / "cell_month"

	_write_city_parquet(city_frame, city_path, "test-commit")
	_write_cell_partitioned(cell_frame, cell_path, "test-commit")

	import pyarrow.parquet as pq

	city_table = pq.read_table(city_path)
	assert tuple(city_table.column_names) == tuple(CITY_MONTH_SCHEMA)
	assert b"climatology_years" in city_table.schema.metadata
	partition_files = list(cell_path.glob("city=dehradun/year=*/part-*.parquet"))
	assert partition_files
	cell_table = pq.read_table(partition_files[0])
	assert tuple(cell_table.column_names) == tuple(CELL_MONTH_SCHEMA)
