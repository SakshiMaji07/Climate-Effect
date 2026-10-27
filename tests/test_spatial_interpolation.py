"""Tests for horizontal interpolation and physically based downscaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import rioxarray  # noqa: F401 - registers the xarray ``rio`` accessor
import xarray as xr
from pyproj import Transformer

from config import get_config
from exceptions import SchemaError
import spatial_interpolation
from spatial_interpolation import (
	coarse_cell_reference_elevation,
	downscale_variable,
	estimate_lapse_rate,
	lapse_adjust,
	reproject_coarse_to_grid,
	to_cell_month_columns,
	_magnus_relative_humidity,
	_conserve_mean,
)


def _grid(
	x: list[float],
	y: list[float],
	crs: str | None = None,
) -> xr.Dataset:
	"""Create a projected target grid for tests."""
	target_crs = crs or get_config().grid.crs["dehradun"]
	return xr.Dataset(coords={"y": y, "x": x}).rio.write_crs(
		target_crs, inplace=False
	)


def _coarse(values: np.ndarray, units: str = "degC") -> xr.Dataset:
	"""Create a two-by-two projected monthly source grid."""
	dataset = xr.Dataset(
		{
			"T2M": xr.DataArray(
				values,
				dims=("y", "x"),
				coords={"y": [0.5, 2.5], "x": [0.5, 2.5]},
				attrs={"units": units},
			)
		}
	)
	return dataset.rio.write_crs(
		get_config().grid.crs["dehradun"], inplace=False
	)


def test_reproject_coarse_to_grid_bilinearly_interpolates_horizontal_only():
	source = _coarse(np.array([[0.0, 2.0], [2.0, 4.0]]))

	result = reproject_coarse_to_grid(source, _grid([0.5, 1.5, 2.5], [2.5, 1.5, 0.5]))

	assert result["T2M"].dims == ("y", "x")
	assert result["T2M"].shape == (3, 3)
	assert result["T2M"].sel(x=1.5, y=1.5).item() == pytest.approx(2.0)
	assert result["T2M"].attrs["units"] == "degC"


def test_reproject_rejects_unsupported_interpolation_method():
	with pytest.raises(SchemaError, match="Unsupported interpolation"):
		reproject_coarse_to_grid(
			_coarse(np.ones((2, 2))), _grid([0.5, 2.5], [2.5, 0.5]), "cubic"
		)


def test_reproject_rejects_missing_source_crs():
	source = xr.Dataset(
		{"T2M": (("y", "x"), np.ones((2, 2)))},
		coords={"y": [0.5, 2.5], "x": [0.5, 2.5]},
	)

	with pytest.raises(SchemaError, match="CRS is missing"):
		reproject_coarse_to_grid(source, _grid([0.5, 2.5], [2.5, 0.5]))


def test_reproject_transforms_geographic_source_to_projected_grid():
	source = xr.Dataset(
		{
			"T2M": xr.DataArray(
				[[0.0, 1.0], [2.0, 3.0]],
				dims=("latitude", "longitude"),
				coords={"latitude": [30.0, 31.0], "longitude": [77.0, 78.0]},
				attrs={"units": "degC"},
			)
		},
		attrs={"crs": "EPSG:4326"},
	)
	transformer = Transformer.from_crs(
		"EPSG:4326", get_config().grid.crs["dehradun"], always_xy=True
	)
	target_x, _ = transformer.transform([77.25, 77.75], [30.5, 30.5])
	_, target_y = transformer.transform([77.5, 77.5], [30.25, 30.75])
	target = _grid(list(target_x), list(target_y))

	result = reproject_coarse_to_grid(source, target)

	assert result["T2M"].shape == (2, 2)
	assert np.isfinite(result["T2M"].values).all()


def test_coarse_cell_reference_elevation_averages_dem_by_footprint():
	dem = xr.DataArray(
		np.arange(9, dtype=np.float64).reshape(3, 3),
		dims=("y", "x"),
		coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5]},
		attrs={"units": "m"},
	).rio.write_crs(get_config().grid.crs["dehradun"], inplace=False)
	source = _coarse(np.ones((2, 2)))

	result = coarse_cell_reference_elevation(dem, source)

	assert result.shape == (2, 2)
	assert result.dims == ("y", "x")
	assert result.attrs["units"] == "m"
	assert np.isfinite(result.values).all()


def test_coarse_cell_reference_rejects_ambiguous_dem_units():
	dem = xr.DataArray(
		np.ones((2, 2)),
		dims=("y", "x"),
		coords={"y": [2.5, 0.5], "x": [0.5, 2.5]},
	).rio.write_crs(get_config().grid.crs["dehradun"], inplace=False)

	with pytest.raises(SchemaError, match="explicitly be metres"):
		coarse_cell_reference_elevation(dem, _coarse(np.ones((2, 2))))


def test_lapse_adjust_gives_configured_temperature_difference_and_preserves_nodata():
	config = get_config()
	dem = xr.DataArray(
		[[0.0, 1000.0], [0.0, np.nan]],
		dims=("y", "x"),
		coords={"y": [1.5, 0.5], "x": [0.5, 1.5]},
		attrs={"units": "m"},
	)
	temperature = xr.DataArray(
		np.full((2, 2), 20.0),
		dims=("y", "x"),
		coords={"y": dem.y, "x": dem.x},
		attrs={"units": "degC"},
	)
	reference = xr.zeros_like(dem)
	rate = pd.Series(
		config.downscaling.default_lapse_rate_c_per_km,
		index=range(1, 13),
	)

	result = lapse_adjust(temperature, dem, reference, rate)

	assert result[0, 1].item() - result[0, 0].item() == pytest.approx(
		config.downscaling.default_lapse_rate_c_per_km
	)
	assert np.isnan(result[1, 1].item())


def test_lapse_adjust_rejects_ambiguous_temperature_units():
	dem = xr.DataArray(
		np.zeros((2, 2)),
		dims=("y", "x"),
		attrs={"units": "m"},
	)
	temperature = xr.DataArray(np.ones((2, 2)), dims=("y", "x"))

	with pytest.raises(SchemaError, match="explicitly in Celsius"):
		lapse_adjust(temperature, dem, dem, pd.Series([-6.5], index=[1]))


def test_station_lapse_rates_are_clamped_and_expanded_by_month():
	station = pd.DataFrame(
		{
			"date": pd.to_datetime(["2020-01-01", "2021-01-01", "2020-07-01", "2021-07-01"]),
			"temperature": [20.0, 0.0, 20.0, 0.0],
			"elevation": [0.0, 1000.0, 0.0, 1000.0],
		}
	)
	station.attrs["temperature_units"] = "degC"
	station.attrs["elevation_units"] = "m"
	coarse_temperature = xr.DataArray(
		np.ones((2, 2)), attrs={"units": "degC"}
	)
	reference = xr.DataArray(np.ones((2, 2)), attrs={"units": "m"})

	rates = estimate_lapse_rate(
		coarse_temperature,
		reference,
		station=station,
		by="season",
	)

	assert rates.loc[12] == pytest.approx(
		get_config().downscaling.lapse_rate_min_c_per_km
	)
	assert rates.loc[6] == pytest.approx(
		get_config().downscaling.lapse_rate_min_c_per_km
	)
	assert rates.index.tolist() == list(range(1, 13))


def test_cell_month_conversion_creates_canonical_spatial_columns():
	downscaled = xr.Dataset(
		{
			"t2m_coarse": (("time", "y", "x"), np.ones((2, 2, 2))),
			"t2m_downscaled": (("time", "y", "x"), np.full((2, 2, 2), 2.0)),
			"RH2M": (("time", "y", "x"), np.full((2, 2, 2), 50.0)),
		},
		coords={
			"time": pd.date_range("2020-01-01", periods=2, freq="MS"),
			"y": [150.0, 50.0],
			"x": [50.0, 150.0],
			"cell_id": (("y", "x"), [[1, 2], [3, 4]]),
		},
		attrs={"city": "dehradun"},
	)

	frame = to_cell_month_columns(downscaled)

	assert len(frame) == 8
	assert {"city", "date", "cell_id", "x", "y", "t2m_coarse", "t2m_downscaled", "RH2M"} <= set(frame)
	assert frame["cell_id"].dtype == np.int64


def test_cell_month_conversion_rejects_non_month_start_dates():
	downscaled = xr.Dataset(
		{"t2m_downscaled": (("time", "y", "x"), np.ones((1, 2, 2)))},
		coords={
			"time": [pd.Timestamp("2020-01-02")],
			"y": [1.5, 0.5],
			"x": [0.5, 1.5],
		},
	)

	with pytest.raises(SchemaError, match="month-start"):
		to_cell_month_columns(downscaled)


def test_downscale_temperature_uses_kelvin_conversion_and_lapse_adjustment(
	monkeypatch: pytest.MonkeyPatch,
):
	"""Exercise the integrated temperature path with in-memory terrain."""
	crs = get_config().grid.crs["dehradun"]
	target_grid = _grid([0.5, 1.5], [3.5, 2.5], crs)
	dem = xr.DataArray(
		[[0.0, 1000.0], [0.0, 1000.0]],
		dims=("y", "x"),
		coords={"y": [3.5, 2.5], "x": [0.5, 1.5]},
		name="dem",
		attrs={"units": "m"},
	).rio.write_crs(crs, inplace=False)
	terrain = xr.Dataset({"dem": dem})
	monkeypatch.setattr(spatial_interpolation, "_canonical_grid", lambda _city: target_grid)
	monkeypatch.setattr(spatial_interpolation, "load_terrain_stack", lambda _path: terrain)
	coarse = xr.Dataset(
		{
			"T2M": xr.DataArray(
				np.full((2, 2), 293.15),
				dims=("y", "x"),
				coords={"y": [0.5, 4.5], "x": [0.5, 4.5]},
				attrs={"units": "K"},
			)
		}
	).rio.write_crs(crs, inplace=False)

	result = downscale_variable("T2M", coarse, "dehradun")

	assert result.name == "t2m_downscaled"
	assert result.attrs["units"] == "degC"
	assert result[0, 1].item() - result[0, 0].item() == pytest.approx(
		get_config().downscaling.default_lapse_rate_c_per_km
	)


def test_downscale_temperature_rejects_missing_unit_metadata(
	monkeypatch: pytest.MonkeyPatch,
):
	crs = get_config().grid.crs["dehradun"]
	target_grid = _grid([0.5, 1.5], [3.5, 2.5], crs)
	dem = xr.DataArray(
		np.zeros((2, 2)),
		dims=("y", "x"),
		coords={"y": [3.5, 2.5], "x": [0.5, 1.5]},
		attrs={"units": "m"},
	).rio.write_crs(crs, inplace=False)
	monkeypatch.setattr(spatial_interpolation, "_canonical_grid", lambda _city: target_grid)
	monkeypatch.setattr(
		spatial_interpolation, "load_terrain_stack", lambda _path: xr.Dataset({"dem": dem})
	)
	coarse = xr.Dataset(
		{
			"T2M": xr.DataArray(
				np.ones((2, 2)),
				dims=("y", "x"),
				coords={"y": [0.5, 4.5], "x": [0.5, 4.5]},
			)
		}
	).rio.write_crs(crs, inplace=False)

	with pytest.raises(SchemaError, match="explicitly be K or"):
		downscale_variable("T2M", coarse, "dehradun")


def test_magnus_humidity_recomputation_preserves_coarse_dew_point():
	adjusted_temperature = xr.DataArray(
		[[10.0]],
		dims=("y", "x"),
		attrs={"units": "degC"},
	)
	reference_temperature = xr.DataArray(
		[[20.0]],
		dims=("y", "x"),
		attrs={"units": "degC"},
	)
	coarse_humidity = xr.DataArray(
		[[50.0]],
		dims=("y", "x"),
		attrs={"units": "%"},
	)

	result = _magnus_relative_humidity(
		adjusted_temperature,
		coarse_humidity,
		reference_temperature,
	)

	assert result.item() > coarse_humidity.item()
	assert result.attrs["humidity_method"] == "dewpoint_preserving_magnus"


def test_precipitation_conservation_rescales_each_represented_coarse_cell():
	crs = get_config().grid.crs["dehradun"]
	target = _grid([0.5, 1.0, 2.0, 2.5], [2.5, 2.0, 1.0, 0.5], crs)
	source = xr.DataArray(
		[[1.0, 2.0], [3.0, 4.0]],
		dims=("y", "x"),
		coords={"y": [0.5, 2.5], "x": [0.5, 2.5]},
	)
	result = xr.DataArray(
		np.full((4, 4), 10.0),
		dims=("y", "x"),
		coords={"y": target.y, "x": target.x},
	)

	conserved = _conserve_mean(
		result, source, target, spatial_interpolation._crs_of(source.rio.write_crs(crs)), False
	)

	assert conserved.sel(x=[0.5, 1.0], y=[2.5, 2.0]).mean().item() == pytest.approx(3.0)
	assert conserved.sel(x=[2.0, 2.5], y=[2.5, 2.0]).mean().item() == pytest.approx(4.0)
