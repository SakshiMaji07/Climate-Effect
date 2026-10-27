"""Horizontal reprojection and terrain-aware climate downscaling."""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

from config import get_config
from exceptions import ConfigError, DataIntegrityError, SchemaError
from logger import get_logger
from raster_processing import load_grid, load_terrain_stack

logger = get_logger(__name__)


def _spatial_dimensions(
	data: xr.DataArray | xr.Dataset,
) -> tuple[str, str, str, str]:
	"""Resolve y/x dimension and coordinate names for a regular raster."""
	x_name = next((name for name in ("x", "lon", "longitude") if name in data.dims), None)
	y_name = next((name for name in ("y", "lat", "latitude") if name in data.dims), None)
	if x_name is None or y_name is None:
		message = "Raster must have regular x/y or longitude/latitude dimensions."
		logger.error(message)
		raise SchemaError(message)
	x_coord = x_name if x_name in data.coords else ""
	y_coord = y_name if y_name in data.coords else ""
	if not x_coord or not y_coord:
		message = "Raster spatial dimensions must have coordinate values."
		logger.error(message)
		raise SchemaError(message)
	return y_name, x_name, y_coord, x_coord


def _crs_of(data: xr.DataArray | xr.Dataset) -> CRS:
	"""Read a raster CRS from rioxarray metadata or explicit attrs."""
	crs_value = data.rio.crs
	if crs_value is None:
		crs_value = data.attrs.get("crs") or data.attrs.get("source_crs")
	if crs_value is None:
		message = "Raster CRS is missing; horizontal reprojection is ambiguous."
		logger.error(message)
		raise SchemaError(message)
	try:
		return CRS.from_user_input(crs_value)
	except (TypeError, ValueError) as error:
		message = f"Raster CRS metadata is invalid: {crs_value!r}."
		logger.error(message)
		raise SchemaError(message) from error


def _axis_values(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
	"""Validate and sort one regularly sampled coordinate axis."""
	axis = np.asarray(values, dtype=np.float64)
	if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
		message = f"Coordinate axis {name!r} requires at least two finite values."
		logger.error(message)
		raise SchemaError(message)
	order = np.argsort(axis)
	sorted_axis = axis[order]
	if np.any(np.diff(sorted_axis) <= 0):
		message = f"Coordinate axis {name!r} must contain unique values."
		logger.error(message)
		raise SchemaError(message)
	return sorted_axis, order


def _target_coordinates(
	grid: xr.Dataset | xr.DataArray,
) -> tuple[np.ndarray, np.ndarray, str, str, CRS, Affine]:
	"""Resolve target center coordinates and georeferencing."""
	y_dim, x_dim, y_name, x_name = _spatial_dimensions(grid)
	x = np.asarray(grid.coords[x_name].values, dtype=np.float64)
	y = np.asarray(grid.coords[y_name].values, dtype=np.float64)
	if x.ndim != 1 or y.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
		message = "Target grid coordinates must be finite one-dimensional axes."
		logger.error(message)
		raise SchemaError(message)
	crs = _crs_of(grid)
	transform = grid.rio.transform(recalc=True)
	return x, y, y_dim, x_dim, crs, transform


def _coarse_source_arrays(
	data: xr.DataArray,
	source_y: str,
	source_x: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], dict[str, np.ndarray]]:
	"""Return a variable's values in ascending source y/x coordinate order."""
	other_dims = tuple(dim for dim in data.dims if dim not in {source_y, source_x})
	ordered = data.transpose(*other_dims, source_y, source_x)
	y_axis, y_order = _axis_values(data.coords[source_y].values, source_y)
	x_axis, x_order = _axis_values(data.coords[source_x].values, source_x)
	values = np.asarray(ordered.values, dtype=np.float64)
	values = np.take(np.take(values, y_order, axis=-2), x_order, axis=-1)
	other_coords = {
		dim: np.asarray(data.coords[dim].values)
		for dim in other_dims
		if dim in data.coords
	}
	return y_axis, x_axis, values, other_dims, other_coords


def _interpolate_values(
	values: np.ndarray,
	source_y: np.ndarray,
	source_x: np.ndarray,
	query_x: np.ndarray,
	query_y: np.ndarray,
	method: str,
) -> np.ndarray:
	"""Interpolate leading-dimensional raster slices at transformed points."""
	points = np.column_stack((query_y.ravel(), query_x.ravel()))
	leading_shape = values.shape[:-2]
	output = np.empty((*leading_shape, *query_x.shape), dtype=np.float64)
	for index in np.ndindex(leading_shape or (1,)):
		slice_index: tuple[int, ...] = () if not leading_shape else index
		interpolator = RegularGridInterpolator(
			(source_y, source_x),
			values[slice_index],
			method=method,
			bounds_error=False,
			fill_value=np.nan,
		)
		output[slice_index] = interpolator(points).reshape(query_x.shape)
	return output


def reproject_coarse_to_grid(
	coarse: xr.Dataset,
	grid: xr.Dataset,
	method: str = "bilinear",
) -> xr.Dataset:
	"""Interpolate coarse raster variables horizontally onto a target grid.

	Temporal and other non-spatial dimensions are retained unchanged. Source
	and target CRS metadata are mandatory; this function does not interpolate
	or resample along time.

	Args:
		coarse: Source dataset with regular spatial coordinates.
		grid: Target grid with regular spatial coordinates and a CRS.
		method: ``bilinear``/``linear`` or ``nearest`` interpolation.

	Returns:
		Dataset on the target grid with source attributes preserved.

	Raises:
		SchemaError: If coordinates, CRS, or interpolation method are invalid.
		DataIntegrityError: If interpolation produces no valid target values.
	"""
	started = time.perf_counter()
	logger.info(
		"Starting coarse-grid interpolation (variables=%d).", len(coarse.data_vars)
	)
	if method not in {"bilinear", "linear", "nearest"}:
		message = f"Unsupported interpolation method {method!r}."
		logger.error(message)
		raise SchemaError(message)
	source_y, source_x, _, _ = _spatial_dimensions(coarse)
	target_x, target_y, target_y_dim, target_x_dim, target_crs, transform = (
		_target_coordinates(grid)
	)
	source_crs = _crs_of(coarse)
	_axis_values(coarse.coords[source_y].values, source_y)
	_axis_values(coarse.coords[source_x].values, source_x)
	target_x_mesh, target_y_mesh = np.meshgrid(target_x, target_y)
	transformer = Transformer.from_crs(target_crs, source_crs, always_xy=True)
	query_x, query_y = transformer.transform(target_x_mesh, target_y_mesh)
	interpolation_method = "linear" if method == "bilinear" else method
	output_vars: dict[str, xr.DataArray] = {}
	for name, variable in coarse.data_vars.items():
		if source_y not in variable.dims or source_x not in variable.dims:
			output_vars[name] = variable.copy(deep=False)
			continue
		y_axis, x_axis, values, other_dims, other_coords = _coarse_source_arrays(
			variable, source_y, source_x
		)
		interpolated = _interpolate_values(
			values, y_axis, x_axis, query_x, query_y, interpolation_method
		)
		if not np.isfinite(interpolated).any():
			message = f"Interpolation of {name!r} produced no valid target cells."
			logger.error(message)
			raise DataIntegrityError(message)
		output_vars[name] = xr.DataArray(
			interpolated,
			dims=(*other_dims, target_y_dim, target_x_dim),
			coords={
				**other_coords,
				target_y_dim: target_y,
				target_x_dim: target_x,
			},
			name=name,
			attrs=variable.attrs.copy(),
		)
	result = xr.Dataset(output_vars, attrs=coarse.attrs.copy())
	result.attrs["source_crs"] = str(source_crs)
	result = result.rio.write_crs(target_crs, inplace=False)
	result = result.rio.set_spatial_dims(x_dim=target_x_dim, y_dim=target_y_dim, inplace=False)
	result = result.rio.write_transform(transform, inplace=False)
	logger.info(
		"Finished coarse-grid interpolation (variables=%d, target_cells=%d) in %.3fs.",
		len(output_vars),
		len(target_x) * len(target_y),
		time.perf_counter() - started,
	)
	return result


def _coarse_cell_indices(
	x: np.ndarray,
	y: np.ndarray,
	coarse_x: np.ndarray,
	coarse_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
	"""Assign projected coordinates to nearest regular-grid cell footprints."""
	def edges(axis: np.ndarray) -> np.ndarray:
		midpoints = (axis[:-1] + axis[1:]) / 2.0
		return np.concatenate(
			(
				[axis[0] - (midpoints[0] - axis[0])],
				midpoints,
				[axis[-1] + (axis[-1] - midpoints[-1])],
			)
		)

	x_edges, y_edges = edges(coarse_x), edges(coarse_y)
	x_index = np.searchsorted(x_edges, x, side="right") - 1
	y_index = np.searchsorted(y_edges, y, side="right") - 1
	valid = (
		(x_index >= 0)
		& (x_index < len(coarse_x))
		& (y_index >= 0)
		& (y_index < len(coarse_y))
	)
	x_index = np.where(valid, x_index, -1)
	y_index = np.where(valid, y_index, -1)
	return y_index, x_index


def _dem_centers_in_crs(
	dem: xr.DataArray,
	target_crs: CRS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]:
	"""Return DEM pixel-center meshes projected into a target CRS."""
	dem_y, dem_x, dem_y_coord, dem_x_coord = _spatial_dimensions(dem)
	dem_crs = _crs_of(dem)
	x_axis = np.asarray(dem.coords[dem_x_coord].values, dtype=np.float64)
	y_axis = np.asarray(dem.coords[dem_y_coord].values, dtype=np.float64)
	x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
	transformer = Transformer.from_crs(dem_crs, target_crs, always_xy=True)
	projected_x, projected_y = transformer.transform(x_mesh, y_mesh)
	return (
		np.asarray(projected_x),
		np.asarray(projected_y),
		x_axis,
		y_axis,
		dem_y,
		dem_x,
	)


def _mask_dem_nodata(dem: xr.DataArray) -> xr.DataArray:
	"""Represent finite raster nodata sentinels as NaN without mutating input."""
	values = dem.astype(np.float64)
	nodata = dem.rio.nodata
	if nodata is None:
		nodata = dem.attrs.get("nodata")
	if nodata is not None and np.isfinite(nodata):
		values = values.where(~np.isclose(values, float(nodata)))
	return values.where(np.isfinite(values))


def coarse_cell_reference_elevation(
	dem: xr.DataArray,
	coarse: xr.Dataset,
) -> xr.DataArray:
	"""Calculate mean DEM elevation inside each coarse-cell footprint.

	By default the reference is the area-weighted DEM mean for each coarse
	cell, using regular projected DEM pixels as equal-area samples. If
	``use_model_orography`` is enabled, the configured model-orography field
	is used instead; geopotential (m²/s²) is converted to metres.

	Args:
		dem: Projected elevation raster in metres.
		coarse: Coarse dataset defining the footprints.

	Returns:
		Reference elevation in metres on the coarse y/x coordinates.

	Raises:
		SchemaError: If DEM/model-orography units or CRS are ambiguous.
		DataIntegrityError: If no DEM cells overlap coarse footprints.
	"""
	started = time.perf_counter()
	logger.info(
		"Starting coarse-cell reference-elevation aggregation (DEM_cells=%d).",
		dem.size,
	)
	config = get_config().downscaling
	coarse_y, coarse_x, y_coord, x_coord = _spatial_dimensions(coarse)
	y_axis, y_order = _axis_values(coarse.coords[y_coord].values, y_coord)
	x_axis, x_order = _axis_values(coarse.coords[x_coord].values, x_coord)
	if config.use_model_orography:
		name = config.model_orography_variable
		if name not in coarse:
			message = f"Configured model-orography variable {name!r} is absent."
			logger.error(message)
			raise SchemaError(message)
		orography = coarse[name]
		units = str(orography.attrs.get("units", "")).strip().lower()
		if units in {"m", "meter", "meters", "metre", "metres"}:
			values = np.asarray(orography.values, dtype=np.float64)
		elif units in {"m2 s-2", "m^2 s^-2", "m**2 s**-2", "j kg-1"}:
			values = np.asarray(orography.values, dtype=np.float64) / 9.80665
		else:
			message = f"Model orography units are ambiguous or unsupported: {units!r}."
			logger.error(message)
			raise SchemaError(message)
		if values.ndim > 2:
			values = np.nanmean(values, axis=tuple(range(values.ndim - 2)))
		values = np.take(np.take(values, y_order, axis=-2), x_order, axis=-1)
		result = xr.DataArray(
			values,
			dims=(coarse_y, coarse_x),
			coords={coarse_y: y_axis, coarse_x: x_axis},
			name="reference_elevation",
			attrs={"units": "m", "source": "model_orography"},
		)
		logger.info(
			"Finished model-orography reference selection (coarse_cells=%d) in %.3fs.",
			result.size,
			time.perf_counter() - started,
		)
		return result
	dem_units = str(dem.attrs.get("units", "")).strip().lower()
	if dem_units not in {"m", "meter", "meters", "metre", "metres"}:
		message = f"DEM units must explicitly be metres, got {dem_units!r}."
		logger.error(message)
		raise SchemaError(message)
	dem_crs = _crs_of(dem)
	if (
		not dem_crs.is_projected
		or not dem_crs.axis_info
		or any(
			not np.isclose(axis.unit_conversion_factor, 1.0)
			for axis in dem_crs.axis_info[:2]
		)
	):
		message = "DEM CRS must be projected in metres for area-weighted aggregation."
		logger.error(message)
		raise SchemaError(message)
	coarse_crs = _crs_of(coarse)
	dem = _mask_dem_nodata(dem)
	projected_x, projected_y, dem_x_axis, dem_y_axis, dem_y_dim, dem_x_dim = (
		_dem_centers_in_crs(dem, coarse_crs)
	)
	y_index, x_index = _coarse_cell_indices(
		projected_x.ravel(), projected_y.ravel(), x_axis, y_axis
	)
	dem_values = np.asarray(
		dem.transpose(dem_y_dim, dem_x_dim).values,
		dtype=np.float64,
	)
	if dem_values.shape != (len(dem_y_axis), len(dem_x_axis)):
		message = "DEM must be a two-dimensional array on its x/y coordinates."
		logger.error(message)
		raise SchemaError(message)
	valid = (
		(y_index >= 0)
		& (x_index >= 0)
		& np.isfinite(dem_values.ravel())
	)
	counts = np.zeros((len(y_axis), len(x_axis)), dtype=np.int64)
	sums = np.zeros_like(counts, dtype=np.float64)
	np.add.at(counts, (y_index[valid], x_index[valid]), 1)
	np.add.at(sums, (y_index[valid], x_index[valid]), dem_values.ravel()[valid])
	means = np.divide(
		sums,
		counts,
		out=np.full_like(sums, np.nan),
		where=counts > 0,
	)
	if not np.isfinite(means).any():
		message = "No valid DEM pixels overlap the coarse-cell footprints."
		logger.error(message)
		raise DataIntegrityError(message)
	result = xr.DataArray(
		means,
		dims=(coarse_y, coarse_x),
		coords={coarse_y: y_axis, coarse_x: x_axis},
		name="reference_elevation",
		attrs={"units": "m", "source": "area_mean_dem"},
	)
	logger.info(
		"Finished DEM reference aggregation (coarse_cells=%d) in %.3fs.",
		result.size,
		time.perf_counter() - started,
	)
	return result


def estimate_lapse_rate(
	coarse_t: xr.DataArray,
	ref_elev: xr.DataArray,
	station: pd.DataFrame | None = None,
	by: Literal["month", "season"] = "month",
) -> pd.Series:
	"""Select monthly lapse rates, optionally estimated from station observations.

	Station data must contain ``date``, ``temperature``, and ``elevation``;
	temperature units are declared by ``station.attrs['temperature_units']``.
	Separate monthly OLS slopes are fit only when a month has two or more
	distinct station elevations, then clamped to configured bounds.

	Args:
		coarse_t: Coarse temperature field, used to validate numeric input.
		ref_elev: Coarse-cell reference elevations in metres.
		station: Optional station observations.
		by: Grouping used to estimate station rates; seasonal rates are expanded
			to their constituent calendar months.

	Returns:
		Series indexed by calendar month (1..12), in °C/km.

	Raises:
		SchemaError: If station columns, units, or grouping are invalid.
	"""
	started = time.perf_counter()
	config = get_config().downscaling
	if by not in {"month", "season"}:
		message = f"Unsupported lapse-rate grouping {by!r}."
		logger.error(message)
		raise SchemaError(message)
	if (
		config.lapse_rate_min_c_per_km > config.lapse_rate_max_c_per_km
		or not (
			config.lapse_rate_min_c_per_km
			<= config.default_lapse_rate_c_per_km
			<= config.lapse_rate_max_c_per_km
		)
	):
		message = "Configured lapse-rate default and bounds are inconsistent."
		logger.error(message)
		raise ConfigError(message)
	logger.info(
		"Starting lapse-rate estimation (station_rows=%d, grouping=%s).",
		0 if station is None else len(station),
		by,
	)
	if not np.issubdtype(coarse_t.dtype, np.number) or not np.issubdtype(
		ref_elev.dtype, np.number
	):
		message = "Temperature and reference elevation must be numeric."
		logger.error(message)
		raise SchemaError(message)
	if str(coarse_t.attrs.get("units", "")).strip().lower() not in {
		"k", "kelvin", "c", "degc", "°c", "celsius"
	}:
		message = "coarse_t units must explicitly be Kelvin or Celsius."
		logger.error(message)
		raise SchemaError(message)
	if str(ref_elev.attrs.get("units", "")).strip().lower() not in {
		"m", "meter", "meters", "metre", "metres"
	}:
		message = "ref_elev units must explicitly be metres."
		logger.error(message)
		raise SchemaError(message)
	rates = pd.Series(
		config.default_lapse_rate_c_per_km,
		index=pd.Index(range(1, 13), name="month"),
		dtype=np.float64,
		name="lapse_rate_c_per_km",
	)
	if station is None:
		logger.info(
			"Finished lapse-rate selection (months=12) in %.3fs.",
			time.perf_counter() - started,
		)
		return rates
	required = {"date", "temperature", "elevation"}
	if not required.issubset(station.columns):
		message = f"Station data is missing columns: {sorted(required - set(station.columns))}."
		logger.error(message)
		raise SchemaError(message)
	units = str(station.attrs.get("temperature_units", "")).strip().lower()
	if units not in {"c", "degc", "°c", "celsius", "k", "kelvin"}:
		message = "Station temperature_units must explicitly be Celsius or Kelvin."
		logger.error(message)
		raise SchemaError(message)
	elevation_units = str(station.attrs.get("elevation_units", "")).strip().lower()
	if elevation_units not in {"m", "meter", "meters", "metre", "metres"}:
		message = "Station elevation_units must explicitly be metres."
		logger.error(message)
		raise SchemaError(message)
	observations = station.loc[:, ["date", "temperature", "elevation"]].copy()
	observations["date"] = pd.to_datetime(observations["date"], errors="coerce")
	if observations["date"].isna().any():
		message = "Station dates contain invalid values."
		logger.error(message)
		raise SchemaError(message)
	for column in ("temperature", "elevation"):
		observations[column] = pd.to_numeric(observations[column], errors="coerce")
	observations = observations.dropna()
	observations = observations.loc[
		np.isfinite(observations[["temperature", "elevation"]]).all(axis=1)
	]
	if units in {"k", "kelvin"}:
		observations["temperature"] -= 273.15
	observations["month"] = observations["date"].dt.month
	if by == "season":
		observations["group"] = ((observations["month"] % 12) // 3).astype(int)
		season_groups = {
			0: (12, 1, 2),
			1: (3, 4, 5),
			2: (6, 7, 8),
			3: (9, 10, 11),
		}
		group_values = observations["group"]
	else:
		group_values = observations["month"]
	for group, subset in observations.groupby(group_values):
		if (
			subset["elevation"].nunique() < 2
			or len(subset) < config.minimum_station_samples
		):
			logger.warning("Insufficient station elevation range for lapse group %s.", group)
			continue
		slope = float(np.polyfit(subset["elevation"], subset["temperature"], 1)[0] * 1000.0)
		clamped = float(
			np.clip(
				slope,
				config.lapse_rate_min_c_per_km,
				config.lapse_rate_max_c_per_km,
			)
		)
		if by == "month":
			rates.loc[int(group)] = clamped
		else:
			for month in season_groups[int(group)]:
				rates.loc[month] = clamped
	logger.info(
		"Finished station lapse-rate estimation (months=12) in %.3fs.",
		time.perf_counter() - started,
	)
	return rates


def _monthly_rate(
	temperature: xr.DataArray,
	rate: pd.Series,
) -> xr.DataArray | float:
	"""Convert a month-indexed rate series to a temperature-shaped rate."""
	if "time" not in temperature.dims or "time" not in temperature.coords:
		return float(rate.mean())
	months = pd.DatetimeIndex(pd.to_datetime(temperature.coords["time"].values)).month
	values = rate.reindex(months).to_numpy(dtype=np.float64)
	return xr.DataArray(
		values,
		dims=("time",),
		coords={"time": temperature.coords["time"]},
		name=rate.name,
	)


def lapse_adjust(
	t_coarse_on_grid: xr.DataArray,
	dem: xr.DataArray,
	ref_elev: xr.DataArray,
	rate: pd.Series,
) -> xr.DataArray:
	"""Apply elevation lapse correction, preserving DEM nodata.

	Args:
		t_coarse_on_grid: Coarse temperature interpolated to the target grid in °C.
		dem: Target-grid DEM in metres.
		ref_elev: Target-grid or broadcastable coarse-cell reference elevation in m.
		rate: Lapse rate by calendar month in °C/km.

	Returns:
		Lapse-adjusted temperature in °C.

	Raises:
		SchemaError: If units or grid dimensions are inconsistent.
	"""
	if str(t_coarse_on_grid.attrs.get("units", "")).lower() not in {
		"c", "degc", "°c", "celsius"
	}:
		message = "lapse_adjust requires temperature explicitly in Celsius."
		logger.error(message)
		raise SchemaError(message)
	if str(dem.attrs.get("units", "")).lower() not in {
		"m", "meter", "meters", "metre", "metres"
	}:
		message = "lapse_adjust requires DEM elevation explicitly in metres."
		logger.error(message)
		raise SchemaError(message)
	if (
		not rate.index.is_unique
		or set(rate.index) != set(range(1, 13))
		or not np.isfinite(rate.to_numpy(dtype=np.float64)).all()
	):
		message = "Lapse-rate Series must contain finite values for months 1 through 12."
		logger.error(message)
		raise SchemaError(message)
	if not {"x", "y"}.issubset(t_coarse_on_grid.dims) or not {
		"x", "y"
	}.issubset(dem.dims):
		message = "Temperature and DEM must have x/y dimensions."
		logger.error(message)
		raise SchemaError(message)
	rate_values = _monthly_rate(t_coarse_on_grid, rate)
	result = t_coarse_on_grid + rate_values * (dem - ref_elev) / 1000.0
	result = result.where(np.isfinite(dem))
	result.name = "t2m_downscaled"
	result.attrs = t_coarse_on_grid.attrs.copy()
	result.attrs["units"] = "degC"
	result.attrs["downscaling_method"] = "lapse_rate"
	return result


def add_topoclimatic_residual_terms(
	temperature: xr.DataArray,
	terrain: xr.Dataset,
	city: str,
) -> xr.DataArray:
	"""Optionally add configured slope/aspect/solar radiative residual effects.

	The correction is disabled by default. When enabled it is the configured
	coefficient times the terrain solar-radiation anomaly in configured units.

	Args:
		temperature: Lapse-adjusted temperature in °C.
		terrain: Target-grid terrain stack with ``solar_rad`` or
			``solar_radiation_proxy`` and optional slope/aspect fields.
		city: Canonical city key used for validation/logging.

	Returns:
		Temperature with optional topoclimatic residual adjustment.

	Raises:
		SchemaError: If the city or required enabled terrain field is invalid.
	"""
	if city not in get_config().grid.crs:
		message = f"Unsupported city {city!r}."
		logger.error(message)
		raise SchemaError(message)
	config = get_config().downscaling
	if not config.topoclimatic_enabled:
		return temperature
	if "slope" not in terrain or not {
		"aspect_sin",
		"aspect_cos",
	}.issubset(terrain.data_vars):
		message = (
			"Topoclimatic adjustment requires slope and aspect_sin/aspect_cos "
			"terrain layers."
		)
		logger.error(message)
		raise SchemaError(message)
	solar_name = next(
		(name for name in ("solar_rad", "solar_radiation_proxy") if name in terrain),
		None,
	)
	if solar_name is None:
		message = "Topoclimatic adjustment requires a solar radiation layer."
		logger.error(message)
		raise SchemaError(message)
	if config.topoclimatic_solar_scale <= 0:
		message = "topoclimatic_solar_scale must be positive."
		logger.error(message)
		raise SchemaError(message)
	solar = terrain[solar_name]
	anomaly = solar - solar.mean(dim=("y", "x"), skipna=True)
	adjusted = temperature + (
		config.topoclimatic_coefficient * anomaly / config.topoclimatic_solar_scale
	)
	adjusted.attrs = temperature.attrs.copy()
	adjusted.attrs["topoclimatic_adjustment"] = True
	adjusted.attrs["topoclimatic_basis"] = (
		"solar_radiation_anomaly_from_slope_aspect_terrain"
	)
	return adjusted


def _temperature_celsius(values: xr.DataArray) -> xr.DataArray:
	"""Convert explicitly unit-tagged Kelvin or Celsius values to Celsius."""
	units = str(values.attrs.get("units", "")).strip().lower()
	if units in {"k", "kelvin"}:
		converted = values.astype(np.float64) - 273.15
	elif units in {"c", "degc", "°c", "celsius"}:
		converted = values.astype(np.float64)
	else:
		message = (
			f"Temperature units for {values.name!r} must explicitly be K or °C; "
			f"got {units!r}."
		)
		logger.error(message)
		raise SchemaError(message)
	converted.attrs = values.attrs.copy()
	converted.attrs["units"] = "degC"
	return converted


def _nearest_coarse_on_grid(
	coarse_values: xr.DataArray,
	grid: xr.Dataset,
	coarse_crs: CRS,
) -> xr.DataArray:
	"""Expand coarse-cell values as piecewise-constant footprint values."""
	source_y, source_x, _, _ = _spatial_dimensions(coarse_values)
	target_x, target_y, target_y_dim, target_x_dim, target_crs, _ = (
		_target_coordinates(grid)
	)
	y_axis, y_order = _axis_values(coarse_values.coords[source_y].values, source_y)
	x_axis, x_order = _axis_values(coarse_values.coords[source_x].values, source_x)
	values = np.asarray(coarse_values.transpose(source_y, source_x).values)
	values = np.take(np.take(values, y_order, axis=0), x_order, axis=1)
	mesh_x, mesh_y = np.meshgrid(target_x, target_y)
	projected_x, projected_y = Transformer.from_crs(
		target_crs, coarse_crs, always_xy=True
	).transform(mesh_x, mesh_y)
	y_index, x_index = _coarse_cell_indices(
		np.asarray(projected_x).ravel(),
		np.asarray(projected_y).ravel(),
		x_axis,
		y_axis,
	)
	valid = (y_index >= 0) & (x_index >= 0)
	output = np.full(y_index.shape, np.nan, dtype=np.float64)
	output[valid] = values[y_index[valid], x_index[valid]]
	return xr.DataArray(
		output.reshape(len(target_y), len(target_x)),
		dims=(target_y_dim, target_x_dim),
		coords={target_y_dim: target_y, target_x_dim: target_x},
		name=coarse_values.name,
		attrs=coarse_values.attrs.copy(),
	)


def _conserve_mean(
	downscaled: xr.DataArray,
	coarse_source: xr.DataArray,
	grid: xr.Dataset,
	coarse_crs: CRS,
	temperature: bool,
) -> xr.DataArray:
	"""Adjust each represented coarse footprint to preserve its source mean."""
	config = get_config().downscaling
	if config.mean_tol < 0 or config.precip_mean_relative_tol < 0:
		message = "Configured conservation tolerances must be non-negative."
		logger.error(message)
		raise ConfigError(message)
	coarse_y, coarse_x, _, _ = _spatial_dimensions(coarse_source)
	target_x, target_y, target_y_dim, target_x_dim, target_crs, _ = (
		_target_coordinates(grid)
	)
	source_y_axis, source_y_order = _axis_values(
		coarse_source.coords[coarse_y].values, coarse_y
	)
	source_x_axis, source_x_order = _axis_values(
		coarse_source.coords[coarse_x].values, coarse_x
	)
	source = coarse_source.transpose(
		*[dim for dim in coarse_source.dims if dim not in {coarse_y, coarse_x}],
		coarse_y,
		coarse_x,
	)
	source_values = np.asarray(source.values, dtype=np.float64)
	source_values = np.take(
		np.take(source_values, source_y_order, axis=-2), source_x_order, axis=-1
	)
	other_dims = tuple(dim for dim in source.dims if dim not in {coarse_y, coarse_x})
	query_x, query_y = Transformer.from_crs(
		target_crs, coarse_crs, always_xy=True
	).transform(*np.meshgrid(target_x, target_y))
	y_index, x_index = _coarse_cell_indices(
		np.asarray(query_x).ravel(),
		np.asarray(query_y).ravel(),
		source_x_axis,
		source_y_axis,
	)
	valid_footprint = (y_index >= 0) & (x_index >= 0)
	downscaled_values = np.asarray(
		downscaled.transpose(*other_dims, target_y_dim, target_x_dim).values,
		dtype=np.float64,
	).copy()
	flat_values = downscaled_values.reshape(
		(-1, len(target_y) * len(target_x))
	)
	flat_source = source_values.reshape((-1, len(source_y_axis) * len(source_x_axis)))
	for layer_index in range(flat_values.shape[0]):
		layer = flat_values[layer_index]
		coarse_layer = flat_source[layer_index]
		for cell_y in range(len(source_y_axis)):
			for cell_x in range(len(source_x_axis)):
				cell_mask = (
					valid_footprint
					& (y_index == cell_y)
					& (x_index == cell_x)
					& np.isfinite(layer)
				)
				if not cell_mask.any():
					continue
				target_mean = float(coarse_layer[cell_y * len(source_x_axis) + cell_x])
				if not np.isfinite(target_mean):
					continue
				current_mean = float(np.mean(layer[cell_mask]))
				if temperature:
					layer[cell_mask] += target_mean - current_mean
				elif current_mean > 0 and target_mean >= 0:
					layer[cell_mask] *= target_mean / current_mean
				elif target_mean == 0:
					layer[cell_mask] = 0.0
				else:
					message = "Precipitation mean cannot be conserved with invalid values."
					logger.error(message)
					raise DataIntegrityError(message)
				verified = float(np.mean(layer[cell_mask]))
				tolerance = (
					config.mean_tol
					if temperature
					else config.precip_mean_relative_tol
					* max(abs(target_mean), np.finfo(np.float64).eps)
				)
				if abs(verified - target_mean) > tolerance:
					message = (
						f"Coarse-cell conservation failed: source={target_mean:.4f}, "
						f"downscaled={verified:.4f}, tolerance={tolerance:.6g}."
					)
					logger.error(message)
					raise DataIntegrityError(message)
	downscaled_values = flat_values.reshape(downscaled_values.shape)
	result = xr.DataArray(
		downscaled_values,
		dims=(*other_dims, target_y_dim, target_x_dim),
		coords={
			**{dim: downscaled.coords[dim] for dim in other_dims if dim in downscaled.coords},
			target_y_dim: target_y,
			target_x_dim: target_x,
		},
		name=downscaled.name,
		attrs=downscaled.attrs.copy(),
	)
	return result


def _canonical_grid(city: str) -> xr.Dataset:
	"""Construct a configured canonical raster grid for a city."""
	config = get_config()
	if (
		city not in config.grid.crs
		or city not in config.grid.shape
		or city not in config.grid.transform
	):
		message = f"Unsupported city {city!r}."
		logger.error(message)
		raise ConfigError(message)
	height, width = config.grid.shape[city]
	transform = config.grid.transform[city]
	x = transform.c + (np.arange(width) + 0.5) * transform.a
	y = transform.f + (np.arange(height) + 0.5) * transform.e
	grid = xr.Dataset(
		coords={"y": y, "x": x},
		attrs={"city": city},
	).rio.write_crs(config.grid.crs[city], inplace=False)
	grid = grid.rio.write_transform(transform, inplace=False)
	loaded = load_grid(grid, city=city)
	if not isinstance(loaded, xr.Dataset):
		message = "Canonical target grid could not be loaded as an xarray.Dataset."
		logger.error(message)
		raise SchemaError(message)
	return loaded


def _find_variable(dataset: xr.Dataset, names: tuple[str, ...]) -> str | None:
	"""Find the first configured candidate variable present in a dataset."""
	return next((name for name in names if name in dataset.data_vars), None)


def _magnus_relative_humidity(
	temperature_c: xr.DataArray,
	relative_humidity: xr.DataArray,
	reference_temperature_c: xr.DataArray,
) -> xr.DataArray:
	"""Preserve dew point while recomputing relative humidity at adjusted T."""
	if str(temperature_c.attrs.get("units", "")).strip().lower() not in {
		"c", "degc", "°c", "celsius"
	}:
		message = "Magnus humidity recomputation requires temperature in Celsius."
		logger.error(message)
		raise SchemaError(message)
	if str(reference_temperature_c.attrs.get("units", "")).strip().lower() not in {
		"c", "degc", "°c", "celsius"
	}:
		message = "Reference temperature must explicitly be in Celsius."
		logger.error(message)
		raise SchemaError(message)
	if not str(relative_humidity.attrs.get("units", "")).strip().lower() in {
		"%", "percent", "percentage"
	}:
		message = "RH2M units must explicitly be percent."
		logger.error(message)
		raise SchemaError(message)
	humidity_fraction = (relative_humidity / 100.0).clip(
		np.finfo(np.float64).eps, 1.0
	)
	magnus_a, magnus_b = 17.625, 243.04
	gamma = np.log(humidity_fraction) + magnus_a * reference_temperature_c / (
		magnus_b + reference_temperature_c
	)
	dewpoint_c = magnus_b * gamma / (magnus_a - gamma)
	adjusted_humidity = 100.0 * np.exp(
		magnus_a * dewpoint_c / (magnus_b + dewpoint_c)
		- magnus_a * temperature_c / (magnus_b + temperature_c)
	)
	result = adjusted_humidity.clip(0.0, 100.0)
	result.name = relative_humidity.name
	result.attrs = relative_humidity.attrs.copy()
	result.attrs["humidity_method"] = "dewpoint_preserving_magnus"
	return result


def _relative_humidity_from_dewpoint(
	temperature_c: xr.DataArray,
	dewpoint_c: xr.DataArray,
) -> xr.DataArray:
	"""Compute relative humidity from temperature and preserved dew point."""
	magnus_a, magnus_b = 17.625, 243.04
	result = 100.0 * np.exp(
		magnus_a * dewpoint_c / (magnus_b + dewpoint_c)
		- magnus_a * temperature_c / (magnus_b + temperature_c)
	)
	result = result.clip(0.0, 100.0)
	result.name = "RH2M"
	result.attrs = {
		"units": "%",
		"humidity_method": "dewpoint_preserving_magnus",
	}
	return result


def downscale_variable(
	var: str,
	coarse: xr.Dataset,
	city: str,
) -> xr.DataArray:
	"""Downscale one supported temperature, humidity, or precipitation variable.

	Temperature uses lapse-rate adjustment and the configured conservation
	check. RH2M is recomputed from coarse dew point and adjusted temperature.
	Precipitation uses bilinear interpolation and optional conserving
	orographic scaling. Terrain is loaded from the configured city stack.

	Args:
		var: Variable name present in the coarse dataset.
		coarse: Coarse climate dataset with explicit source CRS and units.
		city: Canonical city key.

	Returns:
		Downscaled DataArray on the canonical grid.

	Raises:
		SchemaError: If city, variable, grid, or units are invalid.
		DataIntegrityError: If conservation checks fail.
	"""
	started = time.perf_counter()
	config = get_config()
	if city not in config.grid.crs:
		message = f"Unsupported city {city!r}."
		logger.error(message)
		raise SchemaError(message)
	if city not in config.dataset.terrain_path:
		message = f"No terrain-stack path is configured for city {city!r}."
		logger.error(message)
		raise ConfigError(message)
	if var not in coarse.data_vars:
		message = f"Coarse dataset does not contain {var!r}."
		logger.error(message)
		raise SchemaError(message)
	try:
		terrain = load_terrain_stack(str(config.dataset.terrain_path[city]))
	except (OSError, TypeError, ValueError) as error:
		message = f"Could not load configured terrain stack for {city!r}."
		logger.error("%s %s", message, error)
		raise DataIntegrityError(message) from error
	dem_name = next((name for name in ("dem", "elevation") if name in terrain), None)
	if dem_name is None:
		message = "Terrain stack has no DEM/elevation variable."
		logger.error(message)
		raise SchemaError(message)
	dem = _mask_dem_nodata(terrain[dem_name])
	grid = _canonical_grid(city)
	coarse_crs = _crs_of(coarse)
	coarse_on_grid = reproject_coarse_to_grid(coarse, grid, method="bilinear")
	variable = coarse[var]
	downscaling = config.downscaling
	if var in downscaling.temperature_variables:
		temperature_source = _temperature_celsius(variable)
		source_dataset = coarse.copy()
		source_dataset[var] = temperature_source
		coarse_temperature = reproject_coarse_to_grid(
			source_dataset, grid, method="bilinear"
		)[var]
		reference = coarse_cell_reference_elevation(dem, coarse)
		reference_on_grid = _nearest_coarse_on_grid(
			reference, grid, coarse_crs
		)
		rate = estimate_lapse_rate(temperature_source, reference)
		adjusted = lapse_adjust(coarse_temperature, dem, reference_on_grid, rate)
		adjusted = add_topoclimatic_residual_terms(adjusted, terrain, city)
		adjusted = _conserve_mean(
			adjusted, temperature_source, grid, coarse_crs, temperature=True
		)
		adjusted.name = "t2m_downscaled" if var.lower() in {"t2m"} else var
		adjusted.attrs["source_variable"] = var
		result = adjusted
	elif var in downscaling.humidity_variables:
		t_name = _find_variable(coarse, downscaling.temperature_variables)
		if t_name is None:
			message = "RH2M downscaling requires a coarse temperature variable."
			logger.error(message)
			raise SchemaError(message)
		dewpoint_name = _find_variable(coarse, downscaling.dewpoint_variables)
		adjusted_temperature = downscale_variable(t_name, coarse, city)
		if dewpoint_name is not None:
			dewpoint_source = _temperature_celsius(coarse[dewpoint_name])
			dewpoint_dataset = coarse[[dewpoint_name]].copy()
			dewpoint_dataset[dewpoint_name] = dewpoint_source
			dewpoint_grid = reproject_coarse_to_grid(dewpoint_dataset, grid)[
				dewpoint_name
			]
			result = _relative_humidity_from_dewpoint(
				adjusted_temperature, dewpoint_grid
			)
		else:
			coarse_temperature_source = _temperature_celsius(coarse[t_name])
			temperature_dataset = coarse[[t_name]].copy()
			temperature_dataset[t_name] = coarse_temperature_source
			coarse_temperature_grid = reproject_coarse_to_grid(
				temperature_dataset, grid
			)[t_name]
			rh_grid = reproject_coarse_to_grid(coarse[[var]], grid)[var]
			result = _magnus_relative_humidity(
				adjusted_temperature, rh_grid, coarse_temperature_grid
			)
		result = result.where(np.isfinite(dem))
		result.name = "RH2M"
	elif var in downscaling.precipitation_variables:
		precip = coarse_on_grid[var].where(np.isfinite(dem))
		if downscaling.precip_orographic_enabled:
			reference = coarse_cell_reference_elevation(dem, coarse)
			reference_on_grid = _nearest_coarse_on_grid(
				reference, grid, coarse_crs
			)
			factor = (
				1.0
				+ downscaling.precip_orographic_coefficient_per_km
				* (dem - reference_on_grid)
				/ 1000.0
			).clip(min=0.0)
			precip = precip * factor
		result = _conserve_mean(
			precip, variable, grid, coarse_crs, temperature=False
		)
		result.name = "PRECIP"
	else:
		message = (
			f"Variable {var!r} is not configured as temperature, RH2M, or "
			"precipitation."
		)
		logger.error(message)
		raise SchemaError(message)
	result.attrs["city"] = city
	result.attrs["units"] = result.attrs.get("units", variable.attrs.get("units", ""))
	logger.info(
		"Finished downscaling %s for %s (cells=%d) in %.3fs.",
		var,
		city,
		result.size,
		time.perf_counter() - started,
	)
	return result


def to_cell_month_columns(downscaled: xr.Dataset) -> pd.DataFrame:
	"""Flatten downscaled raster variables to cell-month records.

	Spatial ``cell_id`` values are taken from a dataset coordinate when
	available, otherwise assigned in row-major order. All variables must share
	the same ``time``, ``y``, and ``x`` grid.

	Args:
		downscaled: Dataset containing time/y/x downscaled climate fields.

	Returns:
		Long DataFrame with canonical date, cell identifiers, coordinates, and
		downscaled variable columns.

	Raises:
		SchemaError: If the dataset dimensions or time coordinate are invalid.
	"""
	required_dims = {"time", "y", "x"}
	if not required_dims.issubset(downscaled.dims):
		message = f"Downscaled dataset requires dimensions {sorted(required_dims)}."
		logger.error(message)
		raise SchemaError(message)
	if "time" not in downscaled.coords:
		message = "Downscaled dataset requires a time coordinate."
		logger.error(message)
		raise SchemaError(message)
	dates = pd.DatetimeIndex(pd.to_datetime(downscaled["time"].values))
	if dates.hasnans or not dates.is_month_start.all():
		message = "Downscaled time values must be valid month-start dates."
		logger.error(message)
		raise SchemaError(message)
	if dates.has_duplicates or not dates.is_monotonic_increasing:
		message = "Downscaled time values must be unique and chronologically sorted."
		logger.error(message)
		raise SchemaError(message)
	y_values = np.asarray(downscaled["y"].values, dtype=np.float64)
	x_values = np.asarray(downscaled["x"].values, dtype=np.float64)
	x_mesh, y_mesh = np.meshgrid(x_values, y_values)
	cell_count = len(x_values) * len(y_values)
	cell_ids = (
		np.asarray(downscaled["cell_id"].values, dtype=np.int64).reshape(-1)
		if "cell_id" in downscaled.coords
		else np.arange(cell_count, dtype=np.int64)
	)
	if len(cell_ids) != cell_count:
		message = "cell_id coordinate size does not match the spatial grid."
		logger.error(message)
		raise SchemaError(message)
	base = pd.DataFrame(
		{
			"date": np.repeat(dates.to_numpy(dtype="datetime64[ns]"), cell_count),
			"cell_id": np.tile(cell_ids, len(dates)),
			"x": np.tile(x_mesh.ravel(), len(dates)),
			"y": np.tile(y_mesh.ravel(), len(dates)),
		}
	)
	for name, variable in downscaled.data_vars.items():
		if name == "cell_id":
			continue
		if required_dims.issubset(variable.dims):
			base[name] = np.asarray(
				variable.transpose("time", "y", "x").values
			).reshape(-1)
	if "T2M" in base and "t2m_coarse" not in base:
		base["t2m_coarse"] = base["T2M"]
	if "t2m_downscaled" not in base:
		for candidate in ("T2M_downscaled", "t2m", "T2M"):
			if candidate in base:
				base["t2m_downscaled"] = base[candidate]
				break
	base["city"] = str(downscaled.attrs.get("city", ""))
	return base


if __name__ == "__main__":
	"""Exercise the planar lapse-rate response on a 1 km elevation gradient."""
	config = get_config()
	distance_m = 1000.0
	dem = xr.DataArray(
		np.array([[0.0, distance_m], [0.0, distance_m]]),
		dims=("y", "x"),
		coords={"y": [1500.0, 500.0], "x": [500.0, 1500.0]},
		name="dem",
		attrs={"units": "m"},
	).rio.write_crs(config.grid.crs["dehradun"], inplace=False)
	temperature = xr.DataArray(
		np.full((2, 2), 20.0),
		dims=("y", "x"),
		coords={"y": dem.y, "x": dem.x},
		name="t2m_coarse",
		attrs={"units": "degC"},
	)
	reference = xr.DataArray(
		np.zeros((2, 2)),
		dims=("y", "x"),
		coords={"y": dem.y, "x": dem.x},
		attrs={"units": "m"},
	)
	rate = pd.Series(config.downscaling.default_lapse_rate_c_per_km, index=range(1, 13))
	result = lapse_adjust(temperature, dem, reference, rate)
	assert result.shape == dem.shape
	assert result.dims == dem.dims
	assert np.isclose(
		result[0, 1] - result[0, 0],
		config.downscaling.default_lapse_rate_c_per_km,
		atol=1e-6,
	)
