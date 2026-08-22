"""Terrain derivatives and a static clear-sky solar radiation proxy."""

from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import xy
from scipy import ndimage

from config import DEM_SOURCE_PATHS, NODATA, build_target_grid
from raster_processing import load_and_align


DEFAULT_SOLAR_DAYS = [80, 172, 266, 355]
_SOLAR_CONSTANT_W_M2 = 1361.0


def _as_2d(data: xr.DataArray) -> xr.DataArray:
	"""Drop a singleton band dimension while retaining raster metadata."""
	if data.ndim == 3 and "band" in data.dims and data.sizes["band"] == 1:
		return data.isel(band=0, drop=True)
	if data.ndim != 2:
		raise ValueError("DEM and terrain layers must be 2-dimensional rasters.")
	return data


def _raster_metadata(data: xr.DataArray) -> dict:
	metadata = {"crs": data.rio.crs, "transform": data.rio.transform()}
	if data.rio.nodata is not None:
		metadata["nodata"] = data.rio.nodata
	return metadata


def _derived_layer(
	values: np.ndarray, template: xr.DataArray, name: str, nodata: float
) -> xr.DataArray:
	layer = xr.DataArray(
		values,
		dims=template.dims,
		coords={dimension: template.coords[dimension] for dimension in template.dims},
		name=name,
		attrs=template.attrs.copy(),
	)
	layer.attrs["nodata"] = nodata
	layer = layer.rio.write_crs(template.rio.crs, inplace=False)
	return layer.rio.write_transform(template.rio.transform(), inplace=False)


def compute_elevation_slope_aspect(
	dem_native: xr.DataArray,
) -> xr.Dataset:
	"""Compute elevation, Horn slope, aspect, and aspect encodings.

	Aspect is expressed as degrees clockwise from north. Flat terrain below
	0.01 degrees receives nodata for aspect and its sine/cosine encodings.
	"""
	dem = _as_2d(dem_native)
	values = np.asarray(dem.values, dtype=float)
	nodata_value = dem.rio.nodata
	valid = np.isfinite(values)
	if nodata_value is not None and np.isfinite(nodata_value):
		valid &= ~np.isclose(values, nodata_value)

	transform = dem.rio.transform()
	pixel_width = abs(transform.a)
	pixel_height = abs(transform.e)
	if pixel_width == 0 or pixel_height == 0:
		raise ValueError("DEM transform must have non-zero pixel dimensions.")

	# Horn's weighted 3x3 derivatives, evaluated over the complete raster.
	dz_dx_kernel = np.array(
		[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float
	) / (8 * pixel_width)
	dz_dy_kernel = np.array(
		[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float
	) / (8 * pixel_height)
	filled = np.where(valid, values, 0.0)
	dz_dx = ndimage.convolve(filled, dz_dx_kernel, mode="nearest")
	dz_dy = ndimage.convolve(filled, dz_dy_kernel, mode="nearest")
	window_valid = ndimage.convolve(valid.astype(np.uint8), np.ones((3, 3)), mode="nearest") == 9

	slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
	aspect = np.mod(np.degrees(np.arctan2(dz_dx, dz_dy)), 360.0)
	usable = valid & window_valid
	flat = slope < 0.01
	output_nodata = float(NODATA["continuous"])
	aspect_valid = usable & ~flat

	elevation_values = np.where(valid, values, output_nodata)
	slope_values = np.where(usable, slope, output_nodata)
	aspect_values = np.where(aspect_valid, aspect, output_nodata)
	aspect_sin_values = np.where(aspect_valid, np.sin(np.radians(aspect)), output_nodata)
	aspect_cos_values = np.where(aspect_valid, np.cos(np.radians(aspect)), output_nodata)

	return xr.Dataset(
		{
			"elevation": _derived_layer(elevation_values, dem, "elevation", output_nodata),
			"slope": _derived_layer(slope_values, dem, "slope", output_nodata),
			"aspect": _derived_layer(aspect_values, dem, "aspect", output_nodata),
			"aspect_sin": _derived_layer(aspect_sin_values, dem, "aspect_sin", output_nodata),
			"aspect_cos": _derived_layer(aspect_cos_values, dem, "aspect_cos", output_nodata),
		}
	)


def compute_solar_radiation_proxy(
	dem_native: xr.DataArray,
	slope_deg: xr.DataArray,
	aspect_deg: xr.DataArray,
	day_of_year: list[int] | None = None,
	per_day: bool = False,
) -> xr.DataArray:
	"""Compute clear-sky potential daily insolation without shadows.

	The default output is the mean across equinox and solstice dates. Set
	``per_day=True`` to retain a ``day_of_year`` dimension. Atmospheric
	attenuation and terrain shadowing are intentionally excluded.
	"""
	dem = _as_2d(dem_native)
	slope = _as_2d(slope_deg)
	aspect = _as_2d(aspect_deg)
	days = list(DEFAULT_SOLAR_DAYS if day_of_year is None else day_of_year)
	if not days:
		raise ValueError("day_of_year must contain at least one day.")
	if any(day < 1 or day > 366 for day in days):
		raise ValueError("day_of_year values must be between 1 and 366.")

	slope_rad = np.radians(np.asarray(slope.values, dtype=float))
	aspect_rad = np.radians(np.asarray(aspect.values, dtype=float))
	rows, columns = np.indices(dem.shape)
	x_centers, y_centers = xy(dem.rio.transform(), rows, columns, offset="center")
	_lon, lat = Transformer.from_crs(
		dem.rio.crs, "EPSG:4326", always_xy=True
	).transform(x_centers, y_centers)
	latitude = np.radians(np.asarray(lat, dtype=float))
	hour_angles = np.linspace(-np.pi, np.pi, 289)
	solar_days = []
	for day in days:
		declination = np.radians(23.44) * np.sin(2 * np.pi * (day - 81) / 365.0)
		hour_angle = hour_angles[None, None, :]
		sin_altitude = (
			np.sin(latitude)[..., None] * np.sin(declination)
			+ np.cos(latitude)[..., None]
			* np.cos(declination)
			* np.cos(hour_angle)
		)
		cos_altitude = np.sqrt(np.maximum(0.0, 1.0 - sin_altitude**2))
		sin_azimuth = np.cos(declination) * np.sin(hour_angle) / np.maximum(cos_altitude, 1e-12)
		cos_azimuth = (
			sin_altitude * np.sin(latitude)[..., None] - np.sin(declination) * np.cos(latitude)[..., None]
		) / np.maximum(cos_altitude * np.cos(latitude)[..., None], 1e-12)
		incidence = (
			sin_altitude * np.cos(slope_rad)[..., None]
			+ cos_altitude
			* (
				np.sin(slope_rad)[..., None] * np.cos(aspect_rad)[..., None] * cos_azimuth
				+ np.sin(slope_rad)[..., None] * np.sin(aspect_rad)[..., None] * sin_azimuth
			)
		)
		irradiance = _SOLAR_CONSTANT_W_M2 * np.maximum(incidence, 0.0)
		solar_days.append(np.trapezoid(irradiance, hour_angles, axis=-1) * 24 / (2 * np.pi))

	values = np.stack(solar_days, axis=0)
	invalid = ~np.isfinite(np.asarray(dem.values)) | ~np.isfinite(slope.values) | ~np.isfinite(aspect.values)
	values[:, invalid] = np.nan
	if per_day:
		result = xr.DataArray(
			values,
			dims=("day_of_year",) + dem.dims,
			coords={"day_of_year": days, **{dimension: dem.coords[dimension] for dimension in dem.dims}},
			name="solar_radiation_proxy",
			attrs=dem.attrs.copy(),
		)
		result.attrs["nodata"] = float(NODATA["continuous"])
		result = result.rio.write_crs(dem.rio.crs, inplace=False)
		return result.rio.write_transform(dem.rio.transform(), inplace=False)
	return _derived_layer(np.nanmean(values, axis=0), dem, "solar_radiation_proxy", np.nan)


def build_terrain_feature_stack(city_key: str, resolution_m: int) -> xr.Dataset:
	"""Build native-computed terrain features and align them to a target grid."""
	dem_path = DEM_SOURCE_PATHS[city_key]
	source = rioxarray.open_rasterio(dem_path)
	source_2d = _as_2d(source)
	native = load_and_align(
		dem_path,
		source_2d.rio.crs,
		source_2d.rio.transform(),
		source_2d.shape,
		Resampling.bilinear,
		"elevation",
		nodata=NODATA["continuous"],
	)
	native = _as_2d(native)
	native_features = compute_elevation_slope_aspect(native)
	native_solar = compute_solar_radiation_proxy(
		native,
		native_features["slope"],
		native_features["aspect"],
		DEFAULT_SOLAR_DAYS,
	)
	target_transform, target_shape, target_crs = build_target_grid(city_key, resolution_m)
	elevation = load_and_align(
		dem_path,
		target_crs,
		target_transform,
		target_shape,
		Resampling.bilinear,
		"elevation",
		nodata=NODATA["continuous"],
	)
	layers = {"elevation": _as_2d(elevation)}
	for name, variable in {
		"slope": native_features["slope"],
		"aspect": native_features["aspect"],
		"aspect_sin": native_features["aspect_sin"],
		"aspect_cos": native_features["aspect_cos"],
		"solar_radiation_proxy": native_solar,
	}.items():
		layers[name] = _as_2d(
			load_and_align(
				variable,
				target_crs,
				target_transform,
				target_shape,
				Resampling.bilinear,
				"continuous",
				nodata=NODATA["continuous"],
			)
		)
	return xr.Dataset(layers)
