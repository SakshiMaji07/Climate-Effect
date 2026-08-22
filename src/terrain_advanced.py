"""Advanced terrain derivatives: multi-scale TPI and native-resolution TWI."""

from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rioxarray
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter

from config import (
	DEM_SOURCE_PATHS,
	FLOW_ALGORITHM,
	NODATA,
	TPI_SCALES_M,
	build_target_grid,
)
from raster_processing import load_and_align
from terrain_features import compute_elevation_slope_aspect


def _as_2d(data: xr.DataArray) -> xr.DataArray:
	if data.ndim == 3 and "band" in data.dims and data.sizes["band"] == 1:
		return data.isel(band=0, drop=True)
	if data.ndim != 2:
		raise ValueError("Terrain rasters must be 2-dimensional.")
	return data


def _valid_mask(data: xr.DataArray) -> np.ndarray:
	values = np.asarray(data.values, dtype=float)
	valid = np.isfinite(values)
	if data.rio.nodata is not None and np.isfinite(data.rio.nodata):
		valid &= ~np.isclose(values, data.rio.nodata)
	return valid


def _layer(values: np.ndarray, template: xr.DataArray, name: str) -> xr.DataArray:
	result = xr.DataArray(
		values,
		dims=template.dims,
		coords={dimension: template.coords[dimension] for dimension in template.dims},
		name=name,
		attrs=template.attrs.copy(),
	)
	result.attrs["nodata"] = float(NODATA["continuous"])
	result = result.rio.write_crs(template.rio.crs, inplace=False)
	return result.rio.write_transform(template.rio.transform(), inplace=False)


def compute_tpi(
	dem: xr.DataArray, scales_m: list[float], mask_edges: bool = False
) -> xr.Dataset:
	"""Compute nodata-aware multi-scale TPI using reflected uniform windows.

	Each scale is a map-unit radius converted to pixels from the DEM
	resolution. A window containing any nodata is nodata; with ``mask_edges``
	enabled, the corresponding radius-wide border is also nulled.
	"""
	dem = _as_2d(dem)
	values = np.asarray(dem.values, dtype=float)
	valid = _valid_mask(dem)
	result = {}
	resolution_x, resolution_y = dem.rio.resolution()
	pixel_size = (abs(resolution_x) + abs(resolution_y)) / 2
	if pixel_size <= 0:
		raise ValueError("DEM resolution must be non-zero.")

	for scale_m in scales_m:
		if scale_m <= 0:
			raise ValueError("TPI scales must be positive.")
		radius_pixels = max(1, int(ceil(scale_m / pixel_size)))
		window_size = 2 * radius_pixels + 1
		valid_count = uniform_filter(
			valid.astype(float), size=window_size, mode="reflect"
		) * window_size**2
		filled = np.where(valid, values, 0.0)
		window_sum = uniform_filter(
			filled, size=window_size, mode="reflect"
		) * window_size**2
		# Exclude the center cell as specified by the TPI definition.
		neighbor_count = window_size**2 - 1
		complete_window = valid_count >= window_size**2 - 0.5
		tpi = values - (window_sum - np.where(valid, values, 0.0)) / neighbor_count
		output_valid = valid & complete_window
		if mask_edges:
			output_valid[:radius_pixels, :] = False
			output_valid[-radius_pixels:, :] = False
			output_valid[:, :radius_pixels] = False
			output_valid[:, -radius_pixels:] = False
		tpi = np.where(output_valid, tpi, np.nan)
		result[f"tpi_{scale_m:g}m"] = _layer(
			tpi, dem, f"tpi_{scale_m:g}m"
		)
	return xr.Dataset(result)


def _fill_depressions(dem_values: np.ndarray, valid: np.ndarray) -> np.ndarray:
	"""Fill pits with WhiteboxTools' priority-flood implementation for routing.

	WhiteboxTools is isolated here so a different fill backend can replace it.
	Temporary GeoTIFFs bridge xarray and WhiteboxTools. Filled values are
	routing-only and never replace the exported elevation.
	"""
	try:
		from whitebox import WhiteboxTools
	except ImportError as error:
		raise ImportError(
			"compute_flow_and_twi requires WhiteboxTools for depression filling; "
			"install the 'whitebox' requirement."
		) from error
	with TemporaryDirectory(prefix="whitebox_fill_") as directory:
		input_path = Path(directory) / "dem.tif"
		output_path = Path(directory) / "filled.tif"
		_write_whitebox_input(input_path, dem_values, valid)
		wbt = WhiteboxTools()
		wbt.set_verbose_mode(False)
		wbt.fill_depressions(str(input_path), str(output_path))
		filled = _read_whitebox_output(output_path)
	return np.where(valid, filled, np.nan)


def _write_whitebox_input(
	path: Path, values: np.ndarray, valid: np.ndarray
) -> None:
	"""Write a temporary single-band GeoTIFF for a WhiteboxTools operation."""
	array = np.where(valid, values, -9999.0).astype(np.float32)
	with rasterio.open(
		path,
		"w",
		driver="GTiff",
		height=array.shape[0],
		width=array.shape[1],
		count=1,
		dtype=array.dtype,
		crs="EPSG:3857",
		transform=rasterio.Affine.identity(),
		nodata=-9999.0,
	) as destination:
		destination.write(array, 1)


def _read_whitebox_output(path: Path) -> np.ndarray:
	with rasterio.open(path) as source:
		return source.read(1).astype(float)


def _run_flow_accumulation(
	dem_values: np.ndarray, valid: np.ndarray, algorithm: str
) -> np.ndarray:
	"""Run WhiteboxTools flow direction and accumulation via temporary rasters."""
	from whitebox import WhiteboxTools

	with TemporaryDirectory(prefix="whitebox_flow_") as directory:
		input_path = Path(directory) / "filled.tif"
		pointer_path = Path(directory) / "pointer.tif"
		accumulation_path = Path(directory) / "accumulation.tif"
		_write_whitebox_input(input_path, dem_values, valid)
		wbt = WhiteboxTools()
		wbt.set_verbose_mode(False)
		if algorithm == "d8":
			wbt.d8_pointer(str(input_path), str(pointer_path))
			wbt.d8_flow_accumulation(
				str(pointer_path), str(accumulation_path), out_type="cells", pntr=True
			)
		else:
			wbt.d_inf_pointer(str(input_path), str(pointer_path))
			wbt.d_inf_flow_accumulation(
				str(pointer_path), str(accumulation_path), out_type="cells", pntr=True
			)
		accumulation = _read_whitebox_output(accumulation_path)
	return np.where(valid, accumulation, np.nan)


def compute_flow_and_twi(
	dem_native: xr.DataArray, algorithm: str = "d8"
) -> xr.Dataset:
	"""Fill, route, accumulate, and compute TWI on the native DEM.

	WhiteboxTools supplies depression filling, flow direction, and flow
	accumulation. The single ``algorithm`` branch selects D8 or Tarboton's
	D-infinity routing. TWI
	clips ``tan(slope)`` at ``1e-3`` so near-flat cells receive a capped,
	rather than infinite, wetness index.
	"""
	dem = _as_2d(dem_native)
	algorithm = algorithm.strip().lower()
	if algorithm not in {"d8", "dinf"}:
		raise ValueError("algorithm must be 'd8' or 'dinf'.")
	values = np.asarray(dem.values, dtype=float)
	valid = _valid_mask(dem)
	filled_values = _fill_depressions(values, valid)
	accumulation = _run_flow_accumulation(filled_values, valid, algorithm)
	slope = compute_elevation_slope_aspect(
		dem.rio.write_nodata(np.nan, inplace=False)
	)["slope"]
	cell_size = (abs(dem.rio.resolution()[0]) + abs(dem.rio.resolution()[1])) / 2
	specific_area = accumulation * cell_size
	slope_tangent = np.maximum(np.tan(np.radians(np.asarray(slope.values))), 1e-3)
	twi_values = np.log(specific_area / slope_tangent)
	twi_values[~valid] = np.nan
	return xr.Dataset(
		{
			"filled_elevation": _layer(filled_values, dem, "filled_elevation"),
			"flow_accumulation": _layer(accumulation, dem, "flow_accumulation"),
			"slope": slope,
			"twi": _layer(twi_values, dem, "twi"),
		}
	)


def build_advanced_terrain_stack(
	city_key: str, resolution_m: int
) -> xr.Dataset:
	"""Compute native TPI/TWI, then align continuous outputs to target grid."""
	dem_path = DEM_SOURCE_PATHS[city_key]
	source = _as_2d(rioxarray.open_rasterio(dem_path))
	native = _as_2d(
		load_and_align(
			dem_path,
			source.rio.crs,
			source.rio.transform(),
			source.shape,
			Resampling.bilinear,
			"elevation",
			nodata=NODATA["continuous"],
		)
	)
	flow_features = compute_flow_and_twi(native, algorithm=FLOW_ALGORITHM)
	tpi_features = compute_tpi(native, TPI_SCALES_M)
	target_transform, target_shape, target_crs = build_target_grid(city_key, resolution_m)
	aligned = {}
	for name, layer in {"twi": flow_features["twi"], **tpi_features.data_vars}.items():
		aligned[name] = _as_2d(
			load_and_align(
				layer,
				target_crs,
				target_transform,
				target_shape,
				Resampling.bilinear,
				"continuous",
				nodata=NODATA["continuous"],
			)
		)
	return xr.Dataset(aligned)
