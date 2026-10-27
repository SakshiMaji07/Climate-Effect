"""Terrain-agnostic raster loading and target-grid alignment utilities."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from affine import Affine
from pyproj import CRS

from exceptions import SchemaError
from logger import get_logger

logger = get_logger(__name__)


class AlignmentError(AssertionError):
	"""Raised when an aligned raster does not match its requested grid."""


def resampling_for(
	variable_type: str, prefer_cubic: bool = False
) -> rasterio.enums.Resampling:
	"""Return the Part 9.7 resampling method for a variable type."""
	variable_class = variable_type.strip().lower()
	if variable_class == "categorical":
		return rasterio.enums.Resampling.nearest
	if variable_class in {
		"continuous",
		"elevation",
		"slope",
		"aspect", #instead use sin(aspect) and cos(aspect) because it is a circular variable
		"tpi",
		"twi",
		"solar",
		"ndvi",
		"ndbi",
	}:
		return (
			rasterio.enums.Resampling.cubic
			if prefer_cubic
			else rasterio.enums.Resampling.bilinear
		)
	raise ValueError(f"Unsupported variable_type: {variable_type!r}")


def _transforms_match(actual: Affine, expected: Affine) -> bool:
	return bool(np.allclose(tuple(actual), tuple(expected)))


def load_grid(
	source: str | Path | xr.DataArray | xr.Dataset,
	city: str | None = None,
) -> xr.DataArray | xr.Dataset:
	"""Load a raster or dataset while preserving canonical grid metadata.

	Args:
		source: File path or in-memory xarray object.
		city: Optional city key used to label metadata.

	Returns:
		The loaded grid object.
	"""
	if isinstance(source, (str, Path)):
		if source.lower().endswith(".zarr"):
			loaded = xr.open_dataset(source, engine="zarr")
		else:
			loaded = rioxarray.open_rasterio(source)
		if city is not None:
			loaded.attrs["city"] = city
		return loaded
	if isinstance(source, (xr.DataArray, xr.Dataset)):
		if city is not None:
			source.attrs["city"] = city
		return source
	raise TypeError("source must be a file path or xarray object")


def load_terrain_stack(
	source: str | Path | xr.Dataset,
) -> xr.Dataset:
	"""Load an in-memory or Zarr terrain-feature stack.

	Args:
		source: Terrain Dataset or path to a Zarr dataset.

	Returns:
		The loaded terrain feature Dataset.
	"""
	loaded = load_grid(source)
	if not isinstance(loaded, xr.Dataset):
		message = "Terrain stack source must be an xarray.Dataset."
		logger.error(message)
		raise SchemaError(message)
	return loaded


def load_and_align(
	source: str | Path | xr.DataArray,
	target_crs: CRS | str,
	target_transform: Affine,
	target_shape: tuple[int, int],
	resampling: rasterio.enums.Resampling,
	variable_type: str,
	nodata: float | int | None = None,
	clip_bounds: tuple[float, float, float, float] | None = None,
) -> xr.DataArray:
	"""Load a raster and reproject it onto the supplied target grid.

	``clip_bounds`` is interpreted in the source raster CRS and applied
	before reprojection when provided.
	"""
	variable_class = variable_type.strip().lower()
	policy_resampling = resampling_for(variable_type)
	if variable_class == "categorical" and resampling != policy_resampling:
		raise ValueError(
			"Categorical rasters require nearest-neighbor resampling; "
			f"received {resampling!r}."
		)

	source_path = str(source) if isinstance(source, (str, Path)) else "<in-memory>"
	data = rioxarray.open_rasterio(source) if isinstance(source, (str, Path)) else source
	if not isinstance(data, xr.DataArray):
		raise TypeError("source must be a file path or xarray.DataArray")
	if data.rio.crs is None:
		raise ValueError("Source raster has no CRS.")

	original_crs = data.rio.crs
	original_res = data.rio.resolution()
	if clip_bounds is not None:
		data = data.rio.clip_box(*clip_bounds)

	requested_crs = CRS.from_user_input(target_crs)
	aligned = data.rio.reproject(
		requested_crs,
		transform=target_transform,
		shape=target_shape,
		resampling=resampling,
		nodata=nodata,
	)

	if aligned.rio.shape != target_shape:
		raise AlignmentError(
			f"Aligned shape mismatch: expected {target_shape}, "
			f"got {aligned.rio.shape}."
		)
	if not CRS.from_user_input(aligned.rio.crs).equals(requested_crs):
		raise AlignmentError(
			f"Aligned CRS mismatch: expected {requested_crs}, "
			f"got {aligned.rio.crs}."
		)
	if not _transforms_match(aligned.rio.transform(), target_transform):
		raise AlignmentError(
			"Aligned transform mismatch: expected "
			f"{target_transform!r}, got {aligned.rio.transform()!r}."
		)

	aligned.attrs.update(
		{
			"source_path": source_path,
			"original_crs": str(original_crs),
			"original_res": original_res,
			"resampling_method": resampling.name,
			"processed_utc_timestamp": datetime.now(timezone.utc).isoformat(),
		}
	)
	return aligned
