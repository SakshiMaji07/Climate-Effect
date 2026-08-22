"""Reusable, metadata-preserving caches for xarray raster products."""

import base64
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rioxarray
import xarray as xr
from affine import Affine

import config


_GEOTIFF_ATTRS_TAG = "xarray_attrs"
_GEOTIFF_DIMS_TAG = "xarray_dims"
_GEOTIFF_NAME_TAG = "xarray_name"
_CF_BOOKKEEPING_ATTRS = {"_FillValue", "grid_mapping"}
_ZARR_SPATIAL_METADATA = "_raster_cache_spatial_metadata"


def save_to_zarr(ds: xr.Dataset, path: str) -> None:
	"""Write a dataset to Zarr, retaining xarray and spatial metadata."""
	if not isinstance(ds, xr.Dataset):
		raise TypeError("ds must be an xarray.Dataset")
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	spatial_metadata = {}
	for name, variable in ds.data_vars.items():
		if variable.rio.crs is not None:
			spatial_metadata[name] = {
				"crs": variable.rio.crs.to_wkt(),
				"transform": list(variable.rio.transform()),
				"nodata": (
					variable.rio.nodata.item()
					if isinstance(variable.rio.nodata, np.generic)
					else variable.rio.nodata
				),
			}
	to_write = ds.copy(deep=False)
	to_write.attrs = dict(ds.attrs)
	to_write.attrs[_ZARR_SPATIAL_METADATA] = json.dumps(spatial_metadata)
	to_write.to_zarr(output_path, mode="w", zarr_format=2)


def load_from_zarr(path: str) -> xr.Dataset:
	"""Load a fully materialized dataset from a Zarr store."""
	try:
		dataset = xr.open_zarr(path)
	except ImportError as error:
		raise ImportError(
			"Zarr caching requires the 'zarr' package; install project requirements."
		) from error
	dataset = dataset.load()
	spatial_metadata = json.loads(dataset.attrs.pop(_ZARR_SPATIAL_METADATA, "{}"))
	if "spatial_ref" in dataset.data_vars:
		dataset = dataset.drop_vars("spatial_ref")
	for name, metadata in spatial_metadata.items():
		if name not in dataset:
			continue
		variable = dataset[name]
		variable.attrs.pop("_FillValue", None)
		variable.encoding.pop("_FillValue", None)
		variable = variable.rio.write_crs(metadata["crs"], inplace=False)
		variable = variable.rio.write_transform(
			Affine(*metadata["transform"]), inplace=False
		)
		if metadata["nodata"] is not None:
			variable = variable.rio.write_nodata(metadata["nodata"], inplace=False)
		dataset[name] = variable
	return dataset


def _encode_attrs(attrs: dict[str, Any]) -> str:
	"""Encode attrs exactly for storage in GeoTIFF string tags."""
	return base64.b64encode(pickle.dumps(attrs, protocol=pickle.HIGHEST_PROTOCOL)).decode(
		"ascii"
	)


def _decode_attrs(value: str) -> dict[str, Any]:
	return pickle.loads(base64.b64decode(value.encode("ascii")))


def _semantic_attrs(data: xr.DataArray) -> dict[str, Any]:
	return {
		key: value
		for key, value in data.attrs.items()
		if key not in _CF_BOOKKEEPING_ATTRS
	}


def save_to_geotiff(da: xr.DataArray, path: str) -> None:
	"""Write a single- or multi-band DataArray with its raster metadata."""
	if not isinstance(da, xr.DataArray):
		raise TypeError("da must be an xarray.DataArray")
	if da.rio.crs is None:
		raise ValueError("DataArray must have a CRS before GeoTIFF serialization.")
	output_path = Path(path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	da.rio.to_raster(output_path)
	with rasterio.open(output_path, "r+") as dataset:
		dataset.update_tags(
			**{
				_GEOTIFF_ATTRS_TAG: _encode_attrs(dict(da.attrs)),
				_GEOTIFF_DIMS_TAG: json.dumps(list(da.dims)),
				_GEOTIFF_NAME_TAG: da.name or "",
			}
		)


def load_from_geotiff(path: str) -> xr.DataArray:
	"""Read a GeoTIFF and restore the DataArray name, dimensions, and attrs."""
	with rioxarray.open_rasterio(path) as opened:
		data = opened.load()
		with rasterio.open(path) as dataset:
			tags = dataset.tags()
		name = tags.get(_GEOTIFF_NAME_TAG) or None
		dims = tuple(json.loads(tags[_GEOTIFF_DIMS_TAG])) if _GEOTIFF_DIMS_TAG in tags else None
		attrs = _decode_attrs(tags[_GEOTIFF_ATTRS_TAG]) if _GEOTIFF_ATTRS_TAG in tags else dict(data.attrs)

	if dims is not None and len(dims) == 2 and data.sizes.get("band") == 1:
		data = data.isel(band=0, drop=True)
	if dims is not None and len(dims) == data.ndim and tuple(data.dims) != dims:
		data = data.rename(dict(zip(data.dims, dims)))
	data.name = name
	data.attrs = attrs
	return data


def assert_round_trip(original: xr.Dataset | xr.DataArray, restored: xr.Dataset | xr.DataArray) -> None:
	"""Assertion hook for tests covering values and all xarray metadata."""
	if isinstance(original, xr.Dataset) and isinstance(restored, xr.Dataset):
		assert original.attrs == restored.attrs
		original_names = set(original.data_vars) - {"spatial_ref"}
		restored_names = set(restored.data_vars) - {"spatial_ref"}
		assert original_names == restored_names
		for name in original_names:
			assert_round_trip(original[name], restored[name])
		return
	if not isinstance(original, xr.DataArray) or not isinstance(restored, xr.DataArray):
		raise TypeError("original and restored must have the same xarray type")
	assert original.name == restored.name
	assert original.dims == restored.dims
	assert _semantic_attrs(original) == _semantic_attrs(restored)
	np.testing.assert_allclose(original.values, restored.values, equal_nan=True)
	for coordinate in original.coords:
		if coordinate != "spatial_ref" and coordinate in restored.coords:
			xr.testing.assert_equal(original.coords[coordinate], restored.coords[coordinate])
	assert original.rio.crs == restored.rio.crs
	assert np.allclose(tuple(original.rio.transform()), tuple(restored.rio.transform()))
	assert original.rio.nodata == restored.rio.nodata


def cache_key(city_key: str, resolution_m: int, feature_name: str, config_hash: str) -> str:
	"""Return the Part 8 static raster cache name."""
	if not city_key or not feature_name or not config_hash:
		raise ValueError("city_key, feature_name, and config_hash must be non-empty")
	if resolution_m <= 0:
		raise ValueError("resolution_m must be positive")
	return f"{city_key}_{feature_name}_{resolution_m}m_static"


def get_or_compute(
	cache_key: str, compute_fn: Callable[[], xr.Dataset], fmt: str = "zarr"
) -> xr.Dataset:
	"""Load a cached dataset or compute and save it exactly once."""
	if not cache_key:
		raise ValueError("cache_key must be non-empty")
	if not callable(compute_fn):
		raise TypeError("compute_fn must be callable")
	if fmt.lower() != "zarr":
		raise ValueError("Only the 'zarr' cache format is supported for datasets")
	path = Path(config.CACHE_DIR) / cache_key
	if path.exists():
		return load_from_zarr(str(path))
	dataset = compute_fn()
	if not isinstance(dataset, xr.Dataset):
		raise TypeError("compute_fn must return an xarray.Dataset")
	save_to_zarr(dataset, str(path))
	return dataset
