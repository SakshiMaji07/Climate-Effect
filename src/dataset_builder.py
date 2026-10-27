"""Strict city-month and cell-month dataset construction."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr
from affine import Affine

from config import get_config
from data_loader import validate_schema
from data_sanity import run_gate
from exceptions import DataIntegrityError, SchemaError
from logger import get_logger
from temporal_features import add_seasonality

logger = get_logger(__name__)

CITY_MONTH_SCHEMA: dict[str, str] = {
	"city": "string",
	"date": "datetime64[ns]",
	"DTR": "float64",
	"TMAX": "float64",
	"TMIN": "float64",
	"LST": "float64",
	"RH2M": "float64",
	"PRECIP": "float64",
	"precip_anom": "float64",
	"urban_fraction_mean": "float64",
	"urban_fraction_p90": "float64",
	"n_valid_cells": "int64",
	"is_monsoon": "int8",
	"sin_m": "float64",
	"cos_m": "float64",
}

CELL_MONTH_SCHEMA: dict[str, str] = {
	"city": "string",
	"cell_id": "int64",
	"date": "datetime64[ns]",
	"x": "float64",
	"y": "float64",
	"dem": "float64",
	"slope": "float64",
	"aspect_sin": "float64",
	"aspect_cos": "float64",
	"tpi_small": "float64",
	"tpi_large": "float64",
	"twi": "float64",
	"solar_rad": "float64",
	"urban_fraction": "float64",
	"ndvi": "float64",
	"ndbi": "float64",
	"bsi": "float64",
	"t2m_coarse": "float64",
	"t2m_downscaled": "float64",
	"RH2M": "float64",
	"PRECIP": "float64",
	"LST": "float64",
	"DTR_cell": "float64",
	"is_monsoon": "int8",
	"sin_m": "float64",
	"cos_m": "float64",
	"spatial_block": "Int64",
}

_CITY_CLIMATE_COLUMNS = ("DTR", "TMAX", "TMIN", "LST", "RH2M", "PRECIP")
_CANONICAL_CITIES = frozenset({"dehradun", "jammu"})


def _validate_city_month(frame: pd.DataFrame) -> pd.DataFrame:
	"""Validate and cast the exact city-month output schema."""
	validate_schema(frame, tuple(CITY_MONTH_SCHEMA))
	extra = set(frame.columns) - set(CITY_MONTH_SCHEMA)
	if extra:
		message = f"City-month table contains unexpected columns: {sorted(extra)}."
		logger.error(message)
		raise SchemaError(message)
	frame = frame.loc[:, list(CITY_MONTH_SCHEMA)].copy()
	if frame.duplicated(["city", "date"]).any():
		message = "City-month table contains duplicate (city, date) keys."
		logger.error(message)
		raise SchemaError(message)
	if not pd.api.types.is_datetime64_ns_dtype(frame["date"].dtype):
		message = "City-month date must have dtype datetime64[ns]."
		logger.error(message)
		raise SchemaError(message)
	if frame["date"].isna().any() or not (frame["date"].dt.day == 1).all():
		message = "City-month dates must be non-null month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	if frame["city"].isna().any() or not frame["city"].isin(_CANONICAL_CITIES).all():
		message = "City-month table contains missing or invalid city keys."
		logger.error(message)
		raise SchemaError(message)
	frame["city"] = frame["city"].astype("string")
	for column in CITY_MONTH_SCHEMA:
		if column in {"city", "date"}:
			continue
		try:
			frame[column] = frame[column].astype(CITY_MONTH_SCHEMA[column])
		except (TypeError, ValueError) as error:
			message = f"City-month column {column!r} has an invalid dtype."
			logger.error(message)
			raise SchemaError(message) from error
	if not np.allclose(
		frame["DTR"],
		frame["TMAX"] - frame["TMIN"],
		atol=get_config().dataset.dtr_tolerance,
		rtol=0.0,
		equal_nan=True,
	):
		message = (
			"DTR must equal TMAX - TMIN within "
			f"{get_config().dataset.dtr_tolerance:g}."
		)
		logger.error(message)
		raise SchemaError(message)
	return frame


def _area_weights(uf: xr.DataArray, city: str) -> xr.DataArray:
	"""Resolve per-pixel area weights from raster coordinates or configured grid."""
	config = get_config()
	weight_name = config.dataset.area_weights_variable
	if weight_name in uf.coords:
		weights = uf.coords[weight_name]
		if set(weights.dims) != {"y", "x"}:
			message = "Urban-fraction area weights must have y and x dimensions."
			logger.error(message)
			raise SchemaError(message)
		return weights
	if "area_weights" in uf.attrs:
		weights = xr.DataArray(
			np.asarray(uf.attrs["area_weights"], dtype=np.float64),
			dims=("y", "x"),
			coords={"y": uf.y, "x": uf.x},
		)
		return weights
	transform = config.grid.transform.get(city)
	if transform is None:
		message = f"No canonical transform configured for {city!r}."
		logger.error(message)
		raise SchemaError(message)
	pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
	return xr.full_like(uf.isel(time=0, drop=True), pixel_area, dtype=np.float64)


def _validate_month_axis(dates: pd.Series, city: str) -> None:
	"""Log gaps in a monthly date sequence without filling missing periods."""
	unique = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
	if len(unique) < 2:
		return
	expected = pd.date_range(unique.min(), unique.max(), freq="MS")
	missing = expected.difference(unique)
	if len(missing):
		logger.warning(
			"Date continuity gaps for %s: %d missing months between %s and %s.",
			city,
			len(missing),
			unique.min().date(),
			unique.max().date(),
		)


def _assert_uf_values(uf: xr.DataArray, city: str) -> None:
	"""Ensure finite urban fractions lie within the unit interval."""
	values = np.asarray(uf.values, dtype=np.float64)
	finite = values[np.isfinite(values)]
	if np.any((finite < 0.0) | (finite > 1.0)):
		message = f"Urban fraction for {city!r} contains values outside [0, 1]."
		logger.error(message)
		raise SchemaError(message)


def build_city_month(
	climate: pd.DataFrame,
	uf: xr.DataArray,
	city: str,
) -> pd.DataFrame:
	"""Aggregate monthly city climate and area-weighted urban fractions.

	Precipitation anomalies use only the configured reference climatology
	years, grouped by calendar month. The period is included in ``DataFrame.attrs``
	for storage in parquet metadata.

	Args:
		climate: Monthly city climate with date and canonical climate variables.
		uf: Urban fraction with time/y/x dimensions and optional area weights.
		city: Canonical city key.

	Returns:
		City-month dataframe with the exact ``CITY_MONTH_SCHEMA`` columns.

	Raises:
		SchemaError: If the input schema, dates, city, or fractions are invalid.
	"""
	started = time.perf_counter()
	logger.info("Starting city-month aggregation (rows=%d, city=%s).", len(climate), city)
	if city not in _CANONICAL_CITIES:
		message = f"Unsupported city {city!r}."
		logger.error(message)
		raise SchemaError(message)
	validate_schema(climate, ("date", *_CITY_CLIMATE_COLUMNS))
	if not pd.api.types.is_datetime64_ns_dtype(climate["date"].dtype):
		message = "Climate date must have dtype datetime64[ns]."
		logger.error(message)
		raise SchemaError(message)
	climate_frame = climate.copy()
	if "city" in climate_frame and not climate_frame["city"].eq(city).all():
		message = f"Climate input includes rows not belonging to {city!r}."
		logger.error(message)
		raise SchemaError(message)
	if climate_frame["date"].isna().any() or not (climate_frame["date"].dt.day == 1).all():
		message = "Climate dates must be non-null month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	if climate_frame.duplicated(["date"]).any():
		message = f"Climate input contains duplicate dates for {city!r}."
		logger.error(message)
		raise SchemaError(message)
	if not np.allclose(
		climate_frame["DTR"],
		climate_frame["TMAX"] - climate_frame["TMIN"],
		atol=get_config().dataset.dtr_tolerance,
		rtol=0.0,
		equal_nan=True,
	):
		message = (
			"DTR must equal TMAX - TMIN within "
			f"{get_config().dataset.dtr_tolerance:g}."
		)
		logger.error(message)
		raise SchemaError(message)
	if not {"time", "y", "x"}.issubset(uf.dims):
		message = "Urban-fraction data must have time, y, and x dimensions."
		logger.error(message)
		raise SchemaError(message)
	_assert_uf_values(uf, city)
	city_dates = pd.DatetimeIndex(climate_frame["date"])
	uf_dates = pd.DatetimeIndex(uf.time.values).as_unit("ns")
	if not city_dates.equals(uf_dates):
		message = f"Climate and urban-fraction dates do not align for {city!r}."
		logger.error(message)
		raise SchemaError(message)
	if (uf.sizes["y"], uf.sizes["x"]) != get_config().grid.shape[city]:
		message = f"Urban-fraction shape does not match canonical grid for {city!r}."
		logger.error(message)
		raise SchemaError(message)
	weights = _area_weights(uf, city).broadcast_like(uf)
	valid = np.isfinite(uf) & np.isfinite(weights) & (weights > 0)
	valid_area = weights.where(valid, 0).sum(("y", "x"))
	weighted_sum = (uf.where(valid, 0) * weights).sum(("y", "x"))
	mean = (weighted_sum / valid_area).where(valid_area > 0)
	weighted_values = np.asarray(uf.values, dtype=np.float64)
	weight_values = np.asarray(_area_weights(uf, city).values, dtype=np.float64)
	p90_values: list[float] = []
	n_valid_values: list[int] = []
	for time_index in range(uf.sizes["time"]):
		values = weighted_values[time_index]
		valid_pixels = np.isfinite(values) & np.isfinite(weight_values) & (weight_values > 0)
		n_valid_values.append(int(valid_pixels.sum()))
		if not valid_pixels.any():
			p90_values.append(float("nan"))
			continue
		observations = values[valid_pixels]
		pixel_weights = weight_values[valid_pixels]
		order = np.argsort(observations)
		observations = observations[order]
		pixel_weights = pixel_weights[order]
		cumulative_weights = np.cumsum(pixel_weights) / pixel_weights.sum()
		p90_values.append(
			float(observations[min(np.searchsorted(cumulative_weights, 0.9), len(observations) - 1)])
		)

	climate_frame = climate_frame.sort_values("date").reset_index(drop=True)
	climate_frame["precip_anom"] = np.nan
	year_start, year_end = get_config().dataset.climatology_years
	reference = climate_frame.loc[
		climate_frame["date"].dt.year.between(year_start, year_end)
	].copy()
	climatology = reference.groupby(reference["date"].dt.month)["PRECIP"].mean()
	if reference.empty:
		logger.warning(
			"No precipitation observations fall within configured climatology "
			"years %d-%d for %s.",
			year_start,
			year_end,
			city,
		)
	monthly_normals = climate_frame["date"].dt.month.map(climatology)
	climate_frame["precip_anom"] = climate_frame["PRECIP"] - monthly_normals
	if climate_frame["precip_anom"].isna().any():
		logger.warning(
			"Precipitation anomalies are unavailable for %d rows in %s due to "
			"missing reference-month climatologies.",
			int(climate_frame["precip_anom"].isna().sum()),
			city,
		)
	result = pd.DataFrame(
		{
			"city": pd.Series(city, index=climate_frame.index, dtype="string"),
			"date": climate_frame["date"].astype("datetime64[ns]"),
			**{
				column: climate_frame[column].astype("float64")
				for column in _CITY_CLIMATE_COLUMNS
			},
			"precip_anom": climate_frame["precip_anom"].astype("float64"),
			"urban_fraction_mean": np.asarray(mean.values, dtype=np.float64),
			"urban_fraction_p90": np.asarray(p90_values, dtype=np.float64),
			"n_valid_cells": np.asarray(n_valid_values, dtype=np.int64),
		}
	)
	result = add_seasonality(result)
	result.attrs["climatology_years"] = (year_start, year_end)
	result.attrs["city"] = city
	result = _validate_city_month(result.loc[:, list(CITY_MONTH_SCHEMA)])
	_validate_month_axis(result["date"], city)
	logger.info(
		"Finished city-month aggregation (rows=%d) in %.3fs.",
		len(result),
		time.perf_counter() - started,
	)
	return result


def _validate_raster_alignment(
	raster: xr.DataArray | xr.Dataset,
	city: str,
	name: str,
) -> None:
	"""Validate raster shape, CRS, and affine transform against config."""
	config = get_config()
	if not {"y", "x"}.issubset(raster.dims):
		message = f"{name} raster must have y and x dimensions."
		logger.error(message)
		raise SchemaError(message)
	if (raster.sizes["y"], raster.sizes["x"]) != config.grid.shape[city]:
		message = f"{name} shape does not match canonical grid for {city!r}."
		logger.error(message)
		raise SchemaError(message)
	try:
		crs = raster.rio.crs
		transform = raster.rio.transform()
	except AttributeError as error:
		message = f"{name} raster must have CRS and affine metadata."
		logger.error(message)
		raise SchemaError(message) from error
	if crs is None or str(crs) != config.grid.crs[city]:
		message = f"{name} CRS does not match canonical grid for {city!r}."
		logger.error(message)
		raise SchemaError(message)
	if not np.allclose(tuple(transform), tuple(config.grid.transform[city])):
		message = f"{name} transform does not match canonical grid for {city!r}."
		logger.error(message)
		raise SchemaError(message)
	expected_transform: Affine = config.grid.transform[city]
	expected_x = expected_transform.c + expected_transform.a * (
		np.arange(raster.sizes["x"]) + 0.5
	)
	expected_y = expected_transform.f + expected_transform.e * (
		np.arange(raster.sizes["y"]) + 0.5
	)
	for coordinate, expected in (("x", expected_x), ("y", expected_y)):
		if coordinate in raster.coords and not np.allclose(
			np.asarray(raster[coordinate].values, dtype=np.float64),
			expected,
		):
			message = (
				f"{name} {coordinate} coordinates do not match the canonical grid "
				f"for {city!r}."
			)
			logger.error(message)
			raise SchemaError(message)


def _cell_coordinates(
	dataset: xr.Dataset, city: str
) -> tuple[np.ndarray, np.ndarray]:
	"""Resolve projected cell-center x/y coordinates."""
	if "x" in dataset.coords and "y" in dataset.coords:
		x = np.asarray(dataset["x"].values, dtype=np.float64)
		y = np.asarray(dataset["y"].values, dtype=np.float64)
	else:
		transform: Affine = get_config().grid.transform[city]
		x = transform.c + transform.a * (np.arange(dataset.sizes["x"]) + 0.5)
		y = transform.f + transform.e * (np.arange(dataset.sizes["y"]) + 0.5)
	if x.ndim != 1 or y.ndim != 1:
		message = "Canonical grid coordinates x and y must be one-dimensional."
		logger.error(message)
		raise SchemaError(message)
	return x, y


def _flatten_raster(
	data: xr.DataArray,
	dates: pd.DatetimeIndex,
	y_size: int,
	x_size: int,
) -> np.ndarray:
	"""Broadcast a static or temporal raster and return time-major flat values."""
	if "time" not in data.dims:
		data = data.expand_dims(time=dates)
	else:
		data_time = pd.DatetimeIndex(data.time.values).as_unit("ns")
		if not data_time.equals(dates):
			message = "Raster time coordinate does not match climate-cell dates."
			logger.error(message)
			raise SchemaError(message)
	if (data.sizes.get("y"), data.sizes.get("x")) != (y_size, x_size):
		message = "Raster shape does not match the canonical climate-cell grid."
		logger.error(message)
		raise SchemaError(message)
	return np.asarray(data.transpose("time", "y", "x").values).reshape(len(dates), -1)


def _terrain_source(terrain: xr.Dataset, candidate_names: Sequence[str]) -> str | None:
	"""Find the first terrain variable matching a documented alias list."""
	return next((candidate for candidate in candidate_names if candidate in terrain.data_vars), None)


def _validate_cell_month(frame: pd.DataFrame) -> pd.DataFrame:
	"""Validate exact cell-month columns, dtypes, and unique keys."""
	validate_schema(frame, tuple(CELL_MONTH_SCHEMA))
	extra = set(frame.columns) - set(CELL_MONTH_SCHEMA)
	if extra:
		message = f"Cell-month table contains unexpected columns: {sorted(extra)}."
		logger.error(message)
		raise SchemaError(message)
	frame = frame.loc[:, list(CELL_MONTH_SCHEMA)].copy()
	if frame.duplicated(["city", "cell_id", "date"]).any():
		message = "Cell-month table contains duplicate (city, cell_id, date) keys."
		logger.error(message)
		raise SchemaError(message)
	if not pd.api.types.is_datetime64_ns_dtype(frame["date"].dtype):
		message = "Cell-month date must have dtype datetime64[ns]."
		logger.error(message)
		raise SchemaError(message)
	if frame["date"].isna().any() or not (frame["date"].dt.day == 1).all():
		message = "Cell-month dates must be non-null month-start timestamps."
		logger.error(message)
		raise SchemaError(message)
	if not frame["cell_id"].dtype == np.dtype("int64"):
		message = "Cell-month cell_id must have dtype int64."
		logger.error(message)
		raise SchemaError(message)
	for column, dtype in CELL_MONTH_SCHEMA.items():
		if column in {"city", "date", "cell_id"}:
			continue
		try:
			frame[column] = frame[column].astype(dtype)
		except (TypeError, ValueError) as error:
			message = f"Cell-month column {column!r} has an invalid dtype."
			logger.error(message)
			raise SchemaError(message) from error
	if frame["city"].isna().any() or not frame["city"].isin(_CANONICAL_CITIES).all():
		message = "Cell-month table contains missing or invalid city keys."
		logger.error(message)
		raise SchemaError(message)
	frame["city"] = frame["city"].astype("string")
	finite = frame["urban_fraction"].dropna()
	if ((finite < 0) | (finite > 1)).any():
		message = "Cell-month urban_fraction contains values outside [0, 1]."
		logger.error(message)
		raise SchemaError(message)
	return frame


def build_cell_month(
	climate_cells: xr.Dataset,
	uf: xr.DataArray,
	indices: xr.Dataset,
	terrain: xr.Dataset,
	city: str,
) -> pd.DataFrame:
	"""Build the exact long-form monthly cell table.

	Args:
		climate_cells: Monthly cell climate Dataset with a time dimension.
		uf: Urban-fraction DataArray.
		indices: Spectral index Dataset.
		terrain: Static or monthly terrain Dataset.
		city: Canonical city key.

	Returns:
		Cell-month DataFrame with the exact ``CELL_MONTH_SCHEMA`` columns.

	Raises:
		SchemaError: If required inputs, grids, dates, or keys are invalid.
	"""
	started = time.perf_counter()
	logger.info(
		"Starting cell-month assembly (city=%s, times=%d).",
		city,
		climate_cells.sizes.get("time", 0),
	)
	if city not in _CANONICAL_CITIES:
		message = f"Unsupported city {city!r}."
		logger.error(message)
		raise SchemaError(message)
	for name, raster in (
		("climate", climate_cells),
		("urban fraction", uf),
		("indices", indices),
		("terrain", terrain),
	):
		_validate_raster_alignment(raster, city, name)
	if "time" not in climate_cells.dims:
		message = "Climate-cell data must have a time dimension."
		logger.error(message)
		raise SchemaError(message)
	dates = pd.DatetimeIndex(climate_cells.time.values).as_unit("ns")
	if dates.has_duplicates or not dates.is_monotonic_increasing or not (dates.day == 1).all():
		message = "Climate-cell dates must be unique, increasing month-start values."
		logger.error(message)
		raise SchemaError(message)
	for required in ("t2m_coarse", "RH2M", "PRECIP", "LST"):
		if required not in climate_cells.data_vars:
			message = f"Climate-cell dataset missing required variable {required!r}."
			logger.error(message)
			raise SchemaError(message)
	for required in ("ndvi", "ndbi", "bsi"):
		if required not in indices.data_vars:
			message = f"Index dataset missing required variable {required!r}."
			logger.error(message)
			raise SchemaError(message)
	x, y = _cell_coordinates(climate_cells, city)
	y_size = climate_cells.sizes["y"]
	x_size = climate_cells.sizes["x"]
	cell_count = y_size * x_size
	cell_ids = np.arange(cell_count, dtype=np.int64)
	time_count = len(dates)
	terrain_sources = {
		"dem": _terrain_source(terrain, ("dem", "elevation")),
		"slope": _terrain_source(terrain, ("slope",)),
		"aspect_sin": _terrain_source(terrain, ("aspect_sin", "sin_m")),
		"aspect_cos": _terrain_source(terrain, ("aspect_cos", "cos_m")),
		"twi": _terrain_source(terrain, ("twi",)),
		"solar_rad": _terrain_source(terrain, ("solar_rad", "solar_radiation_proxy")),
	}
	missing_terrain = [name for name, source in terrain_sources.items() if source is None]
	tpi_variables = sorted(
		[name for name in terrain.data_vars if name.startswith("tpi_")],
		key=lambda item: float(item.rsplit("_", 1)[-1].removesuffix("m"))
		if item.rsplit("_", 1)[-1].removesuffix("m").replace(".", "", 1).isdigit()
		else float("inf"),
	)
	if missing_terrain or len(tpi_variables) < 2:
		message = (
			f"Terrain dataset missing variables {missing_terrain} or two TPI scales."
		)
		logger.error(message)
		raise SchemaError(message)

	values: dict[str, np.ndarray] = {
		"urban_fraction": _flatten_raster(uf, dates, y_size, x_size),
		"ndvi": _flatten_raster(indices["ndvi"], dates, y_size, x_size),
		"ndbi": _flatten_raster(indices["ndbi"], dates, y_size, x_size),
		"bsi": _flatten_raster(indices["bsi"], dates, y_size, x_size),
	}
	for output, source in terrain_sources.items():
		assert source is not None
		values[output] = _flatten_raster(terrain[source], dates, y_size, x_size)
	values["tpi_small"] = _flatten_raster(
		terrain[tpi_variables[0]], dates, y_size, x_size
	)
	values["tpi_large"] = _flatten_raster(
		terrain[tpi_variables[-1]], dates, y_size, x_size
	)
	climate_names = {
		"t2m_coarse": "t2m_coarse",
		"RH2M": "RH2M",
		"PRECIP": "PRECIP",
		"LST": "LST",
	}
	for output, source in climate_names.items():
		values[output] = _flatten_raster(
			climate_cells[source], dates, y_size, x_size
		)
	for optional, source in (
		("t2m_downscaled", "t2m_downscaled"),
		("DTR_cell", "DTR_cell"),
	):
		if source in climate_cells.data_vars:
			values[optional] = _flatten_raster(
				climate_cells[source], dates, y_size, x_size
			)
		else:
			values[optional] = np.full((time_count, cell_count), np.nan)
	if "spatial_block" in climate_cells.data_vars:
		values["spatial_block"] = _flatten_raster(
			climate_cells["spatial_block"], dates, y_size, x_size
		)
	else:
		values["spatial_block"] = np.full((time_count, cell_count), pd.NA, dtype=object)
	date_values = np.repeat(dates.to_numpy(dtype="datetime64[ns]"), cell_count)
	frame_data: dict[str, Any] = {
		"city": np.full(time_count * cell_count, city, dtype=object),
		"cell_id": np.tile(cell_ids, time_count),
		"date": date_values,
		"x": np.tile(np.repeat(x, y_size), time_count),
		"y": np.tile(np.tile(y, x_size), time_count),
	}
	for column, array in values.items():
		frame_data[column] = np.asarray(array).reshape(-1)
	frame = pd.DataFrame(frame_data)
	seasonality = add_seasonality(pd.DataFrame({"date": frame["date"]}))
	frame["is_monsoon"] = seasonality["is_monsoon"].to_numpy()
	frame["sin_m"] = seasonality["sin_m"].to_numpy()
	frame["cos_m"] = seasonality["cos_m"].to_numpy()
	frame["spatial_block"] = pd.array(frame["spatial_block"], dtype="Int64")
	frame = _validate_cell_month(frame)
	_validate_month_axis(frame["date"], city)
	if len(frame) != time_count * cell_count:
		message = "Cell-month row count failed source reconciliation."
		logger.error(message)
		raise DataIntegrityError(message)
	logger.info(
		"Finished cell-month assembly (rows=%d, cells=%d) in %.3fs.",
		len(frame),
		cell_count,
		time.perf_counter() - started,
	)
	return frame


def _config_hash() -> str:
	"""Compute a stable hash of effective configuration fields."""
	config_json = json.dumps(
		asdict(get_config()),
		sort_keys=True,
		default=str,
	)
	return hashlib.sha256(config_json.encode("utf-8")).hexdigest()


def _git_hash() -> str:
	"""Return the current repository commit identifier or an explicit fallback."""
	try:
		result = subprocess.run(
			["git", "rev-parse", "HEAD"],
			check=True,
			capture_output=True,
			text=True,
			timeout=5,
		)
	except (OSError, subprocess.SubprocessError):
		return "unavailable"
	return result.stdout.strip()


def _parquet_metadata(
	frame: pd.DataFrame,
	schema: dict[str, str],
	git_hash: str,
) -> dict[bytes, bytes]:
	"""Build provenance and schema metadata for a parquet table."""
	metadata: dict[str, str] = {
		"schema": json.dumps(schema, sort_keys=True),
		"build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
		"git_hash": git_hash,
		"config_hash": _config_hash(),
	}
	if "climatology_years" in frame.attrs:
		metadata["climatology_years"] = json.dumps(frame.attrs["climatology_years"])
	return {key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()}


def _log_data_card(frame: pd.DataFrame, table_name: str) -> None:
	"""Log row count and per-column missing-value rates."""
	if frame.empty:
		logger.warning("Data card %s: zero rows.", table_name)
		return
	nan_rates = frame.isna().mean().to_dict()
	logger.info(
		"Data card %s: rows=%d, columns=%d, NaN rates=%s.",
		table_name,
		len(frame),
		len(frame.columns),
		json.dumps(nan_rates, sort_keys=True),
	)


def _calendarize_for_gate(
	climate: pd.DataFrame,
	cities: Sequence[str],
) -> pd.DataFrame:
	"""Insert missing calendar rows as NaNs for Gate 0 only, without output filling."""
	frames: list[pd.DataFrame] = []
	for city in cities:
		city_frame = climate.loc[climate["city"] == city].sort_values("date")
		if city_frame.empty:
			continue
		dates = pd.DatetimeIndex(city_frame["date"])
		calendar = pd.date_range(dates.min(), dates.max(), freq="MS")
		if len(calendar) != len(dates):
			logger.warning(
				"Gate 0 found %d missing calendar months for %s; source rows remain unchanged.",
				len(calendar.difference(dates)),
				city,
			)
		aligned = (
			city_frame.set_index("date")
			.reindex(calendar)
			.rename_axis("date")
			.reset_index()
		)
		aligned["city"] = city
		frames.append(aligned)
	return pd.concat(frames, ignore_index=True) if frames else climate.iloc[0:0].copy()


def _write_city_parquet(frame: pd.DataFrame, path: Path, git_hash: str) -> None:
	"""Write a city-month parquet file with strict schema metadata."""
	try:
		import pyarrow as pa
		import pyarrow.parquet as pq
	except ImportError as error:
		message = "pyarrow is required to write Block C parquet outputs."
		logger.error(message)
		raise DataIntegrityError(message) from error
	schema = pa.schema(
		[
			pa.field("city", pa.string(), nullable=False),
			pa.field("date", pa.timestamp("ns"), nullable=False),
			*[pa.field(name, pa.int8() if name == "is_monsoon" else pa.int64() if name == "n_valid_cells" else pa.float64()) for name in list(CITY_MONTH_SCHEMA)[2:]],
		],
		metadata=_parquet_metadata(frame, CITY_MONTH_SCHEMA, git_hash),
	)
	try:
		table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
		path.parent.mkdir(parents=True, exist_ok=True)
		pq.write_table(table, path)
	except (OSError, ValueError, TypeError, pa.ArrowException) as error:
		message = f"Could not write city-month parquet to {path}."
		logger.error(message)
		raise DataIntegrityError(message) from error


def _write_cell_partitioned(
	frame: pd.DataFrame,
	directory: Path,
	git_hash: str,
) -> None:
	"""Write yearly city-partitioned cell-month parquet chunks."""
	try:
		import pyarrow as pa
		import pyarrow.parquet as pq
	except ImportError as error:
		message = "pyarrow is required to write Block C parquet outputs."
		logger.error(message)
		raise DataIntegrityError(message) from error
	try:
		directory.mkdir(parents=True, exist_ok=True)
	except OSError as error:
		message = f"Could not create cell-month output directory {directory}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	metadata = _parquet_metadata(frame, CELL_MONTH_SCHEMA, git_hash)
	arrow_types = {
		"city": pa.string(),
		"cell_id": pa.int64(),
		"date": pa.timestamp("ns"),
		"spatial_block": pa.int64(),
		"is_monsoon": pa.int8(),
	}
	schema = pa.schema(
		[
			pa.field(column, arrow_types.get(column, pa.float64()), nullable=column not in {"city", "cell_id", "date"})
			for column in CELL_MONTH_SCHEMA
		],
		metadata=metadata,
	)
	rows_per_chunk = get_config().dataset.rows_per_chunk
	if rows_per_chunk < 1:
		message = "dataset.rows_per_chunk must be positive."
		logger.error(message)
		raise SchemaError(message)
	for (city, year), year_frame in frame.assign(
		year=frame["date"].dt.year.astype("int32")
	).groupby(["city", "year"], sort=True):
		partition = directory / f"city={city}" / f"year={year}"
		try:
			partition.mkdir(parents=True, exist_ok=True)
		except OSError as error:
			message = f"Could not create cell-month parquet partition {partition}."
			logger.error(message)
			raise DataIntegrityError(message) from error
		for offset in range(0, len(year_frame), rows_per_chunk):
			chunk = year_frame.iloc[offset : offset + rows_per_chunk].drop(columns="year")
			try:
				table = pa.Table.from_pandas(
					chunk.loc[:, list(CELL_MONTH_SCHEMA)],
					schema=schema,
					preserve_index=False,
					safe=True,
				)
				pq.write_table(
					table, partition / f"part-{offset // rows_per_chunk:05d}.parquet"
				)
			except (OSError, ValueError, TypeError, pa.ArrowException) as error:
				message = f"Could not write cell-month parquet partition {partition}."
				logger.error(message)
				raise DataIntegrityError(message) from error


def run(cities: Sequence[str]) -> dict[str, Path]:
	"""Load configured Block B outputs, gate raw climate series, and write tables.

	Args:
		cities: Canonical city keys to process.

	Returns:
		Paths to the city-month parquet and partitioned cell-month dataset.

	Raises:
		SchemaError: If configured inputs or city keys are invalid.
		DataIntegrityError: If source gates or output writes fail.
	"""
	started = time.perf_counter()
	logger.info("Starting Block C dataset build (cities=%d).", len(cities))
	config = get_config()
	if not cities or len(set(cities)) != len(cities):
		message = "Cities must be a non-empty sequence of unique canonical keys."
		logger.error(message)
		raise SchemaError(message)
	if not set(cities).issubset(_CANONICAL_CITIES):
		message = f"Unsupported city keys: {sorted(set(cities) - _CANONICAL_CITIES)}."
		logger.error(message)
		raise SchemaError(message)
	try:
		import pyarrow as pa
		import pyarrow.parquet as pq
	except ImportError as error:
		message = "pyarrow is required to write Block C parquet outputs."
		logger.error(message)
		raise DataIntegrityError(message) from error

	climate_city_path = config.dataset.city_climate_path
	if not climate_city_path.is_file():
		message = f"Configured city-climate input does not exist: {climate_city_path}."
		logger.error(message)
		raise SchemaError(message)
	try:
		climate_city = pd.read_parquet(climate_city_path)
	except (OSError, ValueError, ImportError) as error:
		message = f"Could not read configured city-climate input: {climate_city_path}."
		logger.error(message)
		raise DataIntegrityError(message) from error
	validate_schema(climate_city, ("city", "date", *_CITY_CLIMATE_COLUMNS))
	raw_columns = [column for column in _CITY_CLIMATE_COLUMNS if column in climate_city]
	if "DTR" not in raw_columns:
		message = "Raw climate forcing is missing DTR required by Gate 0."
		logger.error(message)
		raise SchemaError(message)
	gate_input = _calendarize_for_gate(climate_city, cities)
	run_gate(gate_input, raw_columns)
	git_hash = _git_hash()
	city_frames: list[pd.DataFrame] = []
	cell_frames: list[pd.DataFrame] = []
	for city in cities:
		climate = climate_city.loc[climate_city["city"] == city].copy()
		if climate.empty:
			message = f"No city climate rows found for {city!r}."
			logger.error(message)
			raise SchemaError(message)
		try:
			climate_cells = xr.open_zarr(config.dataset.cell_climate_path[city])
			uf_dataset = xr.open_zarr(config.dataset.urban_fraction_path[city])
			indices = xr.open_zarr(config.dataset.indices_path[city])
			terrain = xr.open_zarr(config.dataset.terrain_path[city])
		except (KeyError, OSError, ValueError) as error:
			message = f"Could not load configured raster inputs for {city!r}."
			logger.error(message)
			raise DataIntegrityError(message) from error
		if "urban_fraction" not in uf_dataset:
			message = f"Urban-fraction store for {city!r} lacks urban_fraction."
			logger.error(message)
			raise SchemaError(message)
		uf = uf_dataset["urban_fraction"]
		city_frame = build_city_month(climate, uf, city)
		cell_frame = build_cell_month(
			climate_cells, uf, indices, terrain, city
		)
		city_valid_count = int(city_frame["n_valid_cells"].sum())
		cell_valid_count = int(cell_frame["urban_fraction"].count())
		if city_valid_count != cell_valid_count:
			message = (
				f"Urban-fraction row-count reconciliation failed for {city}: "
				f"city aggregate={city_valid_count}, cell table={cell_valid_count}."
			)
			logger.error(message)
			raise DataIntegrityError(message)
		city_frames.append(city_frame)
		cell_frames.append(cell_frame)
		if hasattr(climate_cells, "close"):
			climate_cells.close()
		if hasattr(uf_dataset, "close"):
			uf_dataset.close()
		if hasattr(indices, "close"):
			indices.close()
		if hasattr(terrain, "close"):
			terrain.close()

	city_output = pd.concat(city_frames, ignore_index=True)
	city_output.attrs["climatology_years"] = config.dataset.climatology_years
	city_output = _validate_city_month(city_output)
	cell_output = pd.concat(cell_frames, ignore_index=True)
	cell_output = _validate_cell_month(cell_output)
	_log_data_card(city_output, "city_month")
	_log_data_card(cell_output, "cell_month")
	city_path = config.paths.interim / config.dataset.city_month_filename
	cell_path = config.paths.interim / config.dataset.cell_month_directory
	try:
		_write_city_parquet(city_output, city_path, git_hash)
		_write_cell_partitioned(cell_output, cell_path, git_hash)
	except (OSError, ValueError) as error:
		message = "Could not persist Block C parquet outputs."
		logger.error(message)
		raise DataIntegrityError(message) from error
	logger.info(
		"Finished Block C dataset build (city_rows=%d, cell_rows=%d) in %.3fs.",
		len(city_output),
		len(cell_output),
		time.perf_counter() - started,
	)
	return {"city_month": city_path, "cell_month": cell_path}


if __name__ == "__main__":
	"""Build both strict dataset schemas from a compact synthetic example."""
	_config = get_config()
	_city = "dehradun"
	_dates = pd.date_range(
		"2000-01-01", periods=24, freq="MS"
	).as_unit("ns")
	_sine = np.sin(2 * np.pi * np.arange(len(_dates)) / 12)
	_tmin = 5.0 + _sine
	_tmax = _tmin + 10.0 + 0.5 * np.cos(2 * np.pi * np.arange(len(_dates)) / 12)
	_climate = pd.DataFrame(
		{
			"date": _dates,
			"DTR": _tmax - _tmin,
			"TMAX": _tmax,
			"TMIN": _tmin,
			"LST": 15.0 + _sine,
			"RH2M": 60.0 + 10 * _sine,
			"PRECIP": 20.0 + 5 * _sine,
		}
	)
	_transform = _config.grid.transform[_city]
	_x = _transform.c + _transform.a * (np.arange(2) + 0.5)
	_y = _transform.f + _transform.e * (np.arange(2) + 0.5)
	_uf = xr.DataArray(
		np.full((24, 2, 2), 0.4, dtype=np.float32),
		dims=("time", "y", "x"),
		coords={"time": _dates, "y": _y, "x": _x},
		name="urban_fraction",
	).rio.write_crs(_config.grid.crs[_city]).rio.write_transform(_transform)
	_terrain = xr.Dataset(
		{
			"elevation": (("y", "x"), np.ones((2, 2))),
			"slope": (("y", "x"), np.ones((2, 2))),
			"aspect_sin": (("y", "x"), np.zeros((2, 2))),
			"aspect_cos": (("y", "x"), np.ones((2, 2))),
			"tpi_300m": (("y", "x"), np.ones((2, 2))),
			"tpi_3000m": (("y", "x"), np.ones((2, 2))),
			"twi": (("y", "x"), np.ones((2, 2))),
			"solar_radiation_proxy": (("y", "x"), np.ones((2, 2))),
		},
		coords={"y": _y, "x": _x},
	).rio.write_crs(_config.grid.crs[_city]).rio.write_transform(_transform)
	_climate_cells = xr.Dataset(
		{
			"t2m_coarse": (("time", "y", "x"), np.ones((24, 2, 2))),
			"RH2M": (("time", "y", "x"), np.full((24, 2, 2), 60.0)),
			"PRECIP": (("time", "y", "x"), np.full((24, 2, 2), 20.0)),
			"LST": (("time", "y", "x"), np.full((24, 2, 2), 15.0)),
		},
		coords={"time": _dates, "y": _y, "x": _x},
	).rio.write_crs(_config.grid.crs[_city]).rio.write_transform(_transform)
	_indices = xr.Dataset(
		{
			"ndvi": (("time", "y", "x"), np.full((24, 2, 2), 0.4)),
			"ndbi": (("time", "y", "x"), np.full((24, 2, 2), 0.1)),
			"bsi": (("time", "y", "x"), np.full((24, 2, 2), 0.2)),
		},
		coords={"time": _dates, "y": _y, "x": _x},
	).rio.write_crs(_config.grid.crs[_city]).rio.write_transform(_transform)
	_city_frame = build_city_month(_climate, _uf, _city)
	_cell_frame = build_cell_month(
		_climate_cells, _uf, _indices, _terrain, _city
	)
	assert tuple(_city_frame.columns) == tuple(CITY_MONTH_SCHEMA)
	assert tuple(_cell_frame.columns) == tuple(CELL_MONTH_SCHEMA)
	_bad_climate = _climate.copy()
	_bad_climate.loc[0, "DTR"] += 1.0
	try:
		build_city_month(_bad_climate, _uf, _city)
	except SchemaError:
		pass
	else:
		raise AssertionError("Corrupt DTR must fail city-month schema validation.")
