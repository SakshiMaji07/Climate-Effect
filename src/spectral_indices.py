from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from config import get_config
from exceptions import SchemaError
from logger import get_logger
from raster_processing import load_grid

logger = get_logger(__name__)


def _sensor_map(sensor: str) -> dict[str, str]:
    """Return the configured band-name map for a sensor.

    Args:
        sensor: Sensor name such as ``landsat8_9``.

    Returns:
        Mapping from canonical reflectance names to sensor-specific band names.

    Raises:
        SchemaError: If the sensor is not configured.
    """
    band_map = get_config().spectral.band_map
    if sensor not in band_map:
        raise SchemaError(f"Unsupported sensor {sensor!r}.")
    return band_map[sensor]


def _ensure_required_bands(ds: xr.Dataset, sensor: str) -> None:
    """Ensure all reflectance bands required by a sensor are present."""
    missing = [band for band in _sensor_map(sensor).values() if band not in ds.data_vars]
    if missing:
        raise SchemaError(f"Scene missing required reflectance bands: {missing}.")


def _get_city(ds: xr.Dataset) -> str:
    """Return the city label from the dataset metadata or config."""
    city = str(ds.attrs.get("city", "dehradun"))
    if city not in get_config().grid.crs:
        city = "dehradun"
    return city


def _band_name(ds: xr.Dataset, canonical: str, sensor: str) -> str:
    """Resolve the sensor-specific band name for a canonical reflectance band."""
    return _sensor_map(sensor)[canonical]


def _grid_match(ds: xr.Dataset, city: str) -> None:
    """Validate that a dataset is aligned to the canonical city grid."""
    cfg = get_config()
    expected_epsg = int(cfg.grid.crs[city].split(":")[-1])
    if ds.rio.crs is None or ds.rio.crs.to_epsg() != expected_epsg:
        raise SchemaError(f"Grid CRS mismatch for city {city!r}.")
    expected_shape = cfg.grid.shape.get(city, cfg.grid.shape["dehradun"])
    dims = ds.sizes
    if dims.get("y") is None or dims.get("x") is None:
        raise SchemaError(f"Grid dims missing for city {city!r}.")
    if (dims["y"], dims["x"]) != expected_shape:
        raise SchemaError(f"Grid shape mismatch for city {city!r}: {dims}.")


def _apply_reflectance_scale(ds: xr.Dataset, sensor: str) -> xr.Dataset:
    """Apply scale/offset and clamp reflectance to the valid range [0, 1]."""
    cfg = get_config().spectral
    scaled = ds.copy()
    for canonical, band_name in _sensor_map(sensor).items():
        if band_name not in scaled.data_vars:
            continue
        band = scaled[band_name].astype(np.float32)
        band = band * cfg.scale_factors["reflectance"] + cfg.scale_factors["offset"]
        scaled[band_name] = band.clip(0.0, 1.0)
    return scaled


def _mask_qa(ds: xr.Dataset, sensor: str) -> xr.Dataset:
    """Mask cloud, shadow and snow flags using QA_PIXEL/SCL if available."""
    masked = ds.copy()
    qa_vars = [name for name in ("QA_PIXEL", "SCL", "qa") if name in masked.data_vars]
    if not qa_vars:
        return masked
    qa_name = qa_vars[0]
    qa = masked[qa_name]
    cfg = get_config().spectral
    cloud_bit = cfg.qa_cloud_bit
    shadow_bit = cfg.qa_cloud_shadow_bit
    snow_bit = cfg.qa_snow_bit
    cloud_mask = ((qa >> cloud_bit) & 1) == 1
    shadow_mask = ((qa >> shadow_bit) & 1) == 1
    snow_mask = ((qa >> snow_bit) & 1) == 1
    bad = cloud_mask | shadow_mask | snow_mask
    for band_name in _sensor_map(sensor).values():
        if band_name in masked.data_vars:
            masked[band_name] = masked[band_name].where(~bad)
    return masked


def apply_scale_and_mask(ds: xr.Dataset, sensor: str) -> xr.Dataset:
    """Apply scale factors, QA masking, and reflectance clamping.

    Args:
        ds: Dataset containing reflectance bands for one scene.
        sensor: Sensor name (for example ``landsat8_9``).

    Returns:
        A dataset with masked and clamped reflectance values.

    Raises:
        SchemaError: If required bands are missing.
    """
    logger.info("Applying scale and mask for sensor %s.", sensor)
    if not isinstance(ds, xr.Dataset):
        raise SchemaError("apply_scale_and_mask expects an xarray.Dataset.")
    _ensure_required_bands(ds, sensor)
    masked = _mask_qa(ds, sensor)
    return _apply_reflectance_scale(masked, sensor)


def safe_norm_diff(a: xr.DataArray, b: xr.DataArray, eps: float = 1e-6) -> xr.DataArray:
    """Compute a safe normalized difference index.

    Args:
        a: Numerator band.
        b: Denominator band.
        eps: Small value to avoid division by zero.

    Returns:
        Ratio clipped to finite values, with NaNs where the denominator is too small.
    """
    denom = a + b
    mask = denom < eps
    result = (a - b) / (denom + eps)
    return result.where(~mask)


def ndvi(ds: xr.Dataset) -> xr.DataArray:
    """Compute the normalized difference vegetation index."""
    sensor = ds.attrs.get("sensor", "landsat8_9")
    red = ds[_band_name(ds, "red", sensor)]
    nir = ds[_band_name(ds, "nir", sensor)]
    return safe_norm_diff(nir, red).clip(-1.0, 1.0)


def ndbi(ds: xr.Dataset) -> xr.DataArray:
    """Compute the normalized difference built-up index."""
    sensor = ds.attrs.get("sensor", "landsat8_9")
    swir1 = ds[_band_name(ds, "swir1", sensor)]
    nir = ds[_band_name(ds, "nir", sensor)]
    return safe_norm_diff(swir1, nir).clip(-1.0, 1.0)


def ndwi(ds: xr.Dataset) -> xr.DataArray:
    """Compute the normalized difference water index."""
    sensor = ds.attrs.get("sensor", "landsat8_9")
    green = ds[_band_name(ds, "green", sensor)]
    nir = ds[_band_name(ds, "nir", sensor)]
    return safe_norm_diff(green, nir).clip(-1.0, 1.0)


def mndwi(ds: xr.Dataset) -> xr.DataArray:
    """Compute the modified normalized difference water index."""
    sensor = ds.attrs.get("sensor", "landsat8_9")
    green = ds[_band_name(ds, "green", sensor)]
    swir1 = ds[_band_name(ds, "swir1", sensor)]
    return safe_norm_diff(green, swir1).clip(-1.0, 1.0)


def bsi(ds: xr.Dataset) -> xr.DataArray:
    """Compute the bare soil index."""
    sensor = ds.attrs.get("sensor", "landsat8_9")
    swir1 = ds[_band_name(ds, "swir1", sensor)]
    red = ds[_band_name(ds, "red", sensor)]
    nir = ds[_band_name(ds, "nir", sensor)]
    blue = ds[_band_name(ds, "blue", sensor)]
    numerator = (swir1 + red) - (nir + blue)
    denominator = (swir1 + red) + (nir + blue)
    result = numerator / (denominator + 1e-6)
    return result.where(denominator > 1e-6).clip(-1.0, 1.0)


def harmonise_sensors(ds: xr.Dataset, from_sensor: str, to_sensor: str) -> xr.Dataset:
    """Apply band harmonisation between two sensors."""
    if from_sensor == to_sensor:
        return ds.copy()
    cfg = get_config().spectral
    harmonisation = cfg.harmonise.get(from_sensor, {}).get(to_sensor, {})
    if not harmonisation:
        logger.warning("No sensor harmonisation defined for %s -> %s; leaving scene unmodified.", from_sensor, to_sensor)
        return ds.copy()
    out = ds.copy()
    for canonical, coefficient in harmonisation.items():
        source_name = _sensor_map(from_sensor)[canonical]
        if source_name in out.data_vars:
            out[source_name] = out[source_name] * float(coefficient)
    for canonical, target_name in _sensor_map(to_sensor).items():
        if target_name not in out.data_vars and canonical in _sensor_map(from_sensor):
            source_name = _sensor_map(from_sensor)[canonical]
            if source_name in out.data_vars:
                out[target_name] = out[source_name].copy()
    out.attrs["sensor"] = to_sensor
    return out


def compute_index_stack(scene: xr.Dataset, sensor: str) -> xr.Dataset:
    """Compute vegetation and water indices for a single scene.

    Args:
        scene: Dataset containing all sensor bands on the canonical grid.
        sensor: Sensor name.

    Returns:
        Dataset with index variables and a valid mask.

    Raises:
        SchemaError: If required bands or grid metadata are invalid.
    """
    if not isinstance(scene, xr.Dataset):
        raise SchemaError("compute_index_stack expects an xarray.Dataset.")
    city = _get_city(scene)
    _grid_match(scene, city)
    prepared = apply_scale_and_mask(scene, sensor)
    prepared.attrs["sensor"] = sensor
    prepared.attrs["city"] = city
    stack = xr.Dataset(
        {
            "ndvi": ndvi(prepared),
            "ndbi": ndbi(prepared),
            "ndwi": ndwi(prepared),
            "mndwi": mndwi(prepared),
            "bsi": bsi(prepared),
        },
        coords={key: prepared.coords[key] for key in ("time", "y", "x") if key in prepared.coords},
    )
    valid_mask = xr.concat(
        [xr.where(np.isfinite(stack[var]), 1, 0) for var in stack.data_vars],
        dim="index_var",
    ).max("index_var").astype(bool)
    stack["valid_mask"] = valid_mask
    for var in stack.data_vars:
        if var != "valid_mask":
            stack[var] = stack[var].where(valid_mask)
    return stack


def monthly_composite(
    stack: xr.Dataset,
    method: Literal["median", "mean"] = "median",
    min_valid: int = 1,
) -> xr.Dataset:
    """Aggregate a monthly index stack over the time dimension.

    Args:
        stack: Dataset with variables on the ``time`` dimension.
        method: Month aggregation method.
        min_valid: Minimum count of valid pixels needed for a monthly summary.

    Returns:
        Monthly composite dataset with a ``n_valid`` counter variable.
    """
    if "time" not in stack.dims:
        raise SchemaError("monthly_composite requires a time dimension.")
    if method.lower() not in {"median", "mean"}:
        raise SchemaError(f"Unsupported monthly_composite method: {method!r}.")
    valid_counts = stack["valid_mask"].groupby("time.month").sum("time")
    out = xr.Dataset()
    for var in [name for name in stack.data_vars if name != "valid_mask"]:
        reducer = stack[var].groupby("time.month").median if method.lower() == "median" else stack[var].groupby("time.month").mean
        monthly = reducer("time")
        out[var] = monthly.where(valid_counts >= min_valid)
    out["n_valid"] = valid_counts
    return out


def save_index_stack(ds: xr.Dataset, city: str) -> Path:
    """Persist the index stack to a Zarr store under the project interim path.

    Args:
        ds: Dataset to save.
        city: City key used in the output path.

    Returns:
        Path to the written Zarr store.
    """
    cfg = get_config()
    target = cfg.paths.interim / city / "index_stack.zarr"
    target.mkdir(parents=True, exist_ok=True)
    encoding = {var: {"chunks": cfg.spectral.chunking} for var in ds.data_vars}
    ds.to_zarr(target, mode="w", consolidated=True, encoding=encoding)
    return target


if __name__ == "__main__":
    """Validate the spectral index pipeline on synthetic monthly scenes."""
    try:
        cfg = get_config()
        city = "dehradun"
        sensor = "landsat8_9"
        time = np.array(["2020-01-01", "2020-02-01", "2020-03-01"], dtype="datetime64[ns]")
        y = np.arange(2, dtype=float)
        x = np.arange(2, dtype=float)
        values = np.linspace(0.1, 0.9, 12, dtype=np.float32).reshape(3, 2, 2)
        base = xr.DataArray(values, dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x})
        red = base.copy()
        nir = (base + 0.2).clip(0.0, 1.0)
        green = ((base + nir) / 2.0).clip(0.0, 1.0)
        swir1 = (base * 0.8).clip(0.0, 1.0)
        swir2 = (base * 0.9).clip(0.0, 1.0)
        blue = (base * 0.7).clip(0.0, 1.0)
        qa = xr.DataArray(np.zeros_like(values, dtype=np.uint16), dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x})
        scene = xr.Dataset(
            {
                "B2": blue,
                "B3": green,
                "B4": red,
                "B5": nir,
                "B6": swir1,
                "B7": swir2,
                "QA_PIXEL": qa,
            },
            attrs={"city": city, "sensor": sensor},
        )
        scene = load_grid(scene, city=city)
        scene = scene.rio.write_crs(cfg.grid.crs[city])
        scene = scene.rio.write_transform(cfg.grid.transform[city])
        stack = compute_index_stack(scene, sensor)
        assert set(["ndvi", "ndbi", "ndwi", "mndwi", "bsi", "valid_mask"]).issubset(stack.data_vars)
        assert np.all((stack["ndvi"].to_numpy() >= -1.0) & (stack["ndvi"].to_numpy() <= 1.0))
        assert np.all((stack["bsi"].to_numpy() >= -1.0) & (stack["bsi"].to_numpy() <= 1.0))
        assert stack["valid_mask"].dtype == bool
        composite = monthly_composite(stack, method="median", min_valid=1)
        assert "n_valid" in composite.data_vars
        assert composite["n_valid"].sizes.get("month") == 3
    except Exception as exc:  # pragma: no cover - CLI safety net
        logger.error("Synthetic spectral stack validation failed: %s", exc)
        raise SystemExit(1) from exc
