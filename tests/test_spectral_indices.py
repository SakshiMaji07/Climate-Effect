from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from config import get_config
from exceptions import SchemaError
from spectral_indices import (
    apply_scale_and_mask,
    compute_index_stack,
    monthly_composite,
    safe_norm_diff,
)


@pytest.fixture
def synthetic_scene() -> xr.Dataset:
    cfg = get_config()
    time = pd.date_range("2020-01-01", periods=3, freq="MS").values
    y = np.arange(2, dtype=float)
    x = np.arange(2, dtype=float)
    base = np.linspace(0.15, 0.8, 12, dtype=np.float32).reshape(3, 2, 2)
    red = np.clip(base, 0.0, 1.0)
    nir = np.clip(base + 0.20, 0.0, 1.0)
    green = np.clip((red + nir) / 2.0, 0.0, 1.0)
    swir1 = np.clip(base * 0.75, 0.0, 1.0)
    swir2 = np.clip(base * 0.85, 0.0, 1.0)
    blue = np.clip(base * 0.6, 0.0, 1.0)
    ds = xr.Dataset(
        {
            "B2": xr.DataArray(blue, dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x}),
            "B3": xr.DataArray(green, dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x}),
            "B4": xr.DataArray(red, dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x}),
            "B5": xr.DataArray(nir, dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x}),
            "B6": xr.DataArray(swir1, dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x}),
            "B7": xr.DataArray(swir2, dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x}),
            "QA_PIXEL": xr.DataArray(np.zeros((3, 2, 2), dtype=np.uint16), dims=("time", "y", "x"), coords={"time": time, "y": y, "x": x}),
        },
        attrs={"city": "dehradun", "sensor": "landsat8_9"},
    )
    ds = ds.rio.write_crs(cfg.grid.crs["dehradun"])
    ds = ds.rio.write_transform(cfg.grid.transform["dehradun"])
    return ds


def test_safe_norm_diff_returns_zero_when_equal():
    a = xr.DataArray([0.2, 0.4], dims=("x",))
    b = xr.DataArray([0.2, 0.4], dims=("x",))
    result = safe_norm_diff(a, b)
    assert result.values[0] == pytest.approx(0.0)
    assert result.values[1] == pytest.approx(0.0)


def test_apply_scale_and_mask_clamps_and_masks_clouds(synthetic_scene):
    processed = apply_scale_and_mask(synthetic_scene, "landsat8_9")
    assert processed["B4"].min() >= 0.0
    assert processed["B4"].max() <= 1.0


def test_compute_index_stack_returns_expected_dataset(synthetic_scene):
    stack = compute_index_stack(synthetic_scene, "landsat8_9")
    expected = {"ndvi", "ndbi", "ndwi", "mndwi", "bsi", "valid_mask"}
    assert expected.issubset(set(stack.data_vars))
    for var in ["ndvi", "ndbi", "ndwi", "mndwi", "bsi"]:
        values = stack[var].to_numpy()
        assert np.all((values >= -1.0) & (values <= 1.0))


def test_missing_band_raises_schema_error(synthetic_scene):
    dropped = synthetic_scene.drop_vars("B5")
    with pytest.raises(SchemaError, match="required reflectance bands"):
        apply_scale_and_mask(dropped, "landsat8_9")


def test_grid_mismatch_raises_schema_error(synthetic_scene):
    bad = synthetic_scene.copy()
    bad = bad.rio.write_crs("EPSG:4326")
    with pytest.raises(SchemaError, match="Grid CRS mismatch"):
        compute_index_stack(bad, "landsat8_9")


def test_monthly_composite_keeps_nan_months(synthetic_scene):
    stack = compute_index_stack(synthetic_scene, "landsat8_9")
    composite = monthly_composite(stack, method="median", min_valid=1)
    assert "n_valid" in composite.data_vars
    assert composite["n_valid"].sizes.get("month") == 3
