import numpy as np
import pytest
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr
from affine import Affine

import terrain_advanced
from terrain_advanced import compute_flow_and_twi, compute_tpi


def _dem(values):
	return xr.DataArray(values, dims=("y", "x"), name="elevation").rio.write_crs(
		"EPSG:32643"
	).rio.write_transform(Affine(10, 0, 500_000, 0, -10, 3_000_000))


def test_tpi_signs_match_peak_and_pit_interpretation():
	values = np.full((20, 20), 100.0)
	values[10, 10] = 120.0
	values[5, 5] = 80.0
	result = compute_tpi(_dem(values), [10])

	tpi = result["tpi_10m"].values
	assert tpi[10, 10] > 0
	assert tpi[5, 5] < 0


def test_tpi_scales_produce_distinct_patterns():
	rows, columns = np.indices((20, 20))
	broad_hill = 20 * np.exp(-((rows - 10) ** 2 + (columns - 10) ** 2) / 80)
	small_bump = 10 * np.exp(-((rows - 6) ** 2 + (columns - 13) ** 2) / 3)
	result = compute_tpi(_dem(100 + broad_hill + small_bump), [10, 50])

	assert not np.allclose(result["tpi_10m"], result["tpi_50m"], equal_nan=True)


def _patched_twi(monkeypatch, accumulation):
	monkeypatch.setattr(terrain_advanced, "_fill_depressions", lambda values, valid: values)
	monkeypatch.setattr(
		terrain_advanced,
		"_run_flow_accumulation",
		lambda values, valid, algorithm: accumulation,
	)
	return compute_flow_and_twi(_dem(np.full((20, 20), 10.0)))


def test_twi_flat_plane_is_finite_and_bounded(monkeypatch):
	accumulation = np.full((20, 20), 2.0)
	result = _patched_twi(monkeypatch, accumulation)

	assert np.isfinite(result["twi"].values).all()
	assert np.nanmax(result["twi"].values) < 20


def test_twi_has_no_invalid_values_for_valid_inputs(monkeypatch):
	accumulation = np.full((20, 20), 5.0)
	result = _patched_twi(monkeypatch, accumulation)

	assert np.isfinite(result["twi"].values).all()


def test_twi_increases_with_accumulation_at_fixed_slope(monkeypatch):
	low = _patched_twi(monkeypatch, np.full((20, 20), 2.0))["twi"].values
	high = _patched_twi(monkeypatch, np.full((20, 20), 8.0))["twi"].values

	assert np.all(high > low)
