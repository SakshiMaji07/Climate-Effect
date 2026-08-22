import numpy as np
import rioxarray  # noqa: F401 - registers the .rio accessor
import xarray as xr
from affine import Affine

from terrain_features import compute_elevation_slope_aspect


def _dem(values):
	return xr.DataArray(values, dims=("y", "x"), name="elevation").rio.write_crs(
		"EPSG:32643"
	).rio.write_transform(Affine(10, 0, 500_000, 0, -10, 3_000_000))


def test_elevation_is_passed_through_without_scaling():
	values = np.arange(400, dtype=np.float32).reshape(20, 20)
	result = compute_elevation_slope_aspect(_dem(values))

	np.testing.assert_allclose(result["elevation"].values, values)


def test_flat_dem_has_zero_slope_and_nodata_aspect():
	result = compute_elevation_slope_aspect(_dem(np.full((20, 20), 42.0)))

	np.testing.assert_allclose(result["slope"].values, 0.0, atol=1e-10)
	assert np.isnan(result["aspect"].values).all()


def test_constant_gradient_plane_matches_analytic_slope():
	run = 10.0
	rise = 3.0
	columns = np.arange(20, dtype=float) * run
	values = np.broadcast_to(columns * rise / run, (20, 20))
	result = compute_elevation_slope_aspect(_dem(values))
	expected = np.degrees(np.arctan(rise / run))

	np.testing.assert_allclose(result["slope"].values[1:-1, 1:-1], expected, atol=1e-10)


def test_aspect_and_trigonometric_encodings_have_valid_ranges():
	rows, columns = np.indices((20, 20))
	values = rows * 2.0 + columns * 3.0
	result = compute_elevation_slope_aspect(_dem(values))
	valid = np.isfinite(result["aspect"].values)
	aspect = result["aspect"].values[valid]
	sine = result["aspect_sin"].values[valid]
	cosine = result["aspect_cos"].values[valid]

	assert np.all((aspect >= 0) & (aspect < 360))
	assert np.all((sine >= -1) & (sine <= 1))
	assert np.all((cosine >= -1) & (cosine <= 1))
	np.testing.assert_allclose(sine**2 + cosine**2, 1.0, atol=1e-10)
