"""Project configuration for the climate-effect analysis pipeline."""

from math import ceil, floor
from affine import Affine
from pyproj import CRS, Transformer


# These are provisional extents around the thesis coordinates, not surveyed
# administrative boundaries. Replace them with real OSM boundary files when
# those files are available.
CITIES = {
	"dehradun": {
		"bbox_wgs84": (77.85, 30.15, 78.25, 30.55),
		"utm_epsg": None,
	},
	"jammu": {
		"bbox_wgs84": (74.65, 32.55, 75.10, 32.95),
		"utm_epsg": None,
	},
}


def _utm_epsg_for_bbox(bbox_wgs84: tuple[float, float, float, float]) -> int:
	"""Return the northern-hemisphere UTM EPSG code for a WGS84 bbox."""
	minx, _, maxx, _ = bbox_wgs84
	lon_centroid = (minx + maxx) / 2
	return int(32600 + floor((lon_centroid + 180) / 6) + 1)


for _city in CITIES.values():
	_city["utm_epsg"] = _utm_epsg_for_bbox(_city["bbox_wgs84"])

_utm_zones = {city["utm_epsg"] for city in CITIES.values()}
if len(_utm_zones) > 1:
	print(
		"Warning: configured cities fall in different UTM zones; compare "
		"projected results with care."
	)


GRID_RESOLUTIONS_M = [100, 250]


def build_target_grid(
	city_key: str, resolution_m: int
) -> tuple[Affine, tuple[int, int], CRS]:
	"""Build a north-up grid covering a configured city's provisional bbox.

	The grid origin is snapped relative to the fixed UTM reference point
	``(0, 0)``. This makes pixel edges deterministic and aligned whenever
	resolutions share compatible multiples.
	"""
	if city_key not in CITIES:
		raise KeyError(f"Unknown city: {city_key}")
	if resolution_m <= 0:
		raise ValueError("resolution_m must be positive")

	city = CITIES[city_key]
	minx, miny, maxx, maxy = city["bbox_wgs84"]
	transformer = Transformer.from_crs("EPSG:4326", city["utm_epsg"], always_xy=True)
	minx_utm, miny_utm = transformer.transform(minx, miny)
	maxx_utm, maxy_utm = transformer.transform(maxx, maxy)

	origin_x = floor(minx_utm / resolution_m) * resolution_m
	origin_y = ceil(maxy_utm / resolution_m) * resolution_m
	width = ceil((maxx_utm - origin_x) / resolution_m)
	height = ceil((origin_y - miny_utm) / resolution_m)
	transform = Affine(resolution_m, 0, origin_x, 0, -resolution_m, origin_y)
	return transform, (height, width), CRS.from_epsg(city["utm_epsg"])


URBAN_INDEX_THRESHOLDS = {
	"ndbi_min": 0.0,
	"ndvi_max": 0.2,  # Deliberate R4 correction; the original notebook used 0.3.
	"elevation_max_m": 2000,
	"slope_max_deg": 30,
}

NODATA = {"continuous": float("nan"), "categorical": -9999}

# Placeholder paths supplied by the user; replace with local DEM files.
DEM_SOURCE_PATHS = {
	"dehradun": "data/raw/dem/dehradun.tif",
	"jammu": "data/raw/dem/jammu.tif",
}


TPI_SCALES_M = [300, 1000, 3000]
"""Tunable defaults requiring empirical validation, not fixed constants (R3)."""

FLOW_ALGORITHM = "d8"  # Configurable alternative: "dinf".
CACHE_DIR = "data/cache/terrain"
