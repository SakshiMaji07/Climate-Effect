"""Project configuration for the climate-effect analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path

from affine import Affine
from pyproj import CRS, Transformer

from logger import get_logger


logger = get_logger(__name__)


@dataclass(frozen=True)
class SanityConfig:
	"""Thresholds controlling monthly forcing-series validation."""

	min_obs: int = 60
	max_monotonic_fraction: float = 0.85
	max_run: int = 24
	min_seasonal_strength: float = 0.2
	max_constant_diff_fraction: float = 0.5
	max_gap: int = 2
	seasonal_period: int = 12
	signature_tolerance: float = 1e-9
	min_noise_cv: float = 1e-6
	fail_fast: bool = True
	seasonal_variables: tuple[str, ...] = (
		"DTR",
		"TMAX",
		"TMIN",
		"LST",
		"RH2M",
		"PRECIP",
	)
	seed: int = 0


@dataclass(frozen=True)
class PathsConfig:
	"""Configured output directories used by pipeline stages."""

	reports: Path = Path("reports")
	sanity_subdirectory: str = "sanity"
	interim: Path = Path("data/interim")


@dataclass(frozen=True)
class DatasetConfig:
	"""Source, schema, and climatology settings for dataset construction."""

	climatology_years: tuple[int, int] = (2000, 2019)
	n_harmonics: int = 1
	monsoon_months: tuple[int, ...] = (6, 7, 8, 9)
	scale_cols: tuple[str, ...] = ("TMAX", "TMIN", "DTR", "LST", "RH2M", "PRECIP")
	impute_cols: tuple[str, ...] = ("TMAX", "TMIN", "DTR", "LST", "RH2M", "PRECIP")
	dtr_tolerance: float = 1e-3
	city_climate_path: Path = Path("data/interim/climate_city_month.parquet")
	cell_climate_path: dict[str, Path] = field(
		default_factory=lambda: {
			"dehradun": Path("data/interim/climate_cells_dehradun.zarr"),
			"jammu": Path("data/interim/climate_cells_jammu.zarr"),
		}
	)
	urban_fraction_path: dict[str, Path] = field(
		default_factory=lambda: {
			"dehradun": Path("data/interim/urban_classifier/dehradun/urban_fraction.zarr"),
			"jammu": Path("data/interim/urban_classifier/jammu/urban_fraction.zarr"),
		}
	)
	indices_path: dict[str, Path] = field(
		default_factory=lambda: {
			"dehradun": Path("data/interim/spectral_indices/dehradun.zarr"),
			"jammu": Path("data/interim/spectral_indices/jammu.zarr"),
		}
	)
	terrain_path: dict[str, Path] = field(
		default_factory=lambda: {
			"dehradun": Path("data/interim/terrain/dehradun.zarr"),
			"jammu": Path("data/interim/terrain/jammu.zarr"),
		}
	)
	city_month_filename: str = "dataset_city_month.parquet"
	cell_month_directory: str = "dataset_cell_month"
	rows_per_chunk: int = 100_000
	area_weights_variable: str = "area"


@dataclass(frozen=True)
class ValidationConfig:
	"""Leakage-safe validation defaults."""

	min_folds: int = 2
	min_blocks: int = 4
	min_block_m: float = 1_000.0
	variogram_n_lags: int = 20
	variogram_subsample: int = 5_000
	block_n_splits: int = 5
	synthetic_range_m: float = 500.0
	demo_neighbors: int = 1
	elevation_bands: dict[str, tuple[float, float]] = field(
		default_factory=lambda: {
			"default": (1_000.0, 2_000.0),
			"dehradun": (1_000.0, 2_000.0),
			"jammu": (1_000.0, 2_000.0),
		}
	)
	elevation_band_labels: tuple[str, str, str] = (
		"valley_floor",
		"mid_slope",
		"ridge",
	)
	min_elevation_band_cells: int = 1
	reliability_bins: int = 10
	classification_threshold: float = 0.5
	bootstrap_replicates: int = 1_000
	bootstrap_confidence: float = 0.95
	write_figure_data: bool = True
	expected_lapse_rate_c_per_km: float = -6.5
	lapse_tol: float = 2.0
	allowed_inversion_months: tuple[int, ...] = (12, 1)
	physical_temperature_column: str = "y_pred"
	physical_elevation_column: str = "dem"
	physical_control_columns: tuple[str, ...] = ("urban_fraction",)
	dtr_max_c: float = 50.0
	rh_min_percent: float = 0.0
	rh_max_percent: float = 100.0
	moran_permutations: int = 999
	max_morans_i: float = 0.1
	morans_alpha: float = 0.05
	spatial_weights_k: int = 8
	spatial_distance_band_m: float | None = None
	strict: bool = True
	physical_report_directory: Path = Path("reports/physical_checks")


@dataclass(frozen=True)
class EconometricsConfig:
	"""Collinearity and stationarity diagnostic thresholds."""

	vif_max: float = 10.0
	vif_warn: float = 5.0
	condition_index_max: float = 30.0
	variance_proportion_min: float = 0.5
	unit_root_alpha: float = 0.05
	seasonal_period: int = 12
	min_stationarity_obs: int = 36
	correlation_warn: float = 0.9
	min_obs: int = 40
	max_lag_dep: int = 4
	max_lag_x: int = 4
	ic: str = "bic"
	bounds_case: int = 3
	bounds_simulations: int = 20_000
	bounds_seed: int = 0
	narayan_critical_values: dict[str, dict[str, dict[str, dict[str, float]]]] = field(
		default_factory=lambda: {
			"3": {
				"1": {
					"30": {
						"10_lower": 3.332,
						"10_upper": 4.08,
						"5_lower": 3.797,
						"5_upper": 4.855,
						"1_lower": 5.241,
						"1_upper": 6.518,
					}
				}
			}
		}
	)
	dynamic_horizon: int = 24
	count_targets: tuple[str, ...] = ("heat_snap_count", "cold_snap_count")
	panel: bool = False
	output_directory: str = "econometrics"
	residual_lags: int = 12


@dataclass(frozen=True)
class GridConfig:
	"""Canonical spatial metadata for each city."""

	crs: dict[str, str] = field(
		default_factory=lambda: {"dehradun": "EPSG:32644", "jammu": "EPSG:32643"}
	)
	transform: dict[str, Affine] = field(
		default_factory=lambda: {
			"dehradun": Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 350000.0),
			"jammu": Affine(10.0, 0.0, 200000.0, 0.0, -10.0, 3800000.0),
		}
	)
	shape: dict[str, tuple[int, int]] = field(
		default_factory=lambda: {"dehradun": (2, 2), "jammu": (2, 2)}
	)
	block_size: int = 1000


@dataclass(frozen=True)
class SpectralBandMap:
	"""Band mapping by sensor."""

	band_map: dict[str, dict[str, str]] = field(
		default_factory=lambda: {
			"landsat8_9": {
				"blue": "B2",
				"green": "B3",
				"red": "B4",
				"nir": "B5",
				"swir1": "B6",
				"swir2": "B7",
			},
			"landsat5_7": {
				"blue": "B1",
				"green": "B2",
				"red": "B3",
				"nir": "B4",
				"swir1": "B5",
				"swir2": "B7",
			},
			"sentinel2": {
				"blue": "B02",
				"green": "B03",
				"red": "B04",
				"nir": "B08",
				"swir1": "B11",
				"swir2": "B12",
			},
		}
	)
	max_nan_fraction: float = 0.3
	qa_cloud_bit: int = 3
	qa_cloud_shadow_bit: int = 4
	qa_snow_bit: int = 5
	scale_factors: dict[str, float] = field(
		default_factory=lambda: {"reflectance": 1.0, "offset": 0.0}
	)
	chunking: tuple[int, int, int] = (1, 128, 128)
	harmonise: dict[str, dict[str, dict[str, float]]] = field(
		default_factory=lambda: {
			"landsat8_9": {
				"sentinel2": {"blue": 1.0, "green": 1.0, "red": 1.0, "nir": 1.0, "swir1": 1.0, "swir2": 1.0},
			},
			"sentinel2": {
				"landsat8_9": {"blue": 1.0, "green": 1.0, "red": 1.0, "nir": 1.0, "swir1": 1.0, "swir2": 1.0},
			},
		},
	)


@dataclass(frozen=True)
class ClassifierConfig:
	"""Training and output settings for the land-cover classifier."""

	classes: tuple[str, ...] = (
		"urban",
		"bare_rock",
		"riverbed_gravel",
		"terraced_agri",
		"forest",
		"water",
		"other",
	)
	min_samples: int = 100
	min_urban_f1: float = 0.80
	max_jump: float = 0.25
	urban_presence_threshold: float = 0.5
	model: str = "rf"
	n_estimators: int = 300
	max_depth: int | None = None
	xgb_max_depth: int = 6
	n_splits: int = 5
	spatial_context: bool = True
	class_map_resolution_m: int = 10
	output_directory: str = "urban_classifier"


@dataclass(frozen=True)
class DownscalingConfig:
	"""Physical downscaling parameters and conservation tolerances."""

	use_model_orography: bool = False
	model_orography_variable: str = "orography"
	default_lapse_rate_c_per_km: float = -6.5
	lapse_rate_min_c_per_km: float = -12.0
	lapse_rate_max_c_per_km: float = 2.0
	minimum_station_samples: int = 2
	mean_tol: float = 0.3
	precip_mean_relative_tol: float = 1e-6
	topoclimatic_enabled: bool = False
	topoclimatic_coefficient: float = 0.0
	topoclimatic_solar_scale: float = 100.0
	precip_orographic_enabled: bool = False
	precip_orographic_coefficient_per_km: float = 0.0
	temperature_variables: tuple[str, ...] = ("T2M", "t2m", "TMAX", "TMIN", "LST")
	dewpoint_variables: tuple[str, ...] = ("D2M", "d2m", "dewpoint", "dew_point")
	humidity_variables: tuple[str, ...] = ("RH2M", "rh2m")
	precipitation_variables: tuple[str, ...] = ("PRECIP", "precip", "PRECTOTCORR")


@dataclass(frozen=True)
class MLConfig:
	"""Tree-ensemble training, validation, and output settings."""

	feature_columns: tuple[str, ...] = (
		"dem",
		"slope",
		"aspect_sin",
		"aspect_cos",
		"tpi_small",
		"tpi_large",
		"twi",
		"solar_rad",
		"urban_fraction",
		"ndvi",
		"ndbi",
		"bsi",
		"t2m_downscaled",
		"t2m_downscaled_anom",
		"RH2M",
		"is_monsoon",
		"sin_m",
		"cos_m",
	)
	target_columns: dict[str, str] = field(
		default_factory=lambda: {
			"DTR_cell": "DTR_cell",
			"heat_snap_risk": "t2m_downscaled",
			"cold_snap_risk": "t2m_downscaled",
			"precip_extreme": "PRECIP",
		}
	)
	target_percentiles: dict[str, float] = field(
		default_factory=lambda: {
			"heat_snap_risk": 0.9,
			"cold_snap_risk": 0.1,
			"precip_extreme": 0.95,
		}
	)
	validation_months: int = 3
	early_stopping_rounds: int = 20
	prediction_chunk_size: int = 100_000
	importance_repeats: int = 5
	max_importance_share: float = 0.8
	model_directory: Path = Path("models")
	save_fold_models: bool = True
	use_temporal_preprocessor: bool = False
	lag_columns: tuple[str, ...] = ()
	lag_periods: tuple[int, ...] = (1,)
	tuning_trials: int = 20
	tuning_enabled: bool = False
	scale_pos_weight: bool = True
	n_jobs: int = 1
	model_params: dict[str, dict[str, object]] = field(
		default_factory=lambda: {
			"xgb": {
				"n_estimators": 300,
				"max_depth": 6,
				"learning_rate": 0.05,
				"subsample": 0.8,
				"colsample_bytree": 0.8,
				"reg_lambda": 1.0,
			},
			"lgbm": {
				"n_estimators": 300,
				"max_depth": -1,
				"learning_rate": 0.05,
				"subsample": 0.8,
				"colsample_bytree": 0.8,
				"reg_lambda": 1.0,
			},
		}
	)
	monotonic_constraints: dict[str, int] = field(default_factory=dict)
	tuning_search_space: dict[str, tuple[float, float]] = field(
		default_factory=lambda: {
			"learning_rate": (0.01, 0.2),
			"max_depth": (3, 10),
		}
	)


@dataclass(frozen=True)
class Config:
	"""Top-level immutable pipeline configuration."""

	sanity: SanityConfig = field(default_factory=SanityConfig)
	paths: PathsConfig = field(default_factory=PathsConfig)
	dataset: DatasetConfig = field(default_factory=DatasetConfig)
	validation: ValidationConfig = field(default_factory=ValidationConfig)
	econometrics: EconometricsConfig = field(default_factory=EconometricsConfig)
	grid: GridConfig = field(default_factory=GridConfig)
	spectral: SpectralBandMap = field(default_factory=SpectralBandMap)
	classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
	downscaling: DownscalingConfig = field(default_factory=DownscalingConfig)
	ml: MLConfig = field(default_factory=MLConfig)
	seed: int = 0


def get_config() -> Config:
	"""Return the effective project configuration."""
	return Config()


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
	logger.warning(
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
