````markdown
# Mathematical / Algorithmic Specifications

## 9.1 `load_and_align()` contract

```python
load_and_align(
    source,              # path str or xarray/rioxarray
    target_crs,          # pyproj.CRS or EPSG int
    target_transform,    # affine.Affine
    target_shape,        # (height, width)
    resampling,          # Resampling.bilinear | Resampling.nearest
    variable_type,       # "continuous" | "categorical"
    nodata=None,
    clip_bounds=None,    # optional (minx, miny, maxx, maxy)
) -> xarray.DataArray
````

### Processing steps (in order)

1. **Open/validate source:** verify that the source has a CRS — raise `ValueError` if missing.
2. **Validate resampling method:** if `variable_type == "categorical"` and `resampling != Resampling.nearest`, raise an error (prevents R7).
3. **Optional clipping:** clip in the source CRS to a small buffer around `clip_bounds` reprojected back to the source CRS (cheaper than reprojecting the full raster first).
4. **Reproject:**

   ```python
   rioxarray.reproject(
       target_crs,
       transform=target_transform,
       shape=target_shape,
       resampling=resampling,
       nodata=nodata,
   )
   ```
5. **Validate output:** assert:

   * `.shape == target_shape`
   * `.rio.crs == target_crs`
   * `.rio.transform() == target_transform` (within floating tolerance)
6. **Attach attributes:**

   * `source_path`
   * `original_crs`
   * `original_res`
   * `resampling_method`
   * `processed_utc_timestamp`

---

## 9.2 Slope

* **Algorithm:** Horn's 3×3 method (standard, edge-aware, matches common GIS defaults) — explicitly name this choice in code/docstring rather than leaving it implicit as the original did.
* **Units:** degrees (0–90).
* **Edge handling:** border pixels use NumPy `'reflect'` boundary mode by default (documented, configurable); never silently zero-pad (which would bias slope toward 0 at edges).
* **Resolution dependency:** slope magnitude is resolution-dependent (coarser DEM → smoothed slope).

  * Since both 100 m and 250 m grids will be produced, slope must be computed **once from the native/near-native-resolution DEM** and then resampled with **bilinear** (continuous variable) onto each target grid.
  * It must **not** be recomputed independently from an already-resampled 100 m or 250 m DEM, to avoid inconsistent smoothing artifacts between the two grids.
  * This must be explicit in the coding prompt (Part 10).

---

## 9.3 Aspect

* **Convention:** azimuth in degrees, 0° = North, increasing clockwise (90° = East), range `[0, 360)`.
* **Flat terrain:** where slope ≈ 0 (below a small epsilon, e.g. `< 0.01°`), aspect is undefined.

  * Assign **nodata** (not 0°, not -1, to avoid biasing circular statistics).
  * Propagate nodata into `sin`/`cos` too.
* **Radians:** converted internally only for `sin`/`cos`; degrees remain the stored/exported unit for `aspect_deg`.
* **Trig encoding rationale:**
  `x = sin(θ)` and `y = cos(θ)` map the circular 0°/360° discontinuity onto a continuous 2D unit circle. Therefore, any model using these as features sees north-facing slopes (`θ ≈ 0°` and `θ ≈ 359°`) as numerically adjacent rather than maximally distant — standard circular-statistics practice, directly relevant if aspect is ever used in a future ML block.

---

## 9.4 Solar radiation (scope-limited proxy)

### Required inputs

* Slope
* Aspect
* Latitude (from grid coordinates)
* Day-of-year (or a small set of representative days, e.g. solstices/equinoxes, given only monthly climate data exists downstream)

### Assumptions

* **Clear-sky only**
* No cloud/atmospheric transmissivity modelling (unsupported by available data)
* No cast-shadow/horizon occlusion from surrounding terrain (would need viewshed analysis — explicitly out of scope, flagged as a limitation, not silently omitted)

### Formulation

Standard potential clear-sky insolation using:

* Solar declination
* Hour-angle integration over a day
* Modulation by the cosine of the angle of incidence on the tilted (slope/aspect) surface

This is a **simplified Hillshade/insolation index**, not a full radiative-transfer model.

### Temporal resolution

Produce a static seasonal proxy, for example:

* Per-solstice/equinox rasters, or
* A simple annual-mean index

This should match the monthly-at-best granularity of everything else in the project.

**Do not imply sub-daily precision.**

### Limitations to document in output metadata

* No shadowing
* No atmosphere
* Coarse temporal representativeness

---

## 9.5 TPI (multi-scale)

### Definition

[
TPI(x, window) = elevation(x) - mean(elevation\ within\ window\ around\ x,\ excluding\ x)
]

### Window / scale

Use a circular or square neighborhood radius, in map units (meters), defined per grid resolution in `config.py` as a list, for example:

```python
[300, 1000, 3000]  # metres
```

These scales must be explicitly marked as **tunable**, not universally optimal, pending visual/empirical validation against Doon Valley and Jammu-foothill terrain wavelengths (R3).

### Implementation

Use a moving-window mean via:

* `scipy.ndimage.uniform_filter` — fast, vectorized, or
* `xrspatial.focal.mean`

Avoid explicit pixel loops.

### Edge handling

Use `'reflect'` boundary handling and document it.

Alternatively, mask out a border of `window_radius` width as nodata if strict correctness is required.

Make this a **config flag**; do not silently pick one.

### Nodata handling

Any window containing nodata pixels propagates nodata to the center pixel's TPI.

**Never silently average over nodata as if it were 0.**

### Interpretation

* **Positive TPI** → ridge/local high
* **Negative TPI** → valley bottom
* **Near-zero TPI** → flat or mid-slope

Multiple scales are needed because a pixel can simultaneously be a local ridge at a small scale within a larger valley at a large scale.

A single-scale TPI cannot distinguish these regimes, which matters for interpreting where urban expansion is occurring in Himalayan terrain.

---

## 9.6 TWI (Topographic Wetness Index)

### Pipeline

```text
DEM
  ↓
Depression/pit filling
  ↓
Flow direction
  ↓
Flow accumulation
  ↓
Specific catchment area (a)
  ↓
Slope (from 9.2)
  ↓
TWI
```

Depression/pit filling is **required before flow routing** because unfilled pits break flow accumulation.

### Formula

Beven–Kirkby:

[
TWI = \ln\left(\frac{a}{\tan(\beta)}\right)
]

where:

* (a) = upslope contributing area per unit contour width
* (\beta) = local slope in radians

### Numerical stability

Where:

[
\tan(\beta) \approx 0
]

(flat/near-flat cells), clip `tan(β)` to a small epsilon, e.g.:

```python
epsilon = 1e-3
```

before division to avoid `inf`/`NaN`.

Document the clip value and its physical meaning:

> Near-flat cells receive a capped, not infinite, wetness index.

### Flow algorithm

Implement both:

* **D8** — simple, standard, discrete 8-direction flow routing
* **D∞** — Tarboton continuous-angle flow routing, generally more accurate on complex terrain

These must be exposed behind a **single configurable parameter** in `config.py`:

```python
flow_algorithm: "d8" | "dinf"
```

Do **not** duplicate the TWI function for each algorithm.

This satisfies the requirement that the method can be configured rather than duplicated.

### Depressions / sinks

Fill depressions using a standard priority-flood algorithm, for example:

* `whitebox.fill_depressions`

Document that filling alters raw elevation **locally only for routing purposes**.

The filled DEM must **not** replace the exported elevation layer.

### Resolution note

Flow routing and accumulation are highly resolution-sensitive.

Therefore:

1. Compute flow routing and accumulation on the **native-resolution (near-30 m) filled DEM**.
2. Compute TWI from this native-resolution flow accumulation.
3. Resample the resulting TWI using **bilinear interpolation** (continuous variable) onto the:

   * 100 m target grid
   * 250 m target grid

This is analogous to the slope-resolution handling described in Section 9.2.

Computing flow accumulation directly on an already-resampled 100 m or 250 m DEM would materially distort contributing-area estimates and is **explicitly disallowed** in the implementation prompt.

```
```

## 9.7 Resampling policy (explicit, variable-type-driven)

| Variable class | Method | Reason |
|---|---|---|
| Elevation, slope, aspect-derived continuous fields, TPI, TWI, solar proxy | **Bilinear** (default) or **Cubic** (optional, for a smoother DEM-only result where higher-order smoothness is desired) | Continuous, spatially smooth fields — linear/cubic interpolation preserves gradients meaningfully |
| NDVI/NDBI-type continuous spectral indices | **Bilinear** | Continuous, smoothly varying |
| Any classified/categorical layer (e.g., a future land-cover class raster) | **Nearest Neighbor only** | Averaging class codes (e.g., `"urban"=3`, `"forest"=2`) produces meaningless intermediate values (`2.5` is not a real class); nearest-neighbor preserves valid class identity |
| Consequence of resampling a DEM vs. land-cover | Resampling a DEM with bilinear/cubic smooths real elevation transitions — acceptable and expected for continuous terrain. Resampling a categorical layer with anything but nearest neighbor fabricates non-existent classes and silently corrupts any downstream classification. | This is the exact failure the R7 validation guard exists to prevent. |

`load_and_align()` enforces this policy via the required `variable_type` parameter and raises an argument error when the caller requests a mismatched resampling method, rather than relying on the caller to remember the rule.

###Tests

Run with 
$env:PYTHONPATH = "src"
python -m pytest tests -q