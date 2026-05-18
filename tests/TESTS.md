# JAXTPC Test Suite

77 tests covering software correctness and physics validation across all simulation modules. All tests run on CPU with synthetic data.

## Running Tests

```bash
# Fast tests only (~7s)
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v -m "not slow"

# All tests including integration (~90s)
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v

# With coverage report
JAX_PLATFORM_NAME=cpu python3 -m pytest tests/ -v --cov=tools --cov=noise_analysis --cov-report=term-missing
```

## Test Modules

### test_recombination.py (7 tests)

Modified Box model charge recombination (`tools/recombination.py`).

| Test | What it checks |
|------|----------------|
| MIP survival fraction | dE/dx = 2.1 MeV/cm gives R ~ 0.705 (analytical) |
| Higher dE/dx lower survival | dE/dx = 10 produces lower R than dE/dx = 2.1 |
| dE/dx clamping | Very low dE/dx clamps to 1.0 MeV/cm |
| dx = 0 returns zero | No valid step gives zero charge |
| Negative de returns zero | Unphysical input gives zero charge |
| Vectorized correctness | 1000-element batch matches element-wise loop |
| Survival fraction bounds | R stays in [0, 1] for all physical inputs |

### test_drift.py (8 tests)

Electron drift physics (`tools/drift.py`).

| Test | What it checks |
|------|----------------|
| Drift time analytical | Known position gives correct drift distance and time |
| East/West symmetry | Symmetric charges have equal drift distances |
| Charge at anode | Zero drift distance at anode position |
| Plane correction values | Distance subtraction for closer planes |
| Correction clamps | No negative distances after correction |
| Lifetime attenuation formula | exp(-t/lifetime) matches analytical value |
| Zero drift no attenuation | Factor = 1.0 at zero drift |
| Attenuation bounds | Factor in (0, 1] for all distances |

### test_wires.py (13 tests)

Wire geometry and signal calculations (`tools/wires.py`).

| Test | What it checks |
|------|----------------|
| Y-plane closest wire | Correct wire index for angle = 0 |
| Wire distance sign | Half-spacing distance between wires |
| U-plane 60 deg projection | Angled plane wire coordinate projection |
| K-nearest count and order | Shape (n, K) with center wire closest |
| Out-of-bounds wires | OOB indices = -1, distances = NaN |
| Angular scaling at 45 deg | 1/(cos * sin) = 2.0 |
| Angular clipping | theta_y = 0 clips to 5 deg, stays finite |
| Angular symmetry | scaling(-theta) == scaling(theta) |
| Deposit-wire angles | Known geometry gives expected angles |
| Gaussian diffusion normalization | 2D Gaussian integrates to 1.0 |
| Diffusion broadens with drift | Longer drift gives lower peak |
| Signal accumulation | Scatter-add produces correct array values |
| Response drops OOB | Out-of-bounds kernel elements silently dropped |

### test_kernels.py (8 tests)

Response kernel system (`tools/kernels.py`). Uses synthetic DKernel arrays.

| Test | What it checks |
|------|----------------|
| Gaussian kernel normalization | Kernel sums to 1.0 |
| Gaussian kernel symmetry | Point-symmetric: k[i,j] == k[H-1-i, W-1-j] |
| Convolution preserves signal | Total signal preserved for centered kernels |
| Wire count calculation | width = 127, spacing = 0.1 gives 12 wires |
| Interpolation at s = 0 | Returns original kernel shape |
| Interpolation at s = 1 | Returns maximally diffused kernel |
| Batch matches single | vmap result matches individual calls |
| Output shape | Batch returns (N, num_wires, num_time) |

### test_config.py (3 tests)

Configuration NamedTuples and factories (`tools/config.py`).

| Test | What it checks |
|------|----------------|
| K values from sigmas | sigma = 2.5, n_sigma = 3 gives K = 8 |
| K minimum is 1 | sigma = 0 still gives K >= 1 |
| Lifetime unit conversion | 10 ms converts to 10000 us |

### test_geometry.py (8 tests)

Detector geometry parsing (`tools/geometry.py`).

| Test | What it checks |
|------|----------------|
| Dimensions parsing | YAML gives x = y = z = 432 cm |
| Velocity unit conversion | 1.6 mm/us converts to 0.16 cm/us |
| Time parameter calculation | Correct num_time_steps from drift and sampling |
| Y-plane wire count | Reasonable count for detector size |
| U/V plane symmetry | Same wire count for symmetric geometry |
| Diffusion sigma calculation | Analytical sigma values match |
| Missing YAML returns None | Non-existent file handled gracefully |
| Incomplete YAML returns None | Missing required keys handled gracefully |

### test_track_hits.py (5 tests)

Track hit labeling (`tools/track_hits.py`).

| Test | What it checks |
|------|----------------|
| Single track aggregation | 5 hits same location sum to total charge |
| Threshold filtering | Hits below threshold removed |
| Multiple tracks separated | 3 distinct tracks counted correctly |
| Dominant track labeling | Highest-charge track wins at shared location |
| Sparse hits to dense | Known hits produce correct dense array |

### test_sparse_utils.py (4 tests)

Dense/sparse conversion (`tools/sparse_utils.py`).

| Test | What it checks |
|------|----------------|
| Dense-sparse-dense roundtrip | Exact reconstruction |
| Threshold filtering | Only |value| > threshold kept, sign preserved |
| Duplicate index accumulation | Same (wire, time) values summed |
| Empty sparse array | Produces all-zero dense output |

### test_noise.py (10 tests)

Noise generation and threshold analysis (`noise_analysis/`).

| Test | What it checks |
|------|----------------|
| RMS formula matches MicroBooNE | sqrt(x^2 + (y + z*L)^2) |
| RMS increases with wire length | Monotonic increase |
| Generated noise correct RMS | Measured RMS within 15% of target |
| Shaped spectrum | Power at 100 kHz > power at 900 kHz |
| White-only noise is flat | series_rms = 0 gives flat spectrum |
| Reproducibility | Same PRNG key gives identical output |
| Y-plane uniform wire length | All wires equal length for angle = 0 |
| U/V varying wire length | Angled planes have varying lengths |
| Threshold analysis correctness | Correct active count and sparsity |
| Zero threshold full density | All entries active when threshold = 0 |

### test_simulation.py (11 tests)

Integration tests and physics validation (`tools/simulation.py`). Tests marked `slow` require response kernel files.

**Software tests:**

| Test | What it checks |
|------|----------------|
| Padding tier selection | Correct tier for various input sizes |
| Exceeds all tiers | Returns largest tier with truncation |
| East/West splitting | Valid mask sums match side counts |
| Padded data preserves entries | Original values intact after padding |
| Zero charge zero output | de = 0 produces all-zero signals |
| Single deposit localization | Signal peaks near expected location |
| Recombination applied | Signal ratio differs from energy ratio |

**Physics validation:**

| Test | What it checks |
|------|----------------|
| More drift more attenuation | Farther charge has weaker signal |
| Linear charge scaling | Signal scales with recombined charge ratio |
| Y-plane signal polarity | Collection plane has net positive signal |
| Diffusion broadens with distance | Longer drift gives wider signal spread |
