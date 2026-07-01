# Adding a detector geometry

A new detector is, at minimum, a new YAML config. JAXTPC parses it into a plain
dict, derives all geometry from it, and builds one JIT-compiled body that runs
for any number of volumes. No `tools/` code changes are needed for a standard
wire or pixel detector — **provided the new geometry respects the two invariants
below.**

## The two invariants you must respect

### 1. Deposits run in a volume-local, yz-centered, anode-referenced frame

The loader ([`tools/loader.py`][loader], `build_deposit_data`) transforms every
deposit out of global detector coordinates into its volume's local frame before
any physics runs:

```python
x_local = drift_direction * (x_anode_cm*10 - x_global_mm)   # ≥ 0, drift toward 0
y_local = y_global_mm - y_center_mm                         # yz-centered
z_local = z_global_mm - z_center_mm
```

So in local coordinates the anode is at `x_local = 0`, electrons always drift
toward `x_local = 0` regardless of the volume's physical drift direction, and
`(y, z)` are centered on the volume. `x_anode_cm` and `yz_center_cm` are derived
in `create_sim_config()` from each volume's `ranges` and `drift_direction`
(`x_anode = x_max` for `drift_direction: +1`, else `x_min`). See
[Coordinates & frames](../concepts/coordinates.md) for the full transform and
the inverse applied when writing sensor files.

### 2. The JIT body assumes geometry uniformity across volumes

The compiled body is traced **once** and mapped over the volume axis
(`lax.scan` or `vmap`). Because `scan`/`vmap` require uniform shapes across that
axis, the traced graph uses the reference geometry (volume 0's) as fixed
constants for every volume — wire counts, plane angles/spacings, diffusion
dimensions, time-window size. The local-frame transform is exactly what makes
this legal: after centering and anode-referencing, all volumes share the *same*
reference geometry in local coordinates, so one set of constants is correct for
all of them.

!!! warning "All volumes must share reference geometry in the local frame"
    Give every volume in a config the **same** size (drift length, y/z extent),
    the **same** plane layout (angles, spacings, counts), and the **same**
    readout type. Volumes may differ only in their global placement (`ranges`)
    and `drift_direction` — the loader normalizes those away. A config whose
    volumes have genuinely different local geometry (e.g. one wire volume + one
    pixel volume, or different wire spacings per volume) will **not** simulate
    correctly: the body silently applies volume 0's constants to every volume.
    If you truly need heterogeneous modules, run them as separate simulators.

## Step-by-step

### 1. Author the YAML

Copy an existing config in `config/` (e.g. `cubic_wireplane_config.yaml` for
wire, `cubic_pixel_config.yaml` for pixel) and edit it. `generate_detector()`
validates that the five top-level sections
(`volumes`, `readout`, `simulation`, `medium`, `electric_field`) exist and that
each volume has `geometry.ranges`, `geometry.drift_direction`, and **either** a
`planes` list (wire) **or** a `readout.type: pixel` block with `pixel_pitch` +
`pixel_shape`. Everything else is derived. See the full
[config schema](../detector/config-schema.md) for every field.

A minimal single-volume wire volume:

```yaml
volumes:
  - id: 0
    geometry:
      ranges: [[-216.0, 0.0], [-216.0, 216.0], [-216.0, 216.0]]  # cm [x, y, z]
      drift_direction: -1
    planes:
      - {plane_id: 0, angle:  60.0, wire_spacing: 0.3, distance_from_anode: 0.6}
      - {plane_id: 1, angle: -60.0, wire_spacing: 0.3, distance_from_anode: 0.3}
      - {plane_id: 2, angle:   0.0, wire_spacing: 0.3, distance_from_anode: 0.0}
simulation:
  drift: {velocity: 1.6, longitudinal_diffusion: 7.2, transverse_diffusion: 12.0, electron_lifetime: 10.0}
  charge_recombination:
    model: emb
    recomb_parameters: {alpha: 0.93, beta: 0.212, alpha_emb: 0.904, beta_90: 0.204, R_anisotropy: 1.25}
readout: {sampling_rate: 2.0, electrons_per_adc: 182}
electric_field: {field_strength: 500.0}
medium: {type: LAr, temperature: 87.0}
```

To add more volumes, append entries with different `ranges`/`drift_direction`
but **identical** local geometry (same drift length `x_max - x_min`, same y/z
extent, same `planes`).

### 2. Generate + build the config

```python
from tools.geometry import generate_detector, print_detector_summary
from tools.config import create_sim_config, create_sim_params

detector = generate_detector('config/my_detector.yaml')  # validate + parse
cfg = create_sim_config(detector)                         # derive static geometry
params = create_sim_params(detector)                      # dynamic physics scalars

print_detector_summary(detector, cfg)   # inspect derived wire counts, num_time_steps, ...
```

`create_sim_config()` derives per-volume `VolumeGeometry` (wire counts,
diffusion sigmas, anode position, yz-center) and the shared `num_time_steps`
from the longest drift. Check the summary: wire counts and drift lengths should
match your intent, and every volume should report the same derived geometry.

### 3. Smoke-run one synthetic event

```python
import jax
import numpy as np
from tools.simulation import DetectorSimulator
from tools.loader import build_deposit_data

sim = DetectorSimulator(cfg, params)

# A few synthetic deposits inside volume 0's global range (positions in mm)
positions_mm = np.array([[-1080.0, 0.0, 0.0],
                         [-1000.0, 50.0, 50.0]], dtype=np.float32)
de = np.array([2.0, 2.0], dtype=np.float32)   # MeV
dx = np.array([3.0, 3.0], dtype=np.float32)   # mm

deposits = build_deposit_data(positions_mm, de, dx, cfg)
signals, hits, deposits = sim.process_event(deposits, key=jax.random.PRNGKey(0))
```

If the deposits land in the expected wire/time region and no overflow error is
raised, the geometry is wired correctly. To build the same detector end-to-end
in a notebook, see `notebooks/detector/custom_detector.ipynb`.

## Pitfalls

- **Frame assumptions (most common).** If deposits land in the wrong wire/time
  region, or a multi-volume config produces garbage in the non-reference
  volumes, you almost certainly violated invariant 2 (non-uniform local
  geometry) or fed global-frame positions somewhere that expects local. Deposits
  outside every volume's `ranges` are silently dropped by the loader's
  volume mask — check your `ranges` cover the data.
- **Capacity sizing.** The defaults (`total_pad`, `max_keys`, `maxg`, box dims)
  are sized for the reference detector. A larger or busier geometry can overflow
  them and raise a `RuntimeError` at runtime. Size capacities for your data with
  the profiler (`python3 -m profiler.setup_production --data events.h5
  --config config/my_detector.yaml`) — see [Capacities](../concepts/capacities.md).
- **`response_chunk_size` must divide `total_pad`.** A mismatch is rejected when
  building the config.
- **Wire vs pixel is a whole-config choice.** A volume is wire *or* pixel; you
  cannot mix readout types across volumes in one simulator (invariant 2). See
  [wire vs pixel](../detector/wire-vs-pixel.md).

[loader]: https://github.com/DeepLearnPhysics/JAXTPC/blob/main/tools/loader.py
