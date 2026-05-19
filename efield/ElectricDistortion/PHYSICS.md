# Space Charge Effect (SCE) Physics Documentation

## 1. Physical Problem

In a Liquid Argon Time Projection Chamber (LArTPC), cosmic ray muons continuously
ionise the argon, producing electron-ion pairs throughout the active volume.
Electrons are collected at the anode in milliseconds, but the positive argon ions
drift ~10^5 times slower toward the cathode (mm/s vs mm/us). In steady state this
creates a net positive space charge that distorts the nominal electric field,
causing electrons to follow curved trajectories and arrive at displaced positions
on the anode. These spatial distortions must be mapped and corrected to achieve
accurate 3D reconstruction.

### Coordinate Convention (Standard LArTPC)

Following MicroBooNE (arXiv:2008.09765) and all major LArTPC experiments:

- **x = 0**: Anode (wire planes, electron collection)
- **x = Lx**: Cathode (high voltage)
- **Electrons drift in the -x direction** (from cathode toward anode)
- **Positive ions drift in the +x direction** (from anode toward cathode)
- **y, z**: Transverse directions (vertical, beam)

---

## 2. Pipeline Steps

### Step 1: Charge Density

The steady-state positive ion density follows from continuity. Ions are produced
uniformly at rate Q (C/m^3/s) and drift toward the cathode at speed v_ion.
The number of ions passing through a plane at position x equals the integral of
production from 0 to x, giving a linear profile:

```
rho(x) = Q * x / v_ion
```

where:
- Q = volumetric charge production rate (C/m^3/s)
- v_ion = mu_ion * E0 (ion drift speed, cm/s)
- mu_ion = ion mobility (cm^2/V/s)

rho = 0 at the anode (x=0), maximum at the cathode (x=Lx).

**Approximations:**
- Uniform in y and z (uniform cosmic ray flux across transverse plane)
- Constant v_ion (uses nominal field E0; relaxed by self-consistent iteration)
- All produced ions survive (no ion-ion recombination or attachment)
- Steady state (production = removal by drift)

### Step 2: Poisson Equation

The distortion potential dphi (deviation from nominal -E0*x) satisfies:

```
nabla^2(dphi) = -rho / epsilon
```

where epsilon = epsilon_0 * epsilon_r (LAr permittivity).

**Boundary conditions:** Homogeneous Dirichlet on all 6 faces (dphi = 0).
This assumes a perfect field cage that maintains the nominal potential
phi = -E0*x exactly on every boundary surface.

**Solver:** 3D Discrete Sine Transform (Type I, orthonormal) via scipy.fft.
The DST-I naturally satisfies homogeneous Dirichlet BCs. In Fourier space:

```
dphi_hat(l,m,n) = rho_hat(l,m,n) / [epsilon * pi^2 * ((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)]
```

This is spectrally accurate — the only error comes from grid sampling of rho.

### Step 3: Electric Field

The total physical potential is:

```
phi(x,y,z) = -E0 * x + dphi(x,y,z)
```

The electric field E = -grad(phi) gives:

```
Ex = E0 - d(dphi)/dx      (nominal + SCE correction)
Ey =    - d(dphi)/dy       (purely from SCE)
Ez =    - d(dphi)/dz       (purely from SCE)
```

Gradients computed via 2nd-order central finite differences (np.gradient).

**Result:** Ex > E0 near cathode (field strengthened), Ex < E0 near anode
(field weakened). Ey and Ez are non-zero near boundaries, pulling electrons
toward the transverse center of the detector.

### Step 4: Drift Velocity

Electron drift velocity as a function of local E-field magnitude, using the
Walkowiak parameterisation (NIM A 449, 2000) with LArSoft/ICARUS parameters:

```
v_d(E, T) = (P1*(T-T0) + 1) * (P3*E_kV*ln(1 + P4/E_kV) + P5*E_kV^P6) + P2*(T-T0)
```

Parameters:
| Symbol | Value   |
|--------|---------|
| P1     | -0.04640 |
| P2     |  0.01712 |
| P3     |  1.88125 |
| P4     |  0.99408 |
| P5     |  0.01172 |
| P6     |  4.20214 |
| T0     | 105.749 K |

Benchmark at T = 89 K:
- E = 273 V/cm (MicroBooNE): v_d = 1.098 mm/us
- E = 500 V/cm (SBND/ICARUS/JAXTPC): v_d = 1.563 mm/us

### Step 5: Electron Tracing

Each electron is launched from a grid point (x0, y0, z0) and integrated to the
anode using scipy.integrate.solve_ivp (RK45, Dormand-Prince 4th/5th order):

```
dr/dt = -(v_d(|E|) / |E|) * E
```

The electron drifts opposite to the local E-field direction. The E-field at each
position is obtained by trilinear interpolation (RegularGridInterpolator) of the
3D field arrays from Step 3.

**Stopping conditions:**
- Terminal event: x decreases through 0 (electron reaches anode)
- Safety timeout: t_max = 20,000 us (20 ms)
- Zero field guard: |E| < 1e-10 V/cm -> velocity = 0

**Tolerances:** rtol = 1e-6, atol = 1e-6, max_step = 50 us.

**Output:** For each launch point: drift time t_drift, arrival y_anode, arrival z_anode.

### Step 6: Distortion Maps

The spatial distortions represent the difference between reconstructed
(assuming nominal uniform field) and true positions:

```
delta_x = v_nominal * t_drift - x_true
delta_y = y_anode - y_true
delta_z = z_anode - z_true
```

where v_nominal = v_d(E0, T) is the drift velocity at the nominal field.

**Physical meaning:**
- **delta_x > 0**: Electrons arrive later than expected (field weakened near
  anode acts as a bottleneck). Reconstructed x appears further from anode
  than the true position.
- **delta_y, delta_z**: Transverse displacement from true position to arrival
  position. Space charge squeezes electrons toward the detector center.

**Spatial dependence:**
- delta_x: Zero at anode (x=0) and cathode (x=Lx), maximum at x ~ Lx/sqrt(3)
  (~58% of drift distance). The zero at cathode occurs because the strong-field
  and weak-field regions cancel over the full path.
- delta_y, delta_z: Zero at anode (no drift), maximum at cathode (longest path).
  Largest near the transverse walls, zero at the transverse center (by symmetry).

---

## 3. Optional: Self-Consistent Iteration

The base model uses constant v_ion = mu_ion * E0 everywhere. Since the E-field
varies due to space charge, the actual ion speed v_ion(x) = mu_ion * Ex(x) also
varies. This creates a nonlinear feedback: rho depends on v_ion, which depends on
E, which depends on rho.

**Algorithm:**
1. Compute initial E-field with constant v_ion (standard linear model)
2. Iterate:
   a. v_ion(x) = mu_ion * <Ex(x)>_yz  (average over transverse plane)
   b. rho_new(x) = Q * x / v_ion(x)
   c. Under-relaxation: rho = alpha * rho_new + (1-alpha) * rho_old  (alpha=0.5)
   d. Re-solve Poisson, re-compute E-field
   e. Check convergence: max|rho_new - rho| / max|rho| < tol (default 1e-3)
3. Stop when converged or max_iter reached (default 10)

The linearised model is the standard approximation used in published MicroBooNE
simulations. Self-consistent iteration matters when the field varies by >10%
across the drift volume.

---

## 4. Physical Constants and Parameters

### Fundamental
| Constant | Value | Units |
|----------|-------|-------|
| epsilon_0 | 8.854e-12 | F/m |

### Liquid Argon Properties
| Property | Value | Units | Notes |
|----------|-------|-------|-------|
| epsilon_r | 1.505 | - | Relative permittivity |
| mu_ion | 8.0e-4 | cm^2/(V*s) | Positive ion mobility (MicroBooNE standard) |
| Temperature | 89.0 | K | Boiling point at ~1 atm |
| W_ion | 23.6 | eV/pair | Ionisation work function |
| dE/dx (MIP) | 2.1 | MeV/cm | Minimum ionising energy loss |

### Charge Production Rate Q
| Detector | Q (C/m^3/s) | Notes |
|----------|-------------|-------|
| MicroBooNE | 3.0e-11 | Calibrated to published distortion data |
| SBND | 2.0e-10 | Surface at Fermilab |
| ICARUS | 2.0e-10 | Surface at Fermilab |
| DUNE FD | 0.0 | Underground (4300 m.w.e.), negligible cosmics |
| JAXTPC | 3.0e-11 | Surface detector |

Q is the single most uncertain parameter (literature spans factor 2-3x).
Back-of-envelope estimate from cosmic ray physics:
```
Q = muon_flux * dEdx * (1/W_ion) * R_recomb * e
  ~ 1.1e4 /m^2/s * 2.1e8 eV/m * (1/23.6 eV) * 0.64 * 1.6e-19 C
  ~ 1e-10 C/m^3/s  (order of magnitude)
```

### Detector Geometries
| Detector | Lx (cm) | Ly (cm) | Lz (cm) | E0 (V/cm) |
|----------|---------|---------|---------|-----------|
| MicroBooNE | 256 | 233 | 1036 | 273 |
| SBND | 200 | 400 | 500 | 500 |
| ICARUS | 150 | 390 | 1960 | 500 |
| DUNE FD HD | 350 | 1200 | 5800 | 500 |
| JAXTPC | 216 | 432 | 432 | 500 |

---

## 5. Summary of Approximations

1. **Steady state** — ion distribution has reached equilibrium
2. **1D charge density** — rho(x) only, uniform in y,z (uniform cosmic flux)
3. **Linear rho profile** — constant v_ion (relaxed by self-consistent iteration)
4. **Perfect field cage** — Dirichlet BCs enforce nominal potential on all boundaries
5. **No ion losses** — all ions contribute to space charge
6. **No electron diffusion** — deterministic trajectories through E-field
7. **Walkowiak drift velocity** — empirical fit valid at 89 K, 100-1000 V/cm
8. **Field-independent ion mobility** — v_ion = mu_ion * E (linear model)
9. **Single drift volume** — each run handles one side of a dual-drift TPC
10. **No time dependence** — computes the steady-state distortion map only

---

## 6. References

1. MicroBooNE SCE measurement — arXiv:2008.09765 (JINST 15, P12037, 2020)
2. Analytical SCE review — arXiv:2008.10472
3. Walkowiak drift velocity — NIM A 449 (2000) 288-294
4. MicroBooNE early SCE study — arXiv:1511.01563
5. ICARUS SCE calibration — arXiv:2407.11925
