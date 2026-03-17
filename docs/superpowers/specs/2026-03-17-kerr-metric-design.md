# Kerr Metric Implementation

## Goal

Add the Kerr (spinning black hole) metric, producing an asymmetric D-shaped shadow with frame-dragging effects. Default spin `a = 0.998M` (near-extremal).

## New Files

| File | Purpose |
|------|---------|
| `include/grrt/spacetime/kerr.h` | Kerr metric declaration |
| `src/kerr.cpp` | Kerr metric: g_μν, g^μν, horizons, ergosphere, ISCO |

## Modified Files

| File | Change |
|------|--------|
| `include/grrt/math/matrix4.h` | Add general `inverse()` method (full 4×4 Cramer's rule) |
| `include/grrt/spacetime/metric.h` | Add `virtual double isco_radius() const = 0` |
| `include/grrt/spacetime/schwarzschild.h` | Add `isco_radius()` override |
| `src/schwarzschild.cpp` | Implement `isco_radius()` returning `6M` |
| `include/grrt/scene/accretion_disk.h` | Take `const Metric&` reference. Use `metric.isco_radius()`. Make orbit formulas metric-aware. |
| `src/accretion_disk.cpp` | Kerr-aware orbit 4-velocity, energy, angular momentum. Kerr-aware Page-Thorne flux. |
| `src/api.cpp` | Select Kerr or Schwarzschild based on `params->metric_type`. Pass spin parameter. |
| `cli/main.cpp` | Set `metric_type = KERR`, `spin = 0.998` |
| `CMakeLists.txt` | Add `src/kerr.cpp` |

## Component Details

### Matrix4 General Inverse

The Schwarzschild metric is diagonal so `inverse_diagonal()` suffices. Kerr has off-diagonal `g_tφ` terms, requiring a general 4×4 inverse.

For Kerr in Boyer-Lindquist coordinates, the metric is block-diagonal: the `(t, φ)` block is a 2×2 matrix, and `(r, θ)` are diagonal. So the inverse can be computed efficiently:

```
For the (t, φ) 2×2 block:
    det = g_tt × g_φφ - g_tφ²
    g^tt = g_φφ / det
    g^φφ = g_tt / det
    g^tφ = -g_tφ / det

For the diagonal entries:
    g^rr = 1 / g_rr
    g^θθ = 1 / g_θθ
```

However, for generality (future metrics), implement a full 4×4 inverse via cofactor expansion. Add `inverse()` alongside the existing `inverse_diagonal()`.

### Kerr Metric (`spacetime/kerr.h`)

Parameters: mass `M`, spin `a` with `|a| < M`.

Auxiliary quantities:
```
Σ = r² + a² cos²θ
Δ = r² - 2Mr + a²
```

Covariant metric `g_μν`:
```
g_tt = -(1 - 2Mr/Σ)
g_tφ = g_φt = -2Mar sin²θ / Σ
g_rr = Σ / Δ
g_θθ = Σ
g_φφ = (r² + a² + 2Ma²r sin²θ / Σ) sin²θ
```
All other components are zero.

Contravariant metric `g^μν` via the block-diagonal inverse described above.

**Horizon**: `r_+ = M + √(M² - a²)` (outer horizon, used for termination).

**Ergosphere**: `r_ergo(θ) = M + √(M² - a² cos²θ)`. Not needed for rendering but useful for validation.

**ISCO** (prograde, co-rotating orbit):
```
Z1 = 1 + (1 - a²/M²)^(1/3) × [(1 + a/M)^(1/3) + (1 - a/M)^(1/3)]
Z2 = √(3a²/M² + Z1²)
r_isco = M × (3 + Z2 - √((3 - Z1)(3 + Z1 + 2Z2)))
```

For `a = 0`: `r_isco = 6M` (Schwarzschild). For `a = 0.998M`: `r_isco ≈ 1.24M`.

**θ singularity**: Clamp `|sinθ| ≥ 1e-10` as with Schwarzschild. Also clamp `Σ ≥ 1e-20` to avoid division by zero at the ring singularity (`r = 0, θ = π/2` — which geodesics shouldn't reach).

### Metric Interface Changes

Add to `Metric`:
```cpp
virtual double isco_radius() const = 0;
```

`Schwarzschild::isco_radius()` returns `6.0 * mass_`.
`Kerr::isco_radius()` computes the prograde ISCO formula above.

### AccretionDisk Changes

The disk currently hardcodes Schwarzschild orbit formulas. With Kerr:

**Angular velocity**:
```
Ω = √M / (r^(3/2) + a√M)     (prograde)
```

**Specific energy and angular momentum** (Bardeen, Press & Teukolsky 1972, written explicitly in `r`, `M`, `a` to avoid substitution errors):

Define `ω = √(M/r³)` (the Keplerian angular velocity at `r`, ignoring spin corrections). Then:

```
Ω(r) = ω / (1 + a·ω)                                    — prograde angular velocity

E(r) = (1 - 2M/r + a·ω) / √(1 - 3M/r + 2a·ω)          — specific energy

L(r) = √(Mr) (1 - 2a·ω + a²/r²) / √(1 - 3M/r + 2a·ω)  — specific angular momentum
```

where `ω = √(M/r³)` and `a·ω = a√(M/r³)`. At `a = 0` these reduce to the Schwarzschild formulas already in the code.

**Emitter 4-velocity** for circular orbit at `r`:
```
u^t = 1 / √(1 - 3M/r + 2a·ω)
u^φ = Ω × u^t
u^r = u^θ = 0
```

These are used in the redshift calculation `(p_μ u^μ)_emit = p_t u^t + p_φ u^φ`.

The Page-Thorne flux integral `I(r)` uses these updated `E(r)` and `L(r)` — the integral structure is identical, just the functions change.

**Implementation**: Add `spin` parameter to `AccretionDisk` constructor (default `0.0` for Schwarzschild). Add `isco_radius()` to `Metric` interface so the disk calls `metric.isco_radius()` instead of hardcoding `6M`. The `AccretionDisk` constructor signature becomes:
```cpp
AccretionDisk(double mass, double spin, double r_inner, double r_outer,
              double peak_temperature, double isco_radius, int flux_lut_size = 500);
```

The ISCO is passed in from `api.cpp` (computed via `metric->isco_radius()`), so the disk doesn't need to know about the metric interface directly.

**Redshift**: The emitter 4-velocity components change as shown above. The observer remains static at `r = 50M` (well outside the ergosphere for any spin), so the observer 4-velocity is unchanged.

### API Changes

In `grrt_create()`:
```cpp
if (params->metric_type == GRRT_METRIC_KERR) {
    double spin = params->spin > 0.0 ? params->spin : 0.998;
    ctx->metric = std::make_unique<grrt::Kerr>(mass, spin * mass);  // a = spin × M
    disk_spin = spin * mass;
} else {
    ctx->metric = std::make_unique<grrt::Schwarzschild>(mass);
    disk_spin = 0.0;
}
```

The disk receives `spin` (in units of M) to compute Kerr orbits.

## Validation

- **Shadow shape**: At `a = 0.998`, the shadow should be D-shaped — flattened on the prograde (left) side, bulging on the retrograde (right) side. At `a = 0` it should match our existing circular Schwarzschild shadow.
- **ISCO**: For `a = 0.998`, ISCO ≈ `1.24M`. The disk's inner edge should be much closer to the horizon than Schwarzschild's `6M`.
- **Doppler asymmetry**: Should be more pronounced than Schwarzschild because the disk orbits faster at smaller radii.
- **Frame dragging**: Stars near the shadow should show asymmetric lensing patterns.

## What This Defers

- ZAMO observer (use static observer for now)
- Kerr-specific Christoffel symbols (continue using finite differences)
- Adaptive stepping (separate task)
