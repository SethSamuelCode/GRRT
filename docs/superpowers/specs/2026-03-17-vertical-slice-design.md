# Vertical Slice: Schwarzschild Shadow Renderer

## Goal

Render a black hole shadow (silhouette) by tracing photon geodesics through Schwarzschild spacetime. A black circle on a white background, validated against the analytical shadow radius.

## Architecture

```
pixel (i,j)
    → Camera generates initial photon 4-momentum p_μ
    → GeodesicTracer integrates Hamilton's equations via RK4
    → Termination: horizon hit → black, escape → white
    → Write pixel to framebuffer
```

All existing code (`api.h`, `types.h`, `api.cpp`, `cli/main.cpp`, `CMakeLists.txt`) is preserved. New files slot into the existing `include/grrt/` and `src/` structure.

## New Files

| File | Purpose |
|------|---------|
| `include/grrt/math/vec4.h` | Header-only 4-component vector, index access `[0..3]` for `(t,r,θ,φ)`, arithmetic ops |
| `include/grrt/math/matrix4.h` | Header-only 4×4 matrix, metric storage, index contraction `g_μν v^ν`, inverse for diagonal case |
| `include/grrt/spacetime/metric.h` | Abstract `Metric` interface: `g_lower()`, `g_upper()`, `horizon_radius()` |
| `include/grrt/spacetime/schwarzschild.h` | Schwarzschild metric declaration |
| `src/schwarzschild.cpp` | Schwarzschild implementation: diagonal metric, horizon at `r = 2M` |
| `include/grrt/geodesic/integrator.h` | `GeodesicState` struct (position + covariant momentum) |
| `include/grrt/geodesic/rk4.h` | RK4 integrator declaration |
| `src/rk4.cpp` | RK4 fixed-step implementation of Hamilton's equations |
| `include/grrt/geodesic/geodesic_tracer.h` | Tracer declaration: traces one ray to termination |
| `src/geodesic_tracer.cpp` | Tracer implementation: loop RK4 steps, check horizon/escape |
| `include/grrt/camera/camera.h` | Camera declaration: observer position, pixel → initial momentum |
| `src/camera.cpp` | Camera implementation: tetrad construction, ray generation |
| `include/grrt/render/renderer.h` | Renderer declaration |
| `src/renderer.cpp` | Renderer: loop over pixels with OpenMP, call tracer, write black/white |

## Modified Files

| File | Change |
|------|--------|
| `src/api.cpp` | Replace gradient placeholder with real renderer pipeline |
| `CMakeLists.txt` | Add new `.cpp` source files to the `grrt` library target |

## Component Details

### Vec4 (`math/vec4.h`)

Header-only. Stores 4 doubles. Provides:
- `operator[]` for index access
- `operator+`, `operator-`, `operator*` (scalar)
- No dot product here — that requires the metric (not Euclidean)

```cpp
struct Vec4 {
    double data[4]{};
    double& operator[](int i);
    const double& operator[](int i) const;
    Vec4 operator+(const Vec4& other) const;
    Vec4 operator*(double s) const;
};
```

### Matrix4 (`math/matrix4.h`)

Header-only. 4×4 stored as `double m[4][4]`. Provides:
- Index contraction: `contract(Vec4 v)` → computes `g_μν v^ν` (returns covariant vector)
- `inverse()` → for Schwarzschild, exploit diagonal structure (just 1/diagonal)
- Static `diagonal(a, b, c, d)` factory for diagonal metrics

### Schwarzschild Metric

Implements `Metric` interface. In Boyer-Lindquist coordinates `(t, r, θ, φ)`:

```
g_tt = -(1 - 2M/r)
g_rr = 1/(1 - 2M/r)
g_θθ = r²
g_φφ = r² sin²θ
```

All off-diagonal terms are zero. Horizon at `r = 2M`.

### Hamiltonian Geodesic Equations

State vector: `{x^μ, p_μ}` — 8 components (position contravariant, momentum covariant).

Hamilton's equations for null geodesics:
```
dx^μ/dλ =  ∂H/∂p_μ = g^μν p_ν
dp_μ/dλ = -∂H/∂x^μ = -½ (∂g^αβ/∂x^μ) p_α p_β
```

The derivatives `∂g^αβ/∂x^μ` are computed by central finite differences on the inverse metric:
```
∂g^αβ/∂x^μ ≈ (g^αβ(x + ε·e_μ) - g^αβ(x - ε·e_μ)) / (2ε)
```

This is general-purpose — works for any metric without needing analytical Christoffel symbols.

### RK4 Integrator

Fixed step size for the vertical slice. Takes a `GeodesicState` and step size `dλ`, returns new state.

Step size heuristic: `dλ = 0.01 * r` (smaller near the hole where curvature is stronger). This isn't adaptive error control yet — just a sensible scaling.

### Geodesic Tracer

Traces one photon backward from camera:
1. Step with RK4
2. After each step, check:
   - `r < 2M + ε` → **horizon hit** (black pixel)
   - `r > r_escape` (e.g. 1000M) → **escaped** (white pixel)
   - Steps exceeded `max_steps` → black pixel (safety)
3. Return termination type

### Camera

Observer at Boyer-Lindquist coordinates `(r_obs, θ_obs, φ_obs=0)`. Static observer (4-velocity purely in `t` direction).

Tetrad construction (Gram-Schmidt):
- `e_0` = normalized 4-velocity (timelike)
- `e_3` = radial inward (toward hole) — the "forward" direction
- `e_2` = polar direction (up)
- `e_1` = azimuthal direction (right)

For pixel `(i, j)`:
1. Compute screen angles `α`, `β` from FOV
2. Local 3-direction: `d = -cos(β)sin(α) e_1 - sin(β) e_2 + cos(β)cos(α) e_3`
3. 4-momentum: `p^μ = -e_0^μ + d^μ`
4. Lower index: `p_μ = g_μν p^ν`

### Renderer

OpenMP parallel loop over pixels. For each pixel:
1. `camera.ray_for_pixel(i, j)` → initial `GeodesicState`
2. `tracer.trace(state)` → termination type
3. Horizon hit → RGBA `(0,0,0,1)`, escape → RGBA `(1,1,1,1)`

### API Changes

`grrt_render()` in `api.cpp` will:
1. Construct `Schwarzschild` metric from params
2. Construct `Camera` from observer params
3. Construct `GeodesicTracer` with metric and integration params
4. Construct `Renderer` with all of the above
5. Call `renderer.render(framebuffer)`

## Validation

The Schwarzschild shadow has an exact analytical radius:
```
sin(α_shadow) = 3√3 M / r_obs
```

For `M = 1, r_obs = 50`: `α_shadow ≈ 0.1039 rad ≈ 5.95°`. With a 60° FOV and 256px width, the shadow should span roughly `256 × (2 × 5.95 / 60) ≈ 51 pixels` in diameter.

We visually verify the output and can measure the shadow radius in pixels to confirm it matches.

## What This Slice Defers

- Kerr metric (Phase 1 expansion)
- Adaptive step size (Phase 1 expansion)
- Accretion disk, redshift, color (Phase 1 expansion)
- Celestial sphere backgrounds (Phase 1 expansion)
- Tone mapping (Phase 1 expansion)
- CUDA (Phase 2)
- Blender integration (Phase 3)
