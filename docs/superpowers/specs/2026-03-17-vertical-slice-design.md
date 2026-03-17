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
| `include/grrt/math/matrix4.h` | Header-only 4×4 matrix, metric storage, contraction (raise/lower indices), inverse |
| `include/grrt/spacetime/metric.h` | Abstract `Metric` interface: `g_lower()`, `g_upper()`, `horizon_radius()` |
| `include/grrt/spacetime/schwarzschild.h` | Schwarzschild metric declaration |
| `src/schwarzschild.cpp` | Schwarzschild implementation: diagonal metric, horizon at `r = 2M` |
| `include/grrt/geodesic/integrator.h` | `GeodesicState` struct (position + covariant momentum) |
| `include/grrt/geodesic/rk4.h` | RK4 integrator declaration |
| `src/rk4.cpp` | RK4 fixed-step implementation of Hamilton's equations |
| `include/grrt/geodesic/geodesic_tracer.h` | Tracer declaration: traces one ray to termination |
| `src/geodesic_tracer.cpp` | Tracer implementation: loop RK4 steps, check horizon/escape |
| `include/grrt/camera/camera.h` | Camera declaration: observer position + tetrad, pixel → initial momentum |
| `src/camera.cpp` | Camera implementation: tetrad construction, ray generation |
| `include/grrt/render/renderer.h` | Renderer declaration |
| `src/renderer.cpp` | Renderer: loop over pixels with OpenMP, call tracer, write black/white |

**Note on observer.h**: The project spec defines a separate `observer.h` for the tetrad/4-velocity. For this slice, the Camera class owns the tetrad directly. When we expand to support Blender camera offsets, we'll extract the observer into its own class.

## Modified Files

| File | Change |
|------|--------|
| `src/api.cpp` | Replace gradient placeholder with real renderer pipeline. Metric/camera/tracer constructed in `grrt_create()` and stored in `GRRTContext`. `grrt_render()` calls `renderer.render()`. |
| `CMakeLists.txt` | Add new `.cpp` files to `grrt` target. Preserve existing `${CMAKE_BINARY_DIR}/include` path for `grrt_export.h`. |

## Component Details

### Vec4 (`math/vec4.h`)

Header-only. Stores 4 doubles. Provides:
- `operator[]` for index access
- `operator+`, `operator-`, `operator*` (scalar)
- No dot product — that requires the metric (not Euclidean)

```cpp
struct Vec4 {
    double data[4]{};
    double& operator[](int i);
    const double& operator[](int i) const;
    Vec4 operator+(const Vec4& other) const;
    Vec4 operator-(const Vec4& other) const;
    Vec4 operator*(double s) const;
};
```

### Matrix4 (`math/matrix4.h`)

Header-only. 4×4 stored as `double m[4][4]`. Provides:
- `contract(Vec4 v)` → computes `Σ_ν M_μν v^ν` (index lowering when M is `g_lower`, index raising when M is `g_upper`)
- `inverse()` → general 4×4 inverse (for Schwarzschild, diagonal so just `1/diagonal`)
- Static `diagonal(a, b, c, d)` factory for diagonal metrics

Both lowering (`p_μ = g_μν p^ν`) and raising (`p^μ = g^μν p_ν`) use the same `contract()` method — you just call it on the appropriate matrix (`g_lower` or `g_upper`).

### Schwarzschild Metric

Implements `Metric` interface. In Boyer-Lindquist coordinates `(t, r, θ, φ)`:

```
g_tt = -(1 - 2M/r)
g_rr = 1/(1 - 2M/r)
g_θθ = r²
g_φφ = r² sin²θ
```

All off-diagonal terms are zero. Horizon at `r = 2M`.

**θ singularity**: `g_φφ = r² sin²θ` vanishes at the poles (`θ = 0, π`). Clamp `|sinθ| ≥ 1e-10` in the metric evaluation to avoid division by zero in the inverse metric. With the observer at `θ_obs ≈ 80°`, most rays won't reach the poles, but edge rays could.

### Hamiltonian Geodesic Equations

State vector: `{x^μ, p_μ}` — 8 components (position contravariant, momentum covariant).

Hamilton's equations for null geodesics (`H = ½ g^μν p_μ p_ν = 0`):
```
dx^μ/dλ =  ∂H/∂p_μ = g^μν p_ν           (raise momentum with inverse metric)
dp_μ/dλ = -∂H/∂x^μ = -½ Σ_{α,β=0}^{3} (∂g^αβ/∂x^μ) p_α p_β
```

The position derivative `dx^μ/dλ` is computed by contracting `g_upper` with the covariant momentum: `g_upper.contract(p)`.

The momentum derivative requires partial derivatives of `g^αβ` with respect to each coordinate. These are computed by central finite differences:
```
∂g^αβ/∂x^μ ≈ (g^αβ(x + ε·e_μ) - g^αβ(x - ε·e_μ)) / (2ε)
```
with `ε = 1e-6`. The force term sums over all 16 components of `g^αβ` (though for Schwarzschild only the 4 diagonal terms are nonzero).

### RK4 Integrator

Fixed step size for the vertical slice. Takes a `GeodesicState` and step size `dλ`, returns new state.

Step size heuristic: `dλ = 0.01 * r` (smaller steps near the hole where curvature is stronger). This deviates from the project spec's `dλ ∝ r²` — the linear scaling is more conservative near the horizon, which is appropriate for a fixed-step integrator without error control. Adaptive stepping (Phase 1 expansion) will revisit this.

### Geodesic Tracer

Traces one photon backward from camera:
1. Compute step size `dλ = 0.01 * r`
2. Step with RK4
3. After each step, check:
   - `r < 2M + ε` (ε = 0.01) → **horizon hit** (black pixel)
   - `r > r_escape` (1000M) → **escaped** (white pixel)
   - Steps exceeded `max_steps` (default 10000) → black pixel (safety)
4. Return termination type

### Camera

Observer at Boyer-Lindquist coordinates `(r_obs, θ_obs, φ_obs=0)`. Static observer: 4-velocity `u^μ = (1/√(-g_tt), 0, 0, 0)`.

**Tetrad construction** via Gram-Schmidt orthogonalization using the **spacetime metric inner product** `⟨u, v⟩ = g_μν u^μ v^ν` (not Euclidean dot product):

1. `e_0 = u^μ / √|g_μν u^μ u^ν|` — normalized timelike vector
2. Start with coordinate `r`-direction, project out `e_0`, normalize → `e_3` (forward, toward hole)
3. Start with coordinate `θ`-direction, project out `e_0` and `e_3`, normalize → `e_2` (up)
4. Start with coordinate `φ`-direction, project out `e_0`, `e_3`, `e_2`, normalize → `e_1` (right)

Each tetrad vector satisfies `g_μν e_a^μ e_b^ν = η_ab` (Minkowski: `e_0` is timelike with norm -1, spatial vectors have norm +1).

**Pixel → momentum**:

For pixel `(i, j)`:
1. Screen angles: `α = (i - width/2) * fov / width`, `β = (j - height/2) * fov / width`
2. Local 3-direction (unit vector in tetrad frame): `d^μ = -cos(β)sin(α) e_1^μ - sin(β) e_2^μ + cos(β)cos(α) e_3^μ`
3. Null 4-momentum (contravariant): `p^μ = -e_0^μ + d^μ`
   - This is null by construction: `g_μν p^μ p^ν = g_μν(-e_0 + d)^μ(-e_0 + d)^ν = -1 + 1 = 0`
   - The tetrad orthonormality guarantees `d` is a unit spatial vector, so no extra normalization needed
4. Lower index: `p_μ = g_μν p^ν` (this is the covariant momentum stored in `GeodesicState`)

### Renderer

OpenMP parallel loop over pixels. For each pixel:
1. `camera.ray_for_pixel(i, j)` → initial `GeodesicState`
2. `tracer.trace(state)` → termination type
3. Horizon hit → RGBA `(0,0,0,1)`, escape → RGBA `(1,1,1,1)`

### API Changes

`grrt_create()` constructs the metric, camera, tracer, and renderer, storing them in `GRRTContext`.

`grrt_render()` calls `renderer.render(framebuffer)` on the stored objects.

Default integration parameters (used when `GRRTParams` fields are zero-initialized):
- `integrator_max_steps`: 10000
- `integrator_tolerance`: unused in this slice (fixed step)

## Validation

The Schwarzschild shadow has an exact analytical angular radius for a static observer:
```
sin(α_shadow) = (3√3 M / r_obs) × 1/√(1 - 2M/r_obs)
```

The `3√3 M` is the critical impact parameter (photon sphere at `r = 3M`). The `1/√(1 - 2M/r_obs)` factor accounts for the observer's gravitational redshift.

For `M = 1, r_obs = 50`:
- `3√3 / 50 × 1/√(1 - 2/50) ≈ 0.10392 × 1.0204 ≈ 0.10604 rad ≈ 6.07°`
- With 60° FOV and 256px: shadow diameter ≈ `256 × (2 × 6.07 / 60) ≈ 52 pixels`

We visually verify the output and measure the shadow radius in pixels to confirm it matches within a few percent.

## What This Slice Defers

- Kerr metric (Phase 1 expansion)
- Adaptive step size (Phase 1 expansion)
- Accretion disk, redshift, color (Phase 1 expansion)
- Celestial sphere backgrounds (Phase 1 expansion)
- Tone mapping (Phase 1 expansion)
- Observer as separate class (Phase 1 expansion)
- CUDA (Phase 2)
- Blender integration (Phase 3)
