# Accretion Disk, Color Pipeline, and Star Field

## Goal

Add a physically accurate thin accretion disk with Page-Thorne temperature profile, blackbody color rendering, gravitational redshift/Doppler beaming, a procedural star field background, and tone mapping. Transform the black-circle-on-white output into a full-color scientific visualization.

## Architecture

```
Geodesic tracing loop (per step):
    → Check θ crossing π/2 → disk intersection?
        → Yes: compute r_cross, redshift g, emission T(r), color via LUT
              accumulate into ray's color, continue tracing (thin disk, ray passes through)
        → No: continue stepping
    → Check horizon/escape as before
        → Escaped: sample star field at final (θ, φ) direction

Color pipeline:
    disk T(r) → Page-Thorne → T_emitted
    → apply redshift: T_observed = g × T_emitted
    → LUT lookup: T → linear RGB
    → scale by g³ (intensity)
    → tone map (Reinhard) → sRGB gamma → output (CLI only; Blender gets linear HDR)
```

## New Files

| File | Purpose |
|------|---------|
| `include/grrt/math/vec3.h` | Header-only 3D vector for RGB color math |
| `include/grrt/scene/accretion_disk.h` | Disk geometry, Page-Thorne temperature, emission, redshift |
| `src/accretion_disk.cpp` | Disk implementation: flux LUT, 4-velocity, redshift computation |
| `include/grrt/scene/celestial_sphere.h` | Star field catalog and sampling |
| `src/celestial_sphere.cpp` | Star generation and direction → color lookup |
| `include/grrt/color/spectrum.h` | Planck function, temperature → RGB lookup table |
| `src/spectrum.cpp` | CIE color matching integration, LUT construction |
| `include/grrt/render/tonemapper.h` | Reinhard tone mapping + sRGB gamma |
| `src/tonemapper.cpp` | HDR → LDR conversion |

**Note on `redshift.h`**: The project spec lists a separate `redshift.h`/`redshift.cpp`. For this implementation, redshift computation lives inside `AccretionDisk` (it needs the disk's 4-velocity and metric), and the temperature-shift-to-color mapping lives in `SpectrumLUT`. A separate `redshift.h` is not needed. If we later need shared redshift utilities (e.g., for Kerr), we can extract then.

## Modified Files

| File | Change |
|------|--------|
| `include/grrt/geodesic/geodesic_tracer.h` | Replace `RayTermination` return with `TraceResult` struct. New method signature: `TraceResult trace(GeodesicState state, const AccretionDisk* disk) const`. |
| `src/geodesic_tracer.cpp` | Add θ-crossing detection. On disk hit, compute redshift via `AccretionDisk`, accumulate emission, continue tracing. Return accumulated color + termination type + final direction. |
| `include/grrt/render/renderer.h` | Constructor takes `Camera`, `GeodesicTracer`, `AccretionDisk*`, `CelestialSphere*`, `SpectrumLUT*`, `ToneMapper*`. |
| `src/renderer.cpp` | Use TraceResult: accumulated disk color from tracer, add star color for escaped rays, tone map, write to framebuffer. |
| `src/api.cpp` | Add `AccretionDisk`, `CelestialSphere`, `SpectrumLUT`, `ToneMapper` to `GRRTContext`. Construct from params (check `disk_enabled`, `background_type`). Pass to tracer and renderer. |
| `CMakeLists.txt` | Add new `.cpp` files. |

## Component Details

### Vec3 (`math/vec3.h`)

Header-only 3D vector for RGB color math. Stores 3 doubles. Provides:
- `operator+`, `operator-`, `operator*` (scalar), `operator*` (component-wise for color modulation)
- Component access via `[0]`, `[1]`, `[2]` or `.r()`, `.g()`, `.b()`
- Same double-brace aggregate-init pattern as Vec4

### Accretion Disk (`scene/accretion_disk.h`)

#### Geometry

Geometrically thin disk in the equatorial plane (`θ = π/2`). Inner edge at ISCO (`r_isco = 6M` for Schwarzschild), outer edge at `r_outer` (default `20M`).

#### ISCO Calculation

For Schwarzschild: `r_isco = 6M`.

#### Circular Orbit 4-Velocity

Disk material moves on circular geodesics. For Schwarzschild at radius `r`:

```
Ω = √(M / r³)                    — angular velocity
u^t = 1 / √(1 - 3M/r)           — time component
u^φ = Ω × u^t                    — azimuthal component
u^r = u^θ = 0                    — circular orbit
```

The full 4-velocity is `u^μ_emitter = (u^t, 0, 0, u^φ)`.

#### Page-Thorne Temperature Profile

The Page-Thorne (1974) model gives the flux radiated per unit area of the disk. For Schwarzschild:

The specific energy and angular momentum of circular orbits:
```
E(r) = (1 - 2M/r) / √(1 - 3M/r)
L(r) = √(M r) / √(1 - 3M/r)
```

Their derivatives with respect to `r`:
```
E'(r) = dE/dr
L'(r) = dL/dr
```

(Computed analytically or by finite differences.)

The integral function:
```
I(r) = ∫_{r_isco}^{r} [ (E(r') - E_isco) × L'(r') - (L(r') - L_isco) × E'(r') ] dr'
```

where `E_isco = E(r_isco)` and `L_isco = L(r_isco)`.

The flux:
```
F(r) = (3M / (8π r³)) × (1 / (E(r) - Ω(r) × L(r))) × (-dΩ/dr) × I(r)
```

This is the standard Novikov-Thorne form. The flux vanishes at `r_isco` (where `I = 0`, zero-torque inner boundary condition) and falls off at large `r`.

The temperature is:
```
T(r) = T_peak × [ F(r) / F_max ]^{1/4}
```

where `T_peak` is user-specified (default `1e7 K`) and `F_max` is the maximum flux, which occurs around `r ≈ 8.2M` for Schwarzschild.

**Implementation approach**: Precompute `F(r)` by numerical integration (Simpson's rule) at ~500 radial points from `r_isco` to `r_outer`, store as a lookup table, interpolate at runtime. Compute `E'(r)` and `L'(r)` analytically:

```
E'(r) = M(r - 3M) / (r² × (r - 2M) × √(r(r - 3M))) ... [full derivative]
```

Or use central finite differences on `E(r)` and `L(r)` with `ε = 1e-6` — simpler and sufficient given we're precomputing.

#### Intersection Detection

During geodesic tracing, monitor `θ` at each step. When `θ` crosses `π/2`:

1. Detect crossing: `(θ_prev - π/2) × (θ_new - π/2) < 0`
2. Linear interpolation to find fraction: `frac = (π/2 - θ_prev) / (θ_new - θ_prev)`
3. Interpolate `r_cross = r_prev + frac × (r_new - r_prev)` and `φ_cross` similarly
4. Interpolate `p_μ` at crossing: `p_cross = p_prev + frac × (p_new - p_prev)`
5. Check bounds: `r_isco ≤ r_cross ≤ r_outer`
6. If hit: compute redshift and emission at `(r_cross, φ_cross, p_cross)`

**Multiple crossings**: A ray can cross the equatorial plane multiple times. Each valid crossing contributes emission that is accumulated. The tracer continues stepping after a disk crossing (the thin disk is transparent — the ray passes through). In practice the first 1-2 crossings dominate.

#### Redshift Computation

The redshift factor `g` is the ratio of observed to emitted photon energy:

```
g = (p_μ u^μ)_emitter / (p_μ u^μ)_observer
```

**Note**: This convention gives `g > 1` for blueshift (approaching side, higher observed energy) and `g < 1` for redshift (receding side, lower observed energy).

For the observer (static at `r_obs`):
```
(p_μ u^μ)_obs = p_t / √(1 - 2M/r_obs)
```
(Since `u^μ_obs = (1/√(1-2M/r_obs), 0, 0, 0)` and only `p_t` contributes.)

For the emitter (circular orbit at `r_cross`):
```
(p_μ u^μ)_emit = p_t × u^t_emit + p_φ × u^φ_emit
```
where `u^t_emit = 1/√(1-3M/r)` and `u^φ_emit = Ω × u^t_emit`.

**Note on sign**: `p_t` is negative (conserved energy with our sign convention), and `(p_μ u^μ)` is negative for physical photons. The ratio `g` comes out positive because both numerator and denominator are negative.

The observed intensity scales as `I_obs = g³ × I_emitted` (relativistic beaming: `g` for frequency shift, `g²` for solid angle compression, net `g³` for specific intensity).

### Spectrum (`color/spectrum.h`)

#### Planck Function

In wavelength space (matching CIE tabulation):

```
B(λ, T) = (2hc² / λ⁵) / (exp(hc / λkT) - 1)
```

Constants: `h = 6.626e-34 J·s`, `c = 3e8 m/s`, `k = 1.381e-23 J/K`.

#### Temperature → RGB Lookup Table

At initialization, for each of ~1000 temperatures from 1000K to 100,000K:

1. Evaluate `B(λ, T)` at CIE standard wavelengths (380–780nm, 5nm steps = 81 samples)
2. Integrate against CIE 1931 `x̄(λ)`, `ȳ(λ)`, `z̄(λ)` color matching functions → XYZ
3. Convert XYZ → linear sRGB via the standard 3×3 matrix
4. Clamp negative RGB values to 0 (out-of-gamut colors)
5. Normalize so that the brightest component equals 1.0 (preserves chromaticity; intensity handled separately by `g³ × σT⁴`)

Store as `Vec3 color_lut[1000]` (chromaticity only) and `double luminosity_lut[1000]` (relative brightness via Stefan-Boltzmann `σT⁴`).

At runtime, given `T_observed = g × T_emitted`:
- Clamp to LUT range [1000K, 100000K]
- Linear interpolate between nearest entries for both color and luminosity
- Final pixel contribution = `color × luminosity × g³`

#### CIE Color Matching Functions

Hardcode the CIE 1931 2° observer data as a static array (81 entries at 5nm spacing from 380–780nm). This is standard reference data, widely published.

#### XYZ → sRGB Matrix

Standard IEC 61966-2-1 matrix:
```
R =  3.2406 X - 1.5372 Y - 0.4986 Z
G = -0.9689 X + 1.8758 Y + 0.0415 Z
B =  0.0557 X - 0.2040 Y + 1.0570 Z
```

### Celestial Sphere (`scene/celestial_sphere.h`)

#### Star Field

At initialization, generate ~5000 stars with:
- Random `(θ, φ)` positions (uniform on sphere: `θ = acos(1 - 2u)`, `φ = 2π v`)
- Random brightness following a power-law distribution (many dim stars, few bright ones): `brightness = base_brightness × u^(-2.5)`, clamped
- Seed with a fixed value for reproducibility

#### Ray → Star Color

When a ray escapes (`r > r_escape`), extract its asymptotic sky direction from the ray's final position coordinates `(θ_final, φ_final)` — at large `r`, the position coordinates directly give the direction on the celestial sphere.

For each escaped ray:
1. Look up `(θ_final, φ_final)` in a coarse spatial grid (360×180 bins indexed by `(floor(θ × 180/π), floor(φ × 180/π + 180))`)
2. Check stars in that bin and adjacent bins
3. If any star is within angular tolerance `δ = 0.01 rad`, return `Vec3(brightness, brightness, brightness)` (white star)
4. Otherwise return `Vec3(0, 0, 0)` (black background)

The angular tolerance `δ` controls star apparent size. `0.01 rad` gives visible point-like stars at 256px resolution.

### Tone Mapper (`render/tonemapper.h`)

Takes a linear HDR RGB `Vec3` and produces a tone-mapped `Vec3`:

1. **Reinhard** (per-channel): `c_out = c / (1 + c)` for each of R, G, B
2. **sRGB gamma**: `v_out = v^(1/2.2)` (simplified gamma)

**Output format note**: The CLI applies tone mapping + gamma so the PNG looks correct. For Blender integration (Phase 3), the framebuffer should be linear HDR (Blender applies its own color management). The `ToneMapper` is called by the renderer but can be disabled via a flag in `GRRTParams` or made a post-process in the CLI only. For now, we apply it unconditionally in the renderer.

### Tracer Changes

Replace `RayTermination` return with a richer `TraceResult`:

```cpp
struct TraceResult {
    RayTermination termination;  // Horizon, Escaped, MaxSteps (Disk is not a terminal state)
    Vec3 accumulated_color;      // Sum of all disk crossing emissions (linear HDR)
    Vec4 final_position;         // Position at termination
    Vec4 final_momentum;         // Momentum at termination (for star field direction)
};
```

`RayTermination` keeps its existing values: `Horizon`, `Escaped`, `MaxSteps`. Disk crossings are not terminal — the ray continues through the thin disk.

New `trace()` signature:
```cpp
TraceResult trace(GeodesicState state, const AccretionDisk* disk, const SpectrumLUT& spectrum) const;
```

The tracer:
1. Steps with RK4 as before
2. After each step, checks for θ crossing π/2
3. On crossing within disk bounds: computes redshift `g`, temperature `T(r)`, color via spectrum LUT, scales by `g³`, adds to `accumulated_color`
4. Continues stepping (disk is thin/transparent)
5. Checks horizon/escape as before
6. Returns `TraceResult` with final accumulated color and how the ray terminated

If `disk` is null (disk disabled), skip crossing detection.

### Renderer Changes

New constructor:
```cpp
Renderer(const Camera& camera, const GeodesicTracer& tracer,
         const AccretionDisk* disk, const CelestialSphere* sphere,
         const SpectrumLUT& spectrum, const ToneMapper& tonemapper);
```

For each pixel:
1. `camera.ray_for_pixel(i, j)` → initial `GeodesicState`
2. `tracer.trace(state, disk, spectrum)` → `TraceResult`
3. `color = result.accumulated_color` (disk emission, may be zero)
4. If `result.termination == Escaped` and `sphere` is not null: `color = color + sphere->sample(result.final_position)`
5. `color = tonemapper.apply(color)`
6. Write `(color.r, color.g, color.b, 1.0)` as floats to framebuffer

### API Changes

`GRRTContext` gains:
```cpp
std::unique_ptr<grrt::AccretionDisk> disk;          // null if disk_enabled == 0
std::unique_ptr<grrt::CelestialSphere> sphere;      // null if background == BLACK
std::unique_ptr<grrt::SpectrumLUT> spectrum;
std::unique_ptr<grrt::ToneMapper> tonemapper;
```

In `grrt_create()`:
- If `params->disk_enabled`: construct `AccretionDisk` with `mass`, `disk_inner` (0 = ISCO), `disk_outer`, `disk_temperature`
- If `params->background_type == GRRT_BG_STARS`: construct `CelestialSphere` with star field
- Always construct `SpectrumLUT` and `ToneMapper`
- Pass disk (may be null), sphere (may be null), spectrum, tonemapper to `Renderer`
- Pass disk (may be null), spectrum to `GeodesicTracer` (or pass per-call as in the trace signature)

## Validation

- **Doppler beaming**: The approaching side of the disk should be noticeably brighter than the receding side (the side where `p_φ` has the same sign as the orbital `u^φ` gets blueshifted → `g > 1` → brighter)
- **Temperature gradient**: Inner disk should glow white/blue-white, outer disk red/orange
- **Redshift at ISCO**: Material near the inner edge should be dimmer (gravitational redshift) and flux drops to zero exactly at `r_isco`
- **Higher-order images**: A thin bright ring should be visible just outside the shadow edge
- **Einstein ring**: Background stars near the shadow edge should appear duplicated/stretched
- **Page-Thorne zero at ISCO**: `F(r_isco) = 0` because `I(r_isco) = 0`
- **Color correctness**: A ~5800K blackbody should look white/yellowish (like the Sun)

## What This Defers

- Kerr metric (separate expansion — will need Kerr ISCO, Kerr circular orbits, off-diagonal g_tφ in redshift)
- Adaptive step size (separate expansion)
- ACES tone mapping (can swap in later)
- Background texture/equirectangular map (separate, swap celestial sphere mode)
- Linear HDR output mode for Blender (Phase 3, add flag to skip tone mapping)
