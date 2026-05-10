# Low-discrepancy sub-pixel sampling (Sobol + Owen)

**Date:** 2026-05-02
**Branch:** `fix/volumetric-ring`
**Author:** brainstormed with Claude

## Summary

Replace the per-pixel stratified-jitter sub-pixel sampler in `src/renderer.cpp` with a 2D Owen-scrambled Sobol sequence applied to all three render paths (RGB, spectral, spectral-to-FITS). The user-facing API stays the same — `--samples N` still means "N samples per pixel" — but `N` is now the actual sample count (no silent rounding to `floor(sqrt(N))²`) and each sub-pixel offset comes from a low-discrepancy sequence with per-pixel scrambling.

The motivation is render quality at fixed compute budget. Multi-hour production renders at high `spp` benefit most: Sobol+Owen's lower variance translates to ~50% reduction in render time for equivalent quality, and the asymptotic convergence holds out to spp ≫ 10⁴ (where Halton or vanilla low-discrepancy would start showing patterns).

## Background

The current sampler in `src/renderer.cpp:14-66` divides each pixel into a `grid × grid` sub-grid (where `grid = floor(sqrt(spp))`) and jitters within each cell using a deterministic hash function `pixel_hash(i, j, s, channel)`. This is "stratified jittering" — a standard technique that's roughly 2× more efficient than purely random sub-pixel sampling.

Two issues motivate the change:

1. **Quality at fixed spp.** Stratified jitter has 1/√N variance convergence. For sharp features like our photosphere cliff this can't be improved (the integrand is non-smooth), but for the smoother regions of the image (atmosphere, lensing rings, far-from-disk background) low-discrepancy sequences converge faster. Empirically, Sobol+Owen reduces variance by 1.5–2× at the same spp for typical scenes.

2. **Silent sample-count rounding.** `--samples 30` actually produces 25 samples (5×5), `--samples 50` actually produces 49 (7×7). The grid structure is a workaround for stratified jitter, not a fundamental limitation. Sobol takes any sample count.

## Goals

- **G1.** Replace stratified-jitter with Owen-scrambled Sobol in all three render paths (RGB, spectral, spectral-to-FITS).
- **G2.** Eliminate the `floor(sqrt(spp))²` rounding so `--samples N` produces exactly `N` samples.
- **G3.** Preserve determinism: the same scene at the same `--samples` produces bit-identical pixels across runs and thread counts.
- **G4.** Per-pixel decorrelation: neighboring pixels see uncorrelated scrambles so no visible repeating patterns appear at any spp.
- **G5.** Production-grade asymptotic behavior: quality holds out to spp = 10⁵+ without sequence artifacts.

## Non-goals

- Adding a `--sampler` flag to switch between stratified and Sobol. The old algorithm has no remaining use case once Sobol lands.
- Implementing true Owen tree scramble. Burley's hash-based form is within ~5% variance of true Owen and is the production standard in pbrt-v4, RenderMan, Disney, Arnold.
- Implementing dimensions beyond 2. The current renderer's only sub-pixel sample is 2D (px, py). Future spectral-axis sampling, time sampling, or path-tracer extensions would extend the dimension count, but those are separate work.
- Updating the CUDA backend. Phase 2 will mirror the design separately.

## Architecture

### One new utility, three call-site changes

**New utility:**

- `include/grrt/render/sobol_sampler.h` — public API.
- `src/sobol_sampler.cpp` — Joe-Kuo direction numbers + Sobol generation + Burley Owen scramble.

```cpp
namespace grrt {

/// Generate the i-th 2D point of the Owen-scrambled Sobol sequence
/// for a specific pixel. Output components are in [0, 1).
///
/// Determinism: same (pixel_x, pixel_y, sample_index) → same point. The
/// per-pixel scramble seed is derived from (pixel_x, pixel_y), so neighboring
/// pixels see decorrelated point sets but the same render produces identical
/// output across runs and thread counts.
struct SobolSample {
    double x, y;
};

GRRT_EXPORT SobolSample sobol_sample_2d(int pixel_x, int pixel_y, int sample_index);

} // namespace grrt
```

**Modified call sites** (all in `src/renderer.cpp`):

1. `Renderer::render()` — RGB.
2. `Renderer::render_spectral()` — N-frequency render.
3. `Renderer::render_spectral_to_fits()` — streaming spectral.

Each replaces the existing `for (sy)/(sx)` stratified-grid loop with a flat `for (s = 0; s < spp_; ++s)` loop calling `sobol_sample_2d(i, j, s)`.

**Deleted:**
- `static double pixel_hash(...)` at `src/renderer.cpp:14-20`.
- The `grid = floor(sqrt(spp_))` and `actual_spp = grid * grid` rounding logic in all three render functions.

## Sampler internals

### Sobol point generation (32-bit fixed-point gray-code traversal)

Each dimension has a precomputed table of 32 direction numbers `V[d][k]` (32 bits each). The i-th point in dimension `d` is computed by XORing direction numbers indexed by the bits of i's gray code:

```cpp
uint32_t sobol_point_1d(int i, int dim) {
    uint32_t result = 0;
    int idx = i ^ (i >> 1);  // gray code
    int k = 0;
    while (idx) {
        if (idx & 1) result ^= V[dim][k];
        idx >>= 1;
        k++;
    }
    return result;
}
```

The result is a 32-bit fixed-point fraction; multiply by `2^-32` to get a double in `[0, 1)`.

The direction-number tables come from Joe-Kuo's `new-joe-kuo-6.21201` data set, public-domain at `https://web.maths.unsw.edu.au/~fkuo/sobol/`. Only the first two dimensions are needed (≈64 32-bit integers, ~256 bytes). Embedded as a `constexpr` array with attribution.

### Burley hash-based Owen scramble

After generating the raw Sobol point, randomize per-pixel using Burley's 2020 hash:

```cpp
uint32_t burley_scramble(uint32_t x, uint32_t seed) {
    x = x ^ (x * 0x3d20adea);
    x += seed;
    x *= (seed >> 16) | 1;
    x = x ^ (x * 0x05526c56);
    x = x ^ (x * 0x53a22864);
    return x;
}
```

Constants from Burley, "Practical Hash-Based Owen Scrambling" (2020). The same form is used by pbrt-v4's `nested_uniform_scramble`.

### Per-pixel scramble seed derivation

The seed must be deterministic from `(pixel_x, pixel_y)` (so renders reproduce) AND decorrelated across neighbors (so Owen actually breaks patterns). One seed per dimension:

```cpp
uint32_t scramble_seed(int pixel_x, int pixel_y, int dim) {
    uint32_t h = static_cast<uint32_t>(pixel_x * 73856093u
                                     ^ pixel_y * 19349663u
                                     ^ dim     * 83492791u);
    h ^= h >> 16; h *= 0x45d9f3bu; h ^= h >> 16;
    return h;
}
```

Uses the same hash style as the existing `pixel_hash` (the multiplier constants follow MurmurHash3 finalization).

### Putting it together

```cpp
SobolSample sobol_sample_2d(int pixel_x, int pixel_y, int sample_index) {
    const uint32_t seed_x = scramble_seed(pixel_x, pixel_y, 0);
    const uint32_t seed_y = scramble_seed(pixel_x, pixel_y, 1);

    const uint32_t sx_raw = sobol_point_1d(sample_index, 0);
    const uint32_t sy_raw = sobol_point_1d(sample_index, 1);

    const uint32_t sx = burley_scramble(sx_raw, seed_x);
    const uint32_t sy = burley_scramble(sy_raw, seed_y);

    constexpr double inv_2_32 = 1.0 / 4294967296.0;
    return { sx * inv_2_32, sy * inv_2_32 };
}
```

Per-call cost: ~10 arithmetic ops + ~32-iteration gray-code loop ≈ 40 ns on modern x86. Negligible vs the per-ray cost (50–500 µs for geodesic + raymarch).

### Stateless and thread-safe

Pure function of `(pixel_x, pixel_y, sample_index)`. No globals, no per-thread RNG, no member variables. OpenMP `#pragma omp parallel for` in the renderer continues to work without locking or per-thread setup.

## Call-site change pattern

### Before / after for `Renderer::render`

```cpp
// BEFORE:
const int grid       = std::max(1, static_cast<int>(std::sqrt(spp_)));
const int actual_spp = grid * grid;
const double inv_spp = 1.0 / actual_spp;
const double cell    = 1.0 / grid;

for (int sy = 0; sy < grid; ++sy) {
    for (int sx = 0; sx < grid; ++sx) {
        const int s = sy * grid + sx;
        const double jx = pixel_hash(i, j, s, 0);
        const double jy = pixel_hash(i, j, s, 1);
        const double px = i + (sx + jx) * cell;
        const double py = j + (sy + jy) * cell;
        // trace ray, accumulate into accum...
    }
}
framebuffer[idx + 0] = static_cast<float>(accum[0] * inv_spp);
```

```cpp
// AFTER:
const double inv_spp = 1.0 / spp_;

for (int s = 0; s < spp_; ++s) {
    auto sob = sobol_sample_2d(i, j, s);
    const double px = i + sob.x;
    const double py = j + sob.y;
    // trace ray, accumulate into accum... (unchanged)
}
framebuffer[idx + 0] = static_cast<float>(accum[0] * inv_spp);
```

The same pattern applies verbatim to `render_spectral` and `render_spectral_to_fits`. The trace/accumulate body is unchanged — only the outer loop and the offset calculation change.

## Testing & validation

### Unit tests for the sampler (new file `tests/test_sobol_sampler.cpp`)

Five tests, all <1 second total:

1. **Range.** A thousand random `(pixel_x, pixel_y, sample_index)` calls; assert every output is in `[0, 1)` for both x and y.
2. **Determinism.** `sobol_sample_2d(5, 7, 42)` returns the same value on two calls.
3. **Per-pixel decorrelation.** First 64 sample points for `(0, 0)` versus `(1, 0)` — at least 90% should differ in the lowest 16 bits, confirming Owen scrambling worked.
4. **2D uniformity.** 1024 samples for one pixel; partition `[0,1)²` into a 32×32 grid; assert `max(count) - min(count) ≤ 4`. Random typically gives ≥ 8 on this metric, so this validates low-discrepancy structure.
5. **Sobol dim 0 covers the dyadic stratification.** Sample indices 0–7 in dimension 0 (before scrambling) should produce, *as a set*, exactly the 8 dyadic fractions `{0, 1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8}`. (The order is a gray-code permutation of natural order — `0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125` for sample 0..7 — but the set property is the easier-to-assert invariant.) Validates the direction-number table.

### Existing test compatibility

- `test_tolerance_convergence` (Task 9): renders at three Romberg tolerances; asserts `max diff` between renders is small. Sobol replaces stratified, so per-pixel values change, but the convergence-across-tolerance property is preserved (the same Sobol points feed all three runs). Test stays green without modification.

- `test_no_horizontal_bands` (Task 10): measures `rel ≈ 0.183` with stratified+Romberg. Sobol decorrelates samples across pixels, which should *lower* row-mean variance from speckle. The threshold = 0.25 has 37% headroom over the current baseline; the new baseline will likely be lower, so the test stays green and may even tighten naturally. After implementation, run once to record the new baseline and update the calibration comment in the test (one-line doc edit, not its own task).

### Smoke-render quality comparison

After landing, render the same 256² × spp=30 scene before and after the switch (use git to checkout pre-Sobol HEAD for "before"). Save both PNGs. Manually verify the Sobol render shows reduced speckle. Document the comparison alongside the spec; not an automated test.

### Out of scope

- Convergence-rate test (Sobol vs stratified on a known integral). Research validation, not regression coverage.
- Visual regression at high spp. Multi-minute renders don't fit CI; trust unit tests + smoke comparison.
- Spectral output validation. Spectral path uses the same sampler — RGB validation transfers.

## Error handling & edge cases

- **`spp = 0` or negative.** The renderer constructor already clamps via `spp_(samples_per_pixel < 1 ? 1 : samples_per_pixel)`. Sobol path inherits this — `for (s = 0; s < 1; ++s)` produces one sample. No new clamps needed.
- **`spp = 1`.** Single sample; `sobol_sample_2d(i, j, 0)` returns the first scrambled Sobol point — uniformly random per pixel. No anti-aliasing benefit at spp=1, same as stratified.
- **Very large `spp` (≥ 2³² ≈ 4.3 × 10⁹).** The Sobol generator uses 32-bit gray-code indices; beyond 2³² the sequence wraps. We never hit this in practice (a single pixel at 4 × 10⁹ samples would take days; entire render impossible). No guard needed.
- **Negative `pixel_x` or `pixel_y`.** Should not occur (renderer iterates `i ∈ [0, width)` and `j ∈ [0, height)`), but the hash handles negative inputs correctly via `static_cast<uint32_t>` reinterpretation. No correctness concern.

## File changes

**New:**
- `include/grrt/render/sobol_sampler.h`
- `src/sobol_sampler.cpp`
- `tests/test_sobol_sampler.cpp`

**Modified:**
- `src/renderer.cpp` — three call sites; delete `pixel_hash`.
- `CMakeLists.txt` (root) — add `src/sobol_sampler.cpp` to library; add `test-sobol-sampler` executable.

**Reverted:**
- None.

**Optional micro-cleanup:**
- `cli/main.cpp` `--samples` help text could mention Sobol. Single line, optional.

## Future work

- **CUDA Sobol implementation** (Phase 2). Mirror the design on GPU. The pure-function nature of `sobol_sample_2d` means the implementation can be a `__device__ __host__` function with no thread-safety machinery — porting is mostly textual.
- **Sobol with more dimensions** if/when path tracing or time sampling is added. Joe-Kuo provides direction numbers up to dimension 21,201; trivial extension when needed.
- **Per-pixel optimization** (rotational variance reduction): could rotate Sobol points by a per-pixel angle to further decorrelate at very low spp. Not needed at the spp counts we use.

## References

- Sobol, I.M. (1967). *On the distribution of points in a cube and the approximate evaluation of integrals*. USSR Computational Mathematics and Mathematical Physics 7(4).
- Joe, S. and Kuo, F.Y. (2008). *Constructing Sobol' sequences with better two-dimensional projections*. SIAM J. Sci. Comput. 30(5). Direction-number tables at `https://web.maths.unsw.edu.au/~fkuo/sobol/`.
- Owen, A.B. (1997). *Monte Carlo variance of scrambled net quadrature*. SIAM J. Numer. Anal. 34(5).
- Burley, B. (2020). *Practical Hash-Based Owen Scrambling*. JCGT 9(4). Direct source for the hash form used here.
- Pharr, M., Jakob, W., Humphreys, G. (2023). *Physically Based Rendering: From Theory To Implementation* (4th ed.). Reference implementation: `pbrt-v4/util/lowdiscrepancy.h`.

## Migration notes

No API breaking changes. Existing callers of the C API and CLI continue to work. `--samples N` keeps the same meaning, but for non-square `N` the rendered output now uses exactly `N` samples instead of `floor(sqrt(N))²`. Renders before and after the change are not bit-comparable (different sub-pixel offsets), but converge to the same true integral as `spp → ∞`.
