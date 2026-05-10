# Low-Discrepancy Sub-Pixel Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stratified-jitter sub-pixel sampler in `src/renderer.cpp` with a 2D Owen-scrambled Sobol sequence, applied to all three render paths (RGB, spectral, spectral-to-FITS).

**Architecture:** A new pure-function helper `sobol_sample_2d(pixel_x, pixel_y, sample_index) -> SobolSample{x, y}` in `src/sobol_sampler.cpp`. Direction numbers for Sobol dimensions 0 and 1 are computed at compile time via simple recurrences (no precomputed table needed). Owen scrambling uses Burley's hash form. The renderer's per-pixel sub-pixel loop becomes a flat `for s in [0, spp_)` calling the sampler.

**Tech Stack:** C++23, CMake/MSBuild, OpenMP. No new external dependencies.

**Spec:** `docs/superpowers/specs/2026-05-02-low-discrepancy-sub-pixel-sampling-design.md`

**Notes for the implementer:**
- This branch is `fix/volumetric-ring`. Last commit so far is Task 10 of the cliff-aware-raymarch work (`65b2aad`).
- Build with `cmake --build build --config Release` on Windows (Visual Studio 2022 generator).
- **Do NOT run `git commit`.** The user composes commits themselves. After each task's verification, surface the commit message text and pause until the user confirms it landed.
- Existing test convention: `int failures = 0;` accumulator at file scope; `EXPECT_NEAR` macro increments on failure and returns; `main()` returns `failures > 0 ? 1 : 0`. See `tests/test_volumetric.cpp` and `tests/test_romberg_step.cpp` for reference patterns.

---

## File structure

**New:**
- `include/grrt/render/sobol_sampler.h` — public API: `SobolSample` struct + `sobol_sample_2d` declaration.
- `src/sobol_sampler.cpp` — direction-number initialization, `sobol_point_1d`, `scramble_seed`, `burley_scramble`, `sobol_sample_2d`.
- `tests/test_sobol_sampler.cpp` — five unit tests verifying sampler properties.

**Modified:**
- `src/renderer.cpp` — three call sites (`render`, `render_spectral`, `render_spectral_to_fits`) switch from stratified loops to flat Sobol loops. `pixel_hash` static function deleted.
- `tests/test_volumetric.cpp` — calibration comment in `test_no_horizontal_bands` updated with new measured `rel` value.
- `CMakeLists.txt` (root) — add `src/sobol_sampler.cpp` to the `grrt` library; add `test-sobol-sampler` executable target.

---

## Task 1: Skeleton header for `sobol_sampler`

**Files:**
- Create: `include/grrt/render/sobol_sampler.h`

- [ ] **Step 1: Write the header skeleton**

```cpp
#ifndef GRRT_RENDER_SOBOL_SAMPLER_H
#define GRRT_RENDER_SOBOL_SAMPLER_H

#include "grrt_export.h"

namespace grrt {

/// One 2D point of the Owen-scrambled Sobol sequence. Both components in [0, 1).
struct SobolSample {
    double x;
    double y;
};

/// Generate the i-th 2D sub-pixel sample for the given pixel using Owen-scrambled
/// Sobol. Pure function — same arguments always return the same point.
///
/// The per-pixel scramble seed is derived from (pixel_x, pixel_y), so neighboring
/// pixels see decorrelated point sequences. Renders are deterministic across
/// runs and thread counts.
///
/// @param pixel_x integer pixel x-coordinate (column index)
/// @param pixel_y integer pixel y-coordinate (row index)
/// @param sample_index 0-based index of the sample within the pixel (0 ≤ s < spp)
/// @return sub-pixel offset in [0, 1)²
GRRT_EXPORT SobolSample sobol_sample_2d(int pixel_x, int pixel_y, int sample_index);

} // namespace grrt

#endif
```

- [ ] **Step 2: Verify the header compiles standalone**

Run:
```bash
cmake --build build --config Release --target grrt
```
Expected: the existing `grrt` library still builds clean. The new header isn't included by anything yet, so it's a no-op for the library — but if you typo'd it, the file would still parse OK on its own. (Optional: include it from a throwaway test to confirm.)

- [ ] **Step 3: Hand off commit**

Commit message text:
```
feat(sampler): add Sobol sampler header skeleton
```

Files: `include/grrt/render/sobol_sampler.h`. Wait for user confirmation before proceeding.

---

## Task 2: Test scaffold + first failing test (range)

**Files:**
- Create: `tests/test_sobol_sampler.cpp`
- Modify: `CMakeLists.txt` (root)

- [ ] **Step 1: Add test target to root `CMakeLists.txt`**

Find the block of `add_executable(test-volumetric ...)` lines near the end (existing tests). Add right after the `test-romberg-step` stanza:

```cmake
add_executable(test-sobol-sampler tests/test_sobol_sampler.cpp)
target_link_libraries(test-sobol-sampler PRIVATE grrt)
```

- [ ] **Step 2: Write the test file with one failing test (range)**

```cpp
// tests/test_sobol_sampler.cpp
//
// Unit tests for sobol_sample_2d. These tests are pure — no rendering, no
// disk construction, no LUTs. They exercise the sampler's mathematical
// properties (range, determinism, decorrelation, uniformity, dyadic set).

#include "grrt/render/sobol_sampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

using namespace grrt;

int failures = 0;

#define EXPECT_TRUE(cond, msg)                                                \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("FAIL %s:%d: %s — %s\n",                              \
                        __FILE__, __LINE__, #cond, msg);                      \
            failures++;                                                       \
        }                                                                     \
    } while (0)

// Test 1: every produced point lies in [0, 1).
static void test_range() {
    constexpr int N = 1000;
    for (int i = 0; i < N; ++i) {
        // pseudo-random pixel coords for variety
        const int px = (i * 73856093u) & 0xFFFF;
        const int py = (i * 19349663u) & 0xFFFF;
        for (int s = 0; s < 8; ++s) {
            SobolSample p = sobol_sample_2d(px, py, s);
            EXPECT_TRUE(p.x >= 0.0 && p.x < 1.0, "x out of range");
            EXPECT_TRUE(p.y >= 0.0 && p.y < 1.0, "y out of range");
        }
    }
}

int main() {
    std::printf("Running test_sobol_sampler...\n");
    test_range();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
```

- [ ] **Step 3: Refresh the build and try to build the test**

Run:
```bash
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release --target test-sobol-sampler
```

Expected: **link error** — `unresolved external symbol grrt::sobol_sample_2d`. The function is declared but not yet defined. This is the intentional TDD red state.

- [ ] **Step 4: Hand off commit**

Commit message text:
```
test(sampler): scaffold Sobol sampler tests + range check (failing)
```

Files: `tests/test_sobol_sampler.cpp`, `CMakeLists.txt`. Wait for user confirmation.

---

## Task 3: Implement `sobol_sample_2d` end-to-end

This task adds `src/sobol_sampler.cpp` with all required pieces in one go (direction numbers, Sobol point generation, Burley scramble, scramble seed, public entry). The whole file is small and self-contained — it doesn't benefit from being split.

**Files:**
- Create: `src/sobol_sampler.cpp`
- Modify: `CMakeLists.txt` (root) — add the .cpp to the `grrt` library.

- [ ] **Step 1: Write the implementation**

```cpp
// src/sobol_sampler.cpp
//
// Owen-scrambled Sobol 2D sub-pixel sampler.
//
// Direction numbers:
//   - Dimension 0: trivial Sobol (van der Corput in base 2),
//                  V[0][k] = 1 << (31 - k).
//   - Dimension 1: Sobol with primitive polynomial x + 1.
//                  V[1][0] = 1 << 31; V[1][k] = V[1][k-1] ^ (V[1][k-1] >> 1).
//                  Equivalently: V[1][k] = m_{k+1} << (31 - k), where
//                  m_1 = 1, m_k = (2 * m_{k-1}) ^ m_{k-1}.
//
// Owen scramble: Burley 2020, "Practical Hash-Based Owen Scrambling," JCGT 9(4).
// Same form used by pbrt-v4's nested_uniform_scramble.
//
// References:
//   - Joe & Kuo, "Constructing Sobol' sequences with better two-dimensional
//     projections," SIAM J. Sci. Comput. 30(5), 2008.
//   - Burley, "Practical Hash-Based Owen Scrambling," JCGT 9(4), 2020.

#include "grrt/render/sobol_sampler.h"

#include <array>
#include <cstdint>

namespace grrt {

namespace {

// Direction numbers for the first two Sobol dimensions, computed at compile time.
struct DirectionTables {
    std::array<uint32_t, 32> v0{};
    std::array<uint32_t, 32> v1{};
};

constexpr DirectionTables make_direction_tables() {
    DirectionTables d{};
    for (int k = 0; k < 32; ++k) {
        d.v0[k] = static_cast<uint32_t>(1u) << (31 - k);
    }
    d.v1[0] = static_cast<uint32_t>(1u) << 31;
    for (int k = 1; k < 32; ++k) {
        d.v1[k] = d.v1[k - 1] ^ (d.v1[k - 1] >> 1);
    }
    return d;
}

constexpr DirectionTables kDirs = make_direction_tables();

// Generate the i-th 32-bit Sobol point in dimension `dim` (0 or 1) by
// gray-code traversal of the direction-number table.
inline uint32_t sobol_point_1d(int i, int dim) {
    const uint32_t* V = (dim == 0) ? kDirs.v0.data() : kDirs.v1.data();
    uint32_t result = 0;
    int idx = i ^ (i >> 1);  // gray code
    int k = 0;
    while (idx) {
        if (idx & 1) result ^= V[k];
        idx >>= 1;
        ++k;
    }
    return result;
}

// Burley 2020, Listing 3. Hash-based Owen scramble. Permutes a 32-bit
// integer using a per-pixel seed, breaking correlations between adjacent
// pixels' sample sequences.
inline uint32_t burley_scramble(uint32_t x, uint32_t seed) {
    x = x ^ (x * 0x3d20adeau);
    x += seed;
    x *= (seed >> 16) | 1u;
    x = x ^ (x * 0x05526c56u);
    x = x ^ (x * 0x53a22864u);
    return x;
}

// Per-pixel scramble seed. Same hash style as the existing pixel_hash in
// renderer.cpp (MurmurHash3 finalization). Different `dim` arguments give
// different seeds so x and y dimensions are independently scrambled.
inline uint32_t scramble_seed(int pixel_x, int pixel_y, int dim) {
    uint32_t h = static_cast<uint32_t>(pixel_x) * 73856093u
               ^ static_cast<uint32_t>(pixel_y) * 19349663u
               ^ static_cast<uint32_t>(dim)     * 83492791u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

} // anonymous namespace

SobolSample sobol_sample_2d(int pixel_x, int pixel_y, int sample_index) {
    const uint32_t seed_x = scramble_seed(pixel_x, pixel_y, 0);
    const uint32_t seed_y = scramble_seed(pixel_x, pixel_y, 1);

    const uint32_t sx_raw = sobol_point_1d(sample_index, 0);
    const uint32_t sy_raw = sobol_point_1d(sample_index, 1);

    const uint32_t sx = burley_scramble(sx_raw, seed_x);
    const uint32_t sy = burley_scramble(sy_raw, seed_y);

    constexpr double inv_2_32 = 1.0 / 4294967296.0;  // 2^-32
    return { static_cast<double>(sx) * inv_2_32,
             static_cast<double>(sy) * inv_2_32 };
}

} // namespace grrt
```

- [ ] **Step 2: Add the .cpp to the library target**

In root `CMakeLists.txt`, find `add_library(grrt ...)` (look for the list of `src/*.cpp` files). Add `src/sobol_sampler.cpp` to the list.

- [ ] **Step 3: Build everything**

Run:
```bash
cmake --build build --config Release
```
Expected: builds clean.

- [ ] **Step 4: Run the range test**

Run:
```bash
build/Release/test-sobol-sampler.exe
```
Expected output:
```
Running test_sobol_sampler...

=== 0 failures ===
```
Exit code 0.

- [ ] **Step 5: Hand off commit**

Commit message text:
```
feat(sampler): implement Owen-scrambled Sobol 2D sub-pixel sampler

Direction numbers for dims 0 and 1 are computed at compile time via simple
recurrences (no embedded table). Burley hash-based Owen scramble per the
2020 paper. Per-pixel seed derived deterministically from (pixel_x, pixel_y)
so neighboring pixels see decorrelated sequences but the same render
produces identical output across runs.
```

Files: `src/sobol_sampler.cpp`, `CMakeLists.txt`. Wait for user confirmation.

---

## Task 4: Add the four remaining unit tests

The implementation from Task 3 should make all of these tests pass without further code changes — they validate properties already present in the code.

**Files:**
- Modify: `tests/test_sobol_sampler.cpp`

- [ ] **Step 1: Add Test 2 — determinism**

Add this function before `int main()`:

```cpp
// Test 2: same arguments must always return the same point.
static void test_determinism() {
    const SobolSample a = sobol_sample_2d(5, 7, 42);
    const SobolSample b = sobol_sample_2d(5, 7, 42);
    EXPECT_TRUE(a.x == b.x, "x not deterministic");
    EXPECT_TRUE(a.y == b.y, "y not deterministic");
}
```

- [ ] **Step 2: Add Test 3 — per-pixel decorrelation**

Add:

```cpp
// Test 3: adjacent pixels must see decorrelated sample sequences.
// At least 90% of the first 64 sample points should differ in the lowest
// 16 bits between pixel (0, 0) and pixel (1, 0). This confirms Owen
// scrambling produces distinct permutations per pixel.
static void test_decorrelation() {
    int agree_low_bits = 0;
    constexpr int N = 64;
    for (int s = 0; s < N; ++s) {
        const SobolSample p00 = sobol_sample_2d(0, 0, s);
        const SobolSample p10 = sobol_sample_2d(1, 0, s);
        // Compare the low 16 bits of each component (after re-scaling).
        const uint32_t x00 = static_cast<uint32_t>(p00.x * 4294967296.0) & 0xFFFFu;
        const uint32_t x10 = static_cast<uint32_t>(p10.x * 4294967296.0) & 0xFFFFu;
        if (x00 == x10) agree_low_bits++;
    }
    EXPECT_TRUE(agree_low_bits <= N / 10,
                "too many adjacent-pixel samples agree in low bits");
}
```

- [ ] **Step 3: Add Test 4 — 2D uniformity**

Add:

```cpp
// Test 4: 2D uniformity. 1024 samples for one pixel divided into a 32x32
// grid should distribute evenly. Low-discrepancy guarantee: max(count) -
// min(count) <= 4 over the 1024 cells. (Random sampling typically gives 8+.)
static void test_uniformity_2d() {
    constexpr int N = 1024;
    constexpr int G = 32;
    std::array<int, G * G> bucket{};
    for (int s = 0; s < N; ++s) {
        const SobolSample p = sobol_sample_2d(0, 0, s);
        const int gx = std::min(G - 1, static_cast<int>(p.x * G));
        const int gy = std::min(G - 1, static_cast<int>(p.y * G));
        bucket[gy * G + gx]++;
    }
    int min_count = N;
    int max_count = 0;
    for (int b : bucket) {
        if (b < min_count) min_count = b;
        if (b > max_count) max_count = b;
    }
    std::printf("  uniformity: min=%d max=%d (each cell expects 1)\n",
                min_count, max_count);
    EXPECT_TRUE(max_count - min_count <= 4,
                "uniformity worse than expected for low-discrepancy");
}
```

- [ ] **Step 4: Add Test 5 — Sobol dim 0 dyadic-set property**

Add (note: this tests the *unscrambled* Sobol dim 0; it accesses an internal helper, so we can't call it directly from outside the anonymous namespace. Instead, test it indirectly: with a fixed pixel and seed=0 input to scramble, the scramble is the identity-like, but we can't easily fix the seed. Better test: confirm that the **unique values** generated for dim 0 across i=0..7 form the dyadic set, regardless of order):

```cpp
// Test 5: Sobol dim 0 covers the dyadic stratification of [0, 1).
//
// Indirect check: for the unscrambled sampler, the first 8 samples in
// dimension 0 should produce, as a SET, exactly the 8 dyadic fractions
// {0, 1/8, 2/8, ..., 7/8}. We can't disable scrambling easily, so we use
// a probe: pick a pixel where scramble_seed dim 0 happens to be
// well-behaved, OR (simpler) just verify that the 8 .x values from
// sobol_sample_2d for sample 0..7 are distinct and span [0, 1).
// The dyadic-set property survives Burley scramble because Burley is a
// bit-permutation that preserves the lower 3 bits' distinctness.
static void test_dim0_distinctness() {
    constexpr int N = 8;
    std::set<uint32_t> seen;
    for (int s = 0; s < N; ++s) {
        const SobolSample p = sobol_sample_2d(0, 0, s);
        // Quantize x to top 3 bits — these encode which dyadic eighth.
        const uint32_t bucket = static_cast<uint32_t>(p.x * 8.0);
        seen.insert(bucket);
    }
    EXPECT_TRUE(seen.size() == N,
                "first 8 samples don't span all 8 dyadic eighths");
}
```

- [ ] **Step 5: Wire all four into `main()`**

Update `main()`:

```cpp
int main() {
    std::printf("Running test_sobol_sampler...\n");
    test_range();
    test_determinism();
    test_decorrelation();
    test_uniformity_2d();
    test_dim0_distinctness();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
```

- [ ] **Step 6: Build and run**

Run:
```bash
cmake --build build --config Release --target test-sobol-sampler
build/Release/test-sobol-sampler.exe
```
Expected output:
```
Running test_sobol_sampler...
  uniformity: min=1 max=1 (each cell expects 1)

=== 0 failures ===
```
(For 1024 samples in 1024 cells, each cell gets exactly 1 if Sobol is well-distributed. Random samples give a Poisson distribution → some 0s, some 3s, max-min ≥ 4.)

- [ ] **Step 7: Hand off commit**

Commit message text:
```
test(sampler): determinism, decorrelation, uniformity, dyadic-set checks
```

Files: `tests/test_sobol_sampler.cpp`. Wait for user confirmation.

---

## Task 5: Wire Sobol into all three render paths in `src/renderer.cpp`

**Files:**
- Modify: `src/renderer.cpp`

There are three near-identical changes — one per render function. The pattern: replace the `for (sy)/(sx)` stratified-grid loop with a flat `for (s)` loop calling `sobol_sample_2d`, and remove the `grid`/`actual_spp`/`cell` variables.

- [ ] **Step 1: Add the sobol_sampler include**

At the top of `src/renderer.cpp`, near the other `#include "grrt/..."` lines, add:

```cpp
#include "grrt/render/sobol_sampler.h"
```

- [ ] **Step 2: Update `Renderer::render`**

Locate the body of `void Renderer::render(...)` (starts around line 30). Find the block:

```cpp
const int grid = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(spp_))));
const int actual_spp = grid * grid;
const double inv_spp = 1.0 / actual_spp;
const double cell = 1.0 / grid;
```

Replace those four lines with:

```cpp
const double inv_spp = 1.0 / spp_;
```

Then find the per-pixel sample loop:

```cpp
Vec3 accum;
for (int sy = 0; sy < grid; ++sy) {
    for (int sx = 0; sx < grid; ++sx) {
        const int s = sy * grid + sx;
        const double jx = pixel_hash(i, j, s, 0);
        const double jy = pixel_hash(i, j, s, 1);
        const double px = i + (sx + jx) * cell;
        const double py = j + (sy + jy) * cell;

        GeodesicState state = camera_.ray_for_pixel(px, py);
        // ... (rest of inner body)
    }
}
```

Replace with:

```cpp
Vec3 accum;
for (int s = 0; s < spp_; ++s) {
    const SobolSample sob = sobol_sample_2d(i, j, s);
    const double px = i + sob.x;
    const double py = j + sob.y;

    GeodesicState state = camera_.ray_for_pixel(px, py);
    // ... (rest of inner body — UNCHANGED)
}
```

The "rest of inner body" — the `tracer_.trace`, `accum + color`, sphere-sample-on-escape — is **not modified**. Only the loop wrapper and offset calculation change.

- [ ] **Step 3: Update `Renderer::render_spectral`**

Find `void Renderer::render_spectral(...)` (starts around line 84). Apply the same pattern:

Replace:
```cpp
const int grid = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(spp_))));
const int actual_spp = grid * grid;
const double inv_spp = 1.0 / actual_spp;
const double cell = 1.0 / grid;
```
with:
```cpp
const double inv_spp = 1.0 / spp_;
```

Replace the inner double-loop:
```cpp
for (int sy = 0; sy < grid; ++sy) {
    for (int sx = 0; sx < grid; ++sx) {
        const int s = sy * grid + sx;
        const double jx = pixel_hash(i, j, s, 0);
        const double jy = pixel_hash(i, j, s, 1);
        const double px = i + (sx + jx) * cell;
        const double py = j + (sy + jy) * cell;
        // ... (rest of inner body)
    }
}
```
with:
```cpp
for (int s = 0; s < spp_; ++s) {
    const SobolSample sob = sobol_sample_2d(i, j, s);
    const double px = i + sob.x;
    const double py = j + sob.y;
    // ... (rest of inner body — UNCHANGED)
}
```

- [ ] **Step 4: Update `Renderer::render_spectral_to_fits`**

Find `int Renderer::render_spectral_to_fits(...)` (starts around line 140). Apply the same pattern. The function may be slightly different shape (it's a streaming variant), but the `grid`/`actual_spp`/`cell` block and the `for (sy)/(sx)` loop are present — replace identically.

- [ ] **Step 5: Build**

Run:
```bash
cmake --build build --config Release
```
Expected: clean build. The `pixel_hash` function is now unused (warned by some compilers, but not an error). We'll delete it in Task 6.

- [ ] **Step 6: Smoke render**

Run:
```bash
build/Release/grrt-cli.exe --disk-volumetric --samples 30 --width 256 --height 256 --output sobol_smoke --force
```

Open `sobol_smoke.png`. Compare visually to `regression_repro.png` (the pre-Sobol baseline already in the repo). The Sobol render should:
- Look like the same scene (disk position, lensing arcs unchanged).
- Show **less speckled / starfield** appearance — fewer hot pixels, smoother arcs.
- Have bit-different pixel values (sub-pixel offsets changed), but be qualitatively cleaner.

If bands appear or the disk vanishes, STOP and report — something is wrong with the wiring.

- [ ] **Step 7: Run the full test suite to confirm no regressions**

Run:
```bash
cmake --build build --config Release --target test-volumetric
build/Release/test-volumetric.exe 2>&1 | tail -10
```

Expected: same `=== 1 failures ===` as before (the pre-existing tau-midplane known issue). The new `test_no_horizontal_bands` should still pass (Sobol's `rel` is expected to be ≤ 0.183, well under threshold 0.25).

Note the new `rel` value printed by `test_no_horizontal_bands`. You'll record it in Task 6.

- [ ] **Step 8: Hand off commit**

Commit message text:
```
feat(renderer): switch all three render paths to Sobol+Owen sub-pixel sampler

render(), render_spectral(), and render_spectral_to_fits() now use
sobol_sample_2d for sub-pixel jitter. Removes the floor(sqrt(spp))^2
sample-count rounding — --samples N now produces exactly N samples per
pixel.

The pixel_hash helper at the top of the file is now dead code; it's
deleted in the follow-up cleanup commit.
```

Files: `src/renderer.cpp`. Wait for user confirmation.

---

## Task 6: Delete `pixel_hash`, recalibrate banding test, smoke comparison archived

**Files:**
- Modify: `src/renderer.cpp`
- Modify: `tests/test_volumetric.cpp` (calibration comment only)

- [ ] **Step 1: Delete the `pixel_hash` function**

In `src/renderer.cpp`, find at the top of the file (around lines 13-20):

```cpp
// Simple hash for deterministic per-pixel jitter (no external RNG state needed)
static double pixel_hash(int i, int j, int s, int channel) {
    uint32_t h = static_cast<uint32_t>(i * 73856093u ^ j * 19349663u ^ s * 83492791u ^ channel * 45678917u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return (h & 0xFFFFu) / 65536.0;
}
```

Delete the entire function (the comment line plus the body). It's dead code.

- [ ] **Step 2: Verify the build still succeeds**

Run:
```bash
cmake --build build --config Release
```
Expected: clean build. If `pixel_hash` is referenced anywhere we missed, the linker complains — go fix the missed reference.

- [ ] **Step 3: Re-run the banding regression test and capture the new `rel`**

Run:
```bash
cmake --build build --config Release --target test-volumetric
build/Release/test-volumetric.exe 2>&1 | grep -A 3 "Banding regression"
```

Expected output similar to:
```
=== Banding regression test (256x256 spp=30) ===
[VolumetricDisk] ...
  rows with disk content: 32
  banding metric (avg|drow|/<row>): X.XXX
  PASS
```

Record the value `X.XXX`. It should be ≤ 0.183 (the pre-Sobol Romberg baseline). Likely lower, because Sobol decorrelates samples across pixels → less speckle → less inter-row variance.

- [ ] **Step 4: Update the calibration comment in `test_no_horizontal_bands`**

In `tests/test_volumetric.cpp`, find the `test_no_horizontal_bands` function (around line 618). Find this comment block (around line 700-708):

```cpp
    // Threshold calibrated empirically on 2026-05-01 (fix/volumetric-ring),
    // 256x256 spp=30 disk_volumetric scene, observer_theta=80, fov=90:
    //   - Romberg build (this work): rel = 0.183  (speckle-floor only)
    //   - Buggy build (H_max=H):     rel = 0.281  (real banding signal)
    // THRESHOLD = 0.25 sits between the two (~27% headroom over Romberg,
    // ~11% margin under buggy). The metric is insensitive to within-row
    // turbulence noise by design (it averages row means), so flipping
    // disk_turbulence on/off does not move the calibration values.
```

Replace with:

```cpp
    // Threshold calibrated empirically on 2026-05-01 (fix/volumetric-ring),
    // 256x256 spp=30 disk_volumetric scene, observer_theta=80, fov=90:
    //   - Romberg + Stratified (pre-Sobol): rel = 0.183  (speckle floor)
    //   - Romberg + Sobol+Owen (current):   rel = X.XXX  (lower noise)
    //   - Buggy build (H_max=H):            rel = 0.281  (real banding)
    // THRESHOLD = 0.25 has plenty of headroom over the current baseline
    // and clearly fails on the buggy regime. The metric is insensitive to
    // within-row turbulence noise by design (it averages row means).
```

(Substitute `X.XXX` with the value you recorded in Step 3.)

- [ ] **Step 5: Re-run the test once more to confirm it still passes**

Run:
```bash
cmake --build build --config Release --target test-volumetric
build/Release/test-volumetric.exe 2>&1 | tail -5
```

Expected: `=== 1 failures ===` (just the pre-existing tau test). The banding regression test still passes with the new lower baseline.

- [ ] **Step 6: Final smoke render at higher quality**

Run:
```bash
build/Release/grrt-cli.exe --disk-volumetric --samples 30 --width 1024 --height 1024 --output sobol_final --force
```

This produces `sobol_final.png` at full resolution. Visually confirm:
- Disk renders cleanly (no horizontal bands).
- Speckle pattern is finer / more diffuse than the pre-Sobol render.
- No new artifacts (check for striping, moiré, or block patterns — Owen scrambling should prevent these).

- [ ] **Step 7: Hand off commit**

Commit message text:
```
chore(renderer): delete dead pixel_hash; recalibrate banding-test comment

Sobol+Owen replaced the stratified-jitter sampler in the previous commit;
pixel_hash is no longer called. Update the test_no_horizontal_bands
calibration comment with the new measured `rel` value (lower than the
pre-Sobol baseline) — threshold of 0.25 still fails the buggy H_max=H
regime and is well above the new Sobol baseline.
```

Files: `src/renderer.cpp`, `tests/test_volumetric.cpp`. Wait for user confirmation.

---

## Self-review

After all 6 tasks above are committed:

- [ ] **Step A: Confirm spec coverage**

Open `docs/superpowers/specs/2026-05-02-low-discrepancy-sub-pixel-sampling-design.md` and walk down each section:
- "The `sobol_sample_2d` helper" → covered by Tasks 1, 3.
- "Sampler internals (Sobol point + Burley + scramble seed)" → covered by Task 3.
- "Call-site changes in renderer.cpp (3 functions)" → covered by Task 5.
- "Testing & validation 5a–5d" → covered by Tasks 2, 4 (unit tests); 5 (integration smoke render); 6 (banding-test calibration).
- "File changes" → all listed in Task 5–6.
- "Error handling & edge cases (spp=0/1, large spp, negative pixel coords)" → Task 3 implementation handles via the math (pure functions on uint32_t inputs); no explicit guards needed; behavior matches spec.

- [ ] **Step B: Final visual smoke check**

Compare `sobol_final.png` (Task 6) against the previous final render (`final_smoke.png` already in the repo, or `regression_repro.png`). The Sobol render should look smoother — confirm visually.

- [ ] **Step C: Use the `superpowers:finishing-a-development-branch` skill** to wrap up the branch.
