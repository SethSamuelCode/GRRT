# Spectral Output Design

**Date**: 2026-04-05
**Status**: Approved

## Motivation

The current renderer computes blackbody emission and maps it to human-visible RGB via CIE color matching functions (380-780nm). This discards the vast majority of physically interesting emission — inner disk X-rays, outer disk infrared, and redshift-shifted radiation outside the visible band.

The goal is to output a full spectral data cube (width × height × N frequency bins) in FITS format, enabling post-processing with standard astronomy tools (DS9, astropy) to produce:
- Single-frequency monochrome images (like EHT 230 GHz observations)
- False-color composites (like NASA Chandra/Hubble/JWST images)
- Blender-compatible linear RGB (by integrating spectrum × CIE in post-processing)
- Tone-mapped PNG (from any of the above)

## Scope

- Spectral accumulation in the **volumetric disk ray-march only** (CPU backend)
- FITS output with self-describing metadata
- CLI frequency specification (explicit list or log-spaced range)
- C API extensions for spectral rendering
- Legacy RGB mode preserved as fallback when no frequency bins are specified

**Out of scope**: CUDA backend changes, thin disk spectral mode, post-processing scripts.

## Design

### 1. Frequency Bin Specification

Users provide observer-frame frequencies at runtime. Two CLI modes:

- `--frequencies 1e9,1e11,1e13,1e14,1e15` — explicit list of frequencies in Hz
- `--freq-range 1e9 1e18 100` — log-spaced range: min Hz, max Hz, number of bins

Log spacing is the default for ranges because the EM spectrum spans many orders of magnitude and physical quantities (Planck function, opacity) vary smoothly on a log-frequency scale.

In `GRRTParams` (types.h):
```c
int num_frequency_bins;          // 0 = legacy RGB mode
const double* frequency_bins_hz; // Array of observer-frame frequencies in Hz
```

When `num_frequency_bins == 0`, all rendering behavior is identical to the current codebase.

### 2. Ray-March Spectral Accumulation

The core change is in `GeodesicTracer::raymarch_volumetric` (geodesic_tracer.cpp).

**Current state** (lines 142-143):
```cpp
constexpr double nu_obs[3] = {c_cgs / 450e-7, c_cgs / 550e-7, c_cgs / 650e-7};
double J[3] = {0.0, 0.0, 0.0};
double tau_acc[3] = {0.0, 0.0, 0.0};
```

**Spectral variant**: A new method `raymarch_volumetric_spectral` takes the frequency bin array and accumulates into dynamically-sized vectors:
```cpp
void raymarch_volumetric_spectral(GeodesicState& state,
                                  const std::vector<double>& nu_obs,
                                  std::vector<double>& spectral_intensity) const;
```

The inner loop changes from `for (int ch = 0; ch < 3; ch++)` to `for (int ch = 0; ch < num_bins; ch++)`. The physics is identical — Planck evaluation, opacity lookup, and radiative transfer per bin per ray-march step.

The existing 3-channel `raymarch_volumetric` method is preserved unchanged as the RGB fallback.

**Performance**: Geodesic integration dominates render time and is unchanged. The per-bin Planck evaluation (`B(ν,T) = 2hν³/c² × 1/(exp(hν/kT) - 1)`) is a few FP ops per bin. Going from 3 to 200 bins adds roughly 5-15% to total render time. Memory scales as `width × height × num_bins × 8 bytes` (e.g., 1024 × 1024 × 100 bins ≈ 800 MB).

### 3. Renderer Spectral Output

A new method on `Renderer`:
```cpp
void render_spectral(double* spectral_buffer, int width, int height,
                     const std::vector<double>& frequency_bins,
                     ProgressCallback progress_cb = nullptr) const;
```

Key differences from the RGB `render` method:
- **`double*` output** — spectral intensities span huge dynamic range (X-ray vs radio), double precision matters
- **No tone mapping** — raw physical intensities in CGS units (erg/s/cm²/Hz/sr)
- **No `SpectrumLUT` needed** — that's the CIE → RGB mapping, irrelevant here
- **Jitter/SPP identical** — stratified sampling averages across all bins the same way it averages RGB

The `TraceResult` struct is extended with a `std::vector<double>` for per-bin spectral intensities, used only in spectral mode.

Buffer layout: `spectral_buffer[(j * width + i) * num_bins + k]` = intensity at pixel (i,j), frequency bin k.

### 4. FITS Writer

A minimal, dependency-free FITS writer. FITS is a straightforward format: ASCII header in 2880-byte blocks followed by raw big-endian binary data.

New files: `include/grrt/render/fits_writer.h`, `src/fits_writer.cpp`.

The writer is a standalone utility function:
```cpp
struct FITSMetadata {
    double spin;
    double mass;
    double observer_r;
    double observer_theta;
    double fov;
    int samples_per_pixel;
};

void write_fits(const std::string& path,
                const double* data,
                int width, int height, int num_bins,
                const std::vector<double>& frequency_bins_hz,
                const FITSMetadata& metadata);
```

**Primary HDU contents**:
- `BITPIX = -64` (64-bit IEEE double)
- `NAXIS = 3`, `NAXIS1 = width`, `NAXIS2 = height`, `NAXIS3 = num_bins`
- Frequency axis described via WCS keywords: `CRPIX3 = 1`, `CRVAL3 = freq_bins[0]`, `CDELT3 = bin_spacing`, `CTYPE3 = 'FREQ'`, `CUNIT3 = 'Hz'`. For non-uniform spacing (explicit `--frequencies`), the full frequency array is stored in a binary table extension (`FREQ_TABLE`).
- `BUNIT = 'erg/s/cm2/Hz/sr'` — intensity unit
- Render parameters as keywords: `SPIN`, `MASS`, `OBS_R`, `OBS_TH`, `FOV`, `SPP`, etc.

DS9 and astropy read this natively. The axis ordering follows FITS image cube convention.

**CLI integration**: The `--output` flag detects `.fits` extension to trigger spectral mode, or an explicit `--output-fits` flag can be used. Spectral mode requires frequency bins to be specified (error otherwise).

### 5. C API Extensions

Two new functions alongside the existing API:
```c
// Set frequency bins for spectral rendering
void grrt_set_frequency_bins(GRRTContext* ctx,
                             const double* frequencies_hz,
                             int num_bins);

// Render spectral cube (caller allocates width*height*num_bins doubles)
int grrt_render_spectral(GRRTContext* ctx, double* spectral_buffer,
                         int width, int height);
```

No changes to existing functions. No exceptions cross the DLL boundary, consistent with current API design.

## File Changes

| File | Change |
|------|--------|
| `include/grrt/types.h` | Add `num_frequency_bins`, `frequency_bins_hz` to `GRRTParams` |
| `include/grrt/geodesic/geodesic_tracer.h` | Add `raymarch_volumetric_spectral` method, extend `TraceResult` |
| `src/geodesic_tracer.cpp` | Implement spectral ray-march variant |
| `include/grrt/render/renderer.h` | Add `render_spectral` method |
| `src/renderer.cpp` | Implement spectral render loop |
| `include/grrt/render/fits_writer.h` | New — FITS writer interface |
| `src/fits_writer.cpp` | New — FITS writer implementation |
| `include/grrt/api.h` | Add `grrt_set_frequency_bins`, `grrt_render_spectral` |
| `src/api.cpp` | Implement new C API functions |
| `cli/main.cpp` | Add `--frequencies`, `--freq-range`, `.fits` output support |
| `CMakeLists.txt` | Add new source files |
