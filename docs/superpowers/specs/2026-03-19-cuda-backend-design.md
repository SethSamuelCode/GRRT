# Phase 2: CUDA Backend Design

**Date:** 2026-03-19
**Status:** Approved

## Overview

Add a CUDA GPU backend to the general relativistic ray tracer, mirroring the existing CPU architecture with plain structs and enum+switch dispatch instead of virtual functions. The CPU and CUDA backends coexist in a single binary with runtime selection via a `--backend` CLI flag. Correctness (bit-level parity with CPU within floating-point tolerance) is the primary goal; performance optimization comes later.

**Target hardware:** NVIDIA RTX 2080 (Turing, compute capability 7.5), CUDA Toolkit 12.x.

## Architecture: Mirror Approach

Each CPU layer gets a CUDA equivalent in the `cuda/` directory. No shared source between CPU and CUDA — the physics is reimplemented as `__device__` functions using plain structs and `enum class` + `switch` instead of C++ polymorphism. This keeps the CPU code untouched and makes correctness comparison straightforward.

## Directory Structure

```
cuda/
├── cuda_types.h          # Plain structs mirroring CPU types (no virtuals)
├── cuda_math.h           # Vec3/Vec4/Matrix4 as __device__ structs
├── cuda_metric.h         # Schwarzschild & Kerr as device functions + enum MetricType
├── cuda_geodesic.h       # RK4 integrator as device functions
├── cuda_camera.h         # Tetrad + pixel-to-momentum on device
├── cuda_scene.h          # Disk intersection, celestial sphere, ray tracing
├── cuda_color.h          # Spectrum LUT + redshift-to-RGB on device
├── cuda_render.cu        # Main kernel: one thread per pixel
├── cuda_backend.cu       # Host-side orchestration: alloc, launch, copy back
└── cuda_backend.h        # Public header exposing the CUDA backend to C API
```

## Build Integration

- New CMake option: `GRRT_ENABLE_CUDA` (default OFF)
- When ON: `find_package(CUDAToolkit)` locates CUDA Toolkit 12.x
- `cuda/*.cu` files compile via nvcc into the `grrt` library
- All CUDA headers are `.h` files with `__device__` functions — only `.cu` files trigger nvcc
- When OFF: CUDA backend is not compiled; `grrt_cuda_available()` returns 0

## Data Flow & Memory Management

### Host-Side Flow (`cuda_backend.cu`)

1. **Setup:** Allocate device memory for output framebuffer (`float4* d_pixels`, width x height)
2. **Upload:** Copy scene parameters into a single `RenderParams` struct, transfer to device constant memory (`__constant__`)
3. **Launch:** One thread per pixel, 2D grid of thread blocks (16x16 blocks)
4. **Download:** Copy `d_pixels` back to host, convert to output format (PNG/HDR)
5. **Cleanup:** Free device memory

### Memory Layout

| Data | Location | Size | Reason |
|------|----------|------|--------|
| `RenderParams` (scalar params only: metric type, spin, observer, camera, disk config) | `__constant__` memory | ~200 bytes | Small, read-only, broadcast to all threads |
| Accretion disk flux LUT (`flux_lut_`, 500 doubles) | `__constant__` memory | ~4 KB | Read-only, precomputed on host, fits in constant memory |
| Spectrum color LUT (1000 x 3 doubles) | `__constant__` memory | ~24 KB | Read-only lookup, must be flat array (not `std::vector`) |
| Spectrum luminosity LUT (1000 doubles) | `__constant__` memory | ~8 KB | Read-only lookup, must be flat array |
| Output framebuffer | Global memory | width*height*16 bytes | Per-pixel write (`float4` = 16 bytes), copied back to host |
| Geodesic state (position + momentum) | Registers/local | 64 bytes/thread | Per-thread, no sharing needed |
| Cancel flag | Mapped pinned host memory | 4 bytes | Must be accessible by both host and device concurrently |

**Total constant memory usage: ~36 KB** out of 64 KB limit. Safe margin.

**LUT transfer:** The CPU `SpectrumLUT` and `AccretionDisk` use `std::vector` members. The CUDA equivalents must use fixed-size plain arrays (e.g., `double color_lut[1000][3]`, `double flux_lut[500]`). The host copies data from the CPU vectors into these flat arrays before uploading to `__constant__` memory via `cudaMemcpyToSymbol`.

**Kernel access:** `RenderParams` and all LUTs are declared as `__constant__` globals in `cuda_render.cu` and accessed directly by the kernel — they are NOT passed as kernel arguments (kernel argument buffer is limited to ~4 KB).

No shared memory is needed — each pixel's geodesic integration is fully independent (embarrassingly parallel).

### Tile Rendering

Same mechanism as full-frame: launch a smaller grid with pixel offset. Maps directly to the existing `grrt_render_tile` C API call.

## Kernel Design

### Main Kernel

```cpp
// RenderParams and LUTs live in __constant__ memory, accessed directly by the kernel
__constant__ RenderParams d_params;
__constant__ double d_color_lut[1000][3];
__constant__ double d_luminosity_lut[1000];
__constant__ double d_flux_lut[500];

__global__ void render_kernel(float4* output, int width, int height)
```

Each thread:
1. Compute pixel (x, y) from `blockIdx` and `threadIdx`
2. Map pixel to ray direction using camera tetrad
3. Compute initial covariant 4-momentum
4. Integrate geodesic via RK4 with adaptive step doubling
5. At each step: check disk intersection, check horizon, check escape
6. Compute color from hit result (disk emission + redshift, or celestial sphere)
7. Apply tone mapping
8. Write `float4` RGBA to `output[y * width + x]`

### No Virtual Functions — Enum + Switch

```cpp
enum class MetricType { Schwarzschild, Kerr };

__device__ void metric_lower(MetricType type, double spin, const double x[4], double g[4][4]) {
    switch (type) {
        case MetricType::Schwarzschild: /* ... */ break;
        case MetricType::Kerr:          /* ... */ break;
    }
}
```

Same pattern for `metric_upper`, `horizon_radius`, `isco_radius`, and all other metric-dependent functions.

### Double Precision

All GPU computation uses FP64. The RTX 2080 runs FP64 at 1/32 rate of FP32, which is slow but necessary — FP32 accumulates too much error near the event horizon. Correctness is the priority for this phase.

### Adaptive Stepping

Same `dlambda proportional to r^2` rule as CPU. Metric derivatives via central finite differences on the inverse metric, identical to CPU implementation.

### Cancellation & Progress

- Cancel flag uses **mapped pinned host memory** (`cudaHostAlloc` with `cudaHostAllocMapped`) so both the host CPU and GPU threads can access it without `cudaMemcpy`. The host sets the flag; GPU threads poll it periodically (every N integration steps) and early-exit.
- Progress reporting: simple "rendering..." until kernel completes (per-pixel GPU progress is impractical)

## C API Integration

The C API already supports the CUDA backend — no signature changes needed:

- `GRRTBackend` enum (`GRRT_BACKEND_CPU`, `GRRT_BACKEND_CUDA`) already exists in `types.h`
- `GRRTParams` already has a `backend` field (line 50 of `types.h`)
- `grrt_create(const GRRTParams* params)` reads `params->backend` to select the backend
- `grrt_cuda_available()` already declared in `api.h`

The implementation change is inside `grrt_create`: when `params->backend == GRRT_BACKEND_CUDA`, construct a CUDA render context instead of a CPU one. All other API functions (`grrt_render`, `grrt_render_tile`, `grrt_cancel`, etc.) dispatch through the context.

### Fallback Behavior

If `GRRT_BACKEND_CUDA` is requested but CUDA is unavailable (not compiled in, or no GPU at runtime), `grrt_create` returns NULL. Since `grrt_error()` requires a valid context, a new `grrt_last_error(void)` function is added to `api.h` that returns a global/thread-local error string for cases where no context exists.

## CLI Changes

- `--backend cpu|cuda` (default: `cpu`)
- `--validate` — render on both backends, compare output, print error report:
  - Prints per-channel max absolute error and mean absolute error
  - Prints coordinates of worst-case pixel
  - Exits with code 0 if max error < 1e-6, code 1 otherwise
  - Saves a difference image to `<output>_diff.png` (amplified 1000x for visibility)
- `--debug-pixel x,y` — trace single pixel on both backends, dump integration steps

## Correctness Validation Strategy

### Pixel-by-Pixel Comparison

- Render same scene on CPU and CUDA
- Compare raw `float4` arrays (before PNG quantization)
- Report max absolute error and mean absolute error per channel
- Target: max per-pixel error < 1e-6 in linear HDR space

### Incremental Validation Order

1. **Math primitives** — verify Vec4/Matrix4 operations match CPU via test kernel
2. **Metric values** — compute g_uv at known coordinates, compare to CPU
3. **Single geodesic** — trace one ray from a known pixel, compare full trajectory
4. **Full frame** — render complete image, pixel-by-pixel diff
5. **Conserved quantities** — check Hamiltonian constraint H ~ 0 and E, L conservation on GPU

### Debug Output

`--debug-pixel x,y` traces a single pixel on both backends and dumps integration steps to stdout for side-by-side comparison.

### Acceptable Tolerance

CPU and GPU floating-point results will not be bit-identical due to different operation ordering and FMA instructions. Max per-pixel error < 1e-6 in linear HDR space is the acceptance threshold. Pixels exceeding this report their coordinates and integration path.

## What This Design Does NOT Cover

- Performance optimization (occupancy tuning, memory coalescing, FP32 fast paths)
- Multi-GPU support
- CUDA streams / async rendering
- Texture-based celestial sphere (if added later)

These are deferred to after correctness is established.
