# General Relativistic Raytracer — Blender Render Engine Plugin

## Overview

Build a physically accurate GR raytracer as a standalone C++ library, then integrate it into Blender as a custom render engine. Blender provides the UI, scene management, camera controls, animation timeline, and output pipeline. The C++ library handles all the curved-spacetime physics.

The project has three phases:
1. **Core C++ library** — geodesic integration, metrics, disk model, redshift. Produces images via CLI.
2. **CUDA acceleration** — GPU kernel for the same physics, 100–1000× speedup.
3. **Blender integration** — Python render engine plugin that calls the C++ library, giving you Blender's full interactive viewport and animation system.

**Target platform**: Windows (Visual Studio 2022 / CMake). Code should also compile on Linux. Blender plugin targets Blender 5.x (Python 3.11+).

---

## Phase 1: Core C++ Library + CLI

### Goal
A self-contained C++ shared library (`grrt.dll` / `libgrrt.so`) that takes render parameters and produces a float RGB framebuffer. A thin CLI wrapper demonstrates it by saving PNGs. The library's C API is what Phase 3 will call from Python.

### Architecture

```
gr-raytracer/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── grrt/
│   │   ├── api.h              // Public C API for the shared library
│   │   ├── types.h            // Shared structs (RenderParams, MetricType, etc.)
│   │   ├── math/
│   │   │   ├── vec3.h
│   │   │   ├── vec4.h
│   │   │   └── matrix4.h
│   │   ├── spacetime/
│   │   │   ├── metric.h       // Abstract metric interface
│   │   │   ├── schwarzschild.h
│   │   │   ├── kerr.h
│   │   │   └── christoffel.h  // Numerical Christoffel symbols (fallback)
│   │   ├── geodesic/
│   │   │   ├── integrator.h
│   │   │   ├── rk4.h
│   │   │   └── geodesic_tracer.h
│   │   ├── camera/
│   │   │   ├── observer.h
│   │   │   └── camera.h
│   │   ├── scene/
│   │   │   ├── accretion_disk.h
│   │   │   ├── celestial_sphere.h
│   │   │   └── scene.h
│   │   ├── render/
│   │   │   ├── renderer.h
│   │   │   ├── image.h
│   │   │   └── tonemapper.h
│   │   └── color/
│   │       ├── spectrum.h
│   │       └── redshift.h
│   └── grrt_export.h          // DLL export macros
├── src/
│   └── [.cpp files matching headers]
├── cli/
│   └── main.cpp               // CLI wrapper (thin, just parses args and calls API)
├── third_party/
│   └── stb_image_write.h
└── python/                    // Phase 3 lives here
    └── [see Phase 3]
```

### Public C API (`api.h`)

This is the boundary that Python (and CUDA) will call through. Keep it pure C for maximum FFI compatibility.

```cpp
#ifndef GRRT_API_H
#define GRRT_API_H

#include "grrt_export.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Create a render context with given parameters. Returns opaque handle.
GRRT_EXPORT GRRTContext* grrt_create(const GRRTParams* params);

// Destroy a render context.
GRRT_EXPORT void grrt_destroy(GRRTContext* ctx);

// Update parameters on an existing context (e.g. camera moved).
GRRT_EXPORT void grrt_update_params(GRRTContext* ctx, const GRRTParams* params);

// Render a full frame. Writes RGBA float data into the provided buffer.
// Buffer must be at least width * height * 4 * sizeof(float) bytes.
// Returns 0 on success, nonzero on error.
GRRT_EXPORT int grrt_render(GRRTContext* ctx, float* framebuffer);

// Render a rectangular tile (for Blender's tiled rendering).
// (x, y) is the top-left corner of the tile in the full image.
// Writes tile_width * tile_height * 4 floats into the buffer.
GRRT_EXPORT int grrt_render_tile(GRRTContext* ctx, float* buffer,
                                  int x, int y, int tile_width, int tile_height);

// Cancel an in-progress render (thread-safe, can be called from another thread).
GRRT_EXPORT void grrt_cancel(GRRTContext* ctx);

// Query progress of current render (0.0 to 1.0).
GRRT_EXPORT float grrt_progress(const GRRTContext* ctx);

// Get last error message.
GRRT_EXPORT const char* grrt_error(const GRRTContext* ctx);

// Query whether CUDA backend is available.
GRRT_EXPORT int grrt_cuda_available(void);

#ifdef __cplusplus
}
#endif

#endif
```

### Shared Types (`types.h`)

```cpp
typedef enum {
    GRRT_METRIC_SCHWARZSCHILD = 0,
    GRRT_METRIC_KERR = 1
} GRRTMetricType;

typedef enum {
    GRRT_BACKEND_CPU = 0,
    GRRT_BACKEND_CUDA = 1
} GRRTBackend;

typedef enum {
    GRRT_BG_BLACK = 0,
    GRRT_BG_STARS = 1,
    GRRT_BG_TEXTURE = 2
} GRRTBackgroundType;

typedef struct {
    // Image dimensions
    int width;
    int height;

    // Metric
    GRRTMetricType metric_type;
    double mass;            // Black hole mass (default 1.0)
    double spin;            // Kerr spin parameter |a| < M (default 0.998)

    // Observer / camera
    double observer_r;      // Radial coordinate (default 50)
    double observer_theta;  // Polar angle in radians (default ~80°)
    double observer_phi;    // Azimuthal angle in radians (default 0)
    double fov;             // Field of view in radians (default ~60°)

    // Camera orientation overrides (Euler angles relative to default look-at-origin)
    // These allow Blender to control look direction independently of position.
    double cam_yaw;         // Yaw offset in radians (default 0)
    double cam_pitch;       // Pitch offset in radians (default 0)
    double cam_roll;        // Roll offset in radians (default 0)

    // Accretion disk
    int disk_enabled;       // 0 = off, 1 = on
    double disk_inner;      // Inner radius (0 = use ISCO)
    double disk_outer;      // Outer radius (default 20)
    double disk_temperature;// Peak temperature in Kelvin (for color mapping)

    // Background
    GRRTBackgroundType background_type;
    const char* background_texture_path; // For BG_TEXTURE mode

    // Integration quality
    double integrator_tolerance;  // Adaptive step tolerance (default 1e-8)
    int integrator_max_steps;     // Max steps per ray (default 10000)

    // Rendering
    int samples_per_pixel;  // AA samples (default 1)
    int thread_count;       // CPU threads (0 = auto)
    GRRTBackend backend;    // CPU or CUDA

} GRRTParams;

typedef struct GRRTContext GRRTContext; // Opaque
```

### Math Layer (`math/`)

#### `vec4.h`
- 4-component vector with index access `[0..3]` for `(t, r, θ, φ)`.
- Supports both contravariant and covariant semantics (data type only — user tracks index placement).
- Arithmetic: addition, scalar multiplication, dot product with metric.
- Fully header-only for inlining.

#### `matrix4.h`
- 4×4 symmetric matrix for `g_μν`.
- Inverse computation for `g^μν`. Exploit sparsity for diagonal/block-diagonal metrics.
- Contraction: `g_μν v^ν → v_μ`.

#### `vec3.h`
- Standard 3D spatial vector for color math and camera setup.

### Spacetime Layer (`spacetime/`)

#### `metric.h` — Abstract Interface
```cpp
class Metric {
public:
    virtual ~Metric() = default;
    virtual Matrix4 g_lower(const Vec4& x) const = 0;   // g_μν
    virtual Matrix4 g_upper(const Vec4& x) const = 0;   // g^μν
    virtual double horizon_radius(double theta = M_PI/2) const = 0;
    virtual double isco_radius() const = 0;
    virtual bool has_ergosphere() const { return false; }
    virtual double ergosphere_radius(double theta) const { return 0; }
};
```

#### `schwarzschild.h`
Schwarzschild metric in coordinates `(t, r, θ, φ)`:
```
ds² = -(1 - 2M/r) dt² + (1 - 2M/r)⁻¹ dr² + r² dθ² + r² sin²θ dφ²
```
Parameter: mass `M`.

#### `kerr.h`
Kerr metric in Boyer-Lindquist coordinates `(t, r, θ, φ)`:
```
Σ = r² + a² cos²θ
Δ = r² - 2Mr + a²

ds² = -(1 - 2Mr/Σ) dt² - (4Mar sin²θ / Σ) dt dφ
      + (Σ/Δ) dr² + Σ dθ² + (r² + a² + 2Ma²r sin²θ/Σ) sin²θ dφ²
```
Parameters: mass `M`, spin `a` with `|a| < M`.
Implement: `horizon_radius()` → `r_+ = M + √(M² - a²)`, `ergosphere_radius(θ)` → `r_ergo = M + √(M² - a² cos²θ)`, `isco_radius()` using the standard Kerr ISCO formula.

### Geodesic Integration (`geodesic/`)

#### Formulation: Hamiltonian
The Hamiltonian for null geodesics:
```
H = ½ g^μν(x) p_μ p_ν = 0
```
Hamilton's equations give first-order ODEs:
```
dx^μ/dλ =  ∂H/∂p_μ = g^μν p_ν
dp_μ/dλ = -∂H/∂x^μ = -½ (∂g^αβ/∂x^μ) p_α p_β
```
Derivatives `∂g^αβ/∂x^μ` computed by central finite differences on the inverse metric.

**State vector**: 8 components `(t, r, θ, φ, p_t, p_r, p_θ, p_φ)`.

#### `integrator.h`
```cpp
struct GeodesicState {
    Vec4 position;   // x^μ = (t, r, θ, φ)
    Vec4 momentum;   // p_μ (covariant)
};

class Integrator {
public:
    virtual ~Integrator() = default;
    virtual GeodesicState step(const Metric& metric,
                               const GeodesicState& state,
                               double dlambda) const = 0;
};
```

#### `rk4.h` — RK4 with Adaptive Stepping
Standard 4th-order Runge-Kutta on the 8-component Hamiltonian system.

**Adaptive step**: Step doubling to control local error. Target tolerance ~`1e-8`. Step size heuristic: `dlambda ∝ r²` (small near the hole, large far away).

#### `geodesic_tracer.h`
Traces a single geodesic backward from camera until:
1. **Horizon hit** (`r < r_+ + ε`) → black pixel.
2. **Disk hit** (`θ` crosses `π/2`, `r_inner ≤ r ≤ r_outer`) → compute disk emission + redshift.
3. **Escape** (`r > r_max`, e.g. `1000M`) → sample celestial sphere.
4. **Max steps exceeded** → black pixel.

Returns: termination type, final position, redshift factor `g`, disk intersection coordinates.

### Camera and Observer (`camera/`)

#### `observer.h`
Observer characterized by:
- Coordinate position `x^μ_obs`.
- 4-velocity `u^μ_obs` (static observer or ZAMO for Kerr).
- Local orthonormal tetrad `{e_0, e_1, e_2, e_3}` constructed via Gram-Schmidt with the metric inner product.

Tetrad construction:
1. `e_0 = u^μ` (normalized 4-velocity).
2. `e_3` = forward (radially inward toward hole). Normalize.
3. `e_2` = up (along `θ`). Orthogonalize and normalize.
4. `e_1` = right (along `φ`). Orthogonalize and normalize.

#### `camera.h`
Maps pixel `(i, j)` → initial covariant 4-momentum `p_μ`.

1. Screen-space angles: `α = (i - width/2) * fov / width`, `β = (j - height/2) * fov / width`.
2. Apply yaw/pitch/roll offsets from `GRRTParams` (these come from Blender's camera orientation).
3. Local 3-direction: `d = -cos(β)sin(α) e_1 - sin(β) e_2 + cos(β)cos(α) e_3`.
4. 4-momentum: `p^μ = -e_0^μ + d^μ`, then lower index `p_μ = g_μν p^ν`.

### Scene Objects (`scene/`)

#### `accretion_disk.h` — Thin Disk
Geometrically thin, optically thick, equatorial (`θ = π/2`).

- Inner edge at `r_isco`, outer edge at `r_outer`.
- Disk material on circular orbits: `Ω = √(M/r³)` (Schwarzschild) or Kerr equivalent.
- Temperature: `T(r) ∝ r^{-3/4}` (simplified) or Page-Thorne for accuracy.
- Emission: blackbody at local `T`, then redshifted.

**Intersection detection**: Monitor `θ` crossing `π/2`. Interpolate to find `(r_cross, φ_cross)`. Check radial bounds.

**Redshift**: `g = (p_μ u^μ_emitter) / (p_ν u^ν_observer)`. Observed intensity: `I_obs = g³ × I_emitted`.

#### `celestial_sphere.h`
- **Black**: constant background.
- **Stars**: random point sources at `(θ, φ, brightness)`. Match nearest star within angular tolerance when ray escapes.
- **Texture**: equirectangular image sampled at ray's final `(θ, φ)`.

#### `scene.h`
Aggregates metric, disk, background. Provides `trace_ray(state) → RGBA` for the renderer.

### Rendering (`render/`)

#### `renderer.h`
Main render loop over pixels, parallelized with OpenMP:
```cpp
#pragma omp parallel for schedule(dynamic) num_threads(thread_count)
for (int j = 0; j < height; j++)
    for (int i = 0; i < width; i++)
        framebuffer[j*width + i] = scene.trace_ray(camera.ray_for_pixel(i, j));
```

Also implements `render_tile()` for Blender's tiled dispatch.

**Cancellation**: Check an `std::atomic<bool> cancelled` flag periodically (every N rows). Return early if set.

**Progress**: Update an `std::atomic<float> progress` value as rows complete.

#### `image.h`
Float RGBA framebuffer. Methods: `set_pixel`, `get_pixel`, `save_png` (via `stb_image_write`).

#### `tonemapper.h`
HDR → LDR conversion:
- Reinhard: `L_out = L / (1 + L)`.
- Optional ACES filmic curve.
- sRGB gamma.

### Color (`color/`)

#### `redshift.h`
Maps frequency ratio `g` to color shift. Shift blackbody spectrum peak, integrate over CIE color matching functions → XYZ → sRGB. Provide a precomputed lookup table for performance.

#### `spectrum.h`
Planck function `B(ν, T)`. Temperature → RGB lookup table.

### CLI Wrapper (`cli/main.cpp`)

Thin wrapper that parses command-line arguments, populates `GRRTParams`, calls `grrt_create` / `grrt_render` / `grrt_destroy`, and writes the result to PNG.

```
Usage: grrt-cli [options]
  --width N           Image width (default: 1024)
  --height N          Image height (default: 768)
  --metric TYPE       schwarzschild | kerr (default: kerr)
  --mass M            Black hole mass (default: 1.0)
  --spin A            Spin parameter (default: 0.998)
  --observer-r R      Observer radius (default: 50)
  --observer-theta T  Observer polar angle, degrees (default: 80)
  --fov F             Field of view, degrees (default: 60)
  --disk on|off       Accretion disk (default: on)
  --disk-outer R      Disk outer radius (default: 20)
  --background TYPE   black | stars | texture (default: stars)
  --output FILE       Output file (default: output.png)
  --threads N         CPU threads (default: auto)
  --backend cpu|cuda  Compute backend (default: auto)
```

### Build System

```cmake
cmake_minimum_required(VERSION 3.20)
project(gr-raytracer LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)

# Shared library
add_library(grrt SHARED
    src/api.cpp
    src/schwarzschild.cpp
    src/kerr.cpp
    src/rk4.cpp
    src/geodesic_tracer.cpp
    src/observer.cpp
    src/camera.cpp
    src/accretion_disk.cpp
    src/celestial_sphere.cpp
    src/scene.cpp
    src/renderer.cpp
    src/image.cpp
    src/tonemapper.cpp
    src/spectrum.cpp
    src/redshift.cpp
)
target_include_directories(grrt PUBLIC include/ PRIVATE third_party/)
target_compile_definitions(grrt PRIVATE GRRT_BUILDING_DLL)

# Generate export header
include(GenerateExportHeader)
generate_export_header(grrt
    EXPORT_FILE_NAME include/grrt_export.h
    EXPORT_MACRO_NAME GRRT_EXPORT
)

# OpenMP (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(grrt PRIVATE OpenMP::OpenMP_CXX)
endif()

# CLI tool
add_executable(grrt-cli cli/main.cpp)
target_link_libraries(grrt-cli PRIVATE grrt)
target_include_directories(grrt-cli PRIVATE third_party/)
```

### Validation Checklist (Phase 1)

- [ ] Schwarzschild shadow matches `arcsin(3√3 M / r_obs)`.
- [ ] Kerr shadow shape matches known silhouettes for `a = 0, 0.5, 0.998` at various inclinations.
- [ ] Photon ring visible at shadow edge.
- [ ] Disk Doppler beaming: approaching side brighter than receding.
- [ ] Smooth redshift across disk (no discontinuities).
- [ ] Higher-order disk images visible (light looping around the hole).
- [ ] Hamiltonian constraint `H ≈ 0` maintained throughout integration (monitor drift, should stay below `~1e-10`).
- [ ] Conserved quantities `E = -p_t`, `L = p_φ` constant along Schwarzschild geodesics.
- [ ] Library C API works: create context, render, destroy — no leaks (test with AddressSanitizer).
- [ ] Tile rendering produces identical output to full-frame rendering.
- [ ] Cancellation works: calling `grrt_cancel` from another thread stops the render promptly.

---

## Phase 2: CUDA Acceleration

### Goal
Port the geodesic integration to a CUDA kernel. The library auto-detects CUDA availability and selects the backend based on `GRRTParams.backend`. The C API is unchanged — callers don't need to know which backend ran.

### Architecture Changes

```
gr-raytracer/
├── cuda/
│   ├── kernel.cu              // Main render kernel
│   ├── metric_device.cuh      // Device-side metric (Schwarzschild + Kerr)
│   ├── integrator_device.cuh  // Device-side RK4
│   ├── camera_device.cuh      // Device-side ray generation
│   └── cuda_renderer.cu/.cuh  // Implements grrt_render / grrt_render_tile via CUDA
```

### CUDA Kernel

One thread per pixel:
```cpp
__global__ void render_kernel(
    float4* framebuffer,
    int width, int height,
    CameraDeviceParams cam,
    MetricDeviceParams metric,
    DiskDeviceParams disk,
    IntegratorDeviceParams integrator
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= width || j >= height) return;

    GeodesicState state = generate_ray(cam, i, j, width, height);
    RayResult result = trace_geodesic(metric, state, disk, integrator);
    framebuffer[j * width + i] = compute_color(result, disk);
}
```

### Key CUDA Design Decisions

- **No virtual functions on device**: Metric/integrator are device functions or template-specialized kernels. Use `MetricType` enum + `if/switch`, or template the kernel on metric type.
- **Structs not classes**: All device-side data in plain structs passed by value or constant memory.
- **Block size**: 16×16 (256 threads/block).
- **Double precision**: Use `double` throughout. FP32 accumulates too much error for Kerr geodesics near the horizon. Modern GPUs (SM 6.0+) have adequate FP64 throughput.
- **Adaptive stepping on GPU**: Use fixed small step size to avoid warp divergence, or accept divergence with adaptive stepping (profile both, pick whichever is faster for typical scenes).
- **Cancellation**: Host periodically checks a flag and can skip launching further tile kernels. Mid-kernel cancellation isn't practical — just let the current kernel finish.
- **Progress**: Track by tiles completed.

### Build Changes

```cmake
# Optional CUDA support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 23)

    target_sources(grrt PRIVATE
        cuda/kernel.cu
        cuda/cuda_renderer.cu
    )
    target_compile_definitions(grrt PRIVATE GRRT_HAS_CUDA)
    set_target_properties(grrt PROPERTIES CUDA_ARCHITECTURES "75;86;89")
endif()
```

### Validation Checklist (Phase 2)

- [ ] CUDA output matches CPU output within floating-point tolerance for same parameters.
- [ ] Benchmark: measure speedup at 1024×768, 1920×1080, 3840×2160.
- [ ] `grrt_cuda_available()` returns correct result.
- [ ] `backend = GRRT_BACKEND_CUDA` with no GPU falls back to CPU gracefully.
- [ ] All CUDA calls checked with `cudaGetLastError()`.
- [ ] No device memory leaks (profile with `cuda-memcheck` or Compute Sanitizer).

---

## Phase 3: Blender Render Engine Plugin

### Goal
Register a custom render engine in Blender that uses the `grrt` shared library for all pixel computation. Blender handles the UI, camera, animation, and output. The plugin translates Blender's scene state into `GRRTParams` and passes framebuffer data back to Blender.

### Architecture

```
gr-raytracer/
├── python/
│   ├── __init__.py            // Blender addon entry point (bl_info, register, unregister)
│   ├── engine.py              // RenderEngine subclass
│   ├── properties.py          // Custom Blender properties (spin, metric type, etc.)
│   ├── panels.py              // UI panels in Blender's Properties editor
│   ├── operators.py           // Any custom operators (e.g. "set camera to ZAMO orbit")
│   ├── camera_convert.py      // Blender camera → GRRTParams conversion
│   └── grrt_binding.py        // ctypes wrapper around grrt.dll / libgrrt.so
```

### Addon Metadata (`__init__.py`)

```python
bl_info = {
    "name": "GR Raytracer",
    "author": "Seth",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "category": "Render",
    "description": "General relativistic raytracer for black hole visualization",
}

def register():
    # Register all classes: engine, properties, panels, operators
    ...

def unregister():
    ...
```

### ctypes Binding (`grrt_binding.py`)

Wraps the C API using Python's built-in `ctypes`:

```python
import ctypes
import os
import platform

class GRRTParams(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("metric_type", ctypes.c_int),
        ("mass", ctypes.c_double),
        ("spin", ctypes.c_double),
        ("observer_r", ctypes.c_double),
        ("observer_theta", ctypes.c_double),
        ("observer_phi", ctypes.c_double),
        ("fov", ctypes.c_double),
        ("cam_yaw", ctypes.c_double),
        ("cam_pitch", ctypes.c_double),
        ("cam_roll", ctypes.c_double),
        ("disk_enabled", ctypes.c_int),
        ("disk_inner", ctypes.c_double),
        ("disk_outer", ctypes.c_double),
        ("disk_temperature", ctypes.c_double),
        ("background_type", ctypes.c_int),
        ("background_texture_path", ctypes.c_char_p),
        ("integrator_tolerance", ctypes.c_double),
        ("integrator_max_steps", ctypes.c_int),
        ("samples_per_pixel", ctypes.c_int),
        ("thread_count", ctypes.c_int),
        ("backend", ctypes.c_int),
    ]

class GRRTLib:
    """Wrapper around the grrt shared library."""

    def __init__(self):
        # Find the shared library adjacent to the addon
        addon_dir = os.path.dirname(os.path.abspath(__file__))
        if platform.system() == "Windows":
            lib_name = "grrt.dll"
        elif platform.system() == "Darwin":
            lib_name = "libgrrt.dylib"
        else:
            lib_name = "libgrrt.so"

        lib_path = os.path.join(addon_dir, "lib", lib_name)
        self._lib = ctypes.CDLL(lib_path)

        # Bind function signatures
        self._lib.grrt_create.restype = ctypes.c_void_p
        self._lib.grrt_create.argtypes = [ctypes.POINTER(GRRTParams)]

        self._lib.grrt_destroy.restype = None
        self._lib.grrt_destroy.argtypes = [ctypes.c_void_p]

        self._lib.grrt_render.restype = ctypes.c_int
        self._lib.grrt_render.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

        self._lib.grrt_render_tile.restype = ctypes.c_int
        self._lib.grrt_render_tile.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self._lib.grrt_cancel.restype = None
        self._lib.grrt_cancel.argtypes = [ctypes.c_void_p]

        self._lib.grrt_progress.restype = ctypes.c_float
        self._lib.grrt_progress.argtypes = [ctypes.c_void_p]

        self._lib.grrt_cuda_available.restype = ctypes.c_int
        self._lib.grrt_cuda_available.argtypes = []

    def create(self, params: GRRTParams):
        return self._lib.grrt_create(ctypes.byref(params))

    def destroy(self, ctx):
        self._lib.grrt_destroy(ctx)

    def render(self, ctx, width, height):
        buf_size = width * height * 4
        buf = (ctypes.c_float * buf_size)()
        result = self._lib.grrt_render(ctx, buf)
        if result != 0:
            raise RuntimeError("grrt_render failed")
        return buf

    def render_tile(self, ctx, x, y, tile_w, tile_h):
        buf_size = tile_w * tile_h * 4
        buf = (ctypes.c_float * buf_size)()
        result = self._lib.grrt_render_tile(ctx, buf, x, y, tile_w, tile_h)
        if result != 0:
            raise RuntimeError("grrt_render_tile failed")
        return buf

    def cancel(self, ctx):
        self._lib.grrt_cancel(ctx)

    def progress(self, ctx):
        return self._lib.grrt_progress(ctx)

    def cuda_available(self):
        return bool(self._lib.grrt_cuda_available())
```

### Render Engine (`engine.py`)

```python
import bpy
import math
import numpy as np
from . import grrt_binding

class GRRTRenderEngine(bpy.types.RenderEngine):
    bl_idname = "GRRT"
    bl_label = "GR Raytracer"
    bl_use_preview = True        # Support material preview
    bl_use_eevee_viewport = True # Use Eevee for solid/wireframe modes

    def __init__(self):
        super().__init__()
        self._lib = grrt_binding.GRRTLib()
        self._ctx = None

    def __del__(self):
        if self._ctx:
            self._lib.destroy(self._ctx)

    def _build_params(self, scene, camera_obj, width, height, is_preview=False):
        """Convert Blender scene state to GRRTParams."""
        grrt_props = scene.grrt  # Custom property group (see properties.py)

        params = grrt_binding.GRRTParams()
        params.width = width
        params.height = height

        # Metric
        params.metric_type = 1 if grrt_props.metric == 'KERR' else 0
        params.mass = grrt_props.mass
        params.spin = grrt_props.spin

        # Convert Blender camera to GR observer coordinates
        # See camera_convert.py for the full implementation
        from .camera_convert import blender_to_gr_observer
        obs = blender_to_gr_observer(camera_obj, grrt_props)
        params.observer_r = obs['r']
        params.observer_theta = obs['theta']
        params.observer_phi = obs['phi']
        params.cam_yaw = obs['yaw']
        params.cam_pitch = obs['pitch']
        params.cam_roll = obs['roll']

        # FOV from Blender camera
        if camera_obj and camera_obj.data:
            params.fov = camera_obj.data.angle  # Already in radians
        else:
            params.fov = math.radians(60)

        # Disk
        params.disk_enabled = 1 if grrt_props.disk_enabled else 0
        params.disk_inner = grrt_props.disk_inner
        params.disk_outer = grrt_props.disk_outer
        params.disk_temperature = grrt_props.disk_temperature

        # Background
        bg_map = {'BLACK': 0, 'STARS': 1, 'TEXTURE': 2}
        params.background_type = bg_map.get(grrt_props.background, 0)
        if grrt_props.background_texture:
            params.background_texture_path = grrt_props.background_texture.encode('utf-8')

        # Quality (reduce for preview)
        if is_preview:
            params.integrator_tolerance = 1e-5
            params.integrator_max_steps = 2000
            params.samples_per_pixel = 1
        else:
            params.integrator_tolerance = grrt_props.tolerance
            params.integrator_max_steps = grrt_props.max_steps
            params.samples_per_pixel = grrt_props.samples

        # Backend
        params.thread_count = 0  # Auto
        if grrt_props.use_cuda and self._lib.cuda_available():
            params.backend = 1  # CUDA
        else:
            params.backend = 0  # CPU

        return params

    def render(self, depsgraph):
        """Final render (F12)."""
        scene = depsgraph.scene
        scale = scene.render.resolution_percentage / 100.0
        width = int(scene.render.resolution_x * scale)
        height = int(scene.render.resolution_y * scale)

        camera_obj = scene.camera
        params = self._build_params(scene, camera_obj, width, height)

        # Create context and render
        ctx = self._lib.create(params)
        try:
            buf = self._lib.render(ctx, width, height)

            # Convert to numpy and reshape to (height, width, 4)
            pixels = np.ctypeslib.as_array(buf, shape=(height * width * 4,))
            pixels = pixels.reshape((height, width, 4))

            # Pass to Blender's render result
            result = self.begin_result(0, 0, width, height)
            layer = result.layers[0].passes["Combined"]

            # Blender expects a flat list, bottom-to-top row order
            # Flip vertically if our renderer uses top-to-bottom
            pixels_flipped = np.flip(pixels, axis=0).flatten().tolist()
            layer.rect = pixels_flipped

            self.end_result(result)
        finally:
            self._lib.destroy(ctx)

    def view_update(self, context, depsgraph):
        """Called when scene data changes (viewport)."""
        # Mark viewport for redraw
        pass

    def view_draw(self, context, depsgraph):
        """Called to draw the viewport (3D viewport in rendered mode).
        This is where real-time preview happens.
        For now, render at reduced resolution and blit to viewport.
        """
        region = context.region
        scene = depsgraph.scene

        # Render at reduced resolution for interactive speed
        scale = 0.25  # 25% of viewport resolution
        width = max(1, int(region.width * scale))
        height = max(1, int(region.height * scale))

        camera_obj = scene.camera
        if not camera_obj:
            return

        params = self._build_params(scene, camera_obj, width, height, is_preview=True)
        ctx = self._lib.create(params)
        try:
            buf = self._lib.render(ctx, width, height)
            # Upload to GPU texture and draw via bgl / gpu module
            # (Implementation uses Blender's gpu module to blit the texture)
            self._draw_pixels(context, buf, width, height)
        finally:
            self._lib.destroy(ctx)

    def _draw_pixels(self, context, buf, width, height):
        """Upload pixel buffer and draw to viewport using Blender's gpu module."""
        import gpu
        from gpu_extras.presets import draw_texture_2d
        import numpy as np

        pixels = np.ctypeslib.as_array(buf, shape=(height, width, 4))
        pixels_flipped = np.flip(pixels, axis=0).copy()

        # Create GPU texture
        texture = gpu.types.GPUTexture((width, height), format='RGBA32F',
                                        data=gpu.types.Buffer('FLOAT', width * height * 4,
                                                               pixels_flipped.flatten().tolist()))

        # Draw fullscreen in the viewport region
        draw_texture_2d(texture, (0, 0), context.region.width, context.region.height)
```

### Camera Coordinate Conversion (`camera_convert.py`)

This is the trickiest part of the integration — mapping Blender's Cartesian camera to the GR observer's Boyer-Lindquist coordinates.

```python
import math
import mathutils  # Blender's math library

def blender_to_gr_observer(camera_obj, grrt_props):
    """
    Convert Blender camera world transform to GR observer parameters.

    Coordinate mapping convention:
      Blender Z-up → GR polar axis (θ = 0)
      Blender XY plane → GR equatorial plane (θ = π/2)

    The black hole sits at Blender's world origin.
    Blender distance units map to M (geometrized units).

    Returns dict with: r, theta, phi, yaw, pitch, roll
    """
    if camera_obj is None:
        return {
            'r': 50.0, 'theta': math.radians(80), 'phi': 0.0,
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0
        }

    loc = camera_obj.matrix_world.translation

    # Cartesian → spherical
    x, y, z = loc.x, loc.y, loc.z
    r = math.sqrt(x*x + y*y + z*z)
    if r < 1e-10:
        r = 50.0  # Fallback

    theta = math.acos(max(-1, min(1, z / r)))  # Polar angle from Z axis
    phi = math.atan2(y, x)                      # Azimuthal angle

    # Camera orientation: decompose into yaw/pitch/roll relative to
    # the "default" look-at-origin orientation.
    # Default: camera at (r, θ, φ) looking toward origin.
    # Any deviation from that default is expressed as yaw/pitch/roll offsets.
    #
    # 1. Build the "default" rotation matrix (camera at loc, looking at origin, Z-up)
    direction = -loc.normalized()
    up = mathutils.Vector((0, 0, 1))
    right = direction.cross(up).normalized()
    if right.length < 1e-6:
        # Camera is on the Z axis, pick arbitrary up
        right = mathutils.Vector((1, 0, 0))
    up = right.cross(direction).normalized()

    default_rot = mathutils.Matrix((
        (right.x, up.x, direction.x),
        (right.y, up.y, direction.y),
        (right.z, up.z, direction.z),
    )).transposed().to_4x4()

    # 2. Actual camera rotation
    actual_rot = camera_obj.matrix_world.to_3x3().to_4x4()

    # 3. Relative rotation = default⁻¹ × actual
    rel_rot = default_rot.inverted() @ actual_rot
    euler = rel_rot.to_euler('YXZ')  # Yaw-Pitch-Roll convention

    return {
        'r': r,
        'theta': theta,
        'phi': phi,
        'yaw': euler.y,
        'pitch': euler.x,
        'roll': euler.z
    }
```

### Custom Properties (`properties.py`)

```python
import bpy
from bpy.props import (
    FloatProperty, IntProperty, BoolProperty,
    EnumProperty, StringProperty
)

class GRRTSceneProperties(bpy.types.PropertyGroup):
    """Properties attached to bpy.types.Scene as scene.grrt"""

    metric: EnumProperty(
        name="Metric",
        items=[
            ('SCHWARZSCHILD', "Schwarzschild", "Non-rotating black hole"),
            ('KERR', "Kerr", "Rotating black hole"),
        ],
        default='KERR'
    )

    mass: FloatProperty(
        name="Mass (M)", default=1.0, min=0.01, max=100.0,
        description="Black hole mass in geometrized units"
    )

    spin: FloatProperty(
        name="Spin (a/M)", default=0.998, min=0.0, max=0.999,
        description="Dimensionless spin parameter",
        precision=4
    )

    # Disk
    disk_enabled: BoolProperty(name="Accretion Disk", default=True)
    disk_inner: FloatProperty(
        name="Inner Radius", default=0.0, min=0.0,
        description="Inner disk radius (0 = ISCO)"
    )
    disk_outer: FloatProperty(name="Outer Radius", default=20.0, min=1.0, max=200.0)
    disk_temperature: FloatProperty(
        name="Peak Temperature (K)", default=1e7,
        min=1e3, max=1e10,
        description="Peak disk temperature for color mapping"
    )

    # Background
    background: EnumProperty(
        name="Background",
        items=[
            ('BLACK', "Black", "Pure black background"),
            ('STARS', "Star Field", "Procedural star field"),
            ('TEXTURE', "Texture", "Equirectangular background image"),
        ],
        default='STARS'
    )
    background_texture: StringProperty(
        name="Background Texture", subtype='FILE_PATH'
    )

    # Quality
    tolerance: FloatProperty(
        name="Integrator Tolerance", default=1e-8,
        min=1e-12, max=1e-3,
        description="Adaptive step tolerance (smaller = more accurate, slower)"
    )
    max_steps: IntProperty(
        name="Max Steps", default=10000, min=100, max=1000000,
        description="Maximum integration steps per ray"
    )
    samples: IntProperty(
        name="Samples", default=1, min=1, max=64,
        description="Anti-aliasing samples per pixel"
    )

    # Backend
    use_cuda: BoolProperty(
        name="Use CUDA", default=True,
        description="Use GPU acceleration if available"
    )
```

### UI Panels (`panels.py`)

```python
import bpy

class GRRT_PT_main(bpy.types.Panel):
    bl_label = "GR Raytracer"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'GRRT'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        grrt = context.scene.grrt

        # Metric section
        layout.prop(grrt, "metric")
        layout.prop(grrt, "mass")
        if grrt.metric == 'KERR':
            layout.prop(grrt, "spin")


class GRRT_PT_disk(bpy.types.Panel):
    bl_label = "Accretion Disk"
    bl_parent_id = "GRRT_PT_main"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'GRRT'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw_header(self, context):
        self.layout.prop(context.scene.grrt, "disk_enabled", text="")

    def draw(self, context):
        layout = self.layout
        grrt = context.scene.grrt
        layout.active = grrt.disk_enabled
        layout.prop(grrt, "disk_inner")
        layout.prop(grrt, "disk_outer")
        layout.prop(grrt, "disk_temperature")


class GRRT_PT_background(bpy.types.Panel):
    bl_label = "Background"
    bl_parent_id = "GRRT_PT_main"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'GRRT'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        grrt = context.scene.grrt
        layout.prop(grrt, "background")
        if grrt.background == 'TEXTURE':
            layout.prop(grrt, "background_texture")


class GRRT_PT_quality(bpy.types.Panel):
    bl_label = "Quality"
    bl_parent_id = "GRRT_PT_main"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "render"
    COMPAT_ENGINES = {'GRRT'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES

    def draw(self, context):
        layout = self.layout
        grrt = context.scene.grrt
        layout.prop(grrt, "tolerance")
        layout.prop(grrt, "max_steps")
        layout.prop(grrt, "samples")
        layout.separator()
        layout.prop(grrt, "use_cuda")
```

### Installation and Packaging

The addon is distributed as a zip:

```
gr_raytracer_addon.zip
├── gr_raytracer/
│   ├── __init__.py
│   ├── engine.py
│   ├── properties.py
│   ├── panels.py
│   ├── operators.py
│   ├── camera_convert.py
│   ├── grrt_binding.py
│   └── lib/
│       ├── grrt.dll           (Windows)
│       ├── libgrrt.so         (Linux)
│       └── libgrrt.dylib      (macOS, optional)
```

User installs via Edit → Preferences → Add-ons → Install from Disk. The shared library is bundled inside the addon directory.

### Blender Workflow

Once installed:
1. Set render engine to "GR Raytracer" in the Render Properties panel.
2. Place Blender camera anywhere in the scene — its position and orientation map to the GR observer.
3. The origin (0,0,0) is the black hole.
4. Adjust spin, disk, quality in the GR Raytracer panel.
5. Switch viewport shading to "Rendered" for live preview (reduced quality).
6. Press F12 for final render at full quality.
7. Use Blender's timeline to animate: keyframe the camera position, spin parameter, or disk properties.

### Validation Checklist (Phase 3)

- [ ] Addon installs cleanly in Blender 5.x on Windows.
- [ ] "GR Raytracer" appears in render engine dropdown.
- [ ] Custom panels appear in Render Properties when engine is selected.
- [ ] F12 render produces correct image (matches CLI output for same parameters).
- [ ] Viewport "Rendered" mode shows live preview.
- [ ] Camera movement in viewport updates the preview.
- [ ] Changing spin parameter updates preview.
- [ ] Animation render (Ctrl+F12) produces correct frame sequence.
- [ ] CUDA toggle works (falls back to CPU gracefully).
- [ ] Cancelling a render (Esc) works promptly.
- [ ] No crashes or memory leaks over extended use.

---

## General Coding Standards

- **C++23**: Use `<numbers>`, structured bindings, `std::format` where available.
- **Const correctness**: Mark everything `const` that can be.
- **No raw `new`/`delete`**: Use `std::unique_ptr` or stack allocation.
- **Header-only math types**: `Vec3`, `Vec4`, `Matrix4` fully in headers for inlining.
- **Pure C API boundary**: The shared library exports only C functions. All C++ stays internal.
- **Documentation**: Every class/method gets a Doxygen-style comment explaining the physics.
- **Error handling**: Exceptions internally, error codes at the C API boundary. No exceptions cross the DLL boundary.
- **Units**: Geometrized units (`G = c = 1`) throughout. `M` sets the length scale.
- **Python**: Follow Blender addon conventions. PEP 8. No external Python dependencies beyond `ctypes` and `numpy` (both ship with Blender).

## Third-Party Dependencies

| Library | Phase | License | Notes |
|---------|-------|---------|-------|
| stb_image_write | 1 | Public domain | Single header, vendored |
| OpenMP | 1 | System | Optional CPU parallelism |
| CUDA Toolkit 12.x | 2 | NVIDIA EULA | Optional GPU backend |
| Blender 5.x | 3 | GPL | Host application (addon is GPL-compatible) |

**Note on GPL**: Blender addons that use Blender's Python API are generally considered derivative works of Blender and must be GPL-compatible. Since this addon calls an external shared library via `ctypes`, the library itself can be any license, but the Python code should carry a GPL-compatible license.

---

## Stretch Goals

- **Volumetric accretion disk**: Replace thin disk with 3D torus. Ray march through emitting/absorbing medium along geodesics. Integrate with Blender's volume shader for material properties.
- **Arbitrary scene objects**: Allow placing Blender mesh objects near the black hole and lens them correctly (requires BVH queries along geodesic segments — hard).
- **Polarization**: Track polarization vector via parallel transport. Render polarization maps as separate render passes.
- **Multiple black holes**: Approximate metrics for binary systems.
- **Blender node integration**: Expose disk temperature, redshift, and hit type as AOVs (Arbitrary Output Variables) that can be used in Blender's compositor.
- **Video export**: Use Blender's built-in video encoding for flythrough animations.
