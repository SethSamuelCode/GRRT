# CUDA Parity + Modal Integration — Design

**Status:** design only. Implementation plan to be authored separately, after disk/ring-error work lands.
**Date:** 2026-04-27
**Author:** Seth Samuel (designed in collaboration with Claude)

## Background

GRRT today is a working C++23 CPU raytracer with an unfinished CUDA backend that
only handles the basic RGB render path. Several CPU features have outpaced the
GPU side:

- The CPU integrator is now Dormand-Prince RK4(5) with a PI step controller
  (`src/rk4.cpp`); CUDA still uses the older step-doubling RK4
  (`cuda/cuda_geodesic.h`).
- Spectral rendering, streaming-FITS output, progress callbacks, samples-per-pixel
  AA, and tile rendering all exist on CPU only.
- No Python binding or remote-execution path exists yet, despite CLAUDE.md naming
  `python/grrt_binding.py` as the target shape.

The motivating use case: cheap cloud GPU access via [Modal](https://modal.com),
which is Python-only at the job-definition layer. To get there we need a
Python → C++ bridge alongside the CUDA work.

## Goals

Bring the CUDA backend up to **scope-C parity** with CPU:

1. Port Dormand-Prince RK4(5) to CUDA, replacing the existing step-doubling RK4.
2. Add CUDA spectral rendering (in-RAM cube). Output is written to a Modal Volume
   as FITS.
3. Implement the stubbed CUDA tile renderer.
4. Wire progress callbacks through the CUDA path.
5. Add samples-per-pixel anti-aliasing on the CUDA kernel.
6. Build a thin `ctypes` Python binding around the existing C API.
7. Build a Modal entrypoint that runs renders on FP64-strong GPUs.
8. Add a parity test suite that asserts CUDA matches CPU within tolerance.

## Non-goals (deferred to future work)

- Streaming-FITS spectral output on CUDA. Cube assembly happens entirely in
  device + host RAM; no per-row mid-kernel callbacks.
- Setup/render separation in the CUDA backend (LUT uploads remain per-call;
  see `Future work` for the refactor).
- Stateful Modal `Cls` for parameter sweeps (depends on the setup/render split).
- Real-time progress streaming back to a client GUI.
- A Blender plugin or a debug-pixel mode on CUDA.
- CUDA CI automation on Modal.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│ User's Windows machine                                              │
│   modal run scripts/modal_render.py --spin 0.998 --output blackhole │
└────────────────────────┬────────────────────────────────────────────┘
                         │  Modal CLI (gRPC)
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Modal cloud — image: nvidia/cuda:12.8.0-devel + cmake + g++         │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │ Build phase (CPU container, image-build time, cached)     │      │
│  │   cmake -DGRRT_ENABLE_CUDA=ON \                           │      │
│  │         -DCMAKE_CUDA_ARCHITECTURES="80;90;100"            │      │
│  │   ⇒ libgrrt.so (fat binary: A100, H100/H200, B200)        │      │
│  └───────────────────────────────────────────────────────────┘      │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────┐      │
│  │ Run phase  (default GPU = A100 40GB)                      │      │
│  │                                                            │     │
│  │   python ──ctypes──► libgrrt.so ──CUDA──► GPU              │     │
│  │                          │                                 │     │
│  │                          ▼                                 │     │
│  │   FITS / NPY ──────► /mnt/grrt-vol  (Modal Volume)         │     │
│  └────────────────────────────────────────────────────────────┘     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  modal volume get  (on demand)
                           ▼
                       User's machine
```

### Layers (bottom-up)

1. **C++ core** (`src/`, `cuda/`) — existing. CUDA backend grows additively.
2. **C API** (`include/grrt/api.h`) — minor extensions only. New entrypoints
   for spectral-on-CUDA mirror existing CPU signatures.
3. **ctypes Python binding** (`python/grrt/`) — ~150-200 lines. Hand-maintained
   `Structure` for `GRRTParams`. Loads `libgrrt.so` at import. Returns
   `numpy.ndarray` framebuffers, raises `grrt.GRRTError` on non-zero return
   codes.
4. **Modal entrypoint** (`scripts/modal_render.py`) — stateless function plus a
   tiny CPU download helper. Single-image image definition, baked at image-build
   time.
5. **Validation** (`tests/test_cuda_parity.cpp`, extended `cli/main.cpp
   --validate`) — five canonical scenes; live-CLI flag for ad-hoc comparisons.

### Guiding principles

- **CUDA grows; CPU stays unchanged.** Every parity gap is a CUDA-side addition.
- **C API stays minimal and stable.** Python and Modal never see C++ types.
- **No exceptions across the DLL boundary.** Status codes + thread-local
  `grrt_last_error()` only.

## CUDA parity work

All edits live under `cuda/` except for one small `RenderParams` extension and
two new C-API entrypoints.

### Dormand-Prince RK4(5) port

Translate `RK4::dormand_prince_step` and the PI step-size controller from
`src/rk4.cpp` into a `__device__` function `dopri5_adaptive_step` in
`cuda/cuda_geodesic.h`. Butcher-tableau coefficients become `constexpr double`
literals (no `__constant__` upload needed; nvcc folds them).

The kernel call site at `cuda/cuda_render.cu:115` swaps from
`rk4_adaptive_step` to `dopri5_adaptive_step`. The old step-doubling RK4 is
deleted from the CUDA tree.

Risk: the 6-stage tableau increases register pressure; may require tuning
`__launch_bounds__` (currently `256`) downward to keep occupancy.

### Spectral rendering

A separate `__global__ render_spectral_kernel` lives in a new TU
`cuda/cuda_render_spectral.cu`. It owns a new `__constant__ double
d_frequency_bins[MAX_FREQ_BINS]` symbol (cap = 256). Inside the kernel, the
volumetric raymarcher and disk emission functions loop over frequency bins per
sample point, just as their CPU counterparts do.

Output buffer: `double* d_spectral_output` of shape `[height, width, num_freqs]`
in row-major. For 1024² × 100 freqs × 8 bytes = 800 MB — fits A100 40GB with
headroom; A100 80GB / B200 unconstrained.

Host writes the cube to FITS row-by-row via the existing `FITSStreamWriter`
post-render. No new C++ FITS code; this reuses the row-streaming write
loop already in place for the CPU path. The "streaming" we deferred refers to
mid-kernel per-row callbacks, not to the file-format writer.

### Tile rendering

Implement the stub at `cuda/cuda_backend.cu:269` by adding `tile_x_offset`
and `tile_y_offset` fields to `RenderParams` and adjusting the kernel's pixel
indexing:

```cpp
const int i = blockIdx.x * blockDim.x + threadIdx.x + d_params.tile_x_offset;
const int j = blockIdx.y * blockDim.y + threadIdx.y + d_params.tile_y_offset;
```

Output writes go into a tile-shaped `float4*` buffer rather than the full-frame
one. Validation invariant from CLAUDE.md must hold: a 1024² full-frame and a
4×4 grid of stitched 256² tiles must produce **identical bytes**, not just
within tolerance, because per-pixel work is identical regardless of grid shape.

### Progress callback

`CudaRenderContext` gets a sibling to `cancel_flag`: a mapped pinned
`progress_counter` (atomic int). Threads `atomicAdd(&progress_counter, 1)`
every N pixels (N tunable; start at 64 for ~16k updates per 1024² frame).

Host spawns a poller thread before `cudaDeviceSynchronize()` that reads the
counter at ~30 Hz, computes `done / total`, calls the user callback. Thread
joins after sync. Standard CUDA progress pattern; no kernel-side stdio.

### Samples per pixel

Inner loop inside the kernel over `params.samples_per_pixel`. Each iteration
jitters pixel coordinates with a hash-based RNG seeded by `(i, j, sample_idx)`.
Accumulated colors are averaged before write.

No new kernels, no constant-memory changes. Adds register pressure; same
caveat as Dormand-Prince re: `__launch_bounds__`.

### `RenderParams` extensions (single struct, additive)

```cpp
struct RenderParams {
    // ... existing fields ...
    int tile_x_offset = 0;
    int tile_y_offset = 0;
    int samples_per_pixel = 1;
    int num_frequency_bins = 0;     // 0 = RGB render path
};
```

`GRRTParams` (the C-API counterpart) gets the same three fields. CLI gets
`--tile X Y W H` and `--frequencies` (which already exists for the CPU path).

## C API additions

Two new entrypoints in `include/grrt/api.h`:

```c
// Existing tile entrypoint, today CPU-only — gains a CUDA implementation.
int grrt_render_tile(GRRTContext* ctx, float* buffer,
                     int x, int y, int tile_w, int tile_h);

// New: render spectral on CUDA, write to a FITS file at a host path.
// Takes the same frequency-bin buffer set via grrt_set_frequency_bins.
int grrt_render_spectral_to_fits_cuda(GRRTContext* ctx,
                                       const char* output_path,
                                       int width, int height,
                                       grrt_progress_fn progress,
                                       void* user_data);
```

Note: `grrt_render_spectral_to_fits_cuda` is intentionally a separate symbol
from `grrt_render_spectral_to_fits_cb` (the CPU streaming variant) because
their internals differ — the CUDA version assembles the cube in RAM then
writes it; the CPU version streams row-by-row from the renderer. Same file
format on disk.

## Python ctypes binding

### Layout

```
python/
└── grrt/
    ├── __init__.py        # public API: Params, Context, GRRTError, render
    ├── _binding.py        # ctypes loader, struct mirrors, function signatures
    └── _types.py          # MetricType, BackendType enums
```

### Public API

```python
import grrt
import numpy as np

params = grrt.Params(
    width=1024, height=1024,
    metric="kerr", spin=0.998, mass=1.0,
    observer_r=50.0, observer_theta_deg=80.0,
    fov_deg=90.0,
    disk=True, disk_outer=20.0,
    background="stars",
    backend="cuda",
    samples_per_pixel=4,
)

with grrt.Context(params) as ctx:
    img = ctx.render()                          # np.ndarray[H, W, 4], float32
    tile = ctx.render_tile(x=0, y=0, w=256, h=256)
    cube_path = ctx.render_spectral(
        frequencies_hz=np.logspace(9, 18, 100),
        output_fits="/mnt/grrt-vol/run42.fits",
    )
```

### Bindings layer

- Loads `libgrrt.so` (Linux) / `grrt.dll` (Windows) via `ctypes.CDLL`. Search
  order: `GRRT_LIB_PATH` env var → `python/grrt/_lib/` → system loader path.
- `ctypes.Structure` mirror of `GRRTParams`, hand-maintained. Field-name parity
  with the C struct enforced by a unit test that compares `sizeof` and field
  offsets where possible.
- Errors: every C call returning a status (`-1` = error) is wrapped to call
  `grrt_last_error()` and raise `grrt.GRRTError(message)`. No silent failures.
- Buffer hand-off is zero-copy: `numpy.empty((H, W, 4), dtype=np.float32)`
  passed in via `arr.ctypes.data_as(POINTER(c_float))`.
- Progress: `progress_fn` typedef wrapped as `ctypes.CFUNCTYPE(None, c_float,
  c_void_p)`. Python callable can be passed in via `Context.set_progress_fn`.
- Cancellation: `Context.cancel()` calls `grrt_cancel`. SIGTERM handler in the
  Modal function calls it so `function.cancel()` from the client propagates.

### Dependencies

`numpy` + stdlib `ctypes` only. No `cffi`, no `pybind11`, no `cython`. Per
CLAUDE.md.

## Modal integration

### Image

Defined inline in `scripts/modal_render.py`:

```python
grrt_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("cmake", "ninja-build", "g++-12", "git")
    .pip_install("numpy", "modal")
    .add_local_dir(".", remote_path="/build/src", copy=True)
    .run_commands(
        "cmake -S /build/src -B /build/out -G Ninja "
        "  -DGRRT_ENABLE_CUDA=ON "
        "  -DCMAKE_CUDA_ARCHITECTURES='80;90;100' "
        "  -DCMAKE_BUILD_TYPE=Release",
        "cmake --build /build/out --config Release -j",
    )
    .env({"GRRT_LIB_PATH": "/build/out/libgrrt.so",
          "PYTHONPATH": "/build/src/python"})
)
```

First build ~2 min (CUDA toolkit-heavy). Incremental rebuilds on source change
~30 s. Build cache is keyed on the source tarball hash by Modal automatically.

### Volume

```python
grrt_vol = modal.Volume.from_name("grrt-output", create_if_missing=True)
```

Mounted at `/mnt/grrt-vol` inside both render and download functions.

### Functions

**Render (GPU):**

```python
@app.function(
    image=grrt_image,
    gpu="A100",                       # default; override per-call
    volumes={"/mnt/grrt-vol": grrt_vol},
    timeout=3600,
)
def render(params: dict, output_name: str, spectral: bool = False) -> dict:
    import grrt
    p = grrt.Params(**params)
    with grrt.Context(p) as ctx:
        if spectral:
            path = f"/mnt/grrt-vol/{output_name}.fits"
            ctx.render_spectral(
                frequencies_hz=params["frequencies_hz"],
                output_fits=path,
            )
            grrt_vol.commit()
            return {"path": path,
                    "shape": [p.height, p.width, len(params["frequencies_hz"])]}
        else:
            img = ctx.render()
            path = f"/mnt/grrt-vol/{output_name}.npy"
            np.save(path, img)
            grrt_vol.commit()
            return {"path": path, "shape": list(img.shape)}
```

All renders write to the volume, even small RGB images. Uniform pattern. Egress
only happens on explicit download.

**Download:**

Use the Modal CLI directly — `modal volume get grrt-output run42.fits ./run42.fits`.
No download function needed in code.

### Local entrypoint

```python
@app.local_entrypoint()
def main(spin: float = 0.998, output: str = "test",
         spectral: bool = False, gpu: str = "A100"):
    # Build params dict from local args. Schema mirrors grrt.Params kwargs
    # (width, height, metric, spin, observer_r, ...). The full mapping is
    # an implementation-plan concern, not a design-doc concern.
    params = build_params_from_args(spin=spin, ...)
    result = render.with_options(gpu=gpu).remote(params, output, spectral)
    print(f"wrote {result['path']}, shape={result['shape']}")
```

GPU tier override per-call: `--gpu B200` for flagship runs. Default A100 40GB
($2.10/hr) for everyday work.

### What's NOT in scope here

- No web endpoint, scheduled job, or Modal Apps deployment. `modal run` only.
- No client-side progress streaming. Modal logs forward stdout.
- No `Cls`-based stateful sweeps (deferred; see Future work).

## Build system changes

Edits to `CMakeLists.txt`:

```cmake
# Default arch list when CUDA enabled — was "75", now multi-arch
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;90;100")
endif()

if(GRRT_ENABLE_CUDA)
    find_package(CUDAToolkit 12.8 REQUIRED)
    target_sources(grrt PRIVATE
        cuda/cuda_backend.cu
        cuda/cuda_render.cu
        cuda/cuda_render_spectral.cu     # NEW
        cuda/cuda_vol_host_data.cpp
    )
endif()
```

Rationale:

- `80;90;100` covers Modal's FP64-strong tiers (A100, H100/H200, B200).
  Override with `-DCMAKE_CUDA_ARCHITECTURES=75` for a local RTX 2080 build.
- `CUDAToolkit 12.8 REQUIRED` fails fast at configure time if the toolkit is
  too old for SM 100. Friendlier than a cryptic nvcc error.
- `cuda_render_spectral.cu` is a new TU; same constant-memory ownership pattern
  as `cuda_render.cu`.

Untouched: MSVC compile flags, OpenMP setup, CLI executable.

## Validation harness

Two layers, both must pass before declaring parity done.

### Live `--validate` (extended)

Existing CLI flag, extended:

- `--validate` — RGB render, both backends, pixel MAE on linear-HDR
  framebuffer.
- `--validate-spectral` — cube-vs-cube MAE per frequency bin.
- `--validate-tile` — full-frame vs. stitched-tile, must be byte-identical.

All three share one comparator helper in `cli/main.cpp`. Output: `mean=X,
max=Y, p99=Z, pass=N/Y`.

Tolerance: linear-HDR MAE < 1e-4 across the image, max-pixel diff < 1e-3.
Loose enough for FP non-associativity (CUDA reduction order ≠ CPU); tight
enough to catch real drift. **These are starting values** — re-calibrate
empirically once both backends produce a baseline image, since the realistic
drift floor depends on Dormand-Prince's step-acceptance behavior on each
backend's FP unit.

### Test suite (`tests/test_cuda_parity.cpp`)

New CTest target. Five canonical scenes, each <30 s on an A100:

| # | Scene | Tests |
|---|---|---|
| 1 | Schwarzschild edge-on, 256², stars off, no disk | Geodesic integration, photon ring shape |
| 2 | Kerr a=0.998, θ=80°, thin disk | Disk redshift + Doppler, metric derivatives |
| 3 | Kerr a=0.7, edge-on, volumetric disk, low res | Volumetric raymarch, opacity LUTs, noise |
| 4 | Kerr face-on, 4-bin spectral | Spectral path, frequency-bin loop |
| 5 | Kerr edge-on, 1024² full vs. 4×4 tiles of 256² | Tile parity (MAE = 0) |

Skipping rule: detects `cuda_available()` at runtime, SKIPs when no GPU is
present. Test suite is a no-op on Windows local; runs full-fat on Modal.

### Explicitly NOT validated

- Tonemapped output bytes — tonemapping happens after framebuffer comparison.
  We test the physics, not the post-process.
- Star positions on the celestial sphere when count is large — order-dependent
  reduction, covered by image MAE which is order-invariant.
- Per-step debug-pixel output between CPU and CUDA — debug-pixel stays
  CPU-only.

## Future work

Items deliberately deferred. Out of scope for this design but documented so
they're not lost.

### 1. Setup/render split in CUDA backend

Today every `cuda_render` call re-uploads all LUTs (color, luminosity, flux,
stars, volumetric, noise) — hundreds of MB cumulatively for parameter sweeps.
The refactor:

- **`cuda_context_setup(ctx, params)`** — uploads scene LUTs once, at context
  create.
- **`cuda_render(ctx, params, framebuffer)`** — uploads `RenderParams` and
  camera tetrad only, launches kernel, downloads pixels.
- **`cuda_update_params(ctx, params)`** — re-uploads only the LUTs that
  changed when scene parameters mutate between calls.

Touches ~80 lines in `cuda/cuda_backend.cu` and `cuda/cuda_render.cu`.
Prerequisite for stateful Modal `Cls`. Take this on when starting
parameter-sweep / animation work.

### 2. Streaming-FITS spectral on CUDA

Per-row callback from a kernel mid-execution requires either splitting the
kernel by row chunks with sync points, or device-side `cudaStreamAddCallback`
to dump rows. Both add complexity for a feature only used in memory-constrained
cubes. Cross this bridge when you actually OOM on a render.

### 3. Real-time progress streaming back to client

Modal's log forwarding works for "watch a long render", but isn't push-style
progress. Future work for the Blender plugin.

### 4. Modal `Cls` for stateful sweeps

Depends on (1). Big win for parameter sweeps and animation: amortize hundreds
of MB of LUT uploads across many renders.

### 5. CI on Modal

Wire `tests/test_cuda_parity.cpp` into a `@app.function(gpu="A100")` that
runs nightly. Want this once a stable baseline exists; not before, or you're
chasing your own changes.

### 6. Codegen ctypes binding

If `GRRTParams` starts churning, replace the hand-maintained `_binding.py`
struct with a generator script driven by clang's AST or regex over `types.h`.
Until that pain point is real, hand-maintain.

## Validation invariants (from CLAUDE.md, restated)

These must hold throughout the work:

- Hamiltonian constraint `H ≈ 0` below ~1e-10 in integration.
- `E = -p_t`, `L = p_φ` constant along Schwarzschild geodesics.
- Schwarzschild shadow radius = `arcsin(3√3 M / r_obs)`.
- Disk Doppler: approaching side brighter than receding.
- CUDA output matches CPU within FP tolerance.
- Tile output identical to corresponding region of full-frame output.

## Open questions

None at design time. Recorded here so reviewers know it was checked, not
forgotten.
