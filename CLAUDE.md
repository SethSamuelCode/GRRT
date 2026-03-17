# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

General relativistic raytracer: a C++23 shared library (`grrt.dll`/`libgrrt.so`) that traces photon geodesics through curved spacetime (Schwarzschild and Kerr metrics) to render black hole visualizations. Integrates into Blender 5.x as a custom render engine plugin.

Three phases: (1) Core C++ library + CLI, (2) CUDA acceleration, (3) Blender render engine plugin.

## Build Commands

```bash
# Configure and build (Windows with Visual Studio 2022)
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release

# Configure and build (Linux/generic)
cmake -B build
cmake --build build

# Run CLI
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --output test.png

# Build with CUDA (Phase 2, requires CUDA Toolkit 12.x)
cmake -B build -DCMAKE_CUDA_COMPILER=nvcc
cmake --build build --config Release
```

## Architecture

### Layers (bottom-up)

- **Math** (`include/grrt/math/`): Header-only `Vec3`, `Vec4`, `Matrix4`. `Vec4` indexes as `(t, r, θ, φ)`. `Matrix4` is 4x4 symmetric for metric tensors, exploits sparsity for diagonal/block-diagonal metrics.
- **Spacetime** (`include/grrt/spacetime/`): `Metric` abstract interface → `Schwarzschild`, `Kerr` implementations. All in Boyer-Lindquist coordinates. Provides `g_μν`, `g^μν`, horizon/ISCO radii.
- **Geodesic** (`include/grrt/geodesic/`): Hamiltonian formulation for null geodesics. State vector = 8 components `(x^μ, p_μ)`. RK4 with adaptive step doubling (`dlambda ∝ r²`). Metric derivatives via central finite differences on inverse metric.
- **Camera** (`include/grrt/camera/`): Observer with local orthonormal tetrad (Gram-Schmidt). Maps pixel → initial covariant 4-momentum. Supports yaw/pitch/roll offsets for Blender camera control.
- **Scene** (`include/grrt/scene/`): Thin equatorial accretion disk (circular orbits, blackbody emission, redshifted). Celestial sphere (black/stars/texture). `Scene::trace_ray()` aggregates all objects.
- **Render** (`include/grrt/render/`): OpenMP-parallelized pixel loop. Supports full-frame and tiled rendering. Cancellation via `atomic<bool>`, progress via `atomic<float>`. Tone mapping (Reinhard/ACES).
- **Color** (`include/grrt/color/`): Redshift → color via blackbody spectrum + CIE color matching → XYZ → sRGB. Precomputed lookup tables.

### Key boundaries

- **C API** (`include/grrt/api.h`): Pure C interface (`grrt_create`, `grrt_render`, `grrt_render_tile`, `grrt_cancel`, `grrt_destroy`). All C++ is internal. No exceptions cross the DLL boundary.
- **Python binding** (`python/grrt_binding.py`): ctypes wrapper around the C API. No external Python dependencies beyond ctypes and numpy.
- **Blender plugin** (`python/`): `GRRTRenderEngine` subclass handles F12 render and viewport preview. `camera_convert.py` maps Blender Cartesian camera → Boyer-Lindquist observer coordinates.

### CUDA backend (Phase 2)

Lives in `cuda/`. One thread per pixel. No virtual functions on device — uses plain structs and enum+switch. Double precision throughout (FP32 accumulates too much error near horizon). Kernel is templated or switched on metric type.

## Coding Conventions

- **C++23**: Use modern features (`<numbers>`, `std::format`, `std::print`, structured bindings, etc.)
- **Units**: Geometrized units (`G = c = 1`) throughout; `M` sets the length scale
- **Memory**: No raw `new`/`delete`; use `std::unique_ptr` or stack allocation
- **Const correctness**: Mark everything `const` that can be
- **Error handling**: Exceptions internally, error codes at C API boundary
- **Python**: Blender addon conventions, PEP 8, no external deps beyond ctypes/numpy
- **Physics docs**: Every class/method gets a Doxygen comment explaining the physics

## Validation Invariants

- Hamiltonian constraint `H ≈ 0` must stay below ~1e-10 throughout integration
- Conserved quantities `E = -p_t`, `L = p_φ` constant along Schwarzschild geodesics
- Schwarzschild shadow radius = `arcsin(3√3 M / r_obs)`
- Disk Doppler beaming: approaching side brighter than receding
- CUDA output must match CPU output within floating-point tolerance
- Tile rendering must produce identical output to full-frame rendering

## Target Platform

Primary: Windows (Visual Studio 2022 / CMake). Must also compile on Linux. Blender plugin targets Blender 5.x.

## Third-Party Dependencies

- `stb_image_write.h` — vendored in `third_party/`, public domain
- OpenMP — optional CPU parallelism
- CUDA Toolkit 12.x — optional GPU backend (Phase 2)
- Blender 5.x — host application for Phase 3 plugin
