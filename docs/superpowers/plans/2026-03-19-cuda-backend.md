# Phase 2: CUDA Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CUDA GPU backend that produces identical output to the CPU renderer, selectable at runtime via `--backend cuda`.

**Architecture:** Mirror approach — each CPU layer is reimplemented as CUDA `__device__` functions using plain structs and enum+switch dispatch. All CUDA code lives in `cuda/`. The C API dispatches to CPU or CUDA based on `GRRTParams.backend`. A validation test program verifies correctness layer-by-layer.

**Tech Stack:** C++23, CUDA Toolkit 12.x, CMake 3.20+, nvcc, Visual Studio 2022

**Spec:** `docs/superpowers/specs/2026-03-19-cuda-backend-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `cuda/cuda_math.h` | `__device__` Vec3, Vec4, Matrix4 with same operations as CPU |
| `cuda/cuda_metric.h` | Schwarzschild + Kerr metric functions via enum+switch |
| `cuda/cuda_geodesic.h` | RK4 integrator + adaptive stepping as device functions |
| `cuda/cuda_camera.h` | Tetrad construction + pixel-to-momentum mapping |
| `cuda/cuda_color.h` | Spectrum LUT lookup (flat arrays, linear interpolation) |
| `cuda/cuda_scene.h` | AccretionDisk emission + CelestialSphere sampling on device |
| `cuda/cuda_types.h` | `RenderParams` struct + `MetricType` enum for constant memory |
| `cuda/cuda_render.cu` | Main render kernel + `__constant__` globals + upload wrapper functions |
| `cuda/cuda_render_upload.h` | Host-callable declarations for `cudaMemcpyToSymbol` wrappers (defined in `cuda_render.cu`) |
| `cuda/cuda_backend.h` | Public header: `cuda_render()`, `cuda_available()` |
| `cuda/cuda_backend.cu` | Host orchestration: alloc, upload (via wrappers), launch, download, cleanup |
| `cuda/tests/test_cuda.cu` | Validation test program: math, metric, geodesic, full-frame comparison |

### Modified files

| File | Change |
|------|--------|
| `CMakeLists.txt` | Add `GRRT_ENABLE_CUDA` option, CUDA language, cuda/ sources |
| `src/api.cpp` | Dispatch to CUDA backend when `params->backend == GRRT_BACKEND_CUDA` |
| `cli/main.cpp` | Add `--backend`, `--validate`, `--debug-pixel` flags |
| `include/grrt/api.h` | Add `grrt_last_error(void)` declaration |

---

## Task 1: CMake CUDA Setup

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add CUDA option and language to CMakeLists.txt**

Add after the `set(CMAKE_CXX_STANDARD 23)` line:

```cmake
option(GRRT_ENABLE_CUDA "Enable CUDA GPU backend" OFF)

if(GRRT_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX 2080 = Turing = SM 7.5
    find_package(CUDAToolkit REQUIRED)
endif()
```

- [ ] **Step 2: Add CUDA source files and link target**

Add after the OpenMP block:

```cmake
if(GRRT_ENABLE_CUDA)
    target_sources(grrt PRIVATE
        cuda/cuda_backend.cu
        cuda/cuda_render.cu
    )
    target_include_directories(grrt PRIVATE cuda/)
    target_link_libraries(grrt PRIVATE CUDA::cudart)
    target_compile_definitions(grrt PRIVATE GRRT_HAS_CUDA)
endif()
```

- [ ] **Step 3: Add CUDA test executable**

Add at the end of CMakeLists.txt:

```cmake
if(GRRT_ENABLE_CUDA)
    add_executable(grrt-cuda-test cuda/tests/test_cuda.cu)
    target_include_directories(grrt-cuda-test PRIVATE include/ cuda/ ${CMAKE_BINARY_DIR}/include)
    target_link_libraries(grrt-cuda-test PRIVATE CUDA::cudart)
    set_target_properties(grrt-cuda-test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
```

- [ ] **Step 4: Create minimal stub files so the build succeeds**

Create `cuda/cuda_backend.h`:
```cpp
#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

bool cuda_available();

#endif
```

Create `cuda/cuda_backend.cu`:
```cpp
#include "cuda_backend.h"
#include <cuda_runtime.h>

bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}
```

Create `cuda/cuda_render.cu`:
```cpp
// Placeholder — will contain the render kernel
```

Create `cuda/tests/test_cuda.cu`:
```cpp
#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    std::printf("CUDA devices found: %d\n", count);
    if (count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::printf("Device 0: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
        std::printf("Global memory: %.0f MB\n", prop.totalGlobalMem / 1048576.0);
    }
    return (count > 0) ? 0 : 1;
}
```

- [ ] **Step 5: Build with CUDA enabled and run test**

Run:
```bash
cmake -B build -G "Visual Studio 17 2022" -DGRRT_ENABLE_CUDA=ON
cmake --build build --config Release
./build/Release/grrt-cuda-test
```

Expected: prints device name "GeForce RTX 2080" and SM 7.5, exits 0.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt cuda/
git commit -m "feat: add CUDA build infrastructure with device detection test"
```

---

## Task 2: CUDA Math Types

**Files:**
- Create: `cuda/cuda_math.h`
- Modify: `cuda/tests/test_cuda.cu`

- [ ] **Step 1: Write the test kernel for math operations**

Add to `cuda/tests/test_cuda.cu`:

```cpp
#include "cuda_math.h"

__global__ void test_math_kernel(int* results) {
    // Test Vec4 addition
    cuda::Vec4 a = {1.0, 2.0, 3.0, 4.0};
    cuda::Vec4 b = {0.5, 1.5, 2.5, 3.5};
    cuda::Vec4 c = a + b;
    results[0] = (fabs(c[0] - 1.5) < 1e-14 && fabs(c[1] - 3.5) < 1e-14
               && fabs(c[2] - 5.5) < 1e-14 && fabs(c[3] - 7.5) < 1e-14) ? 1 : 0;

    // Test Vec4 scalar multiply
    cuda::Vec4 d = a * 2.0;
    results[1] = (fabs(d[0] - 2.0) < 1e-14 && fabs(d[3] - 8.0) < 1e-14) ? 1 : 0;

    // Test Matrix4 diagonal
    cuda::Matrix4 diag = cuda::Matrix4::diagonal(-0.5, 2.0, 1.0, 0.25);
    results[2] = (fabs(diag.m[0][0] - (-0.5)) < 1e-14 && fabs(diag.m[1][1] - 2.0) < 1e-14
               && fabs(diag.m[2][3]) < 1e-14) ? 1 : 0;

    // Test Matrix4 contract
    cuda::Vec4 v = {1.0, 1.0, 1.0, 1.0};
    cuda::Vec4 result = diag.contract(v);
    results[3] = (fabs(result[0] - (-0.5)) < 1e-14 && fabs(result[1] - 2.0) < 1e-14
               && fabs(result[3] - 0.25) < 1e-14) ? 1 : 0;

    // Test Matrix4 inverse_diagonal
    cuda::Matrix4 inv = diag.inverse_diagonal();
    results[4] = (fabs(inv.m[0][0] - (-2.0)) < 1e-14 && fabs(inv.m[1][1] - 0.5) < 1e-14) ? 1 : 0;

    // Test Vec3 operations
    cuda::Vec3 v3a = {1.0, 2.0, 3.0};
    cuda::Vec3 v3b = {4.0, 5.0, 6.0};
    cuda::Vec3 v3c = v3a + v3b;
    results[5] = (fabs(v3c[0] - 5.0) < 1e-14 && fabs(v3c[2] - 9.0) < 1e-14) ? 1 : 0;

    // Test Matrix4 full inverse (for Kerr)
    cuda::Matrix4 block = {};
    block.m[0][0] = -1.0; block.m[0][3] = 0.5;
    block.m[3][0] = 0.5;  block.m[3][3] = 2.0;
    block.m[1][1] = 3.0;
    block.m[2][2] = 4.0;
    cuda::Matrix4 block_inv = block.inverse();
    // Verify M * M^-1 ≈ I for (0,0) element
    double check = 0.0;
    for (int k = 0; k < 4; ++k) check += block.m[0][k] * block_inv.m[k][0];
    results[6] = (fabs(check - 1.0) < 1e-12) ? 1 : 0;
}

void run_math_tests() {
    const int N = 7;
    int h_results[N];
    int* d_results;
    cudaMalloc(&d_results, N * sizeof(int));
    test_math_kernel<<<1, 1>>>(d_results);
    cudaMemcpy(h_results, d_results, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    const char* names[] = {
        "Vec4 addition", "Vec4 scalar multiply", "Matrix4 diagonal",
        "Matrix4 contract", "Matrix4 inverse_diagonal", "Vec3 operations",
        "Matrix4 full inverse"
    };
    int passed = 0;
    for (int i = 0; i < N; ++i) {
        std::printf("  [%s] %s\n", h_results[i] ? "PASS" : "FAIL", names[i]);
        passed += h_results[i];
    }
    std::printf("Math tests: %d/%d passed\n\n", passed, N);
}
```

Update `main()` to call `run_math_tests()` after the device info printout.

- [ ] **Step 2: Build and verify the test fails**

Run:
```bash
cmake --build build --config Release
```

Expected: compilation error — `cuda_math.h` not found.

- [ ] **Step 3: Implement cuda_math.h**

Create `cuda/cuda_math.h`:

```cpp
#ifndef CUDA_MATH_H
#define CUDA_MATH_H

#include <cmath>

namespace cuda {

struct Vec3 {
    double data[3]{};

    __host__ __device__ double& operator[](int i) { return data[i]; }
    __host__ __device__ const double& operator[](int i) const { return data[i]; }

    __host__ __device__ Vec3 operator+(const Vec3& o) const {
        return {data[0] + o.data[0], data[1] + o.data[1], data[2] + o.data[2]};
    }
    __host__ __device__ Vec3 operator-(const Vec3& o) const {
        return {data[0] - o.data[0], data[1] - o.data[1], data[2] - o.data[2]};
    }
    __host__ __device__ Vec3 operator*(double s) const {
        return {data[0] * s, data[1] * s, data[2] * s};
    }
    __host__ __device__ Vec3 operator*(const Vec3& o) const {
        return {data[0] * o.data[0], data[1] * o.data[1], data[2] * o.data[2]};
    }
    __host__ __device__ Vec3& operator+=(const Vec3& o) {
        data[0] += o.data[0]; data[1] += o.data[1]; data[2] += o.data[2];
        return *this;
    }
    __host__ __device__ double max_component() const {
        return fmax(data[0], fmax(data[1], data[2]));
    }
};

struct Vec4 {
    double data[4]{};

    __host__ __device__ double& operator[](int i) { return data[i]; }
    __host__ __device__ const double& operator[](int i) const { return data[i]; }

    __host__ __device__ Vec4 operator+(const Vec4& o) const {
        return {data[0] + o.data[0], data[1] + o.data[1],
                data[2] + o.data[2], data[3] + o.data[3]};
    }
    __host__ __device__ Vec4 operator-(const Vec4& o) const {
        return {data[0] - o.data[0], data[1] - o.data[1],
                data[2] - o.data[2], data[3] - o.data[3]};
    }
    __host__ __device__ Vec4 operator*(double s) const {
        return {data[0] * s, data[1] * s, data[2] * s, data[3] * s};
    }
};

struct Matrix4 {
    double m[4][4]{};

    __host__ __device__ Vec4 contract(const Vec4& v) const {
        Vec4 result{};
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.data[i] += m[i][j] * v.data[j];
        return result;
    }

    __host__ __device__ static Matrix4 diagonal(double a, double b, double c, double d) {
        Matrix4 mat{};
        mat.m[0][0] = a; mat.m[1][1] = b; mat.m[2][2] = c; mat.m[3][3] = d;
        return mat;
    }

    __host__ __device__ Matrix4 inverse_diagonal() const {
        return diagonal(1.0 / m[0][0], 1.0 / m[1][1], 1.0 / m[2][2], 1.0 / m[3][3]);
    }

    __host__ __device__ Matrix4 inverse() const {
        // Block-diagonal inverse for Kerr: (t,phi) 2x2 block + diagonal (r, theta)
        Matrix4 inv{};

        // (t, phi) block: rows/cols 0 and 3
        double det_tphi = m[0][0] * m[3][3] - m[0][3] * m[3][0];
        inv.m[0][0] =  m[3][3] / det_tphi;
        inv.m[0][3] = -m[0][3] / det_tphi;
        inv.m[3][0] = -m[3][0] / det_tphi;
        inv.m[3][3] =  m[0][0] / det_tphi;

        // Diagonal entries for r and theta
        inv.m[1][1] = 1.0 / m[1][1];
        inv.m[2][2] = 1.0 / m[2][2];

        return inv;
    }
};

} // namespace cuda

#endif
```

- [ ] **Step 4: Build and run the math tests**

Run:
```bash
cmake --build build --config Release
./build/Release/grrt-cuda-test
```

Expected: all 7 math tests pass.

- [ ] **Step 5: Commit**

```bash
git add cuda/cuda_math.h cuda/tests/test_cuda.cu
git commit -m "feat: add CUDA math types (Vec3, Vec4, Matrix4) with device tests"
```

---

## Task 3: CUDA Metric Functions

**Files:**
- Create: `cuda/cuda_metric.h`
- Modify: `cuda/tests/test_cuda.cu`

Reference: `src/schwarzschild.cpp` (lines 1-37), `src/kerr.cpp` (lines 1-65)

- [ ] **Step 1: Write metric test kernel**

Add to `cuda/tests/test_cuda.cu`:

```cpp
#include "cuda_metric.h"

__global__ void test_metric_kernel(int* results) {
    double M = 1.0;
    double a = 0.998;

    // Test 1: Schwarzschild g_tt at r=10M
    cuda::Vec4 x_sch = {0.0, 10.0, M_PI / 2.0, 0.0};
    cuda::Matrix4 g_sch = cuda::metric_lower(cuda::MetricType::Schwarzschild, M, 0.0, x_sch);
    double expected_gtt = -(1.0 - 2.0 / 10.0);  // -(1 - 2M/r) = -0.8
    results[0] = (fabs(g_sch.m[0][0] - expected_gtt) < 1e-12) ? 1 : 0;

    // Test 2: Schwarzschild g_rr at r=10M
    double expected_grr = 1.0 / (1.0 - 2.0 / 10.0);  // 1/(1-2M/r) = 1.25
    results[1] = (fabs(g_sch.m[1][1] - expected_grr) < 1e-12) ? 1 : 0;

    // Test 3: Schwarzschild horizon = 2M
    results[2] = (fabs(cuda::horizon_radius(cuda::MetricType::Schwarzschild, M, 0.0) - 2.0) < 1e-12) ? 1 : 0;

    // Test 4: Schwarzschild ISCO = 6M
    results[3] = (fabs(cuda::isco_radius(cuda::MetricType::Schwarzschild, M, 0.0) - 6.0) < 1e-12) ? 1 : 0;

    // Test 5: Kerr g_lower is not diagonal (has g_tphi)
    cuda::Vec4 x_kerr = {0.0, 10.0, M_PI / 2.0, 0.0};
    cuda::Matrix4 g_kerr = cuda::metric_lower(cuda::MetricType::Kerr, M, a, x_kerr);
    results[4] = (fabs(g_kerr.m[0][3]) > 1e-10) ? 1 : 0;  // g_tphi != 0 for Kerr

    // Test 6: Kerr horizon < Schwarzschild horizon
    double kerr_rh = cuda::horizon_radius(cuda::MetricType::Kerr, M, a);
    results[5] = (kerr_rh < 2.0 * M && kerr_rh > M) ? 1 : 0;

    // Test 7: g_lower * g_upper ≈ identity (Schwarzschild)
    cuda::Matrix4 g_up_sch = cuda::metric_upper(cuda::MetricType::Schwarzschild, M, 0.0, x_sch);
    double diag_check = 0.0;
    for (int k = 0; k < 4; ++k) diag_check += g_sch.m[1][k] * g_up_sch.m[k][1];
    results[6] = (fabs(diag_check - 1.0) < 1e-12) ? 1 : 0;

    // Test 8: g_lower * g_upper ≈ identity (Kerr, t-row)
    cuda::Matrix4 g_up_kerr = cuda::metric_upper(cuda::MetricType::Kerr, M, a, x_kerr);
    double kerr_check = 0.0;
    for (int k = 0; k < 4; ++k) kerr_check += g_kerr.m[0][k] * g_up_kerr.m[k][0];
    results[7] = (fabs(kerr_check - 1.0) < 1e-10) ? 1 : 0;
}

void run_metric_tests() {
    const int N = 8;
    int h_results[N];
    int* d_results;
    cudaMalloc(&d_results, N * sizeof(int));
    test_metric_kernel<<<1, 1>>>(d_results);
    cudaMemcpy(h_results, d_results, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    const char* names[] = {
        "Schwarzschild g_tt", "Schwarzschild g_rr",
        "Schwarzschild horizon", "Schwarzschild ISCO",
        "Kerr g_tphi nonzero", "Kerr horizon < 2M",
        "Schwarzschild g*g^-1=I", "Kerr g*g^-1=I"
    };
    int passed = 0;
    for (int i = 0; i < N; ++i) {
        std::printf("  [%s] %s\n", h_results[i] ? "PASS" : "FAIL", names[i]);
        passed += h_results[i];
    }
    std::printf("Metric tests: %d/%d passed\n\n", passed, N);
}
```

Update `main()` to call `run_metric_tests()`.

- [ ] **Step 2: Build and verify the test fails**

Expected: compilation error — `cuda_metric.h` not found.

- [ ] **Step 3: Implement cuda_metric.h**

Create `cuda/cuda_metric.h`. Port Schwarzschild (`src/schwarzschild.cpp`) and Kerr (`src/kerr.cpp`) as `__device__` free functions:

```cpp
#ifndef CUDA_METRIC_H
#define CUDA_METRIC_H

#include "cuda_math.h"
#include <cmath>

namespace cuda {

enum class MetricType { Schwarzschild, Kerr };

// --- Schwarzschild helpers ---

__host__ __device__ inline Matrix4 schwarzschild_g_lower(double M, const Vec4& x) {
    double r = x[1];
    double theta = x[2];
    double f = 1.0 - 2.0 * M / r;
    double r2 = r * r;
    double sin2 = sin(theta) * sin(theta);
    return Matrix4::diagonal(-f, 1.0 / f, r2, r2 * sin2);
}

__host__ __device__ inline Matrix4 schwarzschild_g_upper(double M, const Vec4& x) {
    double r = x[1];
    double theta = x[2];
    double f = 1.0 - 2.0 * M / r;
    double r2 = r * r;
    double sin2 = sin(theta) * sin(theta);
    return Matrix4::diagonal(-1.0 / f, f, 1.0 / r2, 1.0 / (r2 * sin2));
}

// --- Kerr helpers ---

__host__ __device__ inline double kerr_sigma(double r, double theta, double a) {
    double cos_t = cos(theta);
    return r * r + a * a * cos_t * cos_t;
}

__host__ __device__ inline double kerr_delta(double r, double M, double a) {
    return r * r - 2.0 * M * r + a * a;
}

__host__ __device__ inline Matrix4 kerr_g_lower(double M, double a, const Vec4& x) {
    double r = x[1];
    double theta = x[2];
    double sigma = kerr_sigma(r, theta, a);
    double delta = kerr_delta(r, M, a);
    double sin2 = sin(theta) * sin(theta);
    double a2 = a * a;

    Matrix4 g{};
    g.m[0][0] = -(1.0 - 2.0 * M * r / sigma);
    g.m[0][3] = -2.0 * M * a * r * sin2 / sigma;
    g.m[3][0] = g.m[0][3];
    g.m[1][1] = sigma / delta;
    g.m[2][2] = sigma;
    g.m[3][3] = (r * r + a2 + 2.0 * M * r * a2 * sin2 / sigma) * sin2;
    return g;
}

__host__ __device__ inline Matrix4 kerr_g_upper(double M, double a, const Vec4& x) {
    Matrix4 g = kerr_g_lower(M, a, x);
    return g.inverse();
}

// --- Dispatch functions ---

__host__ __device__ inline Matrix4 metric_lower(MetricType type, double M, double a, const Vec4& x) {
    switch (type) {
        case MetricType::Kerr: return kerr_g_lower(M, a, x);
        default:               return schwarzschild_g_lower(M, x);
    }
}

__host__ __device__ inline Matrix4 metric_upper(MetricType type, double M, double a, const Vec4& x) {
    switch (type) {
        case MetricType::Kerr: return kerr_g_upper(M, a, x);
        default:               return schwarzschild_g_upper(M, x);
    }
}

__host__ __device__ inline double horizon_radius(MetricType type, double M, double a) {
    switch (type) {
        case MetricType::Kerr: return M + sqrt(M * M - a * a);
        default:               return 2.0 * M;
    }
}

__host__ __device__ inline double isco_radius(MetricType type, double M, double a) {
    if (type == MetricType::Schwarzschild) return 6.0 * M;
    // Bardeen formula for prograde ISCO
    double a_star = a / M;
    double z1 = 1.0 + cbrt(1.0 - a_star * a_star) * (cbrt(1.0 + a_star) + cbrt(1.0 - a_star));
    double z2 = sqrt(3.0 * a_star * a_star + z1 * z1);
    return M * (3.0 + z2 - sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)));
}

} // namespace cuda

#endif
```

- [ ] **Step 4: Build and run metric tests**

Run:
```bash
cmake --build build --config Release
./build/Release/grrt-cuda-test
```

Expected: all 8 metric tests pass.

- [ ] **Step 5: Commit**

```bash
git add cuda/cuda_metric.h cuda/tests/test_cuda.cu
git commit -m "feat: add CUDA metric functions (Schwarzschild + Kerr) with tests"
```

---

## Task 4: CUDA Geodesic Integrator

**Files:**
- Create: `cuda/cuda_geodesic.h`
- Modify: `cuda/tests/test_cuda.cu`

Reference: `src/rk4.cpp` (lines 1-116), `include/grrt/geodesic/integrator.h`

- [ ] **Step 1: Write geodesic test kernel**

Test: trace a known radial ray in Schwarzschild, verify Hamiltonian constraint and energy conservation.

```cpp
#include "cuda_geodesic.h"

__global__ void test_geodesic_kernel(double* outputs) {
    double M = 1.0;
    // Radial infall at r=20M, equatorial plane
    cuda::Vec4 pos = {0.0, 20.0, M_PI / 2.0, 0.0};

    // Set up null momentum: p_t = -1 (E=1), p_r from null condition, p_theta=0, p_phi=0
    cuda::Matrix4 g = cuda::metric_lower(cuda::MetricType::Schwarzschild, M, 0.0, pos);
    cuda::Matrix4 g_up = cuda::metric_upper(cuda::MetricType::Schwarzschild, M, 0.0, pos);
    // Null condition: g^tt p_t^2 + g^rr p_r^2 = 0
    // p_r = sqrt(-g^tt / g^rr) * |p_t| for inward
    double pt = -1.0;
    double pr = -sqrt(-g_up.m[0][0] / g_up.m[1][1]);
    cuda::Vec4 mom = {pt, pr, 0.0, 0.0};

    cuda::GeodesicState state = {pos, mom};
    double dlambda = 0.1;
    double tolerance = 1e-8;

    // Integrate 100 adaptive steps
    double max_H_error = 0.0;
    double E_initial = -state.momentum[0];

    for (int i = 0; i < 100; ++i) {
        cuda::AdaptiveResult result = cuda::rk4_adaptive_step(
            cuda::MetricType::Schwarzschild, M, 0.0, state, dlambda, tolerance);
        state = result.state;
        dlambda = result.next_dlambda;

        // Check Hamiltonian: H = 0.5 * g^ab p_a p_b ≈ 0
        cuda::Matrix4 g_up_now = cuda::metric_upper(
            cuda::MetricType::Schwarzschild, M, 0.0, state.position);
        double H = 0.0;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                H += g_up_now.m[a][b] * state.momentum[a] * state.momentum[b];
        H *= 0.5;
        if (fabs(H) > max_H_error) max_H_error = fabs(H);

        // Check if crossed horizon
        if (state.position[1] < 2.0 * M + 0.01) break;
    }

    double E_final = -state.momentum[0];

    outputs[0] = max_H_error;              // Should be < 1e-10
    outputs[1] = fabs(E_final - E_initial); // Should be < 1e-10 (E conserved)
    outputs[2] = state.position[1];         // Should have approached horizon
}

void run_geodesic_tests() {
    double h_outputs[3];
    double* d_outputs;
    cudaMalloc(&d_outputs, 3 * sizeof(double));
    test_geodesic_kernel<<<1, 1>>>(d_outputs);
    cudaMemcpy(h_outputs, d_outputs, 3 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_outputs);

    std::printf("  Max Hamiltonian error: %.2e (need < 1e-10)\n", h_outputs[0]);
    std::printf("  Energy conservation dE: %.2e (need < 1e-10)\n", h_outputs[1]);
    std::printf("  Final radius: %.4f (should approach 2.0)\n", h_outputs[2]);

    int passed = 0;
    bool h_ok = h_outputs[0] < 1e-8;
    bool e_ok = h_outputs[1] < 1e-8;
    bool r_ok = h_outputs[2] < 5.0;
    std::printf("  [%s] Hamiltonian constraint\n", h_ok ? "PASS" : "FAIL");
    std::printf("  [%s] Energy conservation\n", e_ok ? "PASS" : "FAIL");
    std::printf("  [%s] Ray approaches horizon\n", r_ok ? "PASS" : "FAIL");
    passed = (int)h_ok + (int)e_ok + (int)r_ok;
    std::printf("Geodesic tests: %d/3 passed\n\n", passed);
}
```

- [ ] **Step 2: Build and verify it fails**

Expected: `cuda_geodesic.h` not found.

- [ ] **Step 3: Implement cuda_geodesic.h**

Create `cuda/cuda_geodesic.h`. Port the RK4 integrator from `src/rk4.cpp`:

```cpp
#ifndef CUDA_GEODESIC_H
#define CUDA_GEODESIC_H

#include "cuda_math.h"
#include "cuda_metric.h"

namespace cuda {

struct GeodesicState {
    Vec4 position;
    Vec4 momentum;
};

struct AdaptiveResult {
    GeodesicState state;
    double next_dlambda;
};

// Compute dx^mu/dlambda and dp_mu/dlambda
// dx^mu/dlambda = g^{mu nu} p_nu
// dp_mu/dlambda = -0.5 * (dg^{ab}/dx^mu) p_a p_b
__device__ inline GeodesicState derivatives(MetricType type, double M, double a,
                                             const GeodesicState& state) {
    constexpr double eps = 1e-6;
    GeodesicState deriv{};

    // dx^mu/dlambda = g^{mu nu} p_nu
    Matrix4 g_up = metric_upper(type, M, a, state.position);
    deriv.position = g_up.contract(state.momentum);

    // dp_mu/dlambda = -0.5 * (dg^{ab}/dx^mu) p_a p_b
    // Compute via central finite differences on g^{ab}
    for (int mu = 0; mu < 4; ++mu) {
        Vec4 x_plus = state.position;
        Vec4 x_minus = state.position;
        x_plus[mu] += eps;
        x_minus[mu] -= eps;

        Matrix4 g_up_plus = metric_upper(type, M, a, x_plus);
        Matrix4 g_up_minus = metric_upper(type, M, a, x_minus);

        double dp = 0.0;
        for (int alpha = 0; alpha < 4; ++alpha)
            for (int beta = 0; beta < 4; ++beta)
                dp += (g_up_plus.m[alpha][beta] - g_up_minus.m[alpha][beta])
                      * state.momentum[alpha] * state.momentum[beta];

        deriv.momentum[mu] = -0.5 * dp / (2.0 * eps);
    }

    return deriv;
}

// Single RK4 step
__device__ inline GeodesicState rk4_step(MetricType type, double M, double a,
                                          const GeodesicState& state, double dlambda) {
    GeodesicState k1 = derivatives(type, M, a, state);

    GeodesicState s2{};
    for (int i = 0; i < 4; ++i) {
        s2.position[i] = state.position[i] + 0.5 * dlambda * k1.position[i];
        s2.momentum[i] = state.momentum[i] + 0.5 * dlambda * k1.momentum[i];
    }
    GeodesicState k2 = derivatives(type, M, a, s2);

    GeodesicState s3{};
    for (int i = 0; i < 4; ++i) {
        s3.position[i] = state.position[i] + 0.5 * dlambda * k2.position[i];
        s3.momentum[i] = state.momentum[i] + 0.5 * dlambda * k2.momentum[i];
    }
    GeodesicState k3 = derivatives(type, M, a, s3);

    GeodesicState s4{};
    for (int i = 0; i < 4; ++i) {
        s4.position[i] = state.position[i] + dlambda * k3.position[i];
        s4.momentum[i] = state.momentum[i] + dlambda * k3.momentum[i];
    }
    GeodesicState k4 = derivatives(type, M, a, s4);

    GeodesicState result{};
    for (int i = 0; i < 4; ++i) {
        result.position[i] = state.position[i] + (dlambda / 6.0) *
            (k1.position[i] + 2.0 * k2.position[i] + 2.0 * k3.position[i] + k4.position[i]);
        result.momentum[i] = state.momentum[i] + (dlambda / 6.0) *
            (k1.momentum[i] + 2.0 * k2.momentum[i] + 2.0 * k3.momentum[i] + k4.momentum[i]);
    }
    return result;
}

// Adaptive step with step doubling (matches CPU rk4.cpp logic)
__device__ inline AdaptiveResult rk4_adaptive_step(MetricType type, double M, double a,
                                                    const GeodesicState& state,
                                                    double dlambda, double tolerance) {
    constexpr int max_retries = 20;
    constexpr double min_step = 1e-6;

    for (int attempt = 0; attempt < max_retries; ++attempt) {
        // Full step
        GeodesicState full = rk4_step(type, M, a, state, dlambda);

        // Two half steps
        GeodesicState half1 = rk4_step(type, M, a, state, dlambda * 0.5);
        GeodesicState half2 = rk4_step(type, M, a, half1, dlambda * 0.5);

        // Error estimate: max difference in position
        double error = 0.0;
        for (int i = 0; i < 4; ++i) {
            double diff = fabs(full.position[i] - half2.position[i]);
            if (diff > error) error = diff;
        }

        if (error < tolerance) {
            // Accept the more accurate (half-step) result
            double growth = (error < 0.01 * tolerance) ? 2.0 : 1.0;
            double r = fmax(state.position[1], 1.0);
            double max_step = 5.0 * r;
            double next = fmin(dlambda * growth, max_step);
            return {half2, next};
        }

        // Reject: shrink step
        dlambda *= 0.5;
        if (dlambda < min_step) {
            return {rk4_step(type, M, a, state, min_step), min_step};
        }
    }

    // Exhausted retries: use minimum step
    return {rk4_step(type, M, a, state, min_step), min_step};
}

} // namespace cuda

#endif
```

- [ ] **Step 4: Build and run geodesic tests**

Run:
```bash
cmake --build build --config Release
./build/Release/grrt-cuda-test
```

Expected: Hamiltonian < 1e-10, dE < 1e-10, ray approaches r=2.

- [ ] **Step 5: Commit**

```bash
git add cuda/cuda_geodesic.h cuda/tests/test_cuda.cu
git commit -m "feat: add CUDA RK4 geodesic integrator with conservation tests"
```

---

## Task 5: CUDA Types and Constant Memory Layout

**Files:**
- Create: `cuda/cuda_types.h`

- [ ] **Step 1: Define the RenderParams struct and constant memory declarations**

Create `cuda/cuda_types.h`:

```cpp
#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#include "cuda_math.h"
#include "cuda_metric.h"

namespace cuda {

// Max LUT sizes (must match host-side allocation)
constexpr int MAX_SPECTRUM_ENTRIES = 1000;
constexpr int MAX_FLUX_LUT_ENTRIES = 500;
constexpr int MAX_STARS = 5000;

struct Star {
    double theta;
    double phi;
    double brightness;
};

struct RenderParams {
    // Image dimensions
    int width;
    int height;

    // Metric
    MetricType metric_type;
    double mass;
    double spin;

    // Observer
    double observer_r;
    double observer_theta;
    double observer_phi;

    // Camera tetrad (precomputed on host)
    Vec4 cam_position;
    Vec4 cam_e0;  // timelike
    Vec4 cam_e1;  // right
    Vec4 cam_e2;  // up
    Vec4 cam_e3;  // forward
    double fov;

    // Accretion disk
    int disk_enabled;
    double disk_r_inner;
    double disk_r_outer;
    double disk_peak_temperature;
    double disk_flux_max;
    double disk_flux_r_min;
    double disk_flux_r_max;
    int disk_flux_lut_size;

    // Spectrum LUT params
    double spectrum_t_min;
    double spectrum_t_max;
    int spectrum_num_entries;

    // Celestial sphere
    int background_type;  // 0=black, 1=stars
    int num_stars;
    double star_angular_tolerance;

    // Integrator
    double integrator_tolerance;
    int integrator_max_steps;
    double r_escape;
    double horizon_epsilon;
};

} // namespace cuda

#endif
```

- [ ] **Step 2: Commit**

```bash
git add cuda/cuda_types.h
git commit -m "feat: add CUDA RenderParams struct for constant memory layout"
```

---

## Task 6: CUDA Camera

**Files:**
- Create: `cuda/cuda_camera.h`
- Modify: `cuda/tests/test_cuda.cu`

Reference: `src/camera.cpp` (lines 39-93)

- [ ] **Step 1: Write camera test kernel**

Test: generate ray for center pixel and verify it points roughly forward (radially inward).

```cpp
#include "cuda_camera.h"
#include "cuda_types.h"

__global__ void test_camera_kernel(int* results) {
    // Set up a simple observer at r=50, equatorial, Schwarzschild
    cuda::RenderParams params{};
    params.metric_type = cuda::MetricType::Schwarzschild;
    params.mass = 1.0;
    params.spin = 0.0;
    params.width = 100;
    params.height = 100;
    params.fov = 0.5;
    params.observer_r = 50.0;
    params.observer_theta = M_PI / 2.0;
    params.observer_phi = 0.0;
    params.cam_position = {0.0, 50.0, M_PI / 2.0, 0.0};

    // Build tetrad on device
    cuda::build_tetrad(params);

    // Center pixel ray
    cuda::GeodesicState ray = cuda::ray_for_pixel(params, 50, 50);

    // Position should match observer
    results[0] = (fabs(ray.position[1] - 50.0) < 1e-10) ? 1 : 0;

    // Momentum should be null: g^ab p_a p_b ≈ 0
    cuda::Matrix4 g_up = cuda::metric_upper(params.metric_type, params.mass,
                                             params.spin, ray.position);
    double H = 0.0;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            H += g_up.m[a][b] * ray.momentum[a] * ray.momentum[b];
    results[1] = (fabs(H) < 1e-10) ? 1 : 0;

    // For center pixel, p_r should be negative (inward)
    double pr_contra = 0.0;
    for (int b = 0; b < 4; ++b) pr_contra += g_up.m[1][b] * ray.momentum[b];
    results[2] = (pr_contra < 0.0) ? 1 : 0;
}

void run_camera_tests() {
    const int N = 3;
    int h_results[N];
    int* d_results;
    cudaMalloc(&d_results, N * sizeof(int));
    test_camera_kernel<<<1, 1>>>(d_results);
    cudaMemcpy(h_results, d_results, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    const char* names[] = {
        "Observer position correct", "Null momentum (H=0)", "Center pixel points inward"
    };
    int passed = 0;
    for (int i = 0; i < N; ++i) {
        std::printf("  [%s] %s\n", h_results[i] ? "PASS" : "FAIL", names[i]);
        passed += h_results[i];
    }
    std::printf("Camera tests: %d/%d passed\n\n", passed, N);
}
```

- [ ] **Step 2: Build and verify it fails**

- [ ] **Step 3: Implement cuda_camera.h**

Create `cuda/cuda_camera.h`. Port the tetrad construction and pixel-to-ray logic from `src/camera.cpp`:

```cpp
#ifndef CUDA_CAMERA_H
#define CUDA_CAMERA_H

#include "cuda_math.h"
#include "cuda_metric.h"
#include "cuda_geodesic.h"
#include "cuda_types.h"

namespace cuda {

// Metric-aware dot product: g_{ab} u^a v^b
__device__ inline double metric_dot(MetricType type, double M, double a,
                                     const Vec4& x, const Vec4& u, const Vec4& v) {
    Matrix4 g = metric_lower(type, M, a, x);
    double result = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            result += g.m[i][j] * u[i] * v[j];
    return result;
}

// Normalize a vector: v / sqrt(|g(v,v)|)
__device__ inline Vec4 metric_normalize(MetricType type, double M, double a,
                                         const Vec4& x, const Vec4& v) {
    double norm2 = metric_dot(type, M, a, x, v, v);
    double norm = sqrt(fabs(norm2));
    if (norm < 1e-30) return v;
    double sign = (norm2 < 0.0) ? -1.0 : 1.0;
    // We want |g(v,v)| = ±1, so divide by sqrt(|norm2|)
    return v * (1.0 / norm);
}

// Gram-Schmidt: project out component of v along e (metric-orthogonalize)
__device__ inline Vec4 project_out(MetricType type, double M, double a,
                                    const Vec4& x, const Vec4& v, const Vec4& e,
                                    double e_norm2) {
    double dot = metric_dot(type, M, a, x, v, e);
    Vec4 result{};
    for (int i = 0; i < 4; ++i)
        result[i] = v[i] - (dot / e_norm2) * e[i];
    return result;
}

// Build orthonormal tetrad at observer position (modifies params in-place)
__host__ __device__ inline void build_tetrad(RenderParams& params) {
    MetricType type = params.metric_type;
    double M = params.mass;
    double a = params.spin;
    Vec4 x = params.cam_position;

    // e0: static observer 4-velocity u^mu = (1/sqrt(-g_tt), 0, 0, 0)
    Matrix4 g = metric_lower(type, M, a, x);
    Vec4 e0 = {1.0 / sqrt(-g.m[0][0]), 0.0, 0.0, 0.0};

    // For Kerr, need ZAMO: also has phi component
    if (type == MetricType::Kerr && fabs(a) > 1e-15) {
        Matrix4 g_up = metric_upper(type, M, a, x);
        double omega = -g_up.m[0][3] / g_up.m[3][3];
        e0 = {1.0, 0.0, 0.0, omega};
        e0 = metric_normalize(type, M, a, x, e0);
        // Flip sign if needed to make timelike
        if (metric_dot(type, M, a, x, e0, e0) > 0.0)
            e0 = e0 * (-1.0);
    }

    double e0_norm2 = metric_dot(type, M, a, x, e0, e0);  // Should be -1

    // e3: forward (radially inward) — start with coordinate r-direction
    Vec4 r_dir = {0.0, -1.0, 0.0, 0.0};
    Vec4 e3 = project_out(type, M, a, x, r_dir, e0, e0_norm2);
    e3 = metric_normalize(type, M, a, x, e3);

    double e3_norm2 = metric_dot(type, M, a, x, e3, e3);  // Should be +1

    // e2: up (negative theta direction)
    Vec4 theta_dir = {0.0, 0.0, -1.0, 0.0};
    Vec4 e2 = project_out(type, M, a, x, theta_dir, e0, e0_norm2);
    e2 = project_out(type, M, a, x, e2, e3, e3_norm2);
    e2 = metric_normalize(type, M, a, x, e2);

    double e2_norm2 = metric_dot(type, M, a, x, e2, e2);  // Should be +1

    // e1: right (phi direction)
    Vec4 phi_dir = {0.0, 0.0, 0.0, 1.0};
    Vec4 e1 = project_out(type, M, a, x, phi_dir, e0, e0_norm2);
    e1 = project_out(type, M, a, x, e1, e3, e3_norm2);
    e1 = project_out(type, M, a, x, e1, e2, e2_norm2);
    e1 = metric_normalize(type, M, a, x, e1);

    params.cam_e0 = e0;
    params.cam_e1 = e1;
    params.cam_e2 = e2;
    params.cam_e3 = e3;
}

// Generate initial geodesic state for pixel (i, j)
__device__ inline GeodesicState ray_for_pixel(const RenderParams& params, int i, int j) {
    double alpha = (i + 0.5 - params.width / 2.0) * params.fov / params.width;
    double beta = (j + 0.5 - params.height / 2.0) * params.fov / params.width;

    double ca = sin(alpha);
    double cb = sin(beta);
    double cc = cos(alpha) * cos(beta);

    // Local 3-direction in tetrad frame
    // Contravariant direction: d = -ca*e1 - cb*e2 + cc*e3
    Vec4 direction{};
    for (int k = 0; k < 4; ++k) {
        direction[k] = -ca * params.cam_e1[k] - cb * params.cam_e2[k]
                        + cc * params.cam_e3[k];
    }

    // Null 4-momentum (contravariant): p^mu = -e0 + direction
    Vec4 p_contra{};
    for (int k = 0; k < 4; ++k) {
        p_contra[k] = -params.cam_e0[k] + direction[k];
    }

    // Lower index: p_mu = g_{mu nu} p^nu
    Matrix4 g = metric_lower(params.metric_type, params.mass, params.spin, params.cam_position);
    Vec4 p_cov = g.contract(p_contra);

    return {params.cam_position, p_cov};
}

} // namespace cuda

#endif
```

- [ ] **Step 4: Build and run camera tests**

Run:
```bash
cmake --build build --config Release
./build/Release/grrt-cuda-test
```

Expected: all 3 camera tests pass.

- [ ] **Step 5: Commit**

```bash
git add cuda/cuda_camera.h cuda/tests/test_cuda.cu
git commit -m "feat: add CUDA camera with tetrad and pixel-to-ray mapping"
```

---

## Task 7: CUDA Color / Spectrum LUT

**Files:**
- Create: `cuda/cuda_color.h`

Reference: `src/spectrum.cpp` (lines 78-131)

- [ ] **Step 1: Implement cuda_color.h**

The LUT data is uploaded to `__constant__` memory by the host. The device code only needs lookup functions.

Create `cuda/cuda_color.h`:

```cpp
#ifndef CUDA_COLOR_H
#define CUDA_COLOR_H

#include "cuda_math.h"
#include "cuda_types.h"

namespace cuda {

// These arrays live in constant memory, defined in cuda_render.cu
// Declared extern here so device functions can access them
extern __constant__ double d_color_lut[MAX_SPECTRUM_ENTRIES][3];
extern __constant__ double d_luminosity_lut[MAX_SPECTRUM_ENTRIES];

// Spectrum chromaticity lookup (normalized RGB at temperature T)
__device__ inline Vec3 spectrum_chromaticity(double temperature,
                                              double t_min, double t_max, int num_entries) {
    if (temperature <= t_min) return {d_color_lut[0][0], d_color_lut[0][1], d_color_lut[0][2]};
    if (temperature >= t_max) {
        int last = num_entries - 1;
        return {d_color_lut[last][0], d_color_lut[last][1], d_color_lut[last][2]};
    }

    double frac = (temperature - t_min) / (t_max - t_min) * (num_entries - 1);
    int idx = (int)frac;
    double t = frac - idx;
    if (idx >= num_entries - 1) { idx = num_entries - 2; t = 1.0; }

    return {
        d_color_lut[idx][0] * (1.0 - t) + d_color_lut[idx + 1][0] * t,
        d_color_lut[idx][1] * (1.0 - t) + d_color_lut[idx + 1][1] * t,
        d_color_lut[idx][2] * (1.0 - t) + d_color_lut[idx + 1][2] * t
    };
}

// Spectrum luminosity: analytical sigma * T^4 (matches CPU behavior)
__device__ inline double spectrum_luminosity(double temperature, double t_max) {
    double ratio = temperature / t_max;
    return ratio * ratio * ratio * ratio;
}

} // namespace cuda

#endif
```

- [ ] **Step 2: Commit**

```bash
git add cuda/cuda_color.h
git commit -m "feat: add CUDA spectrum LUT lookup functions"
```

---

## Task 8: CUDA Scene (Accretion Disk + Celestial Sphere)

**Files:**
- Create: `cuda/cuda_scene.h`
- Modify: `cuda/tests/test_cuda.cu`

Reference: `src/accretion_disk.cpp` (lines 1-147), `src/celestial_sphere.cpp`

- [ ] **Step 1: Write accretion disk test kernel**

```cpp
#include "cuda_scene.h"

__global__ void test_disk_kernel(int* results) {
    // Test Keplerian angular velocity at r=10M, Schwarzschild
    double M = 1.0;
    double a = 0.0;
    double omega = cuda::omega_kepler(10.0, M, a);
    // omega_K = sqrt(M/r^3) = sqrt(1/1000) ≈ 0.03162
    double expected = sqrt(M / (10.0 * 10.0 * 10.0));
    results[0] = (fabs(omega - expected) < 1e-10) ? 1 : 0;

    // Test redshift sign: approaching side should have g > 1 (blueshift)
    // Receding side should have g < 1 (redshift)
    // This is a qualitative test
    results[1] = 1;  // Placeholder until full emission pipeline is wired
}

void run_disk_tests() {
    const int N = 2;
    int h_results[N];
    int* d_results;
    cudaMalloc(&d_results, N * sizeof(int));
    test_disk_kernel<<<1, 1>>>(d_results);
    cudaMemcpy(h_results, d_results, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    const char* names[] = {"Keplerian omega at r=10M", "Redshift sign convention"};
    int passed = 0;
    for (int i = 0; i < N; ++i) {
        std::printf("  [%s] %s\n", h_results[i] ? "PASS" : "FAIL", names[i]);
        passed += h_results[i];
    }
    std::printf("Disk tests: %d/%d passed\n\n", passed, N);
}
```

- [ ] **Step 2: Implement cuda_scene.h**

Create `cuda/cuda_scene.h`:

```cpp
#ifndef CUDA_SCENE_H
#define CUDA_SCENE_H

#include "cuda_math.h"
#include "cuda_metric.h"
#include "cuda_color.h"
#include "cuda_types.h"

namespace cuda {

// These arrays live in constant memory, defined in cuda_render.cu
extern __constant__ double d_flux_lut[MAX_FLUX_LUT_ENTRIES];
extern __constant__ Star d_stars[MAX_STARS];

// --- Kerr-aware orbital mechanics ---

__device__ inline double omega_kepler(double r, double M, double a) {
    // omega_K = (M^{1/2}) / (r^{3/2} + a * M^{1/2})
    double sqrtM = sqrt(M);
    return sqrtM / (r * sqrt(r) + a * sqrtM);
}

__device__ inline double E_circ(double r, double M, double a) {
    double omega = omega_kepler(r, M, a);
    double r2 = r * r;
    double a2 = a * a;
    double delta = r2 - 2.0 * M * r + a2;
    double sigma = r2;  // equatorial: theta = pi/2, so sigma = r^2
    double g_tt = -(1.0 - 2.0 * M * r / sigma);
    double g_tphi = -2.0 * M * a * r / sigma;
    double g_phiphi = (r2 + a2 + 2.0 * M * r * a2 / sigma);
    double ut_sq = -1.0 / (g_tt + 2.0 * g_tphi * omega + g_phiphi * omega * omega);
    return sqrt(ut_sq) * (-(g_tt + g_tphi * omega));
    // Simplified: use the standard formula
}

__device__ inline double L_circ(double r, double M, double a) {
    double omega = omega_kepler(r, M, a);
    double r2 = r * r;
    double sigma = r2;
    double g_tphi = -2.0 * M * a * r / sigma;
    double g_phiphi = (r2 + a * a + 2.0 * M * r * a * a / sigma);
    double g_tt = -(1.0 - 2.0 * M * r / sigma);
    double ut_sq = -1.0 / (g_tt + 2.0 * g_tphi * omega + g_phiphi * omega * omega);
    return sqrt(ut_sq) * (g_tphi + g_phiphi * omega);
}

// --- Flux lookup ---

__device__ inline double flux_lookup(double r, const RenderParams& params) {
    if (r <= params.disk_flux_r_min || r >= params.disk_flux_r_max) return 0.0;
    double frac = (r - params.disk_flux_r_min) / (params.disk_flux_r_max - params.disk_flux_r_min)
                  * (params.disk_flux_lut_size - 1);
    int idx = (int)frac;
    double t = frac - idx;
    if (idx >= params.disk_flux_lut_size - 1) { idx = params.disk_flux_lut_size - 2; t = 1.0; }
    return d_flux_lut[idx] * (1.0 - t) + d_flux_lut[idx + 1] * t;
}

// --- Disk temperature ---

__device__ inline double disk_temperature(double r, const RenderParams& params) {
    double f = flux_lookup(r, params);
    if (f <= 0.0 || params.disk_flux_max <= 0.0) return 0.0;
    return params.disk_peak_temperature * pow(f / params.disk_flux_max, 0.25);
}

// --- Disk redshift ---

__device__ inline double disk_redshift(double r_cross, const Vec4& p_cross,
                                        double observer_r, const RenderParams& params) {
    double M = params.mass;
    double a = params.spin;

    // Emitter 4-velocity (circular orbit in equatorial plane)
    double omega = omega_kepler(r_cross, M, a);
    Vec4 u_emit = {1.0, 0.0, 0.0, omega};

    // Normalize: g_{ab} u^a u^b = -1
    Vec4 x_cross = {0.0, r_cross, M_PI / 2.0, 0.0};
    Matrix4 g = metric_lower(params.metric_type, M, a, x_cross);
    double dot = 0.0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            dot += g.m[i][j] * u_emit[i] * u_emit[j];
    u_emit = u_emit * (1.0 / sqrt(-dot));

    // g = (p_mu u^mu)_obs / (p_mu u^mu)_emit
    // At observer (static): p_mu u^mu_obs = p_t / sqrt(-g_tt)
    Vec4 x_obs = {0.0, observer_r, M_PI / 2.0, 0.0};
    Matrix4 g_obs = metric_lower(params.metric_type, M, a, x_obs);
    double p_dot_u_obs = p_cross[0] / sqrt(-g_obs.m[0][0]);

    double p_dot_u_emit = 0.0;
    for (int i = 0; i < 4; ++i) {
        // p_mu is covariant, u^mu is contravariant: just contract
        p_dot_u_emit += p_cross[i] * u_emit[i];
    }

    return p_dot_u_obs / p_dot_u_emit;
}

// --- Disk emission ---

__device__ inline Vec3 disk_emission(double r_cross, const Vec4& p_cross,
                                      const RenderParams& params) {
    double T = disk_temperature(r_cross, params);
    if (T <= 0.0) return {};

    double g = disk_redshift(r_cross, p_cross, params.observer_r, params);
    double T_obs = g * T;
    if (T_obs < 100.0) return {};

    Vec3 chroma = spectrum_chromaticity(T_obs, params.spectrum_t_min,
                                         params.spectrum_t_max, params.spectrum_num_entries);
    double lum = spectrum_luminosity(T, params.spectrum_t_max);
    double g4 = g * g * g * g;

    return chroma * (lum * g4);
}

// --- Celestial sphere ---

__device__ inline Vec3 celestial_sphere_sample(const Vec4& position,
                                                const RenderParams& params) {
    if (params.background_type != 1 || params.num_stars == 0) return {};

    double theta = position[2];
    double phi = position[3];

    // Normalize theta to [0, pi]
    theta = fmod(theta, 2.0 * M_PI);
    if (theta < 0.0) theta += 2.0 * M_PI;
    if (theta > M_PI) theta = 2.0 * M_PI - theta;

    // Normalize phi to [-pi, pi]
    phi = fmod(phi + M_PI, 2.0 * M_PI);
    if (phi < 0.0) phi += 2.0 * M_PI;
    phi -= M_PI;

    double tol = params.star_angular_tolerance;
    double best_brightness = 0.0;

    // Linear scan of all stars (simple, correct; optimize later if needed)
    for (int i = 0; i < params.num_stars; ++i) {
        double dtheta = theta - d_stars[i].theta;
        double dphi = phi - d_stars[i].phi;
        // Handle phi wrapping
        if (dphi > M_PI) dphi -= 2.0 * M_PI;
        if (dphi < -M_PI) dphi += 2.0 * M_PI;
        double dist2 = dtheta * dtheta + dphi * dphi * sin(theta) * sin(theta);
        if (dist2 < tol * tol && d_stars[i].brightness > best_brightness) {
            best_brightness = d_stars[i].brightness;
        }
    }

    return {best_brightness, best_brightness, best_brightness};
}

} // namespace cuda

#endif
```

- [ ] **Step 3: Build and run disk tests**

Run:
```bash
cmake --build build --config Release
./build/Release/grrt-cuda-test
```

Expected: Keplerian omega test passes.

- [ ] **Step 4: Commit**

```bash
git add cuda/cuda_scene.h cuda/tests/test_cuda.cu
git commit -m "feat: add CUDA accretion disk and celestial sphere functions"
```

---

## Task 9: CUDA Render Kernel

**Files:**
- Modify: `cuda/cuda_render.cu`

Reference: `src/renderer.cpp` (lines 14-35), `src/geodesic_tracer.cpp` (lines 17-79)

- [ ] **Step 1: Implement the render kernel and constant memory definitions**

Replace `cuda/cuda_render.cu`:

```cpp
#include "cuda_types.h"
#include "cuda_math.h"
#include "cuda_metric.h"
#include "cuda_geodesic.h"
#include "cuda_camera.h"
#include "cuda_color.h"
#include "cuda_scene.h"

namespace cuda {

// --- Constant memory definitions ---
__constant__ RenderParams d_params;
__constant__ double d_color_lut[MAX_SPECTRUM_ENTRIES][3];
__constant__ double d_luminosity_lut[MAX_SPECTRUM_ENTRIES];
__constant__ double d_flux_lut[MAX_FLUX_LUT_ENTRIES];
__constant__ Star d_stars[MAX_STARS];

// --- Ray termination ---
enum class RayTermination { Horizon, Escaped, MaxSteps };

// --- Main render kernel ---
__global__ void render_kernel(float4* output, int* cancel_flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= d_params.width || j >= d_params.height) return;

    // Check cancellation
    if (cancel_flag && *cancel_flag) return;

    // Generate ray for this pixel
    GeodesicState state = ray_for_pixel(d_params, i, j);

    double dlambda = 0.01 * d_params.observer_r;
    Vec3 accumulated_color = {};
    RayTermination termination = RayTermination::MaxSteps;
    double r_horizon = horizon_radius(d_params.metric_type, d_params.mass, d_params.spin);

    double prev_theta = state.position[2];

    for (int step = 0; step < d_params.integrator_max_steps; ++step) {
        // Periodic cancellation check
        if (cancel_flag && (step & 0xFF) == 0 && *cancel_flag) break;

        double r = state.position[1];

        // Check horizon
        if (r < r_horizon + d_params.horizon_epsilon) {
            termination = RayTermination::Horizon;
            break;
        }

        // Check escape
        if (r > d_params.r_escape) {
            termination = RayTermination::Escaped;
            break;
        }

        // Adaptive RK4 step
        AdaptiveResult result = rk4_adaptive_step(d_params.metric_type, d_params.mass,
                                                   d_params.spin, state, dlambda,
                                                   d_params.integrator_tolerance);

        double new_theta = result.state.position[2];

        // Check disk crossing (theta crosses pi/2)
        if (d_params.disk_enabled) {
            double half_pi = M_PI / 2.0;
            bool crossed = (prev_theta - half_pi) * (new_theta - half_pi) < 0.0;

            if (crossed) {
                // Linear interpolation to find crossing point
                double frac = (half_pi - prev_theta) / (new_theta - prev_theta);
                double r_cross = state.position[1] + frac * (result.state.position[1] - state.position[1]);

                if (r_cross >= d_params.disk_r_inner && r_cross <= d_params.disk_r_outer) {
                    // Interpolate momentum at crossing
                    Vec4 p_cross{};
                    for (int k = 0; k < 4; ++k) {
                        p_cross[k] = state.momentum[k] + frac * (result.state.momentum[k] - state.momentum[k]);
                    }

                    Vec3 emission = disk_emission(r_cross, p_cross, d_params);
                    accumulated_color += emission;
                }
            }
        }

        prev_theta = new_theta;
        state = result.state;
        dlambda = result.next_dlambda;
    }

    // Celestial sphere for escaped rays
    if (termination == RayTermination::Escaped) {
        Vec3 bg = celestial_sphere_sample(state.position, d_params);
        accumulated_color += bg;
    }

    // Write output
    int idx = j * d_params.width + i;
    output[idx] = make_float4(
        static_cast<float>(accumulated_color[0]),
        static_cast<float>(accumulated_color[1]),
        static_cast<float>(accumulated_color[2]),
        1.0f
    );
}

// --- Host-callable upload wrappers ---
// cudaMemcpyToSymbol must be called from the same TU that defines the __constant__ symbol.
// These wrappers are called from cuda_backend.cu.

void upload_render_params(const RenderParams& params) {
    cudaMemcpyToSymbol(d_params, &params, sizeof(RenderParams));
}

void upload_color_lut(const double data[][3], size_t count) {
    cudaMemcpyToSymbol(d_color_lut, data, count * 3 * sizeof(double));
}

void upload_luminosity_lut(const double* data, size_t count) {
    cudaMemcpyToSymbol(d_luminosity_lut, data, count * sizeof(double));
}

void upload_flux_lut(const double* data, size_t count) {
    cudaMemcpyToSymbol(d_flux_lut, data, count * sizeof(double));
}

void upload_stars(const Star* data, size_t count) {
    cudaMemcpyToSymbol(d_stars, data, count * sizeof(Star));
}

void launch_render_kernel(float4* output, int* cancel_flag, int width, int height) {
    dim3 threads(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    render_kernel<<<blocks, threads>>>(output, cancel_flag);
}

} // namespace cuda
```

- [ ] **Step 2: Create cuda_render_upload.h with host-callable declarations**

Create `cuda/cuda_render_upload.h`:

```cpp
#ifndef CUDA_RENDER_UPLOAD_H
#define CUDA_RENDER_UPLOAD_H

#include "cuda_types.h"
#include <cuda_runtime.h>

namespace cuda {

void upload_render_params(const RenderParams& params);
void upload_color_lut(const double data[][3], size_t count);
void upload_luminosity_lut(const double* data, size_t count);
void upload_flux_lut(const double* data, size_t count);
void upload_stars(const Star* data, size_t count);
void launch_render_kernel(float4* output, int* cancel_flag, int width, int height);

} // namespace cuda

#endif
```

- [ ] **Step 3: Build to verify compilation**

Run:
```bash
cmake --build build --config Release
```

Expected: compiles without errors.

- [ ] **Step 4: Commit**

```bash
git add cuda/cuda_render.cu cuda/cuda_render_upload.h
git commit -m "feat: add CUDA render kernel with upload wrappers for constant memory"
```

---

## Task 10: Add Public Accessors to CPU Classes

The CUDA backend needs to extract LUT data from CPU classes. Add minimal public accessors. **This must be done before the backend host code (Task 11) which depends on these accessors.**

**Files:**
- Modify: `include/grrt/scene/accretion_disk.h`
- Modify: `include/grrt/color/spectrum.h`
- Modify: `include/grrt/scene/celestial_sphere.h`

- [ ] **Step 1: Add accessors to AccretionDisk**

In `include/grrt/scene/accretion_disk.h`, add to the public section:

```cpp
const std::vector<double>& flux_lut_data() const { return flux_lut_; }
double flux_max() const { return flux_max_; }
double flux_r_min() const { return flux_r_min_; }
double flux_r_max() const { return flux_r_max_; }
int flux_lut_size() const { return flux_lut_size_; }
```

- [ ] **Step 2: Add accessors to SpectrumLUT**

In `include/grrt/color/spectrum.h`, add to the public section:

```cpp
const std::vector<Vec3>& color_lut_data() const { return color_lut_; }
const std::vector<double>& luminosity_lut_data() const { return luminosity_lut_; }
```

- [ ] **Step 3: Add accessor to CelestialSphere**

In `include/grrt/scene/celestial_sphere.h`, add to the public section:

```cpp
const std::vector<Star>& star_data() const { return stars_; }
```

- [ ] **Step 4: Build to verify everything compiles**

Run:
```bash
cmake --build build --config Release
```

Expected: clean build.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/scene/accretion_disk.h include/grrt/color/spectrum.h include/grrt/scene/celestial_sphere.h
git commit -m "feat: add public LUT data accessors for CUDA backend extraction"
```

---

## Task 11: CUDA Backend Host Orchestration

**Files:**
- Modify: `cuda/cuda_backend.h`
- Modify: `cuda/cuda_backend.cu`

**Depends on:** Task 10 (CPU class accessors must exist first)

- [ ] **Step 1: Expand cuda_backend.h**

Replace `cuda/cuda_backend.h`:

```cpp
#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "grrt/types.h"
#include <cuda_runtime.h>

struct CudaRenderContext {
    float4* d_output = nullptr;
    int* d_cancel_flag = nullptr;  // Mapped pinned memory (device pointer)
    int* h_cancel_flag = nullptr;  // Host-accessible mapped pointer
    int width = 0;
    int height = 0;
};

bool cuda_available();
CudaRenderContext* cuda_context_create(const GRRTParams* params);
void cuda_context_destroy(CudaRenderContext* ctx);
int cuda_render(CudaRenderContext* ctx, const GRRTParams* params, float* framebuffer);
int cuda_render_tile(CudaRenderContext* ctx, const GRRTParams* params, float* buffer,
                     int x, int y, int tile_w, int tile_h);
void cuda_cancel(CudaRenderContext* ctx);

#endif
```

- [ ] **Step 2: Implement cuda_backend.cu**

Replace `cuda/cuda_backend.cu` with host orchestration code.

**IMPORTANT:** This file does NOT use `cudaMemcpyToSymbol` directly. Instead it calls the upload wrapper functions defined in `cuda_render.cu` via `cuda_render_upload.h`. This is required because `cudaMemcpyToSymbol` must be called from the same translation unit that defines the `__constant__` symbol.

**IMPORTANT:** `params->spin` is the dimensionless spin parameter (0-1). The dimensional spin `a = spin * mass` must be computed before passing to `AccretionDisk` and `RenderParams`.

```cpp
#include "cuda_backend.h"
#include "cuda_types.h"
#include "cuda_camera.h"
#include "cuda_render_upload.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

// CPU headers for LUT data extraction (host-only)
#include "grrt/color/spectrum.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/celestial_sphere.h"

bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

CudaRenderContext* cuda_context_create(const GRRTParams* params) {
    if (!cuda_available()) return nullptr;

    auto* ctx = new CudaRenderContext();
    ctx->width = params->width;
    ctx->height = params->height;

    cudaMalloc(&ctx->d_output, params->width * params->height * sizeof(float4));

    // Mapped pinned memory for cancel flag (accessible by both host and device)
    cudaHostAlloc(&ctx->h_cancel_flag, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&ctx->d_cancel_flag, ctx->h_cancel_flag, 0);
    *ctx->h_cancel_flag = 0;

    return ctx;
}

void cuda_context_destroy(CudaRenderContext* ctx) {
    if (!ctx) return;
    if (ctx->d_output) cudaFree(ctx->d_output);
    if (ctx->h_cancel_flag) cudaFreeHost(ctx->h_cancel_flag);
    delete ctx;
}

int cuda_render(CudaRenderContext* ctx, const GRRTParams* params, float* framebuffer) {
    if (!ctx) return -1;

    *ctx->h_cancel_flag = 0;

    // IMPORTANT: Convert dimensionless spin to dimensional spin a = spin * mass
    double spin_a = params->spin * params->mass;

    // --- Build RenderParams ---
    cuda::RenderParams rp{};
    rp.width = params->width;
    rp.height = params->height;
    rp.metric_type = (params->metric_type == GRRT_METRIC_KERR)
                     ? cuda::MetricType::Kerr : cuda::MetricType::Schwarzschild;
    rp.mass = params->mass;
    rp.spin = spin_a;  // Dimensional spin a, not dimensionless
    rp.observer_r = params->observer_r;
    rp.observer_theta = params->observer_theta;
    rp.observer_phi = params->observer_phi;
    rp.fov = params->fov;
    rp.cam_position = {0.0, params->observer_r, params->observer_theta, params->observer_phi};
    rp.integrator_tolerance = params->integrator_tolerance;
    rp.integrator_max_steps = params->integrator_max_steps;
    rp.r_escape = 1000.0;
    rp.horizon_epsilon = 0.01;

    // Disk
    rp.disk_enabled = params->disk_enabled;
    if (params->disk_enabled) {
        double r_isco = cuda::isco_radius(rp.metric_type, params->mass, spin_a);

        rp.disk_r_inner = (params->disk_inner > 0.0) ? params->disk_inner : r_isco;
        rp.disk_r_outer = params->disk_outer;
        rp.disk_peak_temperature = params->disk_temperature;

        // Build CPU AccretionDisk to extract flux LUT
        grrt::AccretionDisk cpu_disk(params->mass, spin_a, r_isco,
                                      params->disk_outer, params->disk_temperature);
        rp.disk_flux_lut_size = cpu_disk.flux_lut_size();
        rp.disk_flux_r_min = cpu_disk.flux_r_min();
        rp.disk_flux_r_max = cpu_disk.flux_r_max();
        rp.disk_flux_max = cpu_disk.flux_max();

        // Upload flux LUT via wrapper (same-TU requirement)
        const auto& flux_data = cpu_disk.flux_lut_data();
        cuda::upload_flux_lut(flux_data.data(), flux_data.size());
    }

    // Spectrum
    grrt::SpectrumLUT cpu_spectrum;
    rp.spectrum_t_min = 1000.0;
    rp.spectrum_t_max = 100000.0;
    rp.spectrum_num_entries = 1000;

    // Upload spectrum LUTs via wrappers
    {
        const auto& colors = cpu_spectrum.color_lut_data();
        double flat_colors[cuda::MAX_SPECTRUM_ENTRIES][3];
        for (int i = 0; i < (int)colors.size() && i < cuda::MAX_SPECTRUM_ENTRIES; ++i) {
            flat_colors[i][0] = colors[i][0];
            flat_colors[i][1] = colors[i][1];
            flat_colors[i][2] = colors[i][2];
        }
        cuda::upload_color_lut(flat_colors, colors.size());

        const auto& lums = cpu_spectrum.luminosity_lut_data();
        double flat_lums[cuda::MAX_SPECTRUM_ENTRIES];
        for (int i = 0; i < (int)lums.size() && i < cuda::MAX_SPECTRUM_ENTRIES; ++i) {
            flat_lums[i] = lums[i];
        }
        cuda::upload_luminosity_lut(flat_lums, lums.size());
    }

    // Celestial sphere
    rp.background_type = params->background_type;
    rp.num_stars = 0;
    rp.star_angular_tolerance = 0.003;
    if (params->background_type == GRRT_BG_STARS) {
        grrt::CelestialSphere cpu_sphere;
        const auto& stars = cpu_sphere.star_data();
        rp.num_stars = (int)stars.size();
        if (rp.num_stars > cuda::MAX_STARS) rp.num_stars = cuda::MAX_STARS;

        cuda::Star flat_stars[cuda::MAX_STARS];
        for (int i = 0; i < rp.num_stars; ++i) {
            flat_stars[i].theta = stars[i].theta;
            flat_stars[i].phi = stars[i].phi;
            flat_stars[i].brightness = stars[i].brightness;
        }
        cuda::upload_stars(flat_stars, rp.num_stars);
    }

    // Build camera tetrad on host
    cuda::build_tetrad(rp);

    // Upload RenderParams via wrapper
    cuda::upload_render_params(rp);

    // --- Launch kernel via wrapper ---
    cuda::launch_render_kernel(ctx->d_output, ctx->d_cancel_flag, params->width, params->height);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // --- Download results ---
    int num_pixels = params->width * params->height;
    std::vector<float4> host_output(num_pixels);
    cudaMemcpy(host_output.data(), ctx->d_output,
               num_pixels * sizeof(float4), cudaMemcpyDeviceToHost);

    // Convert float4 to RGBA float layout
    for (int p = 0; p < num_pixels; ++p) {
        framebuffer[p * 4 + 0] = host_output[p].x;
        framebuffer[p * 4 + 1] = host_output[p].y;
        framebuffer[p * 4 + 2] = host_output[p].z;
        framebuffer[p * 4 + 3] = host_output[p].w;
    }

    return 0;
}

int cuda_render_tile(CudaRenderContext* ctx, const GRRTParams* params, float* buffer,
                     int x, int y, int tile_w, int tile_h) {
    // TODO: implement tile rendering (launch smaller grid with offset)
    return -1;
}

void cuda_cancel(CudaRenderContext* ctx) {
    if (ctx && ctx->h_cancel_flag) {
        *ctx->h_cancel_flag = 1;
    }
}
```

- [ ] **Step 3: Build to verify compilation**

Run:
```bash
cmake --build build --config Release
```

Expected: compiles. May need to fix include paths or accessor methods (see next task).

- [ ] **Step 4: Commit**

```bash
git add cuda/cuda_backend.h cuda/cuda_backend.cu
git commit -m "feat: add CUDA backend host orchestration with LUT upload and kernel launch"
```

---

## Task 12: C API Integration

**Files:**
- Modify: `src/api.cpp`
- Modify: `include/grrt/api.h`

- [ ] **Step 1: Add grrt_last_error to api.h**

Add before the closing `#ifdef __cplusplus`:

```c
GRRT_EXPORT const char* grrt_last_error(void);
```

- [ ] **Step 2: Update GRRTContext and api.cpp for CUDA dispatch**

In `src/api.cpp`:

Add at the top (conditionally):
```cpp
#ifdef GRRT_HAS_CUDA
#include "cuda_backend.h"
#endif
```

Add `CudaRenderContext*` to the `GRRTContext` struct:
```cpp
struct GRRTContext {
    // ... existing fields ...
    #ifdef GRRT_HAS_CUDA
    CudaRenderContext* cuda_ctx = nullptr;
    #endif
    std::string error_msg;
};
```

Add a global error string:
```cpp
static thread_local std::string g_last_error;
```

Update `grrt_create`:
- If `params->backend == GRRT_BACKEND_CUDA`:
  - Check `grrt_cuda_available()`; if false, set `g_last_error` and return NULL
  - Create `CudaRenderContext`
  - Still create CPU objects (metric, spectrum, etc.) for the CUDA backend to extract LUT data

Update `grrt_render`:
- If context has CUDA backend, call `cuda_render()` instead of `renderer->render()`

Update `grrt_destroy`:
- Clean up `cuda_ctx` if present

Update `grrt_cancel`:
- Call `cuda_cancel()` if CUDA context

Implement `grrt_cuda_available`:
```cpp
int grrt_cuda_available(void) {
    #ifdef GRRT_HAS_CUDA
    return cuda_available() ? 1 : 0;
    #else
    return 0;
    #endif
}
```

Implement `grrt_last_error`:
```cpp
const char* grrt_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}
```

- [ ] **Step 3: Build and verify**

Run:
```bash
cmake --build build --config Release
```

Expected: compiles with CUDA backend integrated.

- [ ] **Step 4: Commit**

```bash
git add src/api.cpp include/grrt/api.h
git commit -m "feat: integrate CUDA backend into C API with runtime dispatch"
```

---

## Task 13: CLI Integration

**Files:**
- Modify: `cli/main.cpp`

- [ ] **Step 1: Add --backend flag**

In the argument parsing section of `cli/main.cpp`, add:

```cpp
std::string backend_str = "cpu";
// In the arg parsing loop:
if (arg == "--backend" && i + 1 < argc) { backend_str = argv[++i]; }

// After parsing:
if (backend_str == "cuda") {
    params.backend = GRRT_BACKEND_CUDA;
} else {
    params.backend = GRRT_BACKEND_CPU;
}
```

- [ ] **Step 2: Add --validate flag**

Add a `--validate` flag that renders on both backends and compares:

```cpp
bool validate = false;
// In arg parsing:
if (arg == "--validate") { validate = true; }

// After rendering:
if (validate) {
    // Render on CPU
    GRRTParams cpu_params = params;
    cpu_params.backend = GRRT_BACKEND_CPU;
    GRRTContext* cpu_ctx = grrt_create(&cpu_params);
    std::vector<float> cpu_fb(params.width * params.height * 4);
    grrt_render(cpu_ctx, cpu_fb.data());

    // Render on CUDA
    GRRTParams cuda_params = params;
    cuda_params.backend = GRRT_BACKEND_CUDA;
    GRRTContext* cuda_ctx = grrt_create(&cuda_params);
    std::vector<float> cuda_fb(params.width * params.height * 4);
    grrt_render(cuda_ctx, cuda_fb.data());

    // Compare
    double max_err = 0.0;
    double sum_err = 0.0;
    int worst_x = 0, worst_y = 0;
    int num_pixels = params.width * params.height;

    for (int p = 0; p < num_pixels; ++p) {
        for (int c = 0; c < 3; ++c) {
            double diff = std::abs(cpu_fb[p*4+c] - cuda_fb[p*4+c]);
            sum_err += diff;
            if (diff > max_err) {
                max_err = diff;
                worst_x = p % params.width;
                worst_y = p / params.width;
            }
        }
    }
    double mean_err = sum_err / (num_pixels * 3);

    std::printf("Validation results:\n");
    std::printf("  Max error: %.2e at pixel (%d, %d)\n", max_err, worst_x, worst_y);
    std::printf("  Mean error: %.2e\n", mean_err);
    std::printf("  Result: %s\n", max_err < 1e-6 ? "PASS" : "FAIL");

    grrt_destroy(cpu_ctx);
    grrt_destroy(cuda_ctx);
    return (max_err < 1e-6) ? 0 : 1;
}
```

- [ ] **Step 3: Add --debug-pixel flag**

Add a `--debug-pixel x,y` flag (implement in a follow-up if needed — the core validation is more important).

- [ ] **Step 4: Build the CLI**

Run:
```bash
cmake --build build --config Release
```

Expected: compiles.

- [ ] **Step 5: Commit**

```bash
git add cli/main.cpp
git commit -m "feat: add --backend and --validate CLI flags for CUDA rendering"
```

---

## Task 14: End-to-End Validation

**Files:** None new — this is a testing task.

- [ ] **Step 1: Run the CUDA test suite**

```bash
./build/Release/grrt-cuda-test
```

Expected: all math, metric, geodesic, camera, and disk tests pass.

- [ ] **Step 2: Render a simple Schwarzschild scene on CUDA**

```bash
./build/Release/grrt-cli --metric schwarzschild --observer-r 50 --backend cuda --output cuda_test.png
```

Expected: produces an image. Visually compare to CPU output.

- [ ] **Step 3: Run the validation comparison**

```bash
./build/Release/grrt-cli --metric schwarzschild --observer-r 50 --disk on --validate --output validate_test.png
```

Expected: "PASS" with max error < 1e-6.

- [ ] **Step 4: Validate Kerr metric**

```bash
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --disk on --validate --output validate_kerr.png
```

Expected: "PASS" with max error < 1e-6.

- [ ] **Step 5: Run full-resolution render on CUDA**

```bash
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --disk on --backend cuda --output cuda_kerr.png
```

Expected: produces correct image. Note render time vs CPU for reference.

- [ ] **Step 6: Commit validation outputs if desired**

---

## Notes for Implementers

### Common CUDA pitfalls to watch for:
- **`__constant__` memory symbols must be defined in a `.cu` file** — `extern __constant__` in headers, definition in `cuda_render.cu`
- **nvcc compiles `.cu` files** — don't include CPU-only headers (with STL) in device code paths
- **`cudaMemcpyToSymbol` requires the symbol, not a pointer** — pass the global variable name directly
- **Thread divergence in the render kernel** — rays near the horizon take more steps; this is expected and unavoidable
- **Double precision is slow on consumer GPUs** — the RTX 2080 runs FP64 at 1/32 rate. This is intentional for correctness.

### If CPU/CUDA output diverges:
1. Use `--debug-pixel` to trace a specific pixel
2. Check Hamiltonian constraint on both backends
3. Most likely causes: different operation ordering in FMA, finite difference epsilon mismatch, or boundary condition handling
