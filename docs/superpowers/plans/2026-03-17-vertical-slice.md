# Schwarzschild Shadow Vertical Slice — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a Schwarzschild black hole shadow by tracing null geodesics end-to-end, producing a PNG of a black circle on white background.

**Architecture:** Bottom-up: math types → metric → integrator → tracer → camera → renderer → wire into API. Each layer depends only on layers below it.

**Tech Stack:** C++23, CMake, MSVC 2022, OpenMP, stb_image_write

**Spec:** `docs/superpowers/specs/2026-03-17-vertical-slice-design.md`

**Build command:** `build.bat` (or from bash: `"/c/Program Files/CMake/bin/cmake.exe" -B build -G "Visual Studio 17 2022" -S . && "/c/Program Files/CMake/bin/cmake.exe" --build build --config Release`)

**Run command:** `build/Release/grrt-cli.exe` (produces `output.png`)

---

## Chunk 1: Math Types + Schwarzschild Metric

### Task 1: Vec4

**Files:**
- Create: `include/grrt/math/vec4.h`

- [ ] **Step 1: Create `vec4.h`**

Header-only 4-component vector of doubles. Indices 0–3 map to `(t, r, θ, φ)`.

```cpp
#ifndef GRRT_VEC4_H
#define GRRT_VEC4_H

namespace grrt {

struct Vec4 {
    double data[4]{};

    double& operator[](int i) { return data[i]; }
    const double& operator[](int i) const { return data[i]; }

    Vec4 operator+(const Vec4& o) const {
        return {{data[0]+o[0], data[1]+o[1], data[2]+o[2], data[3]+o[3]}};
    }

    Vec4 operator-(const Vec4& o) const {
        return {{data[0]-o[0], data[1]-o[1], data[2]-o[2], data[3]-o[3]}};
    }

    Vec4 operator*(double s) const {
        return {{data[0]*s, data[1]*s, data[2]*s, data[3]*s}};
    }
};

inline Vec4 operator*(double s, const Vec4& v) {
    return v * s;
}

} // namespace grrt

#endif
```

- [ ] **Step 2: Build to verify no compile errors**

Run: `build.bat`
Expected: Compiles successfully (Vec4 isn't used yet, but the header should parse).

### Task 2: Matrix4

**Files:**
- Create: `include/grrt/math/matrix4.h`

- [ ] **Step 1: Create `matrix4.h`**

Header-only 4×4 matrix. Used for metric tensors `g_μν` and `g^μν`.

```cpp
#ifndef GRRT_MATRIX4_H
#define GRRT_MATRIX4_H

#include "grrt/math/vec4.h"

namespace grrt {

struct Matrix4 {
    double m[4][4]{};

    // Contract: result_μ = Σ_ν M_μν v^ν
    // Used for both lowering (g_lower.contract(v)) and raising (g_upper.contract(v))
    Vec4 contract(const Vec4& v) const {
        Vec4 result;
        for (int mu = 0; mu < 4; ++mu) {
            double sum = 0.0;
            for (int nu = 0; nu < 4; ++nu) {
                sum += m[mu][nu] * v[nu];
            }
            result[mu] = sum;
        }
        return result;
    }

    // Create a diagonal matrix
    static Matrix4 diagonal(double a, double b, double c, double d) {
        Matrix4 mat;
        mat.m[0][0] = a;
        mat.m[1][1] = b;
        mat.m[2][2] = c;
        mat.m[3][3] = d;
        return mat;
    }

    // Inverse of a diagonal matrix (sufficient for Schwarzschild)
    // For general metrics (Kerr), this will need a full 4x4 inverse later.
    Matrix4 inverse_diagonal() const {
        return diagonal(
            1.0 / m[0][0],
            1.0 / m[1][1],
            1.0 / m[2][2],
            1.0 / m[3][3]
        );
    }
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Build to verify**

Run: `build.bat`
Expected: Compiles successfully.

### Task 3: Metric Interface

**Files:**
- Create: `include/grrt/spacetime/metric.h`

- [ ] **Step 1: Create `metric.h`**

Abstract base class that all spacetime metrics implement.

```cpp
#ifndef GRRT_METRIC_H
#define GRRT_METRIC_H

#include "grrt/math/matrix4.h"
#include "grrt/math/vec4.h"

namespace grrt {

class Metric {
public:
    virtual ~Metric() = default;

    // Covariant metric tensor g_μν at position x
    virtual Matrix4 g_lower(const Vec4& x) const = 0;

    // Contravariant metric tensor g^μν at position x
    virtual Matrix4 g_upper(const Vec4& x) const = 0;

    // Event horizon radius
    virtual double horizon_radius() const = 0;
};

} // namespace grrt

#endif
```

### Task 4: Schwarzschild Metric

**Files:**
- Create: `include/grrt/spacetime/schwarzschild.h`
- Create: `src/schwarzschild.cpp`
- Modify: `CMakeLists.txt` (add `src/schwarzschild.cpp`)

- [ ] **Step 1: Create `schwarzschild.h`**

```cpp
#ifndef GRRT_SCHWARZSCHILD_H
#define GRRT_SCHWARZSCHILD_H

#include "grrt/spacetime/metric.h"

namespace grrt {

class Schwarzschild : public Metric {
public:
    explicit Schwarzschild(double mass);

    Matrix4 g_lower(const Vec4& x) const override;
    Matrix4 g_upper(const Vec4& x) const override;
    double horizon_radius() const override;

private:
    double mass_;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `schwarzschild.cpp`**

```cpp
#include "grrt/spacetime/schwarzschild.h"
#include <cmath>
#include <algorithm>

namespace grrt {

Schwarzschild::Schwarzschild(double mass) : mass_(mass) {}

Matrix4 Schwarzschild::g_lower(const Vec4& x) const {
    const double r = x[1];
    const double theta = x[2];
    const double sin_theta = std::max(std::abs(std::sin(theta)), 1e-10);

    const double f = 1.0 - 2.0 * mass_ / r;  // 1 - 2M/r

    return Matrix4::diagonal(
        -f,              // g_tt
        1.0 / f,         // g_rr
        r * r,           // g_θθ
        r * r * sin_theta * sin_theta  // g_φφ
    );
}

Matrix4 Schwarzschild::g_upper(const Vec4& x) const {
    // For diagonal metric, g^μν = 1/g_μν
    return g_lower(x).inverse_diagonal();
}

double Schwarzschild::horizon_radius() const {
    return 2.0 * mass_;
}

} // namespace grrt
```

- [ ] **Step 3: Add to CMakeLists.txt**

Change the `add_library` block in `CMakeLists.txt`:

```cmake
add_library(grrt SHARED
    src/api.cpp
    src/schwarzschild.cpp
)
```

- [ ] **Step 4: Build to verify**

Run: `build.bat`
Expected: Compiles and links successfully.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/math/vec4.h include/grrt/math/matrix4.h include/grrt/spacetime/metric.h include/grrt/spacetime/schwarzschild.h src/schwarzschild.cpp CMakeLists.txt
git commit -m "feat: add math types (Vec4, Matrix4) and Schwarzschild metric"
```

---

## Chunk 2: Geodesic Integration

### Task 5: GeodesicState + Integrator Interface

**Files:**
- Create: `include/grrt/geodesic/integrator.h`

- [ ] **Step 1: Create `integrator.h`**

Defines the state vector and abstract integrator interface.

```cpp
#ifndef GRRT_INTEGRATOR_H
#define GRRT_INTEGRATOR_H

#include "grrt/math/vec4.h"
#include "grrt/spacetime/metric.h"

namespace grrt {

struct GeodesicState {
    Vec4 position;  // x^μ = (t, r, θ, φ) — contravariant
    Vec4 momentum;  // p_μ — covariant
};

class Integrator {
public:
    virtual ~Integrator() = default;

    // Advance state by one step of size dlambda
    virtual GeodesicState step(const Metric& metric,
                               const GeodesicState& state,
                               double dlambda) const = 0;
};

} // namespace grrt

#endif
```

### Task 6: RK4 Integrator

**Files:**
- Create: `include/grrt/geodesic/rk4.h`
- Create: `src/rk4.cpp`
- Modify: `CMakeLists.txt` (add `src/rk4.cpp`)

- [ ] **Step 1: Create `rk4.h`**

```cpp
#ifndef GRRT_RK4_H
#define GRRT_RK4_H

#include "grrt/geodesic/integrator.h"

namespace grrt {

class RK4 : public Integrator {
public:
    GeodesicState step(const Metric& metric,
                       const GeodesicState& state,
                       double dlambda) const override;

private:
    // Compute derivatives: (dx^μ/dλ, dp_μ/dλ)
    // Returns a GeodesicState used as the derivative (position_dot, momentum_dot)
    static GeodesicState derivatives(const Metric& metric,
                                     const GeodesicState& state);

    // Finite-difference epsilon for metric derivatives
    static constexpr double fd_epsilon = 1e-6;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/rk4.cpp`**

This is the core physics engine. Hamilton's equations are:
- `dx^μ/dλ = g^μν p_ν` (raise the covariant momentum)
- `dp_μ/dλ = -½ Σ_{α,β} (∂g^αβ/∂x^μ) p_α p_β` (force from curved spacetime)

```cpp
#include "grrt/geodesic/rk4.h"

namespace grrt {

GeodesicState RK4::derivatives(const Metric& metric, const GeodesicState& state) {
    const Vec4& x = state.position;
    const Vec4& p = state.momentum;

    // dx^μ/dλ = g^μν p_ν  (raise momentum with inverse metric)
    Matrix4 g_inv = metric.g_upper(x);
    Vec4 dx = g_inv.contract(p);

    // dp_μ/dλ = -½ Σ_{α,β} (∂g^αβ/∂x^μ) p_α p_β
    // Compute ∂g^αβ/∂x^μ by central finite differences
    Vec4 dp;
    for (int mu = 0; mu < 4; ++mu) {
        // Perturb position along coordinate mu
        Vec4 x_plus = x;
        Vec4 x_minus = x;
        x_plus[mu] += fd_epsilon;
        x_minus[mu] -= fd_epsilon;

        Matrix4 g_inv_plus = metric.g_upper(x_plus);
        Matrix4 g_inv_minus = metric.g_upper(x_minus);

        // Sum over all α, β
        double force = 0.0;
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                double dg = (g_inv_plus.m[a][b] - g_inv_minus.m[a][b]) / (2.0 * fd_epsilon);
                force += dg * p[a] * p[b];
            }
        }
        dp[mu] = -0.5 * force;
    }

    return {dx, dp};
}

GeodesicState RK4::step(const Metric& metric, const GeodesicState& state, double dl) const {
    // Classic RK4: compute 4 derivative evaluations, combine
    auto add = [](const GeodesicState& s, const GeodesicState& ds, double h) -> GeodesicState {
        return {s.position + ds.position * h, s.momentum + ds.momentum * h};
    };

    GeodesicState k1 = derivatives(metric, state);
    GeodesicState k2 = derivatives(metric, add(state, k1, dl * 0.5));
    GeodesicState k3 = derivatives(metric, add(state, k2, dl * 0.5));
    GeodesicState k4 = derivatives(metric, add(state, k3, dl));

    Vec4 new_pos = state.position
        + (k1.position + k2.position * 2.0 + k3.position * 2.0 + k4.position) * (dl / 6.0);
    Vec4 new_mom = state.momentum
        + (k1.momentum + k2.momentum * 2.0 + k3.momentum * 2.0 + k4.momentum) * (dl / 6.0);

    return {new_pos, new_mom};
}

} // namespace grrt
```

- [ ] **Step 3: Add `src/rk4.cpp` to CMakeLists.txt**

```cmake
add_library(grrt SHARED
    src/api.cpp
    src/schwarzschild.cpp
    src/rk4.cpp
)
```

- [ ] **Step 4: Build to verify**

Run: `build.bat`
Expected: Compiles and links successfully.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/geodesic/integrator.h include/grrt/geodesic/rk4.h src/rk4.cpp CMakeLists.txt
git commit -m "feat: add RK4 geodesic integrator with Hamiltonian equations"
```

---

## Chunk 3: Geodesic Tracer + Camera

### Task 7: Geodesic Tracer

**Files:**
- Create: `include/grrt/geodesic/geodesic_tracer.h`
- Create: `src/geodesic_tracer.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `geodesic_tracer.h`**

```cpp
#ifndef GRRT_GEODESIC_TRACER_H
#define GRRT_GEODESIC_TRACER_H

#include "grrt/geodesic/integrator.h"

namespace grrt {

enum class RayTermination {
    Horizon,   // Hit the event horizon
    Escaped,   // Escaped to large radius
    MaxSteps   // Exceeded step limit
};

class GeodesicTracer {
public:
    GeodesicTracer(const Metric& metric, const Integrator& integrator,
                   int max_steps = 10000, double r_escape = 1000.0);

    RayTermination trace(GeodesicState& state) const;

private:
    const Metric& metric_;
    const Integrator& integrator_;
    int max_steps_;
    double r_escape_;
    double horizon_epsilon_ = 0.01;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/geodesic_tracer.cpp`**

```cpp
#include "grrt/geodesic/geodesic_tracer.h"

namespace grrt {

GeodesicTracer::GeodesicTracer(const Metric& metric, const Integrator& integrator,
                               int max_steps, double r_escape)
    : metric_(metric), integrator_(integrator),
      max_steps_(max_steps), r_escape_(r_escape) {}

RayTermination GeodesicTracer::trace(GeodesicState& state) const {
    const double r_horizon = metric_.horizon_radius();

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];

        // Check termination conditions
        if (r < r_horizon + horizon_epsilon_) {
            return RayTermination::Horizon;
        }
        if (r > r_escape_) {
            return RayTermination::Escaped;
        }

        // Step size scales with r: smaller near the hole
        const double dlambda = 0.01 * r;

        state = integrator_.step(metric_, state, dlambda);
    }

    return RayTermination::MaxSteps;
}

} // namespace grrt
```

**Note on backward tracing:** The camera constructs the initial momentum pointing *toward* the hole (via the `e3` tetrad vector = radially inward). With a positive step size, the integrator follows that direction naturally. No sign flip needed — the "backward" aspect is encoded in the momentum direction, not the step sign.

- [ ] **Step 3: Add to CMakeLists.txt**

```cmake
add_library(grrt SHARED
    src/api.cpp
    src/schwarzschild.cpp
    src/rk4.cpp
    src/geodesic_tracer.cpp
)
```

- [ ] **Step 4: Build to verify**

Run: `build.bat`
Expected: Compiles and links.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/geodesic/geodesic_tracer.h src/geodesic_tracer.cpp CMakeLists.txt
git commit -m "feat: add geodesic tracer with horizon/escape termination"
```

### Task 8: Camera

**Files:**
- Create: `include/grrt/camera/camera.h`
- Create: `src/camera.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `camera.h`**

```cpp
#ifndef GRRT_CAMERA_H
#define GRRT_CAMERA_H

#include "grrt/math/vec4.h"
#include "grrt/spacetime/metric.h"
#include "grrt/geodesic/integrator.h"

namespace grrt {

class Camera {
public:
    Camera(const Metric& metric, double r_obs, double theta_obs, double phi_obs,
           double fov, int width, int height);

    // Generate initial geodesic state for pixel (i, j)
    GeodesicState ray_for_pixel(int i, int j) const;

private:
    const Metric& metric_;
    Vec4 position_;  // Observer's 4-position
    double fov_;
    int width_;
    int height_;

    // Orthonormal tetrad at observer's position
    Vec4 e0_;  // Timelike (normalized 4-velocity)
    Vec4 e1_;  // Right (φ direction)
    Vec4 e2_;  // Up (θ direction)
    Vec4 e3_;  // Forward (radially inward, toward hole)

    void build_tetrad();
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/camera.cpp`**

The tetrad construction uses Gram-Schmidt with the spacetime metric inner product.

```cpp
#include "grrt/camera/camera.h"
#include <cmath>

namespace grrt {

// Metric inner product: <u, v> = g_μν u^μ v^ν
static double metric_dot(const Matrix4& g, const Vec4& u, const Vec4& v) {
    double sum = 0.0;
    for (int mu = 0; mu < 4; ++mu)
        for (int nu = 0; nu < 4; ++nu)
            sum += g.m[mu][nu] * u[mu] * v[nu];
    return sum;
}

// Normalize a vector: v / sqrt(|<v,v>|)
static Vec4 metric_normalize(const Matrix4& g, const Vec4& v) {
    double norm2 = metric_dot(g, v, v);
    double norm = std::sqrt(std::abs(norm2));
    return v * (1.0 / norm);
}

// Project out: v - <v, e>/<e, e> * e
static Vec4 project_out(const Matrix4& g, const Vec4& v, const Vec4& e) {
    double ve = metric_dot(g, v, e);
    double ee = metric_dot(g, e, e);
    return v - e * (ve / ee);
}

Camera::Camera(const Metric& metric, double r_obs, double theta_obs, double phi_obs,
               double fov, int width, int height)
    : metric_(metric), fov_(fov), width_(width), height_(height) {
    position_[0] = 0.0;       // t
    position_[1] = r_obs;     // r
    position_[2] = theta_obs; // θ
    position_[3] = phi_obs;   // φ
    build_tetrad();
}

void Camera::build_tetrad() {
    Matrix4 g = metric_.g_lower(position_);

    // e_0: static observer 4-velocity u^μ = (1/√(-g_tt), 0, 0, 0)
    double g_tt = g.m[0][0];
    e0_ = Vec4{{1.0 / std::sqrt(-g_tt), 0.0, 0.0, 0.0}};

    // e_3 (forward = radially inward = -r direction)
    // Start with coordinate r-basis vector, negate for inward
    Vec4 r_dir = {{0.0, -1.0, 0.0, 0.0}};
    Vec4 v3 = project_out(g, r_dir, e0_);
    e3_ = metric_normalize(g, v3);

    // e_2 (up = -θ direction, since θ increases downward from pole)
    Vec4 theta_dir = {{0.0, 0.0, -1.0, 0.0}};
    Vec4 v2 = project_out(g, theta_dir, e0_);
    v2 = project_out(g, v2, e3_);
    e2_ = metric_normalize(g, v2);

    // e_1 (right = φ direction)
    Vec4 phi_dir = {{0.0, 0.0, 0.0, 1.0}};
    Vec4 v1 = project_out(g, phi_dir, e0_);
    v1 = project_out(g, v1, e3_);
    v1 = project_out(g, v1, e2_);
    e1_ = metric_normalize(g, v1);
}

GeodesicState Camera::ray_for_pixel(int i, int j) const {
    // Screen angles
    double alpha = (static_cast<double>(i) - width_ / 2.0) * fov_ / width_;
    double beta = (static_cast<double>(j) - height_ / 2.0) * fov_ / width_;

    // Local 3-direction in tetrad frame
    double ca = std::cos(beta) * std::sin(alpha);
    double cb = std::sin(beta);
    double cc = std::cos(beta) * std::cos(alpha);

    // d^μ = -ca * e1 - cb * e2 + cc * e3  (unit spatial vector in tetrad)
    Vec4 d;
    for (int mu = 0; mu < 4; ++mu) {
        d[mu] = -ca * e1_[mu] - cb * e2_[mu] + cc * e3_[mu];
    }

    // Null 4-momentum (contravariant): p^μ = -e0 + d
    Vec4 p_contra;
    for (int mu = 0; mu < 4; ++mu) {
        p_contra[mu] = -e0_[mu] + d[mu];
    }

    // Lower index: p_μ = g_μν p^ν
    Matrix4 g = metric_.g_lower(position_);
    Vec4 p_cov = g.contract(p_contra);

    return {position_, p_cov};
}

} // namespace grrt
```

- [ ] **Step 3: Add to CMakeLists.txt**

```cmake
add_library(grrt SHARED
    src/api.cpp
    src/schwarzschild.cpp
    src/rk4.cpp
    src/geodesic_tracer.cpp
    src/camera.cpp
)
```

- [ ] **Step 4: Build to verify**

Run: `build.bat`
Expected: Compiles and links.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/camera/camera.h src/camera.cpp CMakeLists.txt
git commit -m "feat: add camera with tetrad construction and ray generation"
```

---

## Chunk 4: Renderer + API Wiring + Validation

### Task 9: Renderer

**Files:**
- Create: `include/grrt/render/renderer.h`
- Create: `src/renderer.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create `renderer.h`**

```cpp
#ifndef GRRT_RENDERER_H
#define GRRT_RENDERER_H

#include "grrt/camera/camera.h"
#include "grrt/geodesic/geodesic_tracer.h"

namespace grrt {

class Renderer {
public:
    Renderer(const Camera& camera, const GeodesicTracer& tracer);

    // Render full frame into RGBA float buffer (width * height * 4 floats)
    void render(float* framebuffer, int width, int height) const;

private:
    const Camera& camera_;
    const GeodesicTracer& tracer_;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/renderer.cpp`**

```cpp
#include "grrt/render/renderer.h"

namespace grrt {

Renderer::Renderer(const Camera& camera, const GeodesicTracer& tracer)
    : camera_(camera), tracer_(tracer) {}

void Renderer::render(float* framebuffer, int width, int height) const {
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            GeodesicState state = camera_.ray_for_pixel(i, j);
            RayTermination result = tracer_.trace(state);

            const int idx = (j * width + i) * 4;
            if (result == RayTermination::Escaped) {
                // White pixel
                framebuffer[idx + 0] = 1.0f;
                framebuffer[idx + 1] = 1.0f;
                framebuffer[idx + 2] = 1.0f;
                framebuffer[idx + 3] = 1.0f;
            } else {
                // Black pixel (horizon hit or max steps)
                framebuffer[idx + 0] = 0.0f;
                framebuffer[idx + 1] = 0.0f;
                framebuffer[idx + 2] = 0.0f;
                framebuffer[idx + 3] = 1.0f;
            }
        }
    }
}

} // namespace grrt
```

- [ ] **Step 3: Add to CMakeLists.txt**

```cmake
add_library(grrt SHARED
    src/api.cpp
    src/schwarzschild.cpp
    src/rk4.cpp
    src/geodesic_tracer.cpp
    src/camera.cpp
    src/renderer.cpp
)
```

- [ ] **Step 4: Build to verify**

Run: `build.bat`
Expected: Compiles and links.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/render/renderer.h src/renderer.cpp CMakeLists.txt
git commit -m "feat: add renderer with OpenMP parallel pixel loop"
```

### Task 10: Wire into API

**Files:**
- Modify: `src/api.cpp`
- Modify: `cli/main.cpp` (set metric to Schwarzschild, set max_steps)

- [ ] **Step 1: Update `src/api.cpp`**

Replace the placeholder gradient with the real rendering pipeline. Store the physics objects in `GRRTContext`.

```cpp
#include "grrt/api.h"
#include "grrt/spacetime/schwarzschild.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/camera/camera.h"
#include "grrt/render/renderer.h"
#include <memory>
#include <print>

struct GRRTContext {
    GRRTParams params;
    std::unique_ptr<grrt::Metric> metric;
    std::unique_ptr<grrt::Integrator> integrator;
    std::unique_ptr<grrt::GeodesicTracer> tracer;
    std::unique_ptr<grrt::Camera> camera;
    std::unique_ptr<grrt::Renderer> renderer;
};

GRRTContext* grrt_create(const GRRTParams* params) {
    auto* ctx = new GRRTContext{};
    ctx->params = *params;

    // Defaults for zero-initialized fields
    double mass = params->mass > 0.0 ? params->mass : 1.0;
    double observer_r = params->observer_r > 0.0 ? params->observer_r : 50.0;
    double observer_theta = params->observer_theta > 0.0 ? params->observer_theta : 1.396;
    double fov = params->fov > 0.0 ? params->fov : 1.047;
    int max_steps = params->integrator_max_steps > 0 ? params->integrator_max_steps : 10000;

    // Build pipeline
    ctx->metric = std::make_unique<grrt::Schwarzschild>(mass);
    ctx->integrator = std::make_unique<grrt::RK4>();
    ctx->tracer = std::make_unique<grrt::GeodesicTracer>(
        *ctx->metric, *ctx->integrator, max_steps);
    ctx->camera = std::make_unique<grrt::Camera>(
        *ctx->metric, observer_r, observer_theta, params->observer_phi,
        fov, params->width, params->height);
    ctx->renderer = std::make_unique<grrt::Renderer>(*ctx->camera, *ctx->tracer);

    std::println("grrt: created context ({}x{}, schwarzschild, M={}, r_obs={})",
                 params->width, params->height, mass, observer_r);
    return ctx;
}

void grrt_destroy(GRRTContext* ctx) {
    delete ctx;
}

void grrt_update_params(GRRTContext* ctx, const GRRTParams* params) {
    ctx->params = *params;
    // TODO: rebuild pipeline when params change
}

int grrt_render(GRRTContext* ctx, float* framebuffer) {
    ctx->renderer->render(framebuffer, ctx->params.width, ctx->params.height);
    std::println("grrt: rendered {}x{} frame", ctx->params.width, ctx->params.height);
    return 0;
}

int grrt_render_tile(GRRTContext* /*ctx*/, float* /*buffer*/,
                     int /*x*/, int /*y*/, int /*tile_width*/, int /*tile_height*/) {
    return 0; // Stub
}

void grrt_cancel(GRRTContext* /*ctx*/) {}

float grrt_progress(const GRRTContext* /*ctx*/) {
    return 1.0f;
}

const char* grrt_error(const GRRTContext* /*ctx*/) {
    return nullptr;
}

int grrt_cuda_available(void) {
    return 0;
}
```

- [ ] **Step 2: Update `cli/main.cpp`**

Set metric to Schwarzschild and ensure max_steps is set:

Change line 12 from:
```cpp
    params.metric_type = GRRT_METRIC_KERR;
```
to:
```cpp
    params.metric_type = GRRT_METRIC_SCHWARZSCHILD;
```

Add after line 18 (`params.fov = 1.047;`):
```cpp
    params.integrator_max_steps = 10000;
```

- [ ] **Step 3: Build**

Run: `build.bat`
Expected: Compiles, links, no errors.

- [ ] **Step 4: Run and validate**

Run: `build/Release/grrt-cli.exe`
Expected output:
- Console prints context creation and render completion
- `output.png` shows a **black circle** (shadow) on a **white background**
- Shadow diameter should be roughly **52 pixels** (see spec validation section)

Open `output.png` and visually verify:
1. There is a circular black region in the center
2. The rest of the image is white
3. The shadow is roughly centered
4. The shadow edge is reasonably sharp (not blurry across many pixels)

- [ ] **Step 5: Commit**

```bash
git add src/api.cpp cli/main.cpp
git commit -m "feat: wire Schwarzschild renderer into API — first real render"
```

### Task 11: Debug and Tune (if needed)

This task is a placeholder. If the output from Task 10 doesn't look right:

- [ ] **Step 1: Check Hamiltonian constraint**

Add a diagnostic to `geodesic_tracer.cpp` that computes `H = ½ g^μν p_μ p_ν` at the start and end of each ray. It should remain near zero (< 1e-6). If it drifts, the integrator has a bug.

- [ ] **Step 2: Check single ray**

Trace a single ray (center pixel, straight at the hole) and print position/momentum at each step. Verify `r` decreases smoothly toward the horizon.

- [ ] **Step 3: Check escape ray**

Trace a ray aimed away from the hole (edge pixel). Verify `r` increases toward `r_escape`.

- [ ] **Step 4: Adjust step size if needed**

If the shadow is wrong, try `dlambda = 0.005 * r` (smaller steps) and see if the result converges. If the render is too slow, try `dlambda = 0.02 * r`.

- [ ] **Step 5: Commit any fixes**

```bash
git add -u
git commit -m "fix: tune geodesic integration for correct shadow rendering"
```
