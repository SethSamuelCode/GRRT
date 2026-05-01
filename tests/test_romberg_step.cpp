// tests/test_romberg_step.cpp
//
// Unit tests for romberg_step using synthetic StepSampler implementations.
// We test against analytic dτ values for closed-form integrands.

#include "grrt/geodesic/romberg_step.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/spacetime/kerr.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

using namespace grrt;

int failures = 0;

#define EXPECT_NEAR(actual, expected, tol)                                    \
    do {                                                                      \
        double a = (actual), e = (expected), t = (tol);                       \
        if (std::abs(a - e) > t) {                                            \
            std::printf("FAIL %s:%d: %s ≈ %s (got %.6e, expected %.6e, tol %.2e)\n", \
                        __FILE__, __LINE__, #actual, #expected, a, e, t);     \
            failures++;                                                       \
            return;                                                           \
        }                                                                     \
    } while (0)

// --- Synthetic samplers ---

// Per-channel constant integrand: integrand[ch] = ks[ch].
struct PerChannelConstantSampler : StepSampler {
    std::vector<double> ks;
    explicit PerChannelConstantSampler(std::vector<double> values)
        : ks(std::move(values)) {}
    bool sample_integrand(const GeodesicState& /*state*/,
                          std::span<const double> channels,
                          std::span<double> integrand) const override {
        for (size_t i = 0; i < channels.size(); ++i) {
            integrand[i] = (i < ks.size()) ? ks[i] : 0.0;
        }
        return true;
    }
};

// Constant integrand: κρ|p·u_emit| = K everywhere. Then dτ = K·Δλ exactly.
struct ConstantSampler : StepSampler {
    double K;
    explicit ConstantSampler(double k) : K(k) {}
    bool sample_integrand(const GeodesicState& /*state*/,
                          std::span<const double> channels,
                          std::span<double> integrand) const override {
        for (size_t i = 0; i < channels.size(); ++i) integrand[i] = K;
        return true;
    }
};

// Smooth nonlinear integrand for order-of-convergence checks.
// integrand = sin(r)  → analytic dτ over a path that doesn't move r much
// is approximately sin(r₀)·ds plus small corrections. We use the magnitude
// of (full - half) to verify the estimator's order.
struct SmoothNonlinearSampler : StepSampler {
    bool sample_integrand(const GeodesicState& state,
                          std::span<const double> channels,
                          std::span<double> integrand) const override {
        const double r = state.position[1];
        for (size_t i = 0; i < channels.size(); ++i) integrand[i] = std::sin(r);
        return true;
    }
};

// Test 1: a constant integrand should give dτ = K·ds exactly,
// and Romberg's full and half estimates should agree (max_err == 0).
static void test_constant_integrand() {
    ConstantSampler sampler{2.5};
    Kerr metric(1.0, 0.0);  // Schwarzschild as a Kerr-with-spin-0
    RK4 integrator;

    // Valid radial null geodesic at r=10 in Schwarzschild (M=1):
    // g^tt p_t^2 + g^rr p_r^2 = 0  →  |p_t|/|p_r| = 1 - 2M/r = 0.8
    GeodesicState start;
    start.position = {0.0, 10.0, 1.5707963267948966, 0.0};
    start.momentum = {-0.8, 1.0, 0.0, 0.0};  // null radial geodesic at r=10, M=1

    constexpr double channels[] = {550e-7};  // one channel, value irrelevant for ConstantSampler
    const double ds = 0.1;

    RombergStep r = romberg_step(start, ds,
                                  std::span<const double>{channels, 1},
                                  sampler, metric, integrator);

    EXPECT_NEAR(r.dtau[0], 2.5 * 0.1, 1e-12);
    EXPECT_NEAR(r.max_err, 0.0,        1e-12);
    EXPECT_NEAR(r.ds_taken, 0.1,       1e-12);
    if (r.n_channels != 1) {
        std::printf("FAIL: n_channels=%d, expected 1\n", r.n_channels);
        failures++;
    }
}

static void test_multi_channel_batching() {
    PerChannelConstantSampler sampler({1.0, 2.5, 7.0});  // 3 channels
    Kerr metric(1.0, 0.0);
    RK4 integrator;

    GeodesicState start;
    start.position = {0.0, 10.0, 1.5707963267948966, 0.0};
    start.momentum = {-0.8, 1.0, 0.0, 0.0};  // valid radial null geodesic at r=10, M=1

    constexpr double channels[] = {450e-7, 550e-7, 650e-7};
    const double ds = 0.1;

    RombergStep r = romberg_step(start, ds,
                                  std::span<const double>{channels, 3},
                                  sampler, metric, integrator);

    EXPECT_NEAR(r.dtau[0], 1.0 * 0.1, 1e-12);
    EXPECT_NEAR(r.dtau[1], 2.5 * 0.1, 1e-12);
    EXPECT_NEAR(r.dtau[2], 7.0 * 0.1, 1e-12);
    EXPECT_NEAR(r.max_err, 0.0,       1e-12);

    if (r.n_channels != 3) {
        std::printf("FAIL: n_channels=%d, expected 3\n", r.n_channels);
        failures++;
    }
}

static void test_romberg_order_convergence() {
    SmoothNonlinearSampler sampler;
    Kerr metric(1.0, 0.0);
    RK4 integrator;

    GeodesicState start;
    start.position = {0.0, 10.0, 1.5707963267948966, 0.0};
    // Give it a non-degenerate radial momentum so the geodesic moves.
    // Note: this momentum is not strictly null (constraint not enforced),
    // but for this test only the integrand sampling needs to work; the
    // geodesic just needs to actually move so r varies.
    start.momentum = {-1.0, 0.5, 0.0, 0.0};

    constexpr double channels[] = {550e-7};

    auto run = [&](double ds) {
        return romberg_step(start, ds, std::span<const double>{channels, 1},
                            sampler, metric, integrator).max_err;
    };

    const double err_h    = run(0.1);
    const double err_h2   = run(0.05);
    const double err_h4   = run(0.025);

    // The full-vs-composite-trapezoid difference scales as O(ds³): the leading
    // truncation terms cancel to leave a ds³ residual, so halving ds → error/8.
    // We check ratio ∈ (2, 16): lower bound catches degenerate (non-moving)
    // geodesics; upper bound catches blow-up. 2× slack around the theoretical
    // value of 8 absorbs geodesic-curvature contributions and higher-order terms.
    const double ratio_1 = err_h    / err_h2;
    const double ratio_2 = err_h2   / err_h4;

    std::printf("test_romberg_order_convergence: ratio_1=%.3f, ratio_2=%.3f\n",
                ratio_1, ratio_2);

    if (!(ratio_1 > 2.0 && ratio_1 < 16.0)) {
        std::printf("FAIL: order ratio_1=%.3f outside [2,16]\n", ratio_1);
        failures++;
    }
    if (!(ratio_2 > 2.0 && ratio_2 < 16.0)) {
        std::printf("FAIL: order ratio_2=%.3f outside [2,16]\n", ratio_2);
        failures++;
    }
}

int main() {
    std::printf("Running test_romberg_step...\n");
    test_constant_integrand();
    test_multi_channel_batching();
    test_romberg_order_convergence();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
