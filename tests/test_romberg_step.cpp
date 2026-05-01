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

int main() {
    std::printf("Running test_romberg_step...\n");
    test_constant_integrand();
    test_multi_channel_batching();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
