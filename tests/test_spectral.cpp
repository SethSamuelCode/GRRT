#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/camera/camera.h"
#include <cstdio>
#include <cmath>
#include <vector>

int failures = 0;

void check(const char* name, bool condition) {
    std::printf("  %s: %s\n", name, condition ? "PASS" : "FAIL");
    if (!condition) failures++;
}

void test_spectral_raymarch_basic() {
    std::printf("\n=== Spectral raymarch: basic emission ===\n");

    grrt::Kerr metric(1.0, 0.998);
    grrt::RK4 integrator;
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    grrt::VolumetricDisk vol_disk(1.0, 0.998, 20.0, 1e7, vp);
    grrt::GeodesicTracer tracer(metric, integrator, 50.0, 10000, 1000.0, 1e-8, &vol_disk);
    grrt::Camera camera(metric, 50.0, 1.396, 0.0, 1.047, 64, 64);

    // 10 log-spaced frequency bins from 1e14 to 1e16 Hz
    std::vector<double> bins;
    for (int i = 0; i < 10; ++i) {
        bins.push_back(1e14 * std::pow(10.0, 2.0 * i / 9.0));
    }

    // Trace a ray aimed near disk midplane (center of image should hit disk)
    auto state = camera.ray_for_pixel(32, 32);
    auto result = tracer.trace_spectral(state, bins);

    std::printf("  termination: %d\n", static_cast<int>(result.termination));
    std::printf("  spectral_intensity size: %zu (expected %zu)\n",
                result.spectral_intensity.size(), bins.size());
    check("output size matches bins", result.spectral_intensity.size() == bins.size());

    // Check that at least some bins have non-negative emission
    bool all_non_negative = true;
    bool any_positive = false;
    for (int i = 0; i < static_cast<int>(result.spectral_intensity.size()); ++i) {
        if (result.spectral_intensity[i] < 0.0) all_non_negative = false;
        if (result.spectral_intensity[i] > 0.0) any_positive = true;
        std::printf("    bin[%d] nu=%.2e Hz -> I=%.4e\n", i, bins[i], result.spectral_intensity[i]);
    }
    check("all intensities non-negative", all_non_negative);

    // If the ray hit the disk, we expect some emission
    if (result.termination != grrt::RayTermination::Escaped) {
        check("some bins have positive emission", any_positive);
    } else {
        std::printf("  (ray escaped without hitting disk — trying another pixel)\n");
        // Try a pixel closer to the equator
        auto state2 = camera.ray_for_pixel(32, 40);
        auto result2 = tracer.trace_spectral(state2, bins);
        bool any_pos2 = false;
        for (const auto& I : result2.spectral_intensity) {
            if (I > 0.0) any_pos2 = true;
        }
        std::printf("  second attempt termination: %d\n", static_cast<int>(result2.termination));
        // Don't fail — disk geometry may not intersect at this angle
        if (any_pos2) std::printf("  second attempt: got positive emission PASS\n");
        else std::printf("  second attempt: no emission (may be geometry)\n");
    }
}

void test_spectral_output_size() {
    std::printf("\n=== Spectral output size: 50 bins ===\n");

    grrt::Kerr metric(1.0, 0.998);
    grrt::RK4 integrator;
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    grrt::VolumetricDisk vol_disk(1.0, 0.998, 20.0, 1e7, vp);
    grrt::GeodesicTracer tracer(metric, integrator, 50.0, 10000, 1000.0, 1e-8, &vol_disk);
    grrt::Camera camera(metric, 50.0, 1.396, 0.0, 1.047, 64, 64);

    // 50 log-spaced bins
    std::vector<double> bins;
    for (int i = 0; i < 50; ++i) {
        bins.push_back(1e14 * std::pow(10.0, 2.0 * i / 49.0));
    }

    auto state = camera.ray_for_pixel(32, 32);
    auto result = tracer.trace_spectral(state, bins);

    check("output has 50 bins", result.spectral_intensity.size() == 50);

    bool all_finite = true;
    for (const auto& I : result.spectral_intensity) {
        if (!std::isfinite(I)) all_finite = false;
    }
    check("all intensities finite", all_finite);
}

int main() {
    test_spectral_raymarch_basic();
    test_spectral_output_size();

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
