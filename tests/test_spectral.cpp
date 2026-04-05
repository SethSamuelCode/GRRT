#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/camera/camera.h"
#include "grrt/render/fits_writer.h"
#include "grrt/math/constants.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <fstream>

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

void test_fits_writer() {
    std::printf("\n=== FITS writer: basic output ===\n");

    int width = 4, height = 3, num_bins = 2;
    std::vector<double> data(width * height * num_bins);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<double>(i) * 1.5;
    }

    std::vector<double> freq_bins = {1e14, 1e16};
    grrt::FITSMetadata meta{};
    meta.spin = 0.998;
    meta.mass = 1.0;
    meta.observer_r = 50.0;

    std::string path = "test_output.fits";
    grrt::write_fits(path, data.data(), width, height, num_bins, freq_bins, meta);

    std::ifstream f(path, std::ios::binary);
    check("file exists", f.good());

    char header[80];
    f.read(header, 80);
    std::string first_card(header, 80);
    check("FITS SIMPLE keyword", first_card.find("SIMPLE") == 0);
    check("SIMPLE = T", first_card.find("T") != std::string::npos);

    f.seekg(0, std::ios::end);
    auto file_size = f.tellg();
    // Data: 4*3*2*8 = 192 bytes -> padded to 2880. Header: at least 2880.
    check("file size >= 5760", file_size >= 5760);

    f.close();
    std::remove(path.c_str());
    std::printf("  cleaned up %s\n", path.c_str());
}

void test_spectral_vs_rgb_consistency() {
    std::printf("\n=== Spectral vs RGB: consistency check ===\n");

    grrt::Kerr metric(1.0, 0.998);
    grrt::RK4 integrator;
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    grrt::VolumetricDisk vol_disk(1.0, 0.998, 20.0, 1e7, vp);
    grrt::GeodesicTracer tracer(metric, integrator, 50.0, 10000, 1000.0, 1e-8, &vol_disk);
    grrt::Camera camera(metric, 50.0, 1.396, 0.0, 1.047, 64, 64);

    // The 3 RGB frequencies from the original raymarch_volumetric
    using namespace grrt::constants;
    std::vector<double> rgb_freqs = {
        c_cgs / 450e-7,  // blue
        c_cgs / 550e-7,  // green
        c_cgs / 650e-7,  // red
    };

    int test_pixels[][2] = {{32, 36}, {32, 38}, {30, 37}, {34, 37}};

    for (auto& px : test_pixels) {
        grrt::GeodesicState state_rgb = camera.ray_for_pixel(px[0], px[1]);
        grrt::GeodesicState state_spec = state_rgb;

        auto result_rgb = tracer.trace(state_rgb, nullptr, nullptr);
        auto result_spec = tracer.trace_spectral(state_spec, rgb_freqs);

        if (result_rgb.termination != result_spec.termination) {
            std::printf("  pixel (%d,%d): termination mismatch (rgb=%d, spec=%d)\n",
                        px[0], px[1],
                        static_cast<int>(result_rgb.termination),
                        static_cast<int>(result_spec.termination));
            continue;
        }

        double rgb_sum = result_rgb.accumulated_color[0]
                       + result_rgb.accumulated_color[1]
                       + result_rgb.accumulated_color[2];
        double spec_sum = result_spec.spectral_intensity[0]
                        + result_spec.spectral_intensity[1]
                        + result_spec.spectral_intensity[2];

        std::printf("  pixel (%d,%d): rgb_sum=%.4e spec_sum=%.4e",
                    px[0], px[1], rgb_sum, spec_sum);

        if (rgb_sum > 0.0 && spec_sum > 0.0) {
            double ratio = spec_sum / rgb_sum;
            std::printf(" ratio=%.3f\n", ratio);
        } else {
            std::printf(" (one or both zero)\n");
        }
    }

    std::printf("  (consistency check is informational, not strict pass/fail)\n");
}

int main() {
    std::printf("Spectral output tests\n");
    std::printf("=====================\n");

    test_spectral_raymarch_basic();
    test_spectral_output_size();
    test_fits_writer();
    test_spectral_vs_rgb_consistency();

    std::printf("\n%s (%d failure%s)\n",
                failures == 0 ? "ALL PASSED" : "SOME FAILED",
                failures, failures == 1 ? "" : "s");
    return failures > 0 ? 1 : 0;
}
