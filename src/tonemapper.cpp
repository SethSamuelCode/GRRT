#include "grrt/render/tonemapper.h"
#include <cmath>
#include <algorithm>

namespace grrt {

void ToneMapper::apply_all(float* framebuffer, int width, int height) const {
    const int num_pixels = width * height;

    // Pass 1: Compute log-average luminance and max luminance
    double log_sum = 0.0;
    double L_max = 0.0;
    int lit_pixels = 0;
    constexpr double delta = 1e-6;
    constexpr double L_threshold = 1e-10;  // Minimum luminance to count as "lit"

    for (int i = 0; i < num_pixels; ++i) {
        int idx = i * 4;
        double R = std::max(0.0, static_cast<double>(framebuffer[idx + 0]));
        double G = std::max(0.0, static_cast<double>(framebuffer[idx + 1]));
        double B = std::max(0.0, static_cast<double>(framebuffer[idx + 2]));

        double L = 0.2126 * R + 0.7152 * G + 0.0722 * B;
        if (L > L_threshold) {
            log_sum += std::log(delta + L);
            ++lit_pixels;
        }
        if (L > L_max) L_max = L;
    }

    // All-black frame: nothing to tone map
    if (L_max <= 0.0 || lit_pixels == 0) return;

    double L_avg = std::exp(log_sum / lit_pixels);
    double exposure = key_value_ / L_avg;
    double L_white = L_max * exposure;
    double L_white2 = L_white * L_white;

    // Pass 2: Apply tone mapping in-place
    auto gamma = [](double v) {
        return std::pow(std::clamp(v, 0.0, 1.0), 1.0 / 2.2);
    };

    for (int i = 0; i < num_pixels; ++i) {
        int idx = i * 4;
        double R = std::max(0.0, static_cast<double>(framebuffer[idx + 0]));
        double G = std::max(0.0, static_cast<double>(framebuffer[idx + 1]));
        double B = std::max(0.0, static_cast<double>(framebuffer[idx + 2]));

        double L = 0.2126 * R + 0.7152 * G + 0.0722 * B;

        if (L <= 0.0) {
            framebuffer[idx + 0] = 0.0f;
            framebuffer[idx + 1] = 0.0f;
            framebuffer[idx + 2] = 0.0f;
            continue;
        }

        // Extended Reinhard with auto white point
        double L_scaled = L * exposure;
        double L_mapped = L_scaled * (1.0 + L_scaled / L_white2) / (1.0 + L_scaled);
        L_mapped = std::clamp(L_mapped, 0.0, 1.0);

        // Scale RGB by luminance ratio to preserve chrominance
        double scale = L_mapped / L;

        framebuffer[idx + 0] = static_cast<float>(gamma(R * scale));
        framebuffer[idx + 1] = static_cast<float>(gamma(G * scale));
        framebuffer[idx + 2] = static_cast<float>(gamma(B * scale));
    }
}

} // namespace grrt
