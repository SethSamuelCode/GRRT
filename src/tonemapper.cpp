#include "grrt/render/tonemapper.h"
#include <cmath>
#include <algorithm>

namespace grrt {

Vec3 ToneMapper::apply(const Vec3& hdr) const {
    // Compute luminance (perceptual brightness)
    double L = 0.2126 * hdr[0] + 0.7152 * hdr[1] + 0.0722 * hdr[2];
    if (L <= 0.0) return {};

    // Exposure: scale down to bring HDR values into tone-mappable range
    double exposure = 1e-8;
    double L_scaled = L * exposure;

    // Luminance-based Reinhard: preserves color ratios better than per-channel
    // Maps [0, ∞) → [0, 1)
    double L_mapped = L_scaled / (1.0 + L_scaled);

    // Scale RGB by the luminance ratio to preserve chrominance
    double scale = L_mapped / L;

    auto gamma = [](double v) {
        return std::pow(std::clamp(v, 0.0, 1.0), 1.0 / 2.2);
    };

    return {{
        gamma(hdr[0] * scale),
        gamma(hdr[1] * scale),
        gamma(hdr[2] * scale)
    }};
}

} // namespace grrt
