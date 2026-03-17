#include "grrt/render/tonemapper.h"
#include <cmath>

namespace grrt {

Vec3 ToneMapper::apply(const Vec3& hdr) const {
    auto reinhard = [](double c) { return c / (1.0 + c); };
    auto gamma = [](double v) { return std::pow(v, 1.0 / 2.2); };

    return {{
        gamma(reinhard(hdr[0])),
        gamma(reinhard(hdr[1])),
        gamma(reinhard(hdr[2]))
    }};
}

} // namespace grrt
