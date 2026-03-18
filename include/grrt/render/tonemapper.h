#ifndef GRRT_TONEMAPPER_H
#define GRRT_TONEMAPPER_H

#include "grrt/math/vec3.h"

namespace grrt {

class ToneMapper {
public:
    // Two-pass auto-exposure tone mapping applied to entire framebuffer
    // Computes log-average luminance, then applies extended Reinhard with auto white point
    void apply_all(float* framebuffer, int width, int height) const;

private:
    double key_value_ = 0.18;  // Photographic middle gray
};

} // namespace grrt

#endif
