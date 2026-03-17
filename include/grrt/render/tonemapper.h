#ifndef GRRT_TONEMAPPER_H
#define GRRT_TONEMAPPER_H

#include "grrt/math/vec3.h"

namespace grrt {

class ToneMapper {
public:
    Vec3 apply(const Vec3& hdr) const;
};

} // namespace grrt

#endif
