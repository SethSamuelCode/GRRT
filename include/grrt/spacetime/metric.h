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
