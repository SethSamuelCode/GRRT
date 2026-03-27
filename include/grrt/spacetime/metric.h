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

    // Innermost stable circular orbit radius
    virtual double isco_radius() const = 0;

    // Geodesic force dp_μ/dλ = ½ ∂g_{αβ}/∂x^μ ẋ^α ẋ^β
    // Default: finite differences on g_lower. Override for analytical derivatives.
    virtual Vec4 geodesic_force(const Vec4& x, const Vec4& velocity) const;

    // Fused derivative computation: computes both dx^μ/dλ and dp_μ/dλ in one call.
    // Shares trig/metric intermediates between g_upper and geodesic_force.
    // Default: calls g_upper + contract + geodesic_force separately.
    struct DerivResult { Vec4 dx; Vec4 dp; };
    virtual DerivResult compute_derivatives(const Vec4& x, const Vec4& p) const;
};

} // namespace grrt

#endif
