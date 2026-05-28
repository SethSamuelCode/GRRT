#ifndef GRRT_GEODESIC_RAYMARCH_STEP_CONTROL_H
#define GRRT_GEODESIC_RAYMARCH_STEP_CONTROL_H

#include <algorithm>
#include <cmath>

namespace grrt {

/// Returns true if a raymarch step whose Cartesian-z endpoints are z0 and z1
/// should be refined (shrunk) for z-resolution.
///
/// The step needs refinement when its signed z-interval
/// [min(z0,z1), max(z0,z1)] overlaps the disk's vertical extent [-env, +env]
/// AND its vertical excursion |z1 - z0| exceeds quarter_H (a quarter scale
/// height). The signed-interval overlap test — NOT endpoint membership — is
/// what catches transversal transits, where both endpoints lie outside the
/// envelope (|z| > env) but the path crosses z = 0 through dense disk material.
///
/// @param z0       Cartesian z (= r*cos(theta)) at the step start.
/// @param z1       Cartesian z at the step end.
/// @param quarter_H  H/4 at a representative radius; the max allowed |Δz|.
/// @param env      Disk vertical envelope z_max(r) + H(r) at that radius.
inline bool step_needs_z_refinement(double z0, double z1,
                                    double quarter_H, double env) {
    const double dz = std::abs(z1 - z0);
    const bool crosses = (std::min(z0, z1) < env) && (std::max(z0, z1) > -env);
    return crosses && dz > quarter_H;
}

} // namespace grrt

#endif
