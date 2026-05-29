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

/// Whether the raymarch should terminate at the disk's outer radial boundary.
///
/// True only when the photon is at/beyond r_max AND moving outward
/// (dr/dλ >= 0) — i.e. genuinely leaving the disk's radial extent. A photon
/// beyond r_max but moving INWARD (dr/dλ < 0) is entering the disk from outside
/// the outer rim; it must keep marching (density is zero out here, so no
/// emission is added or double-counted) or the disk crossing just inside the
/// rim is missed entirely. Using a position-only test (r > r_max) instead bails
/// on these inward-bound rays on the first step, blanking the lensed outer rim.
///
/// @param r           Boyer-Lindquist radius at the current step [M].
/// @param r_max       Disk outer radius r_outer [M].
/// @param dr_dlambda  Contravariant radial velocity dr/dλ at the current step.
inline bool raymarch_exits_outer(double r, double r_max, double dr_dlambda) {
    return r > r_max && dr_dlambda >= 0.0;
}

} // namespace grrt

#endif
