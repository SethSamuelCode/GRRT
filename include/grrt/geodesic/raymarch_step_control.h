#ifndef GRRT_GEODESIC_RAYMARCH_STEP_CONTROL_H
#define GRRT_GEODESIC_RAYMARCH_STEP_CONTROL_H

#include <algorithm>
#include <cmath>

namespace grrt {

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
