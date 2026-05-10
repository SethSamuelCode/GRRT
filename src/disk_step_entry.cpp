#include "grrt/geodesic/disk_step_entry.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/geodesic/rk4.h"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace grrt {

namespace {

constexpr double kHalfPi = std::numbers::pi / 2.0;

/// Tier A endpoint predicate. Returns true if the geodesic step `(prev, curr)`
/// should trigger the volumetric raymarch. Three OR clauses:
///   1. crossed_midplane — sign of (theta - pi/2) flipped between endpoints,
///      indicating the ray passed through the disk's midplane.
///   2. inside_now — `curr` lies inside the disk's volume envelope.
///   3. near_disk — either endpoint sits within H of the disk's z_max envelope
///      AND `curr.r` is within the disk's r-cylinder. The H pairing is
///      asymmetric: H_curr is paired with z_curr, H_prev with z_prev.
///
/// Byte-for-byte equivalent to the inline predicate at the three call sites
/// in src/geodesic_tracer.cpp: lines 192–202 (RGB trace), 441–449 (debug
/// trace), 585–595 (spectral trace). Extracted in Task 2 of the
/// disk-step-entry helper (see docs/superpowers/specs/2026-05-10-disk-step-entry-design.md §5.2).
bool endpoint_predicate(const GeodesicState& prev,
                        const GeodesicState& curr,
                        const VolumetricDisk& disk) {
    const double theta_prev = prev.position[2];
    const double theta_curr = curr.position[2];
    const double r_prev = prev.position[1];
    const double r_curr = curr.position[1];

    const double d_prev = theta_prev - kHalfPi;
    const double d_curr = theta_curr - kHalfPi;

    const double z_prev = r_prev * std::cos(theta_prev);
    const double z_curr = r_curr * std::cos(theta_curr);

    const bool crossed_midplane =
        (d_prev * d_curr < 0.0) && std::abs(d_prev - d_curr) > 1e-12;

    const bool inside_now = disk.inside_volume(r_curr, z_curr);

    const double zm_curr = disk.z_max_at(r_curr);
    const double H_curr  = disk.scale_height(r_curr);
    const double H_prev  = disk.scale_height(r_prev);
    const bool near_disk =
        (std::abs(z_curr) < zm_curr + 1.0 * H_curr
         || std::abs(z_prev) < disk.z_max_at(r_prev) + 1.0 * H_prev)
        && r_curr >= disk.r_horizon()
        && r_curr <= disk.r_max() + 0.5 * disk.outer_taper_width();

    return crossed_midplane || inside_now || near_disk;
}

} // anonymous namespace

DiskStepEntryResult check_disk_step_entry(
    const GeodesicState& prev_state,
    const GeodesicState& new_state,
    double /*dlambda_full*/,
    const VolumetricDisk& disk,
    const Kerr& /*metric*/,
    const RK4& /*integrator*/,
    const DiskStepEntryOptions& /*opts*/)
{
    if (endpoint_predicate(prev_state, new_state, disk)) {
        return { true, new_state, 0 };
    }
    return { false, {}, 0 };
}

} // namespace grrt
