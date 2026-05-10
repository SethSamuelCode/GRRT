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

/// Tier B segment-bound test. Conservative bounding-region check for whether
/// the geodesic step `(prev, curr)` could intersect the disk volume envelope.
/// Returns true if the segment "could" intersect (caller must subdivide);
/// returns false ONLY when the segment is conclusively outside (rejection
/// guaranteed safe).
///
/// Uses a velocity-aware curvature pad: the conservative |z|-min along the
/// segment accounts for trajectories where the chord between endpoints is
/// small but |dz/dλ| is large (e.g. a ray that enters and re-exits the disk
/// top within one step). dz_swing = 0.5 * |Δ(dz/dλ)| * dλ captures this.
/// dz/dλ is derived from RK4::derivatives_kerr — see the same chain rule at
/// src/geodesic_tracer.cpp:147-152 used by the renderer's step clamp.
///
/// See docs/superpowers/specs/2026-05-10-disk-step-entry-design.md §5.3-5.4.
bool segment_could_intersect_disk(const GeodesicState& prev,
                                  const GeodesicState& curr,
                                  double dlambda_full,
                                  const VolumetricDisk& disk,
                                  const Kerr& metric,
                                  double curvature_pad)
{
    const double r_prev = prev.position[1];
    const double r_curr = curr.position[1];
    const double theta_prev = prev.position[2];
    const double theta_curr = curr.position[2];

    const double z_prev = r_prev * std::cos(theta_prev);
    const double z_curr = r_curr * std::cos(theta_curr);

    // r-range bound from the chord. Reject fast if disjoint from disk r-cylinder.
    const double r_min = std::min(r_prev, r_curr);
    const double r_max = std::max(r_prev, r_curr);
    const double disk_r_lo = disk.r_horizon();
    const double disk_r_hi = disk.r_max() + 0.5 * disk.outer_taper_width();
    if (r_max < disk_r_lo || r_min > disk_r_hi) return false;

    // GeodesicState::momentum stores covariant p_μ. The chain rule
    //   dz/dlambda = cos(theta) * dr/dlambda - r * sin(theta) * dtheta/dlambda
    // requires CONTRAVARIANT position-derivatives, which we get from
    // RK4::derivatives_kerr (same source the renderer's step clamp uses
    // at src/geodesic_tracer.cpp:147-152). Using p_μ directly would
    // conflate energy with velocity (units mismatch by ~Σ for Kerr).
    const auto deriv_prev = RK4::derivatives_kerr(metric, prev);
    const auto deriv_curr = RK4::derivatives_kerr(metric, curr);
    const double vz_prev = std::cos(theta_prev) * deriv_prev.position[1]
                         - r_prev * std::sin(theta_prev) * deriv_prev.position[2];
    const double vz_curr = std::cos(theta_curr) * deriv_curr.position[1]
                         - r_curr * std::sin(theta_curr) * deriv_curr.position[2];

    const double dz_chord = std::abs(z_prev - z_curr);
    const double dz_swing = 0.5 * std::abs(vz_prev - vz_curr) * dlambda_full;
    const double pad      = std::max(curvature_pad * dz_chord, dz_swing);

    double abs_z_min = std::min(std::abs(z_prev), std::abs(z_curr)) - pad;
    if (z_prev * z_curr < 0.0) abs_z_min = 0.0;       // crosses midplane
    if (abs_z_min < 0.0)       abs_z_min = 0.0;

    // Disk envelope: max(z_max(r) + 0.5*H(r)) over the segment's r-range.
    // Sample at 3 r-points (endpoints + midpoint) — conservative because
    // LUTs are smooth-ish at the resolution this bound requires.
    auto envelope_at = [&](double r) {
        const double r_clamped = std::clamp(r, disk_r_lo, disk_r_hi);
        return disk.z_max_at(r_clamped) + 0.5 * disk.scale_height(r_clamped);
    };
    const double env_lo  = envelope_at(r_min);
    const double env_hi  = envelope_at(r_max);
    const double env_mid = envelope_at(0.5 * (r_min + r_max));
    const double env_max = std::max({env_lo, env_hi, env_mid});

    return abs_z_min <= env_max;
}

/// Compute adaptive depth_limit such that the smallest substep is on the
/// order of H_min over the segment's r-range. See spec §5.5.
///   needed = ceil(log2(dlambda_full / H_min))
///   depth  = clamp(needed, depth_floor, depth_cap)
/// depth_floor protects against ratios <1 (returns floor instead of 0);
/// depth_cap is the runtime ceiling beyond which conservative policy
/// (§6.1) takes over.
int compute_adaptive_depth(double dlambda_full,
                           double r_prev, double r_curr,
                           const VolumetricDisk& disk,
                           int depth_floor, int depth_cap)
{
    // Floor H at 1e-30 to guard against degenerate LUT entries returning
    // ~0 H, which would blow up the dlambda_full / H_min ratio below.
    const double H_prev = std::max(disk.scale_height(r_prev), 1e-30);
    const double H_curr = std::max(disk.scale_height(r_curr), 1e-30);
    const double H_min  = std::min(H_prev, H_curr);
    if (dlambda_full <= 0.0) return depth_floor;
    // Ratio floor at 1.0 → ceil(log2) ≥ 0 → no negative-depth UB downstream.
    const double ratio = std::max(dlambda_full / H_min, 1.0);
    const int needed = static_cast<int>(std::ceil(std::log2(ratio)));
    return std::clamp(needed, depth_floor, depth_cap);
}

/// Internal result type for the recursive subdivide() helper. Carries the
/// invocation count back up the recursion so callers can attribute the
/// total cost to the per-step call.
struct SubdivResult {
    bool should_raymarch;
    GeodesicState refined;
    int invocations;        // includes this call + recursive children
};

/// Tier C recursive substep with depth_limit. Returns
/// {should_raymarch=true, refined=substep_endpoint} when Tier A fires
/// somewhere in the substep tree. On depth exhaustion: conservative policy
/// (spec §6.1) — return {true, curr}, leaving the raymarch's own
/// is_in_volume() check to short-circuit when there's no actual entry.
///
/// Substepping uses RK4::step_kerr_rkdp45 to match the main loop's DP45
/// integrator family (spec §5.1). The 5th-order solution y5 is used; the
/// returned error_norm is discarded. Future optimization to plain RK4
/// substep is documented in
/// docs/superpowers/optimizations/2026-05-10-disk-step-entry-rk4-substep.md.
/// @pre depth_remaining >= 0. dlambda_remaining is the proper-time duration
/// of the segment under consideration (gets halved each recursion level).
SubdivResult subdivide(const GeodesicState& prev,
                       const GeodesicState& curr,
                       double dlambda_remaining,
                       int depth_remaining,
                       const VolumetricDisk& disk,
                       const Kerr& metric,
                       const RK4& integrator,
                       double curvature_pad)
{
    if (depth_remaining == 0) {
        // Conservative policy: return curr; raymarch handles non-entry cheaply.
        return { true, curr, 1 };
    }

    // Substep using fixed-step Dormand-Prince RK4(5) — same integrator family
    // as the main loop's adaptive_step_kerr_dp45 (spec §5.1). Discard the
    // error_norm; we only need the 5th-order y5 trajectory state. ~50% more
    // derivative evals per substep than plain RK4, but Tier C fires rarely
    // enough that absolute cost is negligible (<1% of total render).
    // See docs/superpowers/optimizations/2026-05-10-disk-step-entry-rk4-substep.md
    // for the perf revisit if this ever shows up in profiling.
    const double dl_half = dlambda_remaining * 0.5;
    GeodesicState mid = integrator.step_kerr_rkdp45(metric, prev, dl_half).y5;

    int invocations = 1;

    // Tier A on each half.
    if (endpoint_predicate(prev, mid, disk)) {
        return { true, mid, invocations };
    }
    if (endpoint_predicate(mid, curr, disk)) {
        return { true, curr, invocations };
    }

    // Tier B on each half — recurse only on halves that might intersect.
    if (segment_could_intersect_disk(prev, mid, dl_half, disk, metric, curvature_pad)) {
        SubdivResult left = subdivide(prev, mid, dl_half, depth_remaining - 1,
                                      disk, metric, integrator, curvature_pad);
        invocations += left.invocations;
        if (left.should_raymarch) {
            return { true, left.refined, invocations };
        }
    }
    if (segment_could_intersect_disk(mid, curr, dl_half, disk, metric, curvature_pad)) {
        SubdivResult right = subdivide(mid, curr, dl_half, depth_remaining - 1,
                                       disk, metric, integrator, curvature_pad);
        invocations += right.invocations;
        if (right.should_raymarch) {
            return { true, right.refined, invocations };
        }
    }

    return { false, {}, invocations };
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
