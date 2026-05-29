#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/geodesic/romberg_step.h"
#include "grrt/geodesic/raymarch_step_control.h"
#include "grrt/geodesic/disk_step_entry.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/color/opacity.h"
#include "grrt/color/spectrum.h"
#include "grrt/math/constants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <span>

namespace {

// Adaptor: bridges StepSampler interface to the production VolumetricDisk + opacity LUTs.
struct VolumetricDiskSampler : grrt::StepSampler {
    const grrt::VolumetricDisk* disk;
    double observer_r;
    double ut_obs;  // observer's 4-velocity time component (cached)

    VolumetricDiskSampler(const grrt::VolumetricDisk* d, double obs_r)
        : disk(d), observer_r(obs_r),
          ut_obs(1.0 / std::sqrt(1.0 - 2.0 / obs_r)) {}

    bool sample_integrand(const grrt::GeodesicState& state,
                          std::span<const double> channels_nu_obs,
                          std::span<double> integrand) const override {
        const double r     = state.position[1];
        const double theta = state.position[2];
        const double phi   = state.position[3];
        const double z     = r * std::cos(theta);

        // Volume / r-extent guards
        if (r <= disk->r_horizon() || r > disk->r_max()) {
            for (size_t i = 0; i < channels_nu_obs.size(); ++i) integrand[i] = 0.0;
            return false;
        }

        const double rho_cgs = disk->density_cgs(r, z, phi);
        const double T_local = disk->temperature(r, std::abs(z));
        if (rho_cgs <= 0.0 || T_local <= 0.0) {
            for (size_t i = 0; i < channels_nu_obs.size(); ++i) integrand[i] = 0.0;
            return false;
        }

        // Redshift factor g = (p·u)_emit / (p·u)_obs.
        double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
        if (r >= disk->r_isco()) {
            disk->circular_velocity(r, ut_emit, uphi_emit);
        } else {
            disk->plunging_velocity(r, theta, ut_emit, ur_emit, uphi_emit);
        }
        const double p_dot_u_emit = state.momentum[0] * ut_emit
                                   + state.momentum[1] * ur_emit
                                   + state.momentum[3] * uphi_emit;
        const double p_dot_u_obs  = state.momentum[0] * ut_obs;
        const double g_factor     = p_dot_u_emit / p_dot_u_obs;

        const auto& luts = disk->opacity_luts();
        const double abs_pue = std::abs(p_dot_u_emit);

        for (size_t i = 0; i < channels_nu_obs.size(); ++i) {
            const double nu_emit = std::abs(g_factor) * channels_nu_obs[i];
            const double kabs    = luts.lookup_kappa_abs(nu_emit, rho_cgs, T_local);
            const double kes     = luts.lookup_kappa_es(rho_cgs, T_local);
            integrand[i] = (kabs + kes) * rho_cgs * abs_pue;
        }
        return true;
    }
};

} // anonymous namespace

namespace grrt {

GeodesicTracer::GeodesicTracer(const Kerr& metric, const RK4& integrator,
                               double observer_r, int max_steps, double r_escape,
                               double tolerance, const VolumetricDisk* vol_disk,
                               double raymarch_tol)
    : metric_(metric), integrator_(integrator),
      observer_r_(observer_r), max_steps_(max_steps), r_escape_(r_escape),
      tolerance_(tolerance), vol_disk_(vol_disk),
      raymarch_tol_(raymarch_tol > 0.0 ? raymarch_tol : 1e-12) {}

TraceResult GeodesicTracer::trace(GeodesicState state,
                                  const AccretionDisk* disk,
                                  const SpectrumLUT* spectrum) const {
    const double r_horizon = metric_.horizon_radius();
    const double half_pi = std::numbers::pi / 2.0;
    Vec3 color;

    // Initial step size — conservative, adapts quickly
    double dlambda = 0.01 * observer_r_;

    // Persistent radiative transfer state across raymarch calls.
    // J = accumulated invariant specific intensity (backward formula).
    // T = remaining transmission (1 = fully transparent, 0 = opaque).
    // Persisting both prevents double-counting on re-entry and ensures
    // correct front-to-back ordering (front disk occludes back disk).
    double running_J[3] = {0.0, 0.0, 0.0};
    double running_T[3] = {1.0, 1.0, 1.0};

    // Finalize volumetric contribution: convert invariant J to observed color.
    constexpr double nu_rgb[3] = {constants::c_cgs / 450e-7,
                                  constants::c_cgs / 550e-7,
                                  constants::c_cgs / 650e-7};
    auto finalize_vol_color = [&]() {
        for (int ch = 0; ch < 3; ++ch)
            color[ch] += running_J[ch] * nu_rgb[ch] * nu_rgb[ch] * nu_rgb[ch];
    };

    GeodesicState prev = state;

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];

        // Check termination
        if (r < r_horizon + horizon_epsilon_) {
            finalize_vol_color();
            return {RayTermination::Horizon, color, state.position, state.momentum};
        }
        if (r > r_escape_) {
            finalize_vol_color();
            return {RayTermination::Escaped, color, state.position, state.momentum};
        }

        prev = state;

        // Clamp step size when near the volumetric disk so that grazing
        // rays cannot overshoot the entire disk volume in a single step.
        // Cost: one derivatives_kerr() call per step when near — negligible
        // vs the 8+ metric evaluations inside the adaptive RK4.
        if (vol_disk_) {
            const double theta = state.position[2];
            const double z = r * std::cos(theta);
            if (r >= vol_disk_->r_horizon() * 0.9
                && r <= vol_disk_->r_max() * 1.5) {
                const double H = vol_disk_->scale_height(r);
                const double zm = vol_disk_->z_max_at(r);
                // Within 3 scale heights of the disk surface?
                if (std::abs(z) < zm + 3.0 * H) {
                    // Estimate dz/dlambda from the geodesic derivatives
                    auto deriv = RK4::derivatives_kerr(metric_, state);
                    const double dr_dl = deriv.position[1];
                    const double dtheta_dl = deriv.position[2];
                    // dz/dl = cos(theta)*dr/dl - r*sin(theta)*dtheta/dl
                    const double dz_dl = std::abs(
                        std::cos(theta) * dr_dl - r * std::sin(theta) * dtheta_dl);
                    if (dz_dl > 1e-20) {
                        // Limit step so z moves at most H/4; floor at H/64
                        // so near-tangential rays (tiny dz_dl) aren't ignored.
                        const double dl_max = std::max(0.25 * H / dz_dl, H / 64.0);
                        dlambda = std::min(dlambda, dl_max);
                    }
                }
            }
        }

        // Adaptive Dormand-Prince 4(5) — 7 derivative evaluations
        // instead of 12 for step-doubling RK4, with PI step control.
        // Capture pre-integrator dλ as a conservative upper bound for the
        // disk-entry helper's Tier B/C segment-bound and substep refinement.
        const double dlambda_used = dlambda;
        {
            auto result = integrator_.adaptive_step_kerr_dp45(metric_, state, dlambda, tolerance_);
            state = result.state;
            dlambda = result.next_dlambda;
        }

        // Volumetric disk entry detection — delegated to check_disk_step_entry,
        // which runs Tier A (endpoint predicate, byte-equivalent to the original
        // inline test), then Tier B (segment-bound) and Tier C (recursive
        // substep) when needed to catch tangential passes and through-bracket
        // entries that the endpoint test alone would miss.
        if (vol_disk_) {
            // Skip all disk interaction if fully opaque — nothing more can contribute.
            const bool opaque = (running_T[0] < 1e-6 && running_T[1] < 1e-6 && running_T[2] < 1e-6);
            if (!opaque) {
                const DiskStepEntryResult entry_check = check_disk_step_entry(
                    prev, state, dlambda_used, *vol_disk_, metric_, integrator_);
                substep_invocation_count_.fetch_add(entry_check.substep_invocations,
                                                    std::memory_order_relaxed);

                if (entry_check.should_raymarch) {
                    const double r_prev = prev.position[1];
                    // Use refined endpoint so the r-range guard tightens after
                    // Tier C subdivision surfaces an interior detection.
                    const double r_new  = entry_check.refined_endpoint.position[1];
                    const double r_lo = std::min(r_prev, r_new);
                    const double r_hi = std::max(r_prev, r_new);
                    if (r_hi >= vol_disk_->r_horizon() && r_lo <= vol_disk_->r_max()) {
                        GeodesicState entry = prev;
                        const double re = entry.position[1];
                        if (re >= vol_disk_->r_horizon() * 0.9
                            && re <= vol_disk_->r_max() * 1.5) {
                            raymarch_volumetric(entry, color, running_J, running_T);
                            // Only revert state to entry if raymarch advanced it. Otherwise the
                            // raymarch early-exited (e.g., is_in_volume(prev) was false because
                            // Tier B/C over-detected on a conservative bound) — keep the
                            // integrator's post-step state to avoid an infinite revert loop.
                            if (entry.position[1] != prev.position[1]
                                || entry.position[2] != prev.position[2]) {
                                state = entry;
                                continue;
                            }
                            // else: fall through; state retains the integrator's post-step value.
                        }
                    }
                }
            }
        }

        // Check for disk crossing (θ crosses π/2) — thin disk only
        if (!vol_disk_ && disk && spectrum) {
            double theta_prev = prev.position[2];
            double theta_new = state.position[2];

            double d_prev = theta_prev - half_pi;
            double d_new = theta_new - half_pi;
            if (d_prev * d_new < 0.0 && std::abs(d_prev - d_new) > 1e-12) {
                double frac = -d_prev / (d_new - d_prev);

                double r_cross = prev.position[1] + frac * (state.position[1] - prev.position[1]);

                Vec4 p_cross;
                for (int mu = 0; mu < 4; ++mu) {
                    p_cross[mu] = prev.momentum[mu] + frac * (state.momentum[mu] - prev.momentum[mu]);
                }

                if (r_cross >= disk->r_inner() && r_cross <= disk->r_outer()) {
                    color += disk->emission(r_cross, p_cross, observer_r_, *spectrum);
                }
            }
        }
    }

    finalize_vol_color();
    return {RayTermination::MaxSteps, color, state.position, state.momentum};
}

void GeodesicTracer::raymarch_volumetric(GeodesicState& state, Vec3& /*color*/,
                                          double J_rgb[3], double T_rgb[3]) const {
    using namespace constants;
    const auto& luts = vol_disk_->opacity_luts();

    constexpr std::array<double, 3> nu_obs_arr = {
        c_cgs / 450e-7, c_cgs / 550e-7, c_cgs / 650e-7
    };
    std::span<const double> ch_span{nu_obs_arr.data(), 3};

    double J[3] = {J_rgb[0], J_rgb[1], J_rgb[2]};
    double T[3] = {T_rgb[0], T_rgb[1], T_rgb[2]};

    VolumetricDiskSampler sampler(vol_disk_, observer_r_);
    const double ut_obs = sampler.ut_obs;

    // Initial step proposal — same heuristics as before.
    double r = state.position[1];
    const double z_start = r * std::cos(state.position[2]);
    const double H_start = vol_disk_->scale_height(r);
    double ds_proposed = vol_disk_->inside_volume(r, z_start)
                       ? H_start / 16.0
                       : std::min(std::abs(z_start) / 8.0, H_start * 2.0);

    int step_count = 0;
    constexpr int MAX_STEPS = 16384;   // headroom for fine transit stepping (thin disks)

    while (step_count < MAX_STEPS) {
        // Hard exits — match prior logic.
        if (r < vol_disk_->r_horizon())                        break;
        // Outer-radius exit is DIRECTION-AWARE: only bail if the photon is
        // genuinely leaving (moving outward). A photon just outside the rim
        // moving inward is entering the disk from outside the outer edge — keep
        // marching (the sampler returns zero out here, so no emission is added
        // or double-counted) so the crossing just inside the rim isn't missed.
        // A position-only test (r > r_max) bails these inward rays on step 0 and
        // blanks the lensed outer rim (≈85% of zero-emission raymarch calls).
        // The inward vacuum march is bounded: the orchestrator only enters here
        // with entry r <= r_max*1.5, r decreases monotonically toward the disk,
        // and MAX_STEPS is the hard backstop.
        if (r > vol_disk_->r_max()) {
            const double dr_dl = RK4::derivatives_kerr(metric_, state).position[1];
            if (raymarch_exits_outer(r, vol_disk_->r_max(), dr_dl)) break;
        }
        if (T[0] < 1e-6 && T[1] < 1e-6 && T[2] < 1e-6)         break;

        // Romberg-controlled step.
        RombergStep rs = romberg_step(state, ds_proposed, ch_span,
                                       sampler, metric_, integrator_);

        // Reject if error exceeds tolerance.
        if (rs.max_err > raymarch_tol_) {
            const double H_local = vol_disk_->scale_height(state.position[1]);
            const double ds_floor = H_local / 256.0;
            if (ds_proposed <= ds_floor) {
                // Already at floor — accept anyway (LUT discontinuity is the cause,
                // not the integrator).
            } else {
                ds_proposed = std::max(ds_proposed * 0.5, ds_floor);
                continue;
            }
        }

        // z-resolution control: a step whose signed z-interval overlaps the
        // disk envelope must not jump more than H/4 in z, or its midpoint
        // (where we sample the source function) won't reliably land inside the
        // disk. Reject and halve ds, reusing the shrink-and-retry loop. Gated
        // on envelope overlap so empty-space steps stay coarse. (Floor uses the
        // midpoint-r scale height; the max_err block above floors on start-r —
        // intentional: each reject loop terminates against its own floor.)
        {
            const double z0 = state.position[1]
                            * std::cos(state.position[2]);
            const double z1 = rs.end_state.position[1]
                            * std::cos(rs.end_state.position[2]);
            const double r_for_H = rs.mid_state.position[1];
            const double H_z   = vol_disk_->scale_height(r_for_H);
            const double env_z = vol_disk_->z_max_at(r_for_H) + H_z;
            const double ds_floor_z = H_z / 256.0;
            if (step_needs_z_refinement(z0, z1, 0.25 * H_z, env_z)
                && ds_proposed > ds_floor_z) {
                ds_proposed = std::max(ds_proposed * 0.5, ds_floor_z);
                continue;
            }
        }
        step_count++;

        // Accepted: per-channel radiative transfer using rs.dtau.
        // Sample the source function at the step MIDPOINT (not the end). For a
        // transversal disk transit the end can lie outside the disk (density 0)
        // even though the path crossed dense material; the midpoint lands at
        // representative density. dtau (the optical depth over the step) is
        // unchanged — only the source sampling point moves. See spec §5.3.
        const GeodesicState& mid = rs.mid_state;
        const double r_mid       = mid.position[1];
        const double theta_mid   = mid.position[2];
        const double phi_mid     = mid.position[3];
        const double z_mid       = r_mid * std::cos(theta_mid);

        const double rho_cgs = vol_disk_->density_cgs(r_mid, z_mid, phi_mid);
        const double T_local = vol_disk_->temperature(r_mid, std::abs(z_mid));
        if (rho_cgs > 0.0 && T_local > 0.0) {
            // Redshift factor at the mid-state.
            double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
            if (r_mid >= vol_disk_->r_isco()) {
                vol_disk_->circular_velocity(r_mid, ut_emit, uphi_emit);
            } else {
                vol_disk_->plunging_velocity(r_mid, theta_mid, ut_emit, ur_emit, uphi_emit);
            }
            const double p_dot_u_emit = mid.momentum[0] * ut_emit
                                       + mid.momentum[1] * ur_emit
                                       + mid.momentum[3] * uphi_emit;
            const double p_dot_u_obs  = mid.momentum[0] * ut_obs;
            const double g_factor     = p_dot_u_emit / p_dot_u_obs;

            for (int ch = 0; ch < 3; ++ch) {
                const double nu_emit = std::abs(g_factor) * nu_obs_arr[ch];
                const double kabs    = luts.lookup_kappa_abs(nu_emit, rho_cgs, T_local);
                const double kes     = luts.lookup_kappa_es(rho_cgs, T_local);
                const double ktot    = kabs + kes;
                const double epsilon = (ktot > 0.0) ? kabs / ktot : 1.0;

                const double Bnu     = planck_nu(nu_emit, T_local);
                const double S       = epsilon * Bnu / (nu_emit * nu_emit * nu_emit);

                const double dtau    = rs.dtau[ch];
                const double exp_dtau = std::exp(-dtau);
                J[ch] += T[ch] * S * (1.0 - exp_dtau);
                T[ch] *= exp_dtau;
            }
        }

        state = rs.end_state;
        r = state.position[1];

        // Step-size growth: well under tolerance → grow. Cap at H/4 while the
        // ray is inside the disk envelope, else cap at H. Point test here (on
        // the post-advance END state), not the interval test used for rejection.
        // This caps thrash only for a ray ALREADY inside the disk. It does NOT
        // prevent grow-then-reject on a transversal *approach* (both step
        // endpoints outside the envelope): there growth is allowed up to H and
        // the z-resolution gate halves it back down on entry — still correct (no
        // emission missed), just a bounded ~log2(H/floor) reject burst. The gate,
        // not this cap, is what guarantees the entry crossing is captured.
        if (rs.max_err < raymarch_tol_ / 8.0) {
            const double z_now   = r * std::cos(state.position[2]);
            const double H_now   = vol_disk_->scale_height(r);
            const double env_now = vol_disk_->z_max_at(r) + H_now;
            const double grow_cap = (std::abs(z_now) < env_now)
                                  ? (0.25 * H_now)
                                  : H_now;
            ds_proposed = std::min(ds_proposed * 2.0, grow_cap);
        }
    }

    // Persist for caller.
    J_rgb[0] = J[0]; J_rgb[1] = J[1]; J_rgb[2] = J[2];
    T_rgb[0] = T[0]; T_rgb[1] = T[1]; T_rgb[2] = T[2];
}

TraceResult GeodesicTracer::trace_debug(GeodesicState state,
                                        const AccretionDisk* disk,
                                        const SpectrumLUT* spectrum) const {
    const double r_horizon = metric_.horizon_radius();
    const double half_pi = std::numbers::pi / 2.0;
    Vec3 color;
    double dlambda = 0.01 * observer_r_;
    double running_J[3] = {0.0, 0.0, 0.0};
    double running_T[3] = {1.0, 1.0, 1.0};
    constexpr double nu_rgb[3] = {constants::c_cgs / 450e-7,
                                  constants::c_cgs / 550e-7,
                                  constants::c_cgs / 650e-7};
    auto finalize_vol_color = [&]() {
        for (int ch = 0; ch < 3; ++ch)
            color[ch] += running_J[ch] * nu_rgb[ch] * nu_rgb[ch] * nu_rgb[ch];
    };
    GeodesicState prev = state;

    std::printf("=== DEBUG PIXEL TRACE ===\n");
    std::printf("  r0=%.4f theta0=%.6f phi0=%.4f\n",
        state.position[1], state.position[2], state.position[3]);
    std::printf("  p0=(%+.4e %+.4e %+.4e %+.4e)\n",
        state.momentum[0], state.momentum[1], state.momentum[2], state.momentum[3]);
    if (vol_disk_) {
        std::printf("  disk: r_in=%.3f r_out=%.3f r_isco=%.3f\n",
            vol_disk_->r_horizon(), vol_disk_->r_max(), vol_disk_->r_isco());
    }
    std::printf("%-6s %-10s %-10s %-10s %-12s %-8s %-8s %-6s\n",
        "step", "r", "theta", "z", "dlambda", "H", "zm", "event");
    std::printf("%.6s %.10s %.10s %.10s %.12s %.8s %.8s %.6s\n",
        "------","----------","----------","----------","------------","--------","--------","------");

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];
        if (r < r_horizon + horizon_epsilon_) {
            std::printf("%-6d %-10.4f  -> HORIZON\n", step, r);
            finalize_vol_color();
            return {RayTermination::Horizon, color, state.position, state.momentum};
        }
        if (r > r_escape_) {
            std::printf("%-6d %-10.4f  -> ESCAPED\n", step, r);
            finalize_vol_color();
            return {RayTermination::Escaped, color, state.position, state.momentum};
        }

        prev = state;

        // Step-size clamping near disk (same logic as trace())
        if (vol_disk_) {
            const double theta = state.position[2];
            const double z = r * std::cos(theta);
            if (r >= vol_disk_->r_horizon() * 0.9 && r <= vol_disk_->r_max() * 1.5) {
                const double H = vol_disk_->scale_height(r);
                const double zm = vol_disk_->z_max_at(r);
                if (std::abs(z) < zm + 3.0 * H) {
                    auto deriv = RK4::derivatives_kerr(metric_, state);
                    const double dz_dl = std::abs(
                        std::cos(theta) * deriv.position[1]
                        - r * std::sin(theta) * deriv.position[2]);
                    if (dz_dl > 1e-20) {
                        const double dl_max = std::max(0.25 * H / dz_dl, H / 64.0);
                        dlambda = std::min(dlambda, dl_max);
                    }
                }
            }
        }

        // Capture pre-integrator dλ as a conservative upper bound for the
        // disk-entry helper's Tier B/C segment-bound and substep refinement.
        const double dlambda_used = dlambda;
        auto result = integrator_.adaptive_step_kerr_dp45(metric_, state, dlambda, tolerance_);
        state = result.state;
        dlambda = result.next_dlambda;

        const double theta_new = state.position[2];
        const double r_new = state.position[1];
        const double z_new = r_new * std::cos(theta_new);

        const char* event = "  ";
        bool should_raymarch = false;
        DiskStepEntryResult entry_check{};
        // Defensive default; only the should_raymarch==true branch reads this,
        // and the helper overwrites both fields atomically before that branch
        // becomes reachable.
        entry_check.refined_endpoint = state;

        // Volumetric disk entry detection — delegated to check_disk_step_entry
        // (Tier A endpoint test + Tier B segment-bound + Tier C recursive substep).
        if (vol_disk_) {
            // Skip all disk interaction if fully opaque — nothing more can contribute.
            const bool opaque = (running_T[0] < 1e-6 && running_T[1] < 1e-6 && running_T[2] < 1e-6);
            if (!opaque) {
                entry_check = check_disk_step_entry(
                    prev, state, dlambda_used, *vol_disk_, metric_, integrator_);
                substep_invocation_count_.fetch_add(entry_check.substep_invocations,
                                                    std::memory_order_relaxed);
                should_raymarch = entry_check.should_raymarch;
                if (should_raymarch) event = "ENTRY";
            }

            const double H = vol_disk_->scale_height(r_new);
            const double zm = vol_disk_->z_max_at(r_new);
            std::printf("%-6d %-10.4f %-10.6f %-10.4f %-12.4e %-8.4f %-8.4f %s\n",
                step, r_new, theta_new, z_new, dlambda, H, zm, event);
        } else {
            const double H = 0.0, zm = 0.0;
            std::printf("%-6d %-10.4f %-10.6f %-10.4f %-12.4e %-8s %-8s %s\n",
                step, r_new, theta_new, z_new, dlambda, "-", "-", event);
        }

        if (should_raymarch && vol_disk_) {
            // Use refined endpoint so the r-range guard tightens after
            // Tier C subdivision surfaces an interior detection.
            const double r_new_refined = entry_check.refined_endpoint.position[1];
            const double r_lo = std::min(prev.position[1], r_new_refined);
            const double r_hi = std::max(prev.position[1], r_new_refined);
            if (r_hi >= vol_disk_->r_horizon() && r_lo <= vol_disk_->r_max()) {
                GeodesicState entry = prev;
                const double re = entry.position[1];
                if (re >= vol_disk_->r_horizon() * 0.9 && re <= vol_disk_->r_max() * 1.5) {
                    std::printf("  -> RAYMARCH entry r=%.4f z=%.4f\n", re,
                        re * std::cos(entry.position[2]));
                    raymarch_volumetric(entry, color, running_J, running_T);
                    // Compute current observed color from J
                    Vec3 cur_color;
                    for (int ch = 0; ch < 3; ++ch)
                        cur_color[ch] = running_J[ch] * nu_rgb[ch] * nu_rgb[ch] * nu_rgb[ch];
                    std::printf("  -> RAYMARCH exit  color=(%.4e %.4e %.4e)\n",
                        cur_color[0], cur_color[1], cur_color[2]);
                    // Only revert state to entry if raymarch advanced it. Otherwise the
                    // raymarch early-exited (e.g., is_in_volume(prev) was false because
                    // Tier B/C over-detected on a conservative bound) — keep the
                    // integrator's post-step state to avoid an infinite revert loop.
                    if (entry.position[1] != prev.position[1]
                        || entry.position[2] != prev.position[2]) {
                        state = entry;
                        continue;
                    }
                    // else: fall through; state retains the integrator's post-step value.
                }
            }
        }

        if (!vol_disk_ && disk && spectrum) {
            const double d_prev = prev.position[2] - half_pi;
            const double d_new = theta_new - half_pi;
            if (d_prev * d_new < 0.0 && std::abs(d_prev - d_new) > 1e-12) {
                const double frac = -d_prev / (d_new - d_prev);
                const double r_cross = prev.position[1] + frac * (r_new - prev.position[1]);
                if (r_cross >= disk->r_inner() && r_cross <= disk->r_outer()) {
                    Vec4 p_cross;
                    for (int mu = 0; mu < 4; ++mu)
                        p_cross[mu] = prev.momentum[mu] + frac * (state.momentum[mu] - prev.momentum[mu]);
                    Vec3 em = disk->emission(r_cross, p_cross, observer_r_, *spectrum);
                    std::printf("  -> THIN DISK HIT r_cross=%.4f emit=(%.4e %.4e %.4e)\n",
                        r_cross, em[0], em[1], em[2]);
                    color += em;
                }
            }
        }
    }

    std::printf("  -> MAX STEPS  color=(%.4e %.4e %.4e)\n", color[0], color[1], color[2]);
    finalize_vol_color();
    return {RayTermination::MaxSteps, color, state.position, state.momentum};
}

SpectralTraceResult GeodesicTracer::trace_spectral(GeodesicState state,
                                                   const std::vector<double>& frequency_bins) const {
    const int num_bins = static_cast<int>(frequency_bins.size());
    const double r_horizon = metric_.horizon_radius();

    std::vector<double> spectral_intensity(num_bins, 0.0);

    if (!vol_disk_ || num_bins == 0) {
        return {RayTermination::Escaped, spectral_intensity, state.position, state.momentum};
    }

    std::vector<double> J(num_bins, 0.0);
    std::vector<double> T_trans(num_bins, 1.0);
    std::vector<double> tau_acc(num_bins, 0.0);

    double dlambda = 0.01 * observer_r_;
    GeodesicState prev = state;
    RayTermination termination = RayTermination::MaxSteps;

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];

        if (r < r_horizon + horizon_epsilon_) {
            termination = RayTermination::Horizon;
            break;
        }
        if (r > r_escape_) {
            termination = RayTermination::Escaped;
            break;
        }

        prev = state;

        // Clamp step size near the volumetric disk (same as trace())
        {
            const double theta = state.position[2];
            const double z = r * std::cos(theta);
            if (r >= vol_disk_->r_horizon() * 0.9
                && r <= vol_disk_->r_max() * 1.5) {
                const double H = vol_disk_->scale_height(r);
                const double zm = vol_disk_->z_max_at(r);
                if (std::abs(z) < zm + 3.0 * H) {
                    auto deriv = RK4::derivatives_kerr(metric_, state);
                    const double dr_dl = deriv.position[1];
                    const double dtheta_dl = deriv.position[2];
                    const double dz_dl = std::abs(
                        std::cos(theta) * dr_dl - r * std::sin(theta) * dtheta_dl);
                    if (dz_dl > 1e-20) {
                        const double dl_max = std::max(0.25 * H / dz_dl, H / 64.0);
                        dlambda = std::min(dlambda, dl_max);
                    }
                }
            }
        }

        // Capture pre-integrator dλ as a conservative upper bound for the
        // disk-entry helper's Tier B/C segment-bound and substep refinement.
        const double dlambda_used = dlambda;
        {
            auto result = integrator_.adaptive_step_kerr_dp45(metric_, state, dlambda, tolerance_);
            state = result.state;
            dlambda = result.next_dlambda;
        }

        // Volumetric disk entry detection — delegated to check_disk_step_entry
        // (Tier A endpoint test + Tier B segment-bound + Tier C recursive substep).
        {
            const DiskStepEntryResult entry_check = check_disk_step_entry(
                prev, state, dlambda_used, *vol_disk_, metric_, integrator_);
            substep_invocation_count_.fetch_add(entry_check.substep_invocations,
                                                std::memory_order_relaxed);
            if (entry_check.should_raymarch) {
                const double r_prev = prev.position[1];
                // Use refined endpoint so the r-range guard tightens after
                // Tier C subdivision surfaces an interior detection.
                const double r_new  = entry_check.refined_endpoint.position[1];
                const double r_lo = std::min(r_prev, r_new);
                const double r_hi = std::max(r_prev, r_new);
                if (r_hi >= vol_disk_->r_horizon() && r_lo <= vol_disk_->r_max()) {
                    GeodesicState entry = prev;
                    const double re = entry.position[1];
                    if (re >= vol_disk_->r_horizon() * 0.9
                        && re <= vol_disk_->r_max() * 1.5) {
                        raymarch_volumetric_spectral(entry, frequency_bins, J, T_trans, tau_acc);
                        // Only revert state to entry if raymarch advanced it. Otherwise the
                        // raymarch early-exited (e.g., is_in_volume(prev) was false because
                        // Tier B/C over-detected on a conservative bound) — keep the
                        // integrator's post-step state to avoid an infinite revert loop.
                        if (entry.position[1] != prev.position[1]
                            || entry.position[2] != prev.position[2]) {
                            state = entry;
                            continue;
                        }
                        // else: fall through; state retains the integrator's post-step value.
                    }
                }
            }
        }
    }

    // Recover observed intensity: I_obs = J * nu_obs^3
    for (int ch = 0; ch < num_bins; ++ch) {
        spectral_intensity[ch] = J[ch] * frequency_bins[ch] * frequency_bins[ch] * frequency_bins[ch];
    }

    return {termination, spectral_intensity, state.position, state.momentum};
}

void GeodesicTracer::raymarch_volumetric_spectral(GeodesicState& state,
                                                   const std::vector<double>& nu_obs,
                                                   std::vector<double>& J,
                                                   std::vector<double>& T_trans,
                                                   std::vector<double>& tau_acc) const {
    using namespace constants;
    const auto& luts = vol_disk_->opacity_luts();
    const int num_bins = static_cast<int>(nu_obs.size());

    // Observer p·u (static observer at observer_r_)
    double ut_obs = 1.0 / std::sqrt(1.0 - 2.0 / observer_r_);

    double r = state.position[1];
    const double z_start = r * std::cos(state.position[2]);
    const double H_start = vol_disk_->scale_height(r);
    double ds = vol_disk_->inside_volume(r, z_start)
              ? H_start / 16.0
              : std::min(std::abs(z_start) / 8.0, H_start * 2.0);
    int step_count = 0;
    constexpr int MAX_STEPS = 4096;
    constexpr double DTAU_TARGET = 0.05;
    bool been_inside = vol_disk_->inside_volume(r, z_start);

    // Median frequency bin index for adaptive step control
    const int med_bin = num_bins / 2;

    while (step_count < MAX_STEPS) {
        GeodesicState new_state = integrator_.step_kerr(metric_, state, ds);
        step_count++;

#ifndef NDEBUG
        {
            auto g_up = metric_.g_upper(new_state.position);
            double H_check = 0.0;
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    H_check += g_up.m[a][b] * new_state.momentum[a] * new_state.momentum[b];
            H_check *= 0.5;
            if (std::abs(H_check) > 1e-10) {
                std::fprintf(stderr, "WARNING: H=%.4e at r=%.4f during spectral raymarch\n",
                             H_check, new_state.position[1]);
            }
        }
#endif

        r = new_state.position[1];
        const double theta = new_state.position[2];
        const double phi = new_state.position[3];
        const double z = r * std::cos(theta);

        // Hard exits — always advance state so the outer loop makes progress.
        if (r < vol_disk_->r_horizon()) { state = new_state; break; }
        if (r > vol_disk_->r_max())     { state = new_state; break; }

        // Early exit when all bins are optically thick
        {
            bool all_thick = true;
            for (int ch = 0; ch < num_bins; ++ch) {
                if (tau_acc[ch] <= 10.0) { all_thick = false; break; }
            }
            if (all_thick) { state = new_state; break; }
        }

        const double H = vol_disk_->scale_height(r);
        if (!vol_disk_->inside_volume(r, z)) {
            const double zm = vol_disk_->z_max_at(r);
            if (been_inside && std::abs(z) > zm + 1.5 * H) { state = new_state; break; }
            if (!been_inside) {
                ds = std::min(std::abs(z) / 8.0, H * 2.0);
                ds = std::max(ds, H / 64.0);
            } else {
                ds = std::clamp(H / 4.0, H / 64.0, H);
            }
            state = new_state;
            continue;
        }
        been_inside = true;

        // Look up local state
        const double rho_cgs = vol_disk_->density_cgs(r, z, phi);
        const double T = vol_disk_->temperature(r, std::abs(z));
        if (rho_cgs <= 0.0 || T <= 0.0) {
            state = new_state;
            continue;
        }

        const double T_turb = T;

        // Compute redshift g = (p·u)_emit / (p·u)_obs
        double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
        if (r >= vol_disk_->r_isco()) {
            vol_disk_->circular_velocity(r, ut_emit, uphi_emit);
        } else {
            vol_disk_->plunging_velocity(r, theta, ut_emit, ur_emit, uphi_emit);
        }

        const double p_dot_u_emit = new_state.momentum[0] * ut_emit
                                  + new_state.momentum[1] * ur_emit
                                  + new_state.momentum[3] * uphi_emit;
        const double p_dot_u_obs = new_state.momentum[0] * ut_obs;
        const double g = p_dot_u_emit / p_dot_u_obs;

        const double ds_proper = std::abs(p_dot_u_emit) * std::abs(ds);

        // Per-channel radiative transfer
        for (int ch = 0; ch < num_bins; ++ch) {
            const double nu_emit = std::abs(g) * nu_obs[ch];

            const double kabs = luts.lookup_kappa_abs(nu_emit, rho_cgs, T_turb);
            const double kes = luts.lookup_kappa_es(rho_cgs, T_turb);
            const double ktot = kabs + kes;
            const double epsilon = (ktot > 0.0) ? kabs / ktot : 1.0;

            const double dtau = ktot * rho_cgs * ds_proper;
            tau_acc[ch] += dtau;

            // Invariant source: S = epsilon * B_nu(nu_emit, T) / nu_emit^3
            const double Bnu = planck_nu(nu_emit, T_turb);
            const double S = epsilon * Bnu / (nu_emit * nu_emit * nu_emit);

            const double exp_dtau = std::exp(-dtau);
            // Backward accumulation: emission weighted by accumulated
            // transmission from material already in front.
            J[ch] += T_trans[ch] * S * (1.0 - exp_dtau);
            T_trans[ch] *= exp_dtau;
        }

        // Adaptive step control using median frequency bin
        const double nu_med_emit = std::abs(g) * nu_obs[med_bin];
        const double alpha_tot = (luts.lookup_kappa_abs(nu_med_emit, rho_cgs, T_turb)
                                + luts.lookup_kappa_es(rho_cgs, T_turb)) * rho_cgs;
        double ds_tau = (alpha_tot > 0.0)
                      ? DTAU_TARGET / alpha_tot
                      : ds * 2.0;

        const double ds_geo = 0.1 * std::max(r - vol_disk_->r_horizon(), 0.5);
        ds = std::min(ds_tau, ds_geo);
        ds = std::clamp(ds, H / 64.0, H);

        state = new_state;
    }
}

} // namespace grrt
