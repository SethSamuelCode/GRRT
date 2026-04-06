#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/color/opacity.h"
#include "grrt/color/spectrum.h"
#include "grrt/math/constants.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>

namespace grrt {

GeodesicTracer::GeodesicTracer(const Kerr& metric, const RK4& integrator,
                               double observer_r, int max_steps, double r_escape,
                               double tolerance, const VolumetricDisk* vol_disk)
    : metric_(metric), integrator_(integrator),
      observer_r_(observer_r), max_steps_(max_steps), r_escape_(r_escape),
      tolerance_(tolerance), vol_disk_(vol_disk) {}

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
                // Within 6 scale heights of the disk surface?
                if (std::abs(z) < zm + 6.0 * H) {
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

        // Adaptive RK4 step — concrete Kerr, no virtual dispatch
        {
            auto result = integrator_.adaptive_step_kerr(metric_, state, dlambda, tolerance_);
            state = result.state;
            dlambda = result.next_dlambda;
        }

        // Volumetric disk entry detection.  Three cases to catch:
        // (a) θ crossed π/2 (midplane crossing — most common)
        // (b) Endpoint landed inside the volume
        // (c) Ray passed through the volume without crossing the midplane
        //     (tangential pass: both endpoints outside, but the minimum |z|
        //     along the step was inside ±3H).  We detect this conservatively
        //     by checking if either endpoint's |z| is within 3H of the
        //     midplane at the endpoint's r.
        if (vol_disk_) {
            // Skip all disk interaction if fully opaque — nothing more can contribute.
            const bool opaque = (running_T[0] < 1e-6 && running_T[1] < 1e-6 && running_T[2] < 1e-6);
            if (!opaque) {
                const double theta_prev = prev.position[2];
                const double theta_new = state.position[2];
                const double d_prev = theta_prev - half_pi;
                const double d_new = theta_new - half_pi;
                const double r_new = state.position[1];
                const double r_prev = prev.position[1];

                const double z_new = r_new * std::cos(theta_new);
                const double z_prev = r_prev * std::cos(theta_prev);
                const bool crossed_midplane = (d_prev * d_new < 0.0)
                                           && std::abs(d_prev - d_new) > 1e-12;
                const bool inside_now = vol_disk_->inside_volume(r_new, z_new);
                const double zm_new = vol_disk_->z_max_at(r_new);
                const double H_new = vol_disk_->scale_height(r_new);
                const double H_prev = vol_disk_->scale_height(r_prev);
                const bool near_disk = (std::abs(z_new) < zm_new + 2.0 * H_new
                                     || std::abs(z_prev) < vol_disk_->z_max_at(r_prev) + 2.0 * H_prev)
                                    && r_new >= vol_disk_->r_horizon()
                                    && r_new <= vol_disk_->r_max();
                const bool should_raymarch = crossed_midplane || inside_now || near_disk;

                if (should_raymarch) {
                    const double r_lo = std::min(r_prev, r_new);
                    const double r_hi = std::max(r_prev, r_new);
                    if (r_hi >= vol_disk_->r_horizon() && r_lo <= vol_disk_->r_max()) {
                        GeodesicState entry = prev;
                        const double re = entry.position[1];
                        if (re >= vol_disk_->r_horizon() * 0.9
                            && re <= vol_disk_->r_max() * 1.5) {
                            raymarch_volumetric(entry, color, running_J, running_T);
                            state = entry;
                            continue;
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

void GeodesicTracer::raymarch_volumetric(GeodesicState& state, Vec3& color,
                                          double J_rgb[3], double T_rgb[3]) const {
    using namespace constants;
    const auto& luts = vol_disk_->opacity_luts();

    // Three RGB channels at 450nm, 550nm, 650nm
    constexpr double nu_obs[3] = {c_cgs / 450e-7, c_cgs / 550e-7, c_cgs / 650e-7};

    // Backward radiative transfer accumulation:
    //   J += T * S * (1 - exp(-dtau))   (new emission, attenuated by material in front)
    //   T *= exp(-dtau)                  (transmission decreases)
    // Persisted across calls so repeated re-entries and separate transits
    // (front disk → back disk) are composited with correct occlusion.
    double J[3] = {J_rgb[0], J_rgb[1], J_rgb[2]};
    double T[3] = {T_rgb[0], T_rgb[1], T_rgb[2]};

    // Observer p·u (static observer at observer_r_)
    // In geometric units with M=1, g_tt = -(1 - 2M/r)
    double ut_obs = 1.0 / std::sqrt(1.0 - 2.0 / observer_r_);

    double r = state.position[1];
    const double z_start = r * std::cos(state.position[2]);
    const double H_start = vol_disk_->scale_height(r);
    // If starting outside the volume, use coarse steps to approach quickly;
    // if already inside, use fine steps for accurate radiative transfer.
    double ds = vol_disk_->inside_volume(r, z_start)
              ? H_start / 16.0
              : std::min(std::abs(z_start) / 8.0, H_start * 2.0);
    int step_count = 0;
    constexpr int MAX_STEPS = 4096;
    constexpr double DTAU_TARGET = 0.05;  // target optical depth per step
    bool been_inside = vol_disk_->inside_volume(r, z_start);

    while (step_count < MAX_STEPS) {
        GeodesicState new_state = integrator_.step_kerr(metric_, state, ds);
        step_count++;

#ifndef NDEBUG
        // Check Hamiltonian constraint H = 1/2 g^{ab} p_a p_b ~ 0
        {
            auto g_up = metric_.g_upper(new_state.position);
            double H_check = 0.0;
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    H_check += g_up.m[a][b] * new_state.momentum[a] * new_state.momentum[b];
            H_check *= 0.5;
            if (std::abs(H_check) > 1e-10) {
                std::fprintf(stderr, "WARNING: H=%.4e at r=%.4f during raymarch\n",
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
        // Exit when cumulative transmission is negligible (opaque).
        if (T[0] < 1e-6 && T[1] < 1e-6 && T[2] < 1e-6) { state = new_state; break; }

        const double H = vol_disk_->scale_height(r);
        if (!vol_disk_->inside_volume(r, z)) {
            // Only exit when leaving after having been inside the volume,
            // and we're at 3H beyond the disk surface. This is wider than
            // the outer loop's 2H near_disk trigger, giving a 1H gap that
            // prevents the outer loop from immediately re-invoking the
            // raymarcher after this exit.
            const double zm = vol_disk_->z_max_at(r);
            if (been_inside && std::abs(z) > zm + 3.0 * H) { state = new_state; break; }
            // Still approaching or close — use coarse steps when far,
            // fine steps when near the disk surface.
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
        const double T_local = vol_disk_->temperature(r, std::abs(z));
        if (rho_cgs <= 0.0 || T_local <= 0.0) {
            state = new_state;
            continue;
        }

        const double T_turb = T_local;

        // Compute redshift g = (p·u)_emit / (p·u)_obs
        double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
        if (r >= vol_disk_->r_isco()) {
            vol_disk_->circular_velocity(r, ut_emit, uphi_emit);
        } else {
            vol_disk_->plunging_velocity(r, theta, ut_emit, ur_emit, uphi_emit);
        }

        // p·u (covariant momentum · contravariant velocity)
        const double p_dot_u_emit = new_state.momentum[0] * ut_emit
                                  + new_state.momentum[1] * ur_emit
                                  + new_state.momentum[3] * uphi_emit;
        const double p_dot_u_obs = new_state.momentum[0] * ut_obs;
        const double g = p_dot_u_emit / p_dot_u_obs;

        // Proper distance along ray
        const double ds_proper = std::abs(p_dot_u_emit) * std::abs(ds);

        // Per-channel radiative transfer
        for (int ch = 0; ch < 3; ch++) {
            const double nu_emit = std::abs(g) * nu_obs[ch];

            const double kabs = luts.lookup_kappa_abs(nu_emit, rho_cgs, T_turb);
            const double kes = luts.lookup_kappa_es(rho_cgs, T_turb);
            const double ktot = kabs + kes;
            const double epsilon = (ktot > 0.0) ? kabs / ktot : 1.0;

            const double dtau = ktot * rho_cgs * ds_proper;

            // Invariant source: S = epsilon * B_nu(nu_emit, T) / nu_emit^3
            const double Bnu = planck_nu(nu_emit, T_turb);
            const double S = epsilon * Bnu / (nu_emit * nu_emit * nu_emit);

            const double exp_dtau = std::exp(-dtau);
            // Backward accumulation: new emission weighted by what's
            // already in front (T), then reduce transmission.
            J[ch] += T[ch] * S * (1.0 - exp_dtau);
            T[ch] *= exp_dtau;
        }

        // Smooth adaptive step control: adjust ds so dtau ≈ DTAU_TARGET
        const double alpha_tot = (luts.lookup_kappa_abs(std::abs(g) * nu_obs[1], rho_cgs, T_turb)
                                + luts.lookup_kappa_es(rho_cgs, T_turb)) * rho_cgs;
        double ds_tau = (alpha_tot > 0.0)
                      ? DTAU_TARGET / alpha_tot  // step that gives target dtau
                      : ds * 2.0;

        const double ds_geo = 0.1 * std::max(r - vol_disk_->r_horizon(), 0.5);
        ds = std::min(ds_tau, ds_geo);
        // H already declared above in the outside-volume check
        ds = std::clamp(ds, H / 64.0, H);

        state = new_state;
    }

    // Persist J and T back to caller for correct compositing across calls.
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
                if (std::abs(z) < zm + 6.0 * H) {
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

        auto result = integrator_.adaptive_step_kerr(metric_, state, dlambda, tolerance_);
        state = result.state;
        dlambda = result.next_dlambda;

        const double theta_new = state.position[2];
        const double r_new = state.position[1];
        const double z_new = r_new * std::cos(theta_new);
        const double z_prev2 = prev.position[1] * std::cos(prev.position[2]);

        const char* event = "  ";
        bool should_raymarch = false;

        if (vol_disk_) {
            // Skip all disk interaction if fully opaque — nothing more can contribute.
            const bool opaque = (running_T[0] < 1e-6 && running_T[1] < 1e-6 && running_T[2] < 1e-6);
            if (!opaque) {
                const double d_prev = prev.position[2] - half_pi;
                const double d_new = theta_new - half_pi;
                const bool crossed = (d_prev * d_new < 0.0) && std::abs(d_prev - d_new) > 1e-12;
                const bool inside_now = vol_disk_->inside_volume(r_new, z_new);
                const double zm_new = vol_disk_->z_max_at(r_new);
                const double H_new = vol_disk_->scale_height(r_new);
                const double H_prev2 = vol_disk_->scale_height(prev.position[1]);
                const bool near = (std::abs(z_new) < zm_new + 2.0 * H_new
                                || std::abs(z_prev2) < vol_disk_->z_max_at(prev.position[1]) + 2.0 * H_prev2)
                               && r_new >= vol_disk_->r_horizon() && r_new <= vol_disk_->r_max();
                should_raymarch = crossed || inside_now || near;

                if (crossed)     event = "CROSS";
                else if (inside_now) event = "INSIDE";
                else if (near)   event = "NEAR";
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
            const double r_lo = std::min(prev.position[1], r_new);
            const double r_hi = std::max(prev.position[1], r_new);
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
                    state = entry;
                    continue;
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
    const double half_pi = std::numbers::pi / 2.0;

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
                if (std::abs(z) < zm + 6.0 * H) {
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

        {
            auto result = integrator_.adaptive_step_kerr(metric_, state, dlambda, tolerance_);
            state = result.state;
            dlambda = result.next_dlambda;
        }

        // Volumetric disk entry detection — identical to trace()
        {
            const double theta_prev = prev.position[2];
            const double theta_new = state.position[2];
            const double d_prev = theta_prev - half_pi;
            const double d_new = theta_new - half_pi;
            const double r_new = state.position[1];
            const double r_prev = prev.position[1];

            const double z_new = r_new * std::cos(theta_new);
            const double z_prev = r_prev * std::cos(theta_prev);
            const bool crossed_midplane = (d_prev * d_new < 0.0)
                                       && std::abs(d_prev - d_new) > 1e-12;
            const bool inside_now = vol_disk_->inside_volume(r_new, z_new);
            const double zm_new = vol_disk_->z_max_at(r_new);
            const double H_new = vol_disk_->scale_height(r_new);
            const double H_prev_s = vol_disk_->scale_height(r_prev);
            const bool near_disk = (std::abs(z_new) < zm_new + 2.0 * H_new
                                 || std::abs(z_prev) < vol_disk_->z_max_at(r_prev) + 2.0 * H_prev_s)
                                && r_new >= vol_disk_->r_horizon()
                                && r_new <= vol_disk_->r_max();
            const bool should_raymarch = crossed_midplane || inside_now || near_disk;

            if (should_raymarch) {
                const double r_lo = std::min(r_prev, r_new);
                const double r_hi = std::max(r_prev, r_new);
                if (r_hi >= vol_disk_->r_horizon() && r_lo <= vol_disk_->r_max()) {
                    GeodesicState entry = prev;
                    const double re = entry.position[1];
                    if (re >= vol_disk_->r_horizon() * 0.9
                        && re <= vol_disk_->r_max() * 1.5) {
                        raymarch_volumetric_spectral(entry, frequency_bins, J, T_trans, tau_acc);
                        state = entry;
                        continue;
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
            if (been_inside && std::abs(z) > zm + 3.0 * H) { state = new_state; break; }
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
