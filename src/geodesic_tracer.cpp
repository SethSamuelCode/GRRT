#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/geodesic/rk4.h"
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

GeodesicTracer::GeodesicTracer(const Metric& metric, const Integrator& integrator,
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

    // Try to use adaptive stepping if the integrator is RK4
    const auto* rk4 = dynamic_cast<const RK4*>(&integrator_);

    // Initial step size — conservative, adapts quickly
    double dlambda = 0.01 * observer_r_;

    GeodesicState prev = state;

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];

        // Check termination
        if (r < r_horizon + horizon_epsilon_) {
            return {RayTermination::Horizon, color, state.position, state.momentum};
        }
        if (r > r_escape_) {
            return {RayTermination::Escaped, color, state.position, state.momentum};
        }

        prev = state;

        // Adaptive step (or fixed fallback)
        if (rk4) {
            auto result = rk4->adaptive_step(metric_, state, dlambda, tolerance_);
            state = result.state;
            dlambda = result.next_dlambda;
        } else {
            state = integrator_.step(metric_, state, 0.005 * r);
        }

        // Volumetric disk: detect midplane crossing (θ crosses π/2),
        // then interpolate to the crossing point and start raymarching.
        // This is robust even when the adaptive step overshoots the thin
        // ±3H volume entirely.
        if (vol_disk_) {
            const double theta_prev = prev.position[2];
            const double theta_new = state.position[2];
            const double d_prev = theta_prev - half_pi;
            const double d_new = theta_new - half_pi;

            const double z_new = state.position[1] * std::cos(theta_new);
            const bool crossed_midplane = (d_prev * d_new < 0.0)
                                       && std::abs(d_prev - d_new) > 1e-12;
            const bool inside_vol = vol_disk_->inside_volume(state.position[1], z_new);

            if (crossed_midplane || inside_vol) {
                // Interpolate state to the midplane crossing point so the
                // raymarch begins inside the disk, not far above it.
                GeodesicState entry = state;
                if (crossed_midplane) {
                    const double frac = -d_prev / (d_new - d_prev);
                    for (int mu = 0; mu < 4; ++mu) {
                        entry.position[mu] = prev.position[mu]
                            + frac * (state.position[mu] - prev.position[mu]);
                        entry.momentum[mu] = prev.momentum[mu]
                            + frac * (state.momentum[mu] - prev.momentum[mu]);
                    }
                }

                const double r_entry = entry.position[1];
                if (r_entry >= vol_disk_->r_horizon()
                    && r_entry <= vol_disk_->r_max()) {
                    raymarch_volumetric(entry, color);
                    state = entry;
                    continue;
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

    return {RayTermination::MaxSteps, color, state.position, state.momentum};
}

void GeodesicTracer::raymarch_volumetric(GeodesicState& state, Vec3& color) const {
    using namespace constants;
    const auto& luts = vol_disk_->opacity_luts();

    // Three RGB channels at 450nm, 550nm, 650nm
    constexpr double nu_obs[3] = {c_cgs / 450e-7, c_cgs / 550e-7, c_cgs / 650e-7};

    // Invariant J per channel
    double J[3] = {0.0, 0.0, 0.0};

    // Observer p·u (static observer at observer_r_)
    // In geometric units with M=1, g_tt = -(1 - 2M/r)
    double ut_obs = 1.0 / std::sqrt(1.0 - 2.0 / observer_r_);

    double r = state.position[1];
    double ds = vol_disk_->scale_height(r) / 16.0;  // finer initial step
    int step_count = 0;
    constexpr int MAX_STEPS = 4096;
    constexpr double DTAU_TARGET = 0.05;  // target optical depth per step
    double tau_acc[3] = {0.0, 0.0, 0.0};

    while (step_count < MAX_STEPS) {
        GeodesicState new_state = integrator_.step(metric_, state, ds);
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

        // Hard exits
        if (r < vol_disk_->r_horizon()) break;
        if (r > vol_disk_->r_max()) break;  // left the disk radially
        if (tau_acc[0] > 10.0 && tau_acc[1] > 10.0 && tau_acc[2] > 10.0) break;

        // Outside the vertical extent: the ray oscillates in theta so
        // it can leave ±3H and come back.  Only exit after |z| > 6H
        // (well beyond the disk surface where re-entry is impossible).
        const double H = vol_disk_->scale_height(r);
        if (!vol_disk_->inside_volume(r, z)) {
            if (std::abs(z) > 6.0 * H) break;  // truly gone
            // Still close — keep stepping through empty space
            ds = std::clamp(H / 4.0, H / 64.0, H);
            state = new_state;
            continue;
        }

        // Look up local state
        const double rho_cgs = vol_disk_->density_cgs(r, z, phi);
        const double T = vol_disk_->temperature(r, std::abs(z));
        if (rho_cgs <= 0.0 || T <= 0.0) {
            state = new_state;
            continue;
        }

        const double T_turb = T; // Simplified for now

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
            tau_acc[ch] += dtau;

            // Invariant source: S = epsilon * B_nu(nu_emit, T) / nu_emit^3
            const double Bnu = planck_nu(nu_emit, T_turb);
            const double S = epsilon * Bnu / (nu_emit * nu_emit * nu_emit);

            const double exp_dtau = std::exp(-dtau);
            J[ch] = J[ch] * exp_dtau + S * (1.0 - exp_dtau);
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

    // Recover observed intensity: I_obs = J * nu_obs^3
    for (int ch = 0; ch < 3; ch++) {
        color[ch] += J[ch] * nu_obs[ch] * nu_obs[ch] * nu_obs[ch];
    }
}

} // namespace grrt
