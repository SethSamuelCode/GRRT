#include "grrt/geodesic/rk4.h"
#include <cmath>
#include <algorithm>

namespace grrt {

// ---------- Generic Metric path (virtual, not hot) ----------

GeodesicState RK4::step(const Metric& metric, const GeodesicState& state, double dl) const {
    auto deriv = [&](const GeodesicState& s) {
        auto [dx, dp] = metric.compute_derivatives(s.position, s.momentum);
        return GeodesicState{dx, dp};
    };
    auto add = [](const GeodesicState& s, const GeodesicState& ds, double h) {
        return GeodesicState{s.position + ds.position * h, s.momentum + ds.momentum * h};
    };

    GeodesicState k1 = deriv(state);
    GeodesicState k2 = deriv(add(state, k1, dl * 0.5));
    GeodesicState k3 = deriv(add(state, k2, dl * 0.5));
    GeodesicState k4 = deriv(add(state, k3, dl));

    Vec4 new_pos = state.position
        + (k1.position + k2.position * 2.0 + k3.position * 2.0 + k4.position) * (dl / 6.0);
    Vec4 new_mom = state.momentum
        + (k1.momentum + k2.momentum * 2.0 + k3.momentum * 2.0 + k4.momentum) * (dl / 6.0);

    return {new_pos, new_mom};
}

// ---------- Kerr-specific: derivatives_kerr and step_kerr are inline in rk4.h ----------

AdaptiveResult RK4::adaptive_step_kerr(const Kerr& metric, const GeodesicState& state,
                                       double dlambda, double tolerance) const {
    constexpr double dl_min = 1e-6;
    constexpr int max_retries = 20;
    constexpr double eps = 1e-10;

    double dl = dlambda;

    for (int retry = 0; retry < max_retries; ++retry) {
        GeodesicState s_full = step_kerr(metric, state, dl);

        GeodesicState s_mid = step_kerr(metric, state, dl * 0.5);
        GeodesicState s_half = step_kerr(metric, s_mid, dl * 0.5);

        double err = 0.0;
        for (int i = 1; i < 4; ++i) {
            double diff = std::abs(s_full.position[i] - s_half.position[i]);
            double scale = std::abs(s_half.position[i]) + eps;
            err = std::max(err, diff / scale);
        }
        for (int i = 0; i < 4; ++i) {
            double diff = std::abs(s_full.momentum[i] - s_half.momentum[i]);
            double scale = std::abs(s_half.momentum[i]) + eps;
            err = std::max(err, diff / scale);
        }

        if (err <= tolerance) {
            double next_dl = dl;
            if (err < tolerance * 0.01) {
                next_dl = dl * 2.0;
            }
            double r = std::abs(s_half.position[1]);
            next_dl = std::min(next_dl, 5.0 * std::max(r, 1.0));
            return {s_half, next_dl};
        }

        dl *= 0.5;
        if (dl < dl_min) {
            double r = std::abs(s_half.position[1]);
            double next_dl = std::min(dl_min, 5.0 * std::max(r, 1.0));
            return {s_half, next_dl};
        }
    }

    GeodesicState s_mid = step_kerr(metric, state, dl * 0.5);
    GeodesicState s_half = step_kerr(metric, s_mid, dl * 0.5);
    return {s_half, dl};
}

// ---------- Dormand-Prince 4(5) adaptive controller ----------
// 7 derivative evaluations per attempt (6 stages + 1 FSAL) vs 12 for
// step-doubling RK4.  Smooth PI step-size control replaces the old
// conservative "double or stay" logic.

AdaptiveResult RK4::adaptive_step_kerr_dp45(const Kerr& metric,
                                             const GeodesicState& state,
                                             double dlambda,
                                             double tolerance) const {
    constexpr double dl_min = 1e-6;
    constexpr int max_retries = 20;
    // PI controller constants (standard for RK45):
    //   safety factor 0.9, exponent 0.2 = 1/(p+1) for p=4
    constexpr double safety = 0.9;
    constexpr double p_exp = 0.2;       // 1/5
    constexpr double grow_max = 5.0;    // max step growth factor
    constexpr double shrink_max = 0.2;  // max step shrink factor (don't shrink below 1/5)

    double dl = dlambda;

    for (int retry = 0; retry < max_retries; ++retry) {
        auto result = step_kerr_rkdp45(metric, state, dl);

        if (result.error_norm <= tolerance) {
            // Accepted — compute next step size via PI controller.
            double factor = (result.error_norm > 1e-30)
                          ? safety * std::pow(tolerance / result.error_norm, p_exp)
                          : grow_max;
            factor = std::clamp(factor, 1.0, grow_max);  // only grow on accept
            double next_dl = dl * factor;

            // Cap to geometric scale: don't overshoot in curved regions
            double r = std::abs(result.y5.position[1]);
            next_dl = std::min(next_dl, 5.0 * std::max(r, 1.0));

            return {result.y5, next_dl};
        }

        // Rejected — shrink step and retry.
        double factor = safety * std::pow(tolerance / result.error_norm, p_exp);
        factor = std::clamp(factor, shrink_max, 1.0);
        dl *= factor;

        if (dl < dl_min) {
            // Step too small — accept whatever we have.
            double r = std::abs(result.y5.position[1]);
            double next_dl = std::min(dl_min, 5.0 * std::max(r, 1.0));
            return {result.y5, next_dl};
        }
    }

    // Exhausted retries — take the last result.
    auto result = step_kerr_rkdp45(metric, state, dl);
    return {result.y5, dl};
}

} // namespace grrt
