#ifndef GRRT_RK4_H
#define GRRT_RK4_H

#include "grrt/geodesic/integrator.h"
#include "grrt/spacetime/kerr.h"
#include "grrt_export.h"

namespace grrt {

class GRRT_EXPORT RK4 : public Integrator {
public:
    // Virtual override for generic Metric (backward compat, not on hot path)
    GeodesicState step(const Metric& metric,
                       const GeodesicState& state,
                       double dlambda) const override;

    // Concrete Kerr methods — force-inlined for the hot path
#ifdef _MSC_VER
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    static GeodesicState derivatives_kerr(const Kerr& metric,
                                          const GeodesicState& state) {
        auto [dx, dp] = metric.compute_derivatives_inline(state.position, state.momentum);
        return {dx, dp};
    }

#ifdef _MSC_VER
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    GeodesicState step_kerr(const Kerr& metric,
                            const GeodesicState& state,
                            double dl) const {
        auto add = [](const GeodesicState& s, const GeodesicState& ds, double h) -> GeodesicState {
            return {s.position + ds.position * h, s.momentum + ds.momentum * h};
        };

        GeodesicState k1 = derivatives_kerr(metric, state);
        GeodesicState k2 = derivatives_kerr(metric, add(state, k1, dl * 0.5));
        GeodesicState k3 = derivatives_kerr(metric, add(state, k2, dl * 0.5));
        GeodesicState k4 = derivatives_kerr(metric, add(state, k3, dl));

        Vec4 new_pos = state.position
            + (k1.position + k2.position * 2.0 + k3.position * 2.0 + k4.position) * (dl / 6.0);
        Vec4 new_mom = state.momentum
            + (k1.momentum + k2.momentum * 2.0 + k3.momentum * 2.0 + k4.momentum) * (dl / 6.0);

        return {new_pos, new_mom};
    }

    AdaptiveResult adaptive_step_kerr(const Kerr& metric,
                                      const GeodesicState& state,
                                      double dlambda,
                                      double tolerance) const;

    /// Dormand-Prince RK4(5) embedded method — 6 derivative evaluations
    /// produce both a 4th-order and 5th-order solution.  The difference
    /// gives an error estimate for free (no step-doubling needed).
    /// Returns {y5 (5th-order, local extrapolation), y4, error_norm}.
    struct RKDP45Result {
        GeodesicState y5;       // 5th-order solution (use this)
        double error_norm;      // max relative error |y5 - y4| / scale
    };

#ifdef _MSC_VER
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    RKDP45Result step_kerr_rkdp45(const Kerr& metric,
                                   const GeodesicState& state,
                                   double dl) const {
        // Dormand-Prince Butcher tableau coefficients
        // c2=1/5, c3=3/10, c4=4/5, c5=8/9, c6=1, c7=1
        //
        // a21 = 1/5
        // a31 = 3/40,      a32 = 9/40
        // a41 = 44/45,     a42 = -56/15,    a43 = 32/9
        // a51 = 19372/6561, a52 = -25360/2187, a53 = 64448/6561, a54 = -212/729
        // a61 = 9017/3168, a62 = -355/33,   a63 = 46732/5247, a64 = 49/176, a65 = -5103/18656
        // 5th-order weights (b):
        // b1=35/384, b3=500/1113, b4=125/192, b5=-2187/6784, b6=11/84
        // 4th-order weights (b*):
        // b*1=5179/57600, b*3=7571/16695, b*4=393/640, b*5=-92097/339200, b*6=187/2100, b*7=1/40
        //
        // Error = y5 - y4, using e_i = b_i - b*_i:
        // e1=71/57600, e3=-71/16695, e4=71/1920, e5=-17253/339200, e6=22/525, e7=-1/40

        auto add2 = [](const GeodesicState& s,
                       const GeodesicState& a, double ha,
                       const GeodesicState& b, double hb) -> GeodesicState {
            return {s.position + a.position * ha + b.position * hb,
                    s.momentum + a.momentum * ha + b.momentum * hb};
        };

        // Stage 1
        GeodesicState k1 = derivatives_kerr(metric, state);

        // Stage 2: y + h*(1/5)*k1
        GeodesicState s2 = {state.position + k1.position * (dl / 5.0),
                            state.momentum + k1.momentum * (dl / 5.0)};
        GeodesicState k2 = derivatives_kerr(metric, s2);

        // Stage 3: y + h*(3/40*k1 + 9/40*k2)
        GeodesicState s3 = {state.position + k1.position * (dl * 3.0/40.0) + k2.position * (dl * 9.0/40.0),
                            state.momentum + k1.momentum * (dl * 3.0/40.0) + k2.momentum * (dl * 9.0/40.0)};
        GeodesicState k3 = derivatives_kerr(metric, s3);

        // Stage 4: y + h*(44/45*k1 - 56/15*k2 + 32/9*k3)
        GeodesicState s4 = {state.position + k1.position * (dl * 44.0/45.0) + k2.position * (dl * -56.0/15.0) + k3.position * (dl * 32.0/9.0),
                            state.momentum + k1.momentum * (dl * 44.0/45.0) + k2.momentum * (dl * -56.0/15.0) + k3.momentum * (dl * 32.0/9.0)};
        GeodesicState k4 = derivatives_kerr(metric, s4);

        // Stage 5: y + h*(19372/6561*k1 - 25360/2187*k2 + 64448/6561*k3 - 212/729*k4)
        GeodesicState s5 = {state.position + k1.position * (dl * 19372.0/6561.0) + k2.position * (dl * -25360.0/2187.0)
                                           + k3.position * (dl * 64448.0/6561.0) + k4.position * (dl * -212.0/729.0),
                            state.momentum + k1.momentum * (dl * 19372.0/6561.0) + k2.momentum * (dl * -25360.0/2187.0)
                                           + k3.momentum * (dl * 64448.0/6561.0) + k4.momentum * (dl * -212.0/729.0)};
        GeodesicState k5 = derivatives_kerr(metric, s5);

        // Stage 6: y + h*(9017/3168*k1 - 355/33*k2 + 46732/5247*k3 + 49/176*k4 - 5103/18656*k5)
        GeodesicState s6 = {state.position + k1.position * (dl * 9017.0/3168.0) + k2.position * (dl * -355.0/33.0)
                                           + k3.position * (dl * 46732.0/5247.0) + k4.position * (dl * 49.0/176.0)
                                           + k5.position * (dl * -5103.0/18656.0),
                            state.momentum + k1.momentum * (dl * 9017.0/3168.0) + k2.momentum * (dl * -355.0/33.0)
                                           + k3.momentum * (dl * 46732.0/5247.0) + k4.momentum * (dl * 49.0/176.0)
                                           + k5.momentum * (dl * -5103.0/18656.0)};
        GeodesicState k6 = derivatives_kerr(metric, s6);

        // 5th-order solution (local extrapolation — use this result):
        // y5 = y + h*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
        Vec4 y5_pos = state.position + k1.position * (dl * 35.0/384.0)
                                     + k3.position * (dl * 500.0/1113.0)
                                     + k4.position * (dl * 125.0/192.0)
                                     + k5.position * (dl * -2187.0/6784.0)
                                     + k6.position * (dl * 11.0/84.0);
        Vec4 y5_mom = state.momentum + k1.momentum * (dl * 35.0/384.0)
                                     + k3.momentum * (dl * 500.0/1113.0)
                                     + k4.momentum * (dl * 125.0/192.0)
                                     + k5.momentum * (dl * -2187.0/6784.0)
                                     + k6.momentum * (dl * 11.0/84.0);

        // Error estimate: e = y5 - y4 = h * sum(e_i * k_i)
        // e1=71/57600, e3=-71/16695, e4=71/1920, e5=-17253/339200, e6=22/525, e7=-1/40
        // k7 = derivatives_kerr(metric, {y5_pos, y5_mom})  — the FSAL stage
        GeodesicState k7 = derivatives_kerr(metric, {y5_pos, y5_mom});

        Vec4 err_pos = k1.position * (dl * 71.0/57600.0)
                     + k3.position * (dl * -71.0/16695.0)
                     + k4.position * (dl * 71.0/1920.0)
                     + k5.position * (dl * -17253.0/339200.0)
                     + k6.position * (dl * 22.0/525.0)
                     + k7.position * (dl * -1.0/40.0);
        Vec4 err_mom = k1.momentum * (dl * 71.0/57600.0)
                     + k3.momentum * (dl * -71.0/16695.0)
                     + k4.momentum * (dl * 71.0/1920.0)
                     + k5.momentum * (dl * -17253.0/339200.0)
                     + k6.momentum * (dl * 22.0/525.0)
                     + k7.momentum * (dl * -1.0/40.0);

        // Compute max relative error across spatial coordinates + all momenta
        constexpr double eps = 1e-10;
        double err = 0.0;
        for (int i = 1; i < 4; ++i) {
            double scale = std::abs(y5_pos[i]) + eps;
            err = std::max(err, std::abs(err_pos[i]) / scale);
        }
        for (int i = 0; i < 4; ++i) {
            double scale = std::abs(y5_mom[i]) + eps;
            err = std::max(err, std::abs(err_mom[i]) / scale);
        }

        return {GeodesicState{y5_pos, y5_mom}, err};
    }

    /// Adaptive Dormand-Prince 4(5) — drop-in replacement for
    /// adaptive_step_kerr with ~2× fewer metric evaluations per step.
    AdaptiveResult adaptive_step_kerr_dp45(const Kerr& metric,
                                           const GeodesicState& state,
                                           double dlambda,
                                           double tolerance) const;

private:
    static constexpr double fd_epsilon = 1e-6;
};

} // namespace grrt

#endif
