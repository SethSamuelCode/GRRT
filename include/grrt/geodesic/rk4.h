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

private:
    static constexpr double fd_epsilon = 1e-6;
};

} // namespace grrt

#endif
