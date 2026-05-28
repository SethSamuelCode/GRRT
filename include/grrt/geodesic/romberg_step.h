#ifndef GRRT_GEODESIC_ROMBERG_STEP_H
#define GRRT_GEODESIC_ROMBERG_STEP_H

#include "grrt/geodesic/integrator.h"
#include "grrt_export.h"

#include <array>
#include <span>

namespace grrt {

class Kerr;
class RK4;

/// Maximum number of control channels carried through a single helper call.
/// Covers RGB (3) and modest spectral outputs without heap allocation.
/// Spectral callers wanting more bins must either raise this constant
/// or split their bins across multiple helper calls.
constexpr int MAX_ROMBERG_CHANNELS = 32;

/// Result of one Romberg-controlled raymarch step.
struct RombergStep {
    GeodesicState end_state;                           ///< Geodesic state at end of accepted half-step path.
    GeodesicState mid_state;                           ///< Geodesic state at the step midpoint (junction of the two half-steps).
    std::array<double, MAX_ROMBERG_CHANNELS> dtau;     ///< Per-channel Δτ from the half-step pass (more accurate).
    double max_err;                                    ///< Max over channels of |Δτ_full − Δτ_half|.
    double ds_taken;                                   ///< = ds_proposed (helper does not shrink; caller does).
    int n_channels;                                    ///< Count of valid entries in dtau[].
};

/// Sampler interface: callers provide one of these to romberg_step()
/// so the helper can query the integrand κρ|p·u_emit| at any state.
/// VolumetricDiskSampler in geodesic_tracer.cpp wraps the production
/// VolumetricDisk; tests provide synthetic implementations.
struct GRRT_EXPORT StepSampler {
    StepSampler();
    virtual ~StepSampler();

    /// Sample the per-channel integrand at a geodesic state.
    /// integrand[ch] = κ_total(ν_emit, ρ, T) · ρ · |p·u_emit|
    /// where ν_emit = |g| · channels_nu_obs[ch] and g is the redshift factor.
    /// If the state is outside the optically active region, the sampler
    /// must zero the integrand[] entries and return false.
    /// Returns true when the integrand was sampled.
    virtual bool sample_integrand(const GeodesicState& state,
                                  std::span<const double> channels_nu_obs,
                                  std::span<double> integrand) const = 0;
};

/// Take one Romberg-controlled raymarch step.
/// Caller manages step proposal/growth/shrinkage between calls.
/// Helper does ONE geodesic full step + TWO half-steps to estimate error.
GRRT_EXPORT RombergStep romberg_step(
    const GeodesicState& start_state,
    double ds_proposed,
    std::span<const double> channels_nu_obs,
    const StepSampler& sampler,
    const Kerr& metric,
    const RK4& integrator);

} // namespace grrt

#endif
