// src/romberg_step.cpp
//
// Implementation of one Romberg-controlled raymarch step.
// Uses a full trapezoidal pass and a composite half-step pass to
// estimate the integration error for caller-driven step control.

#include "grrt/geodesic/romberg_step.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/spacetime/kerr.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace grrt {

// Out-of-line defaulted ctor/dtor for the abstract class StepSampler.
// Required because StepSampler is GRRT_EXPORT (DLL boundary) — MSVC
// requires the vtable's destructor be defined in the DLL itself, not
// only as an inline default in the header.
StepSampler::StepSampler() = default;
StepSampler::~StepSampler() = default;

RombergStep romberg_step(
    const GeodesicState& start_state,
    double ds_proposed,
    std::span<const double> channels_nu_obs,
    const StepSampler& sampler,
    const Kerr& metric,
    const RK4& integrator)
{
    RombergStep out{};
    out.ds_taken = ds_proposed;
    out.n_channels = static_cast<int>(channels_nu_obs.size());

    if (out.n_channels <= 0) {
        // Empty channel list: nothing to integrate. Still advance state.
        out.end_state = integrator.step_kerr(metric, start_state, ds_proposed);
        out.max_err = 0.0;
        return out;
    }

    // Per-sample integrand storage (sized to MAX_ROMBERG_CHANNELS).
    std::array<double, MAX_ROMBERG_CHANNELS> i_start{};
    std::array<double, MAX_ROMBERG_CHANNELS> i_mid{};
    std::array<double, MAX_ROMBERG_CHANNELS> i_end_full{};
    std::array<double, MAX_ROMBERG_CHANNELS> i_end_half{};

    // Spans bounded to the actual channel count.
    const auto nc = static_cast<size_t>(out.n_channels);
    std::span<double> span_start  {i_start.data(),    nc};
    std::span<double> span_mid    {i_mid.data(),      nc};
    std::span<double> span_end_f  {i_end_full.data(), nc};
    std::span<double> span_end_h  {i_end_half.data(), nc};

    // Sample at start (shared between full and half passes).
    sampler.sample_integrand(start_state, channels_nu_obs, span_start);

    // --- Full step pass ---
    const GeodesicState end_full = integrator.step_kerr(metric, start_state, ds_proposed);
    sampler.sample_integrand(end_full, channels_nu_obs, span_end_f);

    // Trapezoidal Δτ_full[ch] = 0.5 · (i_start + i_end_full) · ds
    std::array<double, MAX_ROMBERG_CHANNELS> dtau_full{};
    for (int ch = 0; ch < out.n_channels; ++ch) {
        dtau_full[ch] = 0.5 * (i_start[ch] + i_end_full[ch]) * ds_proposed;
    }

    // --- Half-step pass (two steps of ds/2) ---
    const double half = 0.5 * ds_proposed;
    const GeodesicState mid      = integrator.step_kerr(metric, start_state, half);
    const GeodesicState end_half = integrator.step_kerr(metric, mid,         half);
    sampler.sample_integrand(mid,      channels_nu_obs, span_mid);
    sampler.sample_integrand(end_half, channels_nu_obs, span_end_h);

    // Composite trapezoid Δτ_half[ch] = 0.5·(i_start + 2·i_mid + i_end_half) · half
    for (int ch = 0; ch < out.n_channels; ++ch) {
        out.dtau[ch] = 0.5 * (i_start[ch] + 2.0 * i_mid[ch] + i_end_half[ch]) * half;
    }

    // --- Error estimate ---
    double err = 0.0;
    for (int ch = 0; ch < out.n_channels; ++ch) {
        err = std::max(err, std::abs(dtau_full[ch] - out.dtau[ch]));
    }
    out.max_err  = err;
    out.end_state = end_half;  // half-step path is more accurate

    return out;
}

} // namespace grrt
