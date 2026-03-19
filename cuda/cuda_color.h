#ifndef CUDA_COLOR_H
#define CUDA_COLOR_H

#include "cuda_math.h"
#include "cuda_types.h"

namespace cuda {

// These arrays live in __constant__ memory, defined in cuda_render.cu.
// Declared extern here so device functions can access them.
extern __constant__ double d_color_lut[MAX_SPECTRUM_ENTRIES][3];
extern __constant__ double d_luminosity_lut[MAX_SPECTRUM_ENTRIES];

/// Spectrum chromaticity lookup (normalized RGB at temperature T).
/// Linear interpolation over the precomputed LUT.
__device__ inline Vec3 spectrum_chromaticity(double temperature,
                                              double t_min, double t_max, int num_entries) {
    // Clamp to range
    if (temperature <= t_min) return {d_color_lut[0][0], d_color_lut[0][1], d_color_lut[0][2]};
    if (temperature >= t_max) {
        int last = num_entries - 1;
        return {d_color_lut[last][0], d_color_lut[last][1], d_color_lut[last][2]};
    }

    double frac = (temperature - t_min) / (t_max - t_min) * (num_entries - 1);
    int idx = (int)frac;
    double t = frac - idx;
    if (idx >= num_entries - 1) { idx = num_entries - 2; t = 1.0; }

    return {
        d_color_lut[idx][0] * (1.0 - t) + d_color_lut[idx + 1][0] * t,
        d_color_lut[idx][1] * (1.0 - t) + d_color_lut[idx + 1][1] * t,
        d_color_lut[idx][2] * (1.0 - t) + d_color_lut[idx + 1][2] * t
    };
}

/// Spectrum luminosity: analytical sigma * T^4 (matches CPU behavior).
/// Normalized relative to t_max so values near the ceiling are ~1.0.
__device__ inline double spectrum_luminosity(double temperature, double t_max) {
    double ratio = temperature / t_max;
    return ratio * ratio * ratio * ratio;
}

} // namespace cuda

#endif
