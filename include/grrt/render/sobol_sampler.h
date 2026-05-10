#ifndef GRRT_RENDER_SOBOL_SAMPLER_H
#define GRRT_RENDER_SOBOL_SAMPLER_H

#include "grrt_export.h"

namespace grrt {

/// One 2D point of the Owen-scrambled Sobol sequence. Both components in [0, 1).
struct SobolSample {
    double x;
    double y;
};

/// Generate the i-th 2D sub-pixel sample for the given pixel using Owen-scrambled
/// Sobol. Pure function — same arguments always return the same point.
///
/// The per-pixel scramble seed is derived from (pixel_x, pixel_y), so neighboring
/// pixels see decorrelated point sequences. Renders are deterministic across
/// runs and thread counts.
///
/// @param pixel_x integer pixel x-coordinate (column index)
/// @param pixel_y integer pixel y-coordinate (row index)
/// @param sample_index 0-based index of the sample within the pixel (0 <= s < spp)
/// @return sub-pixel offset in [0, 1)^2
[[nodiscard]] GRRT_EXPORT SobolSample sobol_sample_2d(int pixel_x, int pixel_y, int sample_index);

} // namespace grrt

#endif
