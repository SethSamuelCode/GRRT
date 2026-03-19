#ifndef CUDA_RENDER_UPLOAD_H
#define CUDA_RENDER_UPLOAD_H

/// @file cuda_render_upload.h
/// @brief Host-callable declarations for CUDA constant/device memory upload functions
///        and render kernel launch.
///
/// The implementations live in cuda_render.cu (same translation unit as the
/// __constant__/__device__ symbol definitions, which is required by cudaMemcpyToSymbol).

#include "cuda_types.h"
#include <cuda_runtime.h>

namespace cuda {

/// @brief Upload RenderParams to device constant memory.
void upload_render_params(const RenderParams& params);

/// @brief Upload chromaticity (normalized RGB) lookup table to constant memory.
/// @param data   Array of [count][3] floats
/// @param count  Number of entries (must be <= MAX_SPECTRUM_ENTRIES)
void upload_color_lut(const float data[][3], size_t count);

/// @brief Upload luminosity lookup table to constant memory.
/// @param data   Array of count floats
/// @param count  Number of entries (must be <= MAX_SPECTRUM_ENTRIES)
void upload_luminosity_lut(const float* data, size_t count);

/// @brief Upload Novikov-Thorne flux lookup table to constant memory.
/// @param data   Array of count doubles
/// @param count  Number of entries (must be <= MAX_FLUX_LUT_ENTRIES)
void upload_flux_lut(const double* data, size_t count);

/// @brief Upload star catalog to device global memory.
/// @param data   Array of count Star structs
/// @param count  Number of stars (must be <= MAX_STARS)
void upload_stars(const Star* data, size_t count);

/// @brief Launch the render kernel with a 16x16 thread block configuration.
/// @param output      Device float4 buffer (width * height elements)
/// @param cancel_flag Device int pointer (nullable); set to nonzero to cancel
/// @param width       Image width in pixels
/// @param height      Image height in pixels
void launch_render_kernel(float4* output, int* cancel_flag, int width, int height);

} // namespace cuda

#endif // CUDA_RENDER_UPLOAD_H
