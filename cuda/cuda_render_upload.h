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

/// @brief Upload star catalog (sorted by bucket) to device global memory.
/// @param data   Array of count Star structs, sorted by spatial bucket
/// @param count  Number of stars (must be <= MAX_STARS)
void upload_stars(const Star* data, size_t count);

/// @brief Upload star spatial grid prefix-sum offsets to constant memory.
/// @param offsets  Array of (STAR_GRID_CELLS + 1) ints
/// @param count    Number of entries (STAR_GRID_CELLS + 1)
void upload_star_grid(const int* offsets, size_t count);

/// @brief Upload all volumetric disk LUTs and noise permutation table to device memory.
///
/// Each LUT is allocated on the device with cudaMalloc; the __device__ pointer
/// symbols in cuda_render.cu are then set to point at the allocations.
/// Call free_volumetric_luts() after the kernel completes to release device memory.
void upload_volumetric_luts(const double* H_data, size_t H_size,
                             const double* rho_mid_data, size_t rho_mid_size,
                             const double* rho_prof_data, size_t rho_prof_size,
                             const double* T_prof_data, size_t T_prof_size,
                             const double* kabs_data, size_t kabs_size,
                             const double* kes_data, size_t kes_size,
                             const double* kross_data, size_t kross_size,
                             const double* mu_data, size_t mu_size,
                             const int* perm_data);

/// @brief Free all device memory allocated by upload_volumetric_luts().
void free_volumetric_luts();

/// @brief Launch the render kernel with a 16x16 thread block configuration.
/// @param output      Device float4 buffer (width * height elements)
/// @param cancel_flag Device int pointer (nullable); set to nonzero to cancel
/// @param width       Image width in pixels
/// @param height      Image height in pixels
void launch_render_kernel(float4* output, int* cancel_flag, int width, int height);

} // namespace cuda

#endif // CUDA_RENDER_UPLOAD_H
