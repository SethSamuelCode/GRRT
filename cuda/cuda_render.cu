/// @file cuda_render.cu
/// @brief Main CUDA render kernel and host-callable constant/device memory upload wrappers.
///
/// This file has three responsibilities:
///   1. Define __constant__ and __device__ memory symbols (declared extern in headers)
///   2. Implement the render kernel (one thread per pixel)
///   3. Implement host-callable upload wrappers for cudaMemcpyToSymbol
///
/// CRITICAL: cudaMemcpyToSymbol must be called from the SAME translation unit that
/// defines the symbol. That's why the upload wrappers live here, not in cuda_backend.cu.

#include "cuda_types.h"
#include "cuda_math.h"
#include "cuda_metric.h"
#include "cuda_geodesic.h"
#include "cuda_camera.h"
#include "cuda_color.h"
#include "cuda_scene.h"
#include "cuda_volumetric_disk.h"
#include "cuda_noise.h"
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Constant/device memory definitions
// ---------------------------------------------------------------------------
// d_color_lut, d_luminosity_lut, d_flux_lut, d_stars are declared extern in
// cuda_color.h and cuda_scene.h. With CUDA_SEPARABLE_COMPILATION enabled
// (-rdc=true), nvcc correctly handles the extern + definition pattern.
__constant__ float cuda::d_color_lut[cuda::MAX_SPECTRUM_ENTRIES][3];
__constant__ float cuda::d_luminosity_lut[cuda::MAX_SPECTRUM_ENTRIES];
__constant__ double cuda::d_flux_lut[cuda::MAX_FLUX_LUT_ENTRIES];

// d_stars remains __device__ (global memory) — 60 KB with float fields.
// Stars are sorted by spatial bucket for O(1) average lookup.
__device__ cuda::Star cuda::d_stars[cuda::MAX_STARS];

// CSR prefix-sum offset array for star spatial grid (8 KB in constant memory)
__constant__ int cuda::d_star_grid_offset[cuda::STAR_GRID_CELLS + 1];

// Volumetric disk LUTs (global memory, device pointers)
__device__ double* cuda::d_vol_H_lut = nullptr;
__device__ double* cuda::d_vol_rho_mid_lut = nullptr;
__device__ double* cuda::d_vol_rho_profile_lut = nullptr;
__device__ double* cuda::d_vol_T_profile_lut = nullptr;
__device__ double* cuda::d_opacity_kappa_abs_lut = nullptr;
__device__ double* cuda::d_opacity_kappa_es_lut = nullptr;
__device__ double* cuda::d_opacity_kappa_ross_lut = nullptr;
__device__ double* cuda::d_opacity_mu_lut = nullptr;

// Noise permutation table (constant memory)
__constant__ int cuda::d_noise_perm[512];

namespace cuda {

// d_params has no extern declaration in any header, so we define it directly
// inside the namespace block.
__constant__ RenderParams d_params;

// ---------------------------------------------------------------------------
// Ray termination categories
// ---------------------------------------------------------------------------
enum class RayTermination { Horizon, Escaped, MaxSteps };

// ---------------------------------------------------------------------------
// Render kernel: one thread per pixel
// ---------------------------------------------------------------------------

/// @brief Main render kernel. Each thread traces one photon geodesic for pixel (i, j).
///
/// Algorithm:
///   1. Generate initial null geodesic from camera tetrad (ray_for_pixel)
///   2. Integrate with adaptive RK4, checking for:
///      - Horizon crossing (ray absorbed by black hole)
///      - Disk crossing (equatorial plane intersection within disk bounds)
///      - Escape (r > r_escape)
///   3. For escaped rays, sample the celestial sphere
///   4. Write accumulated linear-HDR color to output buffer
///
/// @param output      Device float4 buffer (width * height), stores linear HDR color
/// @param cancel_flag Device int pointer; if non-null and *cancel_flag != 0, threads exit
__global__ void render_kernel(float4* output, int* cancel_flag) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= d_params.width || j >= d_params.height) return;
    if (cancel_flag && *cancel_flag) return;

    // Generate initial null geodesic for this pixel
    GeodesicState state = ray_for_pixel(d_params, i, j);

    double dlambda = 0.01 * d_params.observer_r;
    Vec3 accumulated_color = {};
    RayTermination termination = RayTermination::MaxSteps;
    const double r_horizon = horizon_radius(d_params.metric_type, d_params.mass, d_params.spin);
    double prev_theta = state.position[2];

    for (int step = 0; step < d_params.integrator_max_steps; ++step) {
        // Periodic cancellation check (every 256 steps to avoid memory traffic)
        if (cancel_flag && (step & 0xFF) == 0 && *cancel_flag) break;

        const double r = state.position[1];

        // Check horizon crossing
        if (r < r_horizon + d_params.horizon_epsilon) {
            termination = RayTermination::Horizon;
            break;
        }

        // Check escape
        if (r > d_params.r_escape) {
            termination = RayTermination::Escaped;
            break;
        }

        // Adaptive RK4 step
        AdaptiveResult result = rk4_adaptive_step(d_params.metric_type, d_params.mass,
                                                   d_params.spin, state, dlambda,
                                                   d_params.integrator_tolerance);
        const double new_theta = result.state.position[2];

        // Volumetric disk: detect midplane crossing (θ crosses π/2)
        // or direct volume entry, then interpolate to crossing and raymarch.
        if (d_params.disk_volumetric) {
            constexpr double half_pi = M_PI / 2.0;
            const double d_prev = prev_theta - half_pi;
            const double d_new = new_theta - half_pi;
            const double z_new = result.state.position[1] * cos(new_theta);
            const bool crossed = (d_prev * d_new < 0.0)
                              && fabs(d_prev - d_new) > 1e-12;
            const bool inside = vol_inside(result.state.position[1], z_new, d_params);

            if (crossed || inside) {
                // Interpolate to the midplane crossing so raymarch starts
                // inside the disk, not far above/below it.
                GeodesicState entry = result.state;
                if (crossed) {
                    const double frac = -d_prev / (d_new - d_prev);
                    for (int mu = 0; mu < 4; ++mu) {
                        entry.position[mu] = state.position[mu]
                            + frac * (result.state.position[mu] - state.position[mu]);
                        entry.momentum[mu] = state.momentum[mu]
                            + frac * (result.state.momentum[mu] - state.momentum[mu]);
                    }
                }

                const double r_entry = entry.position[1];
                if (r_entry >= d_params.disk_r_horizon
                    && r_entry <= d_params.disk_r_outer) {
                    vol_raymarch(entry, accumulated_color, d_params);
                    state = entry;
                    prev_theta = state.position[2];
                    dlambda = result.next_dlambda;
                    continue;
                }
            }
        }

        // Check equatorial thin-disk crossing (theta crosses pi/2)
        if (!d_params.disk_volumetric && d_params.disk_enabled) {
            constexpr double half_pi = M_PI / 2.0;
            const bool crossed = (prev_theta - half_pi) * (new_theta - half_pi) < 0.0;
            if (crossed) {
                // Linear interpolation to find crossing point
                const double frac = (half_pi - prev_theta) / (new_theta - prev_theta);
                const double r_cross = state.position[1]
                                       + frac * (result.state.position[1] - state.position[1]);

                if (r_cross >= d_params.disk_r_inner && r_cross <= d_params.disk_r_outer) {
                    // Interpolate momentum at crossing
                    Vec4 p_cross{};
                    for (int k = 0; k < 4; ++k) {
                        p_cross[k] = state.momentum[k]
                                     + frac * (result.state.momentum[k] - state.momentum[k]);
                    }

                    Vec3 emission = disk_emission(r_cross, p_cross,
                                                   d_params.observer_r, d_params);
                    accumulated_color += emission;
                }
            }
        }

        prev_theta = new_theta;
        state = result.state;
        dlambda = result.next_dlambda;
    }

    // Celestial sphere contribution for escaped rays
    if (termination == RayTermination::Escaped) {
        Vec3 bg = celestial_sphere_sample(state.position, d_params);
        accumulated_color += bg;
    }

    // Write linear HDR color to output buffer (row-major order)
    const int idx = j * d_params.width + i;
    output[idx] = make_float4(
        static_cast<float>(accumulated_color[0]),
        static_cast<float>(accumulated_color[1]),
        static_cast<float>(accumulated_color[2]),
        1.0f
    );
}

// ---------------------------------------------------------------------------
// Host-callable upload wrappers
// ---------------------------------------------------------------------------
// These MUST be in the same translation unit as the __constant__/__device__
// symbol definitions, because cudaMemcpyToSymbol resolves symbols at link time
// within a single .cu file.

void upload_render_params(const RenderParams& params) {
    cudaMemcpyToSymbol(d_params, &params, sizeof(RenderParams));
}

void upload_color_lut(const float data[][3], size_t count) {
    cudaMemcpyToSymbol(d_color_lut, data, count * 3 * sizeof(float));
}

void upload_luminosity_lut(const float* data, size_t count) {
    cudaMemcpyToSymbol(d_luminosity_lut, data, count * sizeof(float));
}

void upload_flux_lut(const double* data, size_t count) {
    cudaMemcpyToSymbol(d_flux_lut, data, count * sizeof(double));
}

void upload_stars(const Star* data, size_t count) {
    cudaMemcpyToSymbol(d_stars, data, count * sizeof(Star));
}

void upload_star_grid(const int* offsets, size_t count) {
    cudaMemcpyToSymbol(d_star_grid_offset, offsets, count * sizeof(int));
}

// ---------------------------------------------------------------------------
// Volumetric disk LUT upload / free
// ---------------------------------------------------------------------------

// Host-side copies of device pointers (for cleanup)
static double* h_vol_H_ptr = nullptr;
static double* h_vol_rho_mid_ptr = nullptr;
static double* h_vol_rho_profile_ptr = nullptr;
static double* h_vol_T_profile_ptr = nullptr;
static double* h_opacity_kappa_abs_ptr = nullptr;
static double* h_opacity_kappa_es_ptr = nullptr;
static double* h_opacity_kappa_ross_ptr = nullptr;
static double* h_opacity_mu_ptr = nullptr;

/// @brief Helper macro: allocate device memory, copy data, and set the __device__ pointer.
///
/// cudaMemcpyToSymbol requires the symbol name at compile time, so we use a macro
/// rather than a function to ensure the symbol is resolved correctly.
#define UPLOAD_VOL_LUT(symbol, h_ptr, data, count)                     \
    do {                                                                \
        size_t bytes = (count) * sizeof(double);                        \
        cudaMalloc(&(h_ptr), bytes);                                    \
        cudaMemcpy((h_ptr), (data), bytes, cudaMemcpyHostToDevice);     \
        double* tmp = (h_ptr);                                          \
        cudaMemcpyToSymbol(symbol, &tmp, sizeof(double*));              \
    } while (0)

void upload_volumetric_luts(const double* H_data, size_t H_size,
                             const double* rho_mid_data, size_t rho_mid_size,
                             const double* rho_prof_data, size_t rho_prof_size,
                             const double* T_prof_data, size_t T_prof_size,
                             const double* kabs_data, size_t kabs_size,
                             const double* kes_data, size_t kes_size,
                             const double* kross_data, size_t kross_size,
                             const double* mu_data, size_t mu_size,
                             const int* perm_data) {
    UPLOAD_VOL_LUT(d_vol_H_lut, h_vol_H_ptr, H_data, H_size);
    UPLOAD_VOL_LUT(d_vol_rho_mid_lut, h_vol_rho_mid_ptr, rho_mid_data, rho_mid_size);
    UPLOAD_VOL_LUT(d_vol_rho_profile_lut, h_vol_rho_profile_ptr, rho_prof_data, rho_prof_size);
    UPLOAD_VOL_LUT(d_vol_T_profile_lut, h_vol_T_profile_ptr, T_prof_data, T_prof_size);
    UPLOAD_VOL_LUT(d_opacity_kappa_abs_lut, h_opacity_kappa_abs_ptr, kabs_data, kabs_size);
    UPLOAD_VOL_LUT(d_opacity_kappa_es_lut, h_opacity_kappa_es_ptr, kes_data, kes_size);
    UPLOAD_VOL_LUT(d_opacity_kappa_ross_lut, h_opacity_kappa_ross_ptr, kross_data, kross_size);
    UPLOAD_VOL_LUT(d_opacity_mu_lut, h_opacity_mu_ptr, mu_data, mu_size);

    // Upload noise permutation table to constant memory
    cudaMemcpyToSymbol(d_noise_perm, perm_data, 512 * sizeof(int));
}

#undef UPLOAD_VOL_LUT

void free_volumetric_luts() {
    auto free_one = [](double** h_ptr) {
        if (*h_ptr) { cudaFree(*h_ptr); *h_ptr = nullptr; }
    };
    free_one(&h_vol_H_ptr);
    free_one(&h_vol_rho_mid_ptr);
    free_one(&h_vol_rho_profile_ptr);
    free_one(&h_vol_T_profile_ptr);
    free_one(&h_opacity_kappa_abs_ptr);
    free_one(&h_opacity_kappa_es_ptr);
    free_one(&h_opacity_kappa_ross_ptr);
    free_one(&h_opacity_mu_ptr);
}

void launch_render_kernel(float4* output, int* cancel_flag, int width, int height) {
    // 32-wide ensures warp threads are horizontally adjacent pixels (similar rays = less divergence)
    dim3 threads(32, 8);
    dim3 blocks((width + 31) / 32, (height + 7) / 8);
    render_kernel<<<blocks, threads>>>(output, cancel_flag);
}

} // namespace cuda
