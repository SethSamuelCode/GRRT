#include "cuda_backend.h"
#include "cuda_types.h"
#include "cuda_camera.h"
#include "cuda_render_upload.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

// CPU headers for LUT data extraction (host-only)
#include "grrt/color/spectrum.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/celestial_sphere.h"

bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

CudaRenderContext* cuda_context_create(const GRRTParams* params) {
    if (!cuda_available()) return nullptr;

    auto* ctx = new CudaRenderContext();
    ctx->width = params->width;
    ctx->height = params->height;

    cudaMalloc(&ctx->d_output, params->width * params->height * sizeof(float4));

    // Mapped pinned memory for cancel flag (accessible by both host and device)
    cudaHostAlloc(&ctx->h_cancel_flag, sizeof(int), cudaHostAllocMapped);
    cudaHostGetDevicePointer(&ctx->d_cancel_flag, ctx->h_cancel_flag, 0);
    *ctx->h_cancel_flag = 0;

    return ctx;
}

void cuda_context_destroy(CudaRenderContext* ctx) {
    if (!ctx) return;
    if (ctx->d_output) cudaFree(ctx->d_output);
    if (ctx->h_cancel_flag) cudaFreeHost(ctx->h_cancel_flag);
    delete ctx;
}

int cuda_render(CudaRenderContext* ctx, const GRRTParams* params, float* framebuffer) {
    if (!ctx) return -1;

    *ctx->h_cancel_flag = 0;

    // Compute dimensional spin a = spin * mass (Kerr only; Schwarzschild uses a=0)
    double mass = params->mass > 0.0 ? params->mass : 1.0;
    double spin_a = 0.0;
    if (params->metric_type == GRRT_METRIC_KERR) {
        double spin_param = params->spin > 0.0 ? params->spin : 0.998;
        spin_a = spin_param * mass;
    }

    double observer_r = params->observer_r > 0.0 ? params->observer_r : 50.0;
    double observer_theta = params->observer_theta > 0.0 ? params->observer_theta : 1.396;
    double fov = params->fov > 0.0 ? params->fov : 1.047;
    int max_steps = params->integrator_max_steps > 0 ? params->integrator_max_steps : 10000;
    double tolerance = params->integrator_tolerance > 0.0 ? params->integrator_tolerance : 1e-8;

    // --- Build RenderParams ---
    cuda::RenderParams rp{};
    rp.width = params->width;
    rp.height = params->height;
    rp.metric_type = (params->metric_type == GRRT_METRIC_KERR)
                     ? cuda::MetricType::Kerr : cuda::MetricType::Schwarzschild;
    rp.mass = mass;
    rp.spin = spin_a;  // Dimensional spin a, not dimensionless
    rp.observer_r = observer_r;
    rp.observer_theta = observer_theta;
    rp.observer_phi = params->observer_phi;
    rp.fov = fov;
    rp.cam_position = {0.0, observer_r, observer_theta, params->observer_phi};
    rp.integrator_tolerance = tolerance;
    rp.integrator_max_steps = max_steps;
    rp.r_escape = 1000.0;
    rp.horizon_epsilon = 0.01;

    // Disk
    rp.disk_enabled = params->disk_enabled;
    if (params->disk_enabled) {
        double r_isco = cuda::isco_radius(rp.metric_type, mass, spin_a);
        double disk_temp = params->disk_temperature > 0.0 ? params->disk_temperature : 1e7;
        double disk_outer = params->disk_outer > 0.0 ? params->disk_outer : 20.0;

        rp.disk_r_inner = (params->disk_inner > 0.0) ? params->disk_inner : r_isco;
        rp.disk_r_outer = disk_outer;
        rp.disk_peak_temperature = disk_temp;

        // Build CPU AccretionDisk to extract flux LUT
        grrt::AccretionDisk cpu_disk(mass, spin_a, r_isco, disk_outer, disk_temp);
        rp.disk_flux_lut_size = cpu_disk.flux_lut_size();
        rp.disk_flux_r_min = cpu_disk.flux_r_min();
        rp.disk_flux_r_max = cpu_disk.flux_r_max();
        rp.disk_flux_max = cpu_disk.flux_max();

        // Upload flux LUT via wrapper (same-TU requirement)
        const auto& flux_data = cpu_disk.flux_lut_data();
        cuda::upload_flux_lut(flux_data.data(), flux_data.size());
    }

    // Spectrum
    grrt::SpectrumLUT cpu_spectrum;
    rp.spectrum_t_min = 1000.0;
    rp.spectrum_t_max = 100000.0;
    rp.spectrum_num_entries = 1000;

    // Upload spectrum LUTs via wrappers (float precision — perceptual data)
    {
        const auto& colors = cpu_spectrum.color_lut_data();
        int n = std::min((int)colors.size(), cuda::MAX_SPECTRUM_ENTRIES);
        std::vector<float> flat_colors(n * 3);
        for (int i = 0; i < n; ++i) {
            flat_colors[i * 3 + 0] = static_cast<float>(colors[i][0]);
            flat_colors[i * 3 + 1] = static_cast<float>(colors[i][1]);
            flat_colors[i * 3 + 2] = static_cast<float>(colors[i][2]);
        }
        cuda::upload_color_lut(reinterpret_cast<const float(*)[3]>(flat_colors.data()), n);

        const auto& lums = cpu_spectrum.luminosity_lut_data();
        int nl = std::min((int)lums.size(), cuda::MAX_SPECTRUM_ENTRIES);
        std::vector<float> flat_lums(nl);
        for (int i = 0; i < nl; ++i) {
            flat_lums[i] = static_cast<float>(lums[i]);
        }
        cuda::upload_luminosity_lut(flat_lums.data(), nl);
    }

    // Celestial sphere
    rp.background_type = params->background_type;
    rp.num_stars = 0;
    rp.star_angular_tolerance = 0.003;
    if (params->background_type == GRRT_BG_STARS) {
        grrt::CelestialSphere cpu_sphere;
        const auto& stars = cpu_sphere.star_data();
        rp.num_stars = (int)stars.size();
        if (rp.num_stars > cuda::MAX_STARS) rp.num_stars = cuda::MAX_STARS;

        std::vector<cuda::Star> flat_stars(rp.num_stars);
        for (int i = 0; i < rp.num_stars; ++i) {
            flat_stars[i].theta = static_cast<float>(stars[i].theta);
            flat_stars[i].phi = static_cast<float>(stars[i].phi);
            flat_stars[i].brightness = static_cast<float>(stars[i].brightness);
        }
        cuda::upload_stars(flat_stars.data(), rp.num_stars);
    }

    // Build camera tetrad on host
    cuda::build_tetrad(rp);

    // Upload RenderParams via wrapper
    cuda::upload_render_params(rp);

    // --- Launch kernel via wrapper ---
    cuda::launch_render_kernel(ctx->d_output, ctx->d_cancel_flag, params->width, params->height);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA kernel error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // --- Download results directly into framebuffer ---
    // float4 is layout-compatible with float[4] (x,y,z,w = R,G,B,A)
    int num_pixels = params->width * params->height;
    cudaMemcpy(framebuffer, ctx->d_output,
               num_pixels * sizeof(float4), cudaMemcpyDeviceToHost);

    return 0;
}

int cuda_render_tile(CudaRenderContext* /*ctx*/, const GRRTParams* /*params*/, float* /*buffer*/,
                     int /*x*/, int /*y*/, int /*tile_w*/, int /*tile_h*/) {
    // TODO: implement tile rendering (launch smaller grid with offset)
    return -1;
}

void cuda_cancel(CudaRenderContext* ctx) {
    if (ctx && ctx->h_cancel_flag) {
        *ctx->h_cancel_flag = 1;
    }
}
