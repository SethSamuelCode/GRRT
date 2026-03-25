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
#include "cuda_vol_host_data.h"

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

    // Volumetric disk
    rp.disk_volumetric = params->disk_volumetric;
    if (params->disk_volumetric) {
        double disk_outer = params->disk_outer > 0.0 ? params->disk_outer : 20.0;
        double disk_temp = params->disk_temperature > 0.0 ? params->disk_temperature : 1e7;
        double vol_alpha = params->disk_alpha > 0.0 ? params->disk_alpha : 0.1;
        double vol_turb = params->disk_turbulence > 0.0 ? params->disk_turbulence : 0.4;
        unsigned int vol_seed = params->disk_seed > 0 ? static_cast<unsigned int>(params->disk_seed) : 42u;

        // Build VolumetricDisk on CPU (in a separate .cpp TU to avoid nvcc C++20 issues)
        VolDiskHostData vd = build_vol_disk_host_data(mass, spin_a, disk_outer, disk_temp,
                                                       vol_alpha, vol_turb, vol_seed);

        // Fill RenderParams volumetric fields
        rp.disk_r_isco = vd.r_isco;
        rp.disk_r_horizon = vd.r_horizon;
        rp.disk_taper_width = vd.taper_width;
        rp.disk_E_isco = vd.E_isco;
        rp.disk_L_isco = vd.L_isco;
        rp.disk_rho_scale = vd.rho_scale;
        rp.disk_turbulence = vd.turbulence;
        rp.disk_noise_scale = vd.noise_scale;
        rp.disk_r_inner = vd.r_horizon;
        rp.disk_r_outer = disk_outer;

        // LUT grid parameters
        rp.vol_n_r = vd.n_r;
        rp.vol_n_z = vd.n_z;
        rp.vol_r_min = vd.r_min;
        rp.vol_r_max = vd.r_max;

        // Opacity LUT grid parameters
        rp.opacity_n_nu = vd.opacity_n_nu;
        rp.opacity_n_rho = vd.opacity_n_rho;
        rp.opacity_n_T = vd.opacity_n_T;
        rp.opacity_log_nu_min = vd.opacity_log_nu_min;
        rp.opacity_log_nu_max = vd.opacity_log_nu_max;
        rp.opacity_log_rho_min = vd.opacity_log_rho_min;
        rp.opacity_log_rho_max = vd.opacity_log_rho_max;
        rp.opacity_log_T_min = vd.opacity_log_T_min;
        rp.opacity_log_T_max = vd.opacity_log_T_max;

        // Upload all LUTs to device memory
        cuda::upload_volumetric_luts(
            vd.H_lut.data(), vd.H_lut.size(),
            vd.rho_mid_lut.data(), vd.rho_mid_lut.size(),
            vd.rho_profile_lut.data(), vd.rho_profile_lut.size(),
            vd.T_profile_lut.data(), vd.T_profile_lut.size(),
            vd.kappa_abs_lut.data(), vd.kappa_abs_lut.size(),
            vd.kappa_es_lut.data(), vd.kappa_es_lut.size(),
            vd.kappa_ross_lut.data(), vd.kappa_ross_lut.size(),
            vd.mu_lut.data(), vd.mu_lut.size(),
            vd.perm_table.data()
        );
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

    // Celestial sphere with spatial bucketing
    rp.background_type = params->background_type;
    rp.num_stars = 0;
    rp.star_angular_tolerance = 0.003;
    if (params->background_type == GRRT_BG_STARS) {
        grrt::CelestialSphere cpu_sphere;
        const auto& stars = cpu_sphere.star_data();
        rp.num_stars = (int)stars.size();
        if (rp.num_stars > cuda::MAX_STARS) rp.num_stars = cuda::MAX_STARS;

        // Build spatial grid: assign each star to a bucket
        constexpr int GT = cuda::STAR_GRID_THETA;
        constexpr int GP = cuda::STAR_GRID_PHI;
        std::vector<int> bucket(rp.num_stars);
        std::vector<int> cell_count(cuda::STAR_GRID_CELLS, 0);

        for (int i = 0; i < rp.num_stars; ++i) {
            int tb = (int)(stars[i].theta * GT / M_PI);
            int pb = (int)((stars[i].phi + M_PI) * GP / (2.0 * M_PI));
            if (tb < 0) tb = 0; if (tb >= GT) tb = GT - 1;
            if (pb < 0) pb = 0; if (pb >= GP) pb = GP - 1;
            bucket[i] = tb * GP + pb;
            cell_count[bucket[i]]++;
        }

        // Build prefix-sum offset array
        std::vector<int> offsets(cuda::STAR_GRID_CELLS + 1, 0);
        for (int c = 0; c < cuda::STAR_GRID_CELLS; ++c)
            offsets[c + 1] = offsets[c] + cell_count[c];

        // Sort stars by bucket (stable, preserves original order within each cell)
        std::vector<int> write_pos(offsets.begin(), offsets.end() - 1);
        std::vector<cuda::Star> sorted_stars(rp.num_stars);
        for (int i = 0; i < rp.num_stars; ++i) {
            int cell = bucket[i];
            int pos = write_pos[cell]++;
            sorted_stars[pos].theta = static_cast<float>(stars[i].theta);
            sorted_stars[pos].phi = static_cast<float>(stars[i].phi);
            sorted_stars[pos].brightness = static_cast<float>(stars[i].brightness);
        }

        cuda::upload_stars(sorted_stars.data(), rp.num_stars);
        cuda::upload_star_grid(offsets.data(), offsets.size());
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
        if (params->disk_volumetric) cuda::free_volumetric_luts();
        return -1;
    }

    // Free volumetric LUTs now that the kernel has completed
    if (params->disk_volumetric) {
        cuda::free_volumetric_luts();
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
