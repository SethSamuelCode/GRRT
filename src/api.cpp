#include "grrt/api.h"
#include "grrt/spacetime/schwarzschild.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/camera/camera.h"
#include "grrt/render/renderer.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/scene/celestial_sphere.h"
#include "grrt/color/spectrum.h"
#include "grrt/render/tonemapper.h"
#ifdef GRRT_HAS_CUDA
#include "cuda_backend.h"
#endif
#include <memory>
#include <print>
#include <string>

static thread_local std::string g_last_error;

struct GRRTContext {
    GRRTParams params;
    std::unique_ptr<grrt::Metric> metric;
    std::unique_ptr<grrt::Integrator> integrator;
    std::unique_ptr<grrt::GeodesicTracer> tracer;
    std::unique_ptr<grrt::Camera> camera;
    std::unique_ptr<grrt::AccretionDisk> disk;
    std::unique_ptr<grrt::VolumetricDisk> vol_disk;
    std::unique_ptr<grrt::CelestialSphere> sphere;
    std::unique_ptr<grrt::SpectrumLUT> spectrum;
    std::unique_ptr<grrt::ToneMapper> tonemapper;
    std::unique_ptr<grrt::Renderer> renderer;
#ifdef GRRT_HAS_CUDA
    CudaRenderContext* cuda_ctx = nullptr;
#endif
    std::string error_msg;
};

GRRTContext* grrt_create(const GRRTParams* params) {
    auto* ctx = new GRRTContext{};
    ctx->params = *params;

    double mass = params->mass > 0.0 ? params->mass : 1.0;
    double observer_r = params->observer_r > 0.0 ? params->observer_r : 50.0;
    double observer_theta = params->observer_theta > 0.0 ? params->observer_theta : 1.396;
    double fov = params->fov > 0.0 ? params->fov : 1.047;
    int max_steps = params->integrator_max_steps > 0 ? params->integrator_max_steps : 10000;
    double tolerance = params->integrator_tolerance > 0.0 ? params->integrator_tolerance : 1e-8;

    // Core physics — select metric
    double spin_a = 0.0;
    if (params->metric_type == GRRT_METRIC_KERR) {
        double spin_param = params->spin > 0.0 ? params->spin : 0.998;
        spin_a = spin_param * mass;
        ctx->metric = std::make_unique<grrt::Kerr>(mass, spin_a);
    } else {
        ctx->metric = std::make_unique<grrt::Schwarzschild>(mass);
    }
    ctx->integrator = std::make_unique<grrt::RK4>();
    ctx->camera = std::make_unique<grrt::Camera>(
        *ctx->metric, observer_r, observer_theta, params->observer_phi,
        fov, params->width, params->height);

    // Color pipeline
    ctx->spectrum = std::make_unique<grrt::SpectrumLUT>();
    ctx->tonemapper = std::make_unique<grrt::ToneMapper>();

    // Accretion disk (optional)
    if (params->disk_enabled) {
        double disk_temp = params->disk_temperature > 0.0 ? params->disk_temperature : 1e7;
        double disk_outer = params->disk_outer > 0.0 ? params->disk_outer : 20.0;
        double isco = ctx->metric->isco_radius();
        double r_inner = params->disk_inner > 0.0 ? params->disk_inner : isco;
        ctx->disk = std::make_unique<grrt::AccretionDisk>(
            mass, spin_a, r_inner, disk_outer, disk_temp);

        if (params->disk_volumetric) {
            // Derive peak_temperature from mass and Eddington fraction if provided
            double vol_disk_temp = params->disk_temperature > 0 ? params->disk_temperature : 1e7;
            if (params->mass_solar > 0.0 && params->eddington_fraction > 0.0
                && params->disk_temperature <= 0.0) {
                // Radiative efficiency: η = 1 - E_isco (Bardeen, Press & Teukolsky 1972)
                // E_isco is the specific orbital energy at the ISCO.
                // Re-derived here because this runs before VolumetricDisk construction.
                // The same formula exists in VolumetricDisk::E_isco().
                const double a_star = std::abs(params->spin);
                const double Z1 = 1.0 + std::cbrt(1.0 - a_star * a_star)
                                       * (std::cbrt(1.0 + a_star) + std::cbrt(1.0 - a_star));
                const double Z2 = std::sqrt(3.0 * a_star * a_star + Z1 * Z1);
                const double r_isco_M = 3.0 + Z2 - std::sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2));
                const double v_isco = 1.0 / std::sqrt(r_isco_M);
                const double E_isco = (1.0 - 2.0*v_isco*v_isco + a_star*v_isco*v_isco*v_isco)
                                    / std::sqrt(1.0 - 3.0*v_isco*v_isco + 2.0*a_star*v_isco*v_isco*v_isco);
                const double eta = 1.0 - E_isco;
                vol_disk_temp = 5.0e7 * std::pow(eta, 0.25)
                              * std::pow(params->mass_solar, -0.25)
                              * std::pow(params->eddington_fraction, 0.25);
                std::printf("[GRRT] Derived T_peak = %.0f K from M=%.1f M_sun, f_Edd=%.3f, eta=%.4f\n",
                            vol_disk_temp, params->mass_solar, params->eddington_fraction, eta);
            }

            grrt::VolumetricParams vp;
            vp.alpha = params->disk_alpha;
            vp.turbulence = params->disk_turbulence;
            vp.seed = static_cast<uint32_t>(params->disk_seed);
            vp.noise_scale = params->disk_noise_scale;
            vp.noise_octaves = params->disk_noise_octaves;
            ctx->vol_disk = std::make_unique<grrt::VolumetricDisk>(
                params->mass, params->spin,
                params->disk_outer > 0 ? params->disk_outer : 30.0,
                vol_disk_temp,
                vp);
        }
    }

    // Geodesic tracer (created after volumetric disk so it can reference it)
    ctx->tracer = std::make_unique<grrt::GeodesicTracer>(
        *ctx->metric, *ctx->integrator, observer_r, max_steps, 1000.0, tolerance,
        ctx->vol_disk.get());

    // Background (optional)
    if (params->background_type == GRRT_BG_STARS) {
        ctx->sphere = std::make_unique<grrt::CelestialSphere>();
    }

    // Renderer
    int spp = params->samples_per_pixel > 0 ? params->samples_per_pixel : 1;
    ctx->renderer = std::make_unique<grrt::Renderer>(
        *ctx->camera, *ctx->tracer,
        ctx->disk.get(), ctx->sphere.get(),
        ctx->spectrum.get(), *ctx->tonemapper, spp);

    const char* metric_name = params->metric_type == GRRT_METRIC_KERR ? "kerr" : "schwarzschild";
    const char* backend_name = "cpu";

#ifdef GRRT_HAS_CUDA
    if (params->backend == GRRT_BACKEND_CUDA) {
        if (!cuda_available()) {
            g_last_error = "CUDA backend requested but no CUDA device available";
            delete ctx;
            return nullptr;
        }
        ctx->cuda_ctx = cuda_context_create(params);
        if (!ctx->cuda_ctx) {
            g_last_error = "Failed to create CUDA render context";
            delete ctx;
            return nullptr;
        }
        backend_name = "cuda";
    }
#endif

    std::println("grrt: created context ({}x{}, {}, M={}, a={}, r_obs={}, disk={}, stars={}, backend={})",
                 params->width, params->height, metric_name, mass, spin_a, observer_r,
                 params->disk_enabled ? "on" : "off",
                 params->background_type == GRRT_BG_STARS ? "on" : "off",
                 backend_name);
    return ctx;
}

void grrt_destroy(GRRTContext* ctx) {
#ifdef GRRT_HAS_CUDA
    if (ctx && ctx->cuda_ctx) {
        cuda_context_destroy(ctx->cuda_ctx);
        ctx->cuda_ctx = nullptr;
    }
#endif
    delete ctx;
}

void grrt_update_params(GRRTContext* ctx, const GRRTParams* params) {
    ctx->params = *params;
}

int grrt_render(GRRTContext* ctx, float* framebuffer) {
#ifdef GRRT_HAS_CUDA
    if (ctx->cuda_ctx) {
        int result = cuda_render(ctx->cuda_ctx, &ctx->params, framebuffer);
        if (result != 0) {
            ctx->error_msg = "CUDA render failed";
            return result;
        }
        std::println("grrt: rendered {}x{} frame (cuda)", ctx->params.width, ctx->params.height);
        return 0;
    }
#endif
    ctx->renderer->render(framebuffer, ctx->params.width, ctx->params.height);
    std::println("grrt: rendered {}x{} frame (cpu)", ctx->params.width, ctx->params.height);
    return 0;
}

int grrt_render_tile(GRRTContext* /*ctx*/, float* /*buffer*/,
                     int /*x*/, int /*y*/, int /*tile_width*/, int /*tile_height*/) {
    return 0;
}

void grrt_cancel(GRRTContext* ctx) {
#ifdef GRRT_HAS_CUDA
    if (ctx->cuda_ctx) {
        cuda_cancel(ctx->cuda_ctx);
        return;
    }
#endif
}

float grrt_progress(const GRRTContext* /*ctx*/) {
    return 1.0f;
}

const char* grrt_error(const GRRTContext* ctx) {
    return ctx->error_msg.empty() ? nullptr : ctx->error_msg.c_str();
}

int grrt_cuda_available(void) {
#ifdef GRRT_HAS_CUDA
    return cuda_available() ? 1 : 0;
#else
    return 0;
#endif
}

const char* grrt_last_error(void) {
    return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

void grrt_tonemap(float* framebuffer, int width, int height) {
    grrt::ToneMapper tonemapper;
    tonemapper.apply_all(framebuffer, width, height);
}
