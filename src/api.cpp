#include "grrt/api.h"
#include "grrt/spacetime/schwarzschild.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/camera/camera.h"
#include "grrt/render/renderer.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/celestial_sphere.h"
#include "grrt/color/spectrum.h"
#include "grrt/render/tonemapper.h"
#include <memory>
#include <print>

struct GRRTContext {
    GRRTParams params;
    std::unique_ptr<grrt::Metric> metric;
    std::unique_ptr<grrt::Integrator> integrator;
    std::unique_ptr<grrt::GeodesicTracer> tracer;
    std::unique_ptr<grrt::Camera> camera;
    std::unique_ptr<grrt::AccretionDisk> disk;
    std::unique_ptr<grrt::CelestialSphere> sphere;
    std::unique_ptr<grrt::SpectrumLUT> spectrum;
    std::unique_ptr<grrt::ToneMapper> tonemapper;
    std::unique_ptr<grrt::Renderer> renderer;
};

GRRTContext* grrt_create(const GRRTParams* params) {
    auto* ctx = new GRRTContext{};
    ctx->params = *params;

    double mass = params->mass > 0.0 ? params->mass : 1.0;
    double observer_r = params->observer_r > 0.0 ? params->observer_r : 50.0;
    double observer_theta = params->observer_theta > 0.0 ? params->observer_theta : 1.396;
    double fov = params->fov > 0.0 ? params->fov : 1.047;
    int max_steps = params->integrator_max_steps > 0 ? params->integrator_max_steps : 10000;

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
    ctx->tracer = std::make_unique<grrt::GeodesicTracer>(
        *ctx->metric, *ctx->integrator, observer_r, max_steps);
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
    }

    // Background (optional)
    if (params->background_type == GRRT_BG_STARS) {
        ctx->sphere = std::make_unique<grrt::CelestialSphere>();
    }

    // Renderer
    ctx->renderer = std::make_unique<grrt::Renderer>(
        *ctx->camera, *ctx->tracer,
        ctx->disk.get(), ctx->sphere.get(),
        ctx->spectrum.get(), *ctx->tonemapper);

    const char* metric_name = params->metric_type == GRRT_METRIC_KERR ? "kerr" : "schwarzschild";
    std::println("grrt: created context ({}x{}, {}, M={}, a={}, r_obs={}, disk={}, stars={})",
                 params->width, params->height, metric_name, mass, spin_a, observer_r,
                 params->disk_enabled ? "on" : "off",
                 params->background_type == GRRT_BG_STARS ? "on" : "off");
    return ctx;
}

void grrt_destroy(GRRTContext* ctx) {
    delete ctx;
}

void grrt_update_params(GRRTContext* ctx, const GRRTParams* params) {
    ctx->params = *params;
}

int grrt_render(GRRTContext* ctx, float* framebuffer) {
    ctx->renderer->render(framebuffer, ctx->params.width, ctx->params.height);
    std::println("grrt: rendered {}x{} frame", ctx->params.width, ctx->params.height);
    return 0;
}

int grrt_render_tile(GRRTContext* /*ctx*/, float* /*buffer*/,
                     int /*x*/, int /*y*/, int /*tile_width*/, int /*tile_height*/) {
    return 0;
}

void grrt_cancel(GRRTContext* /*ctx*/) {}

float grrt_progress(const GRRTContext* /*ctx*/) {
    return 1.0f;
}

const char* grrt_error(const GRRTContext* /*ctx*/) {
    return nullptr;
}

int grrt_cuda_available(void) {
    return 0;
}
