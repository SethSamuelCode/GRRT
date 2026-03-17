#include "grrt/api.h"
#include "grrt/spacetime/schwarzschild.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/camera/camera.h"
#include "grrt/render/renderer.h"
#include <memory>
#include <print>

struct GRRTContext {
    GRRTParams params;
    std::unique_ptr<grrt::Metric> metric;
    std::unique_ptr<grrt::Integrator> integrator;
    std::unique_ptr<grrt::GeodesicTracer> tracer;
    std::unique_ptr<grrt::Camera> camera;
    std::unique_ptr<grrt::Renderer> renderer;
};

GRRTContext* grrt_create(const GRRTParams* params) {
    auto* ctx = new GRRTContext{};
    ctx->params = *params;

    // Defaults for zero-initialized fields
    double mass = params->mass > 0.0 ? params->mass : 1.0;
    double observer_r = params->observer_r > 0.0 ? params->observer_r : 50.0;
    double observer_theta = params->observer_theta > 0.0 ? params->observer_theta : 1.396;
    double fov = params->fov > 0.0 ? params->fov : 1.047;
    int max_steps = params->integrator_max_steps > 0 ? params->integrator_max_steps : 10000;

    // Build pipeline
    ctx->metric = std::make_unique<grrt::Schwarzschild>(mass);
    ctx->integrator = std::make_unique<grrt::RK4>();
    ctx->tracer = std::make_unique<grrt::GeodesicTracer>(
        *ctx->metric, *ctx->integrator, max_steps);
    ctx->camera = std::make_unique<grrt::Camera>(
        *ctx->metric, observer_r, observer_theta, params->observer_phi,
        fov, params->width, params->height);
    ctx->renderer = std::make_unique<grrt::Renderer>(*ctx->camera, *ctx->tracer);

    std::println("grrt: created context ({}x{}, schwarzschild, M={}, r_obs={})",
                 params->width, params->height, mass, observer_r);
    return ctx;
}

void grrt_destroy(GRRTContext* ctx) {
    delete ctx;
}

void grrt_update_params(GRRTContext* ctx, const GRRTParams* params) {
    ctx->params = *params;
    // TODO: rebuild pipeline when params change
}

int grrt_render(GRRTContext* ctx, float* framebuffer) {
    ctx->renderer->render(framebuffer, ctx->params.width, ctx->params.height);
    std::println("grrt: rendered {}x{} frame", ctx->params.width, ctx->params.height);
    return 0;
}

int grrt_render_tile(GRRTContext* /*ctx*/, float* /*buffer*/,
                     int /*x*/, int /*y*/, int /*tile_width*/, int /*tile_height*/) {
    return 0; // Stub
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
