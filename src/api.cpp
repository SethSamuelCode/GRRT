#include "grrt/api.h"
#include <cstring>
#include <print>
#include <vector>

struct GRRTContext {
    GRRTParams params;
    std::vector<float> framebuffer;
};

GRRTContext* grrt_create(const GRRTParams* params) {
    auto* ctx = new GRRTContext{};
    ctx->params = *params;
    ctx->framebuffer.resize(params->width * params->height * 4);
    std::println("grrt: created context ({}x{}, metric={})",
                 params->width, params->height,
                 params->metric_type == GRRT_METRIC_KERR ? "kerr" : "schwarzschild");
    return ctx;
}

void grrt_destroy(GRRTContext* ctx) {
    delete ctx;
}

void grrt_update_params(GRRTContext* ctx, const GRRTParams* params) {
    ctx->params = *params;
}

int grrt_render(GRRTContext* ctx, float* framebuffer) {
    const int w = ctx->params.width;
    const int h = ctx->params.height;

    // Placeholder: render a gradient to prove the pipeline works
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const int idx = (j * w + i) * 4;
            framebuffer[idx + 0] = static_cast<float>(i) / static_cast<float>(w);  // R
            framebuffer[idx + 1] = static_cast<float>(j) / static_cast<float>(h);  // G
            framebuffer[idx + 2] = 0.2f;                                            // B
            framebuffer[idx + 3] = 1.0f;                                            // A
        }
    }

    std::println("grrt: rendered {}x{} frame (placeholder gradient)", w, h);
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
