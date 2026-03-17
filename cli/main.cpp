#include "grrt/api.h"
#include <print>
#include <vector>

int main() {
    GRRTParams params{};
    params.width = 256;
    params.height = 256;
    params.metric_type = GRRT_METRIC_KERR;
    params.mass = 1.0;
    params.spin = 0.998;
    params.observer_r = 50.0;
    params.observer_theta = 1.396; // ~80 degrees
    params.observer_phi = 0.0;
    params.fov = 1.047; // ~60 degrees

    std::println("gr-raytracer CLI");
    std::println("================");

    GRRTContext* ctx = grrt_create(&params);
    if (!ctx) {
        std::println(stderr, "Failed to create render context");
        return 1;
    }

    std::vector<float> framebuffer(params.width * params.height * 4);
    int result = grrt_render(ctx, framebuffer.data());

    if (result == 0) {
        std::println("Render complete. ({} pixels)", params.width * params.height);
        // TODO: save to PNG via stb_image_write
    } else {
        std::println(stderr, "Render failed: {}", grrt_error(ctx));
    }

    grrt_destroy(ctx);
    return result;
}
