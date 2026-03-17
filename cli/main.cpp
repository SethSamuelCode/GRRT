#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "grrt/api.h"
#include <print>
#include <vector>

int main() {
    GRRTParams params{};
    params.width = 256;
    params.height = 256;
    params.metric_type = GRRT_METRIC_SCHWARZSCHILD;
    params.mass = 1.0;
    params.spin = 0.998;
    params.observer_r = 50.0;
    params.observer_theta = 1.396; // ~80 degrees
    params.observer_phi = 0.0;
    params.fov = 1.047; // ~60 degrees
    params.integrator_max_steps = 10000;

    const char* output_path = "output.png";

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
        // Convert float RGBA [0,1] to uint8 RGBA [0,255]
        std::vector<unsigned char> pixels(params.width * params.height * 4);
        for (int i = 0; i < params.width * params.height * 4; ++i) {
            float v = framebuffer[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            pixels[i] = static_cast<unsigned char>(v * 255.0f);
        }

        stbi_write_png(output_path, params.width, params.height, 4,
                       pixels.data(), params.width * 4);
        std::println("Saved to {}", output_path);
    } else {
        std::println(stderr, "Render failed: {}", grrt_error(ctx));
    }

    grrt_destroy(ctx);
    return result;
}
