#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "grrt/api.h"
#include <print>
#include <vector>
#include <cmath>

// Draw a circle overlay onto RGBA byte buffer
static void draw_circle(std::vector<unsigned char>& pixels, int width, int height,
                        double center_x, double center_y, double radius,
                        unsigned char r, unsigned char g, unsigned char b,
                        double thickness = 1.5) {
    int r_min = static_cast<int>(std::max(center_y - radius - thickness - 1, 0.0));
    int r_max = static_cast<int>(std::min(center_y + radius + thickness + 1, static_cast<double>(height - 1)));
    int c_min = static_cast<int>(std::max(center_x - radius - thickness - 1, 0.0));
    int c_max = static_cast<int>(std::min(center_x + radius + thickness + 1, static_cast<double>(width - 1)));

    for (int j = r_min; j <= r_max; ++j) {
        for (int i = c_min; i <= c_max; ++i) {
            double dx = i - center_x;
            double dy = j - center_y;
            double dist = std::sqrt(dx * dx + dy * dy);
            if (std::abs(dist - radius) < thickness) {
                int idx = (j * width + i) * 4;
                pixels[idx + 0] = r;
                pixels[idx + 1] = g;
                pixels[idx + 2] = b;
                pixels[idx + 3] = 255;
            }
        }
    }
}

int main() {
    GRRTParams params{};
    params.width = 1024;
    params.height = 1024;
    params.metric_type = GRRT_METRIC_KERR;
    params.mass = 1.0;
    params.spin = 0.998;
    params.observer_r = 50.0;
    params.observer_theta = 1.396; // ~80 degrees
    params.observer_phi = 0.0;
    params.fov = 1.5708; // ~90 degrees
    params.integrator_max_steps = 10000;
    params.disk_enabled = 1;
    params.disk_inner = 0.0;
    params.disk_outer = 20.0;
    params.disk_temperature = 1e7;
    params.background_type = GRRT_BG_STARS;

    bool draw_horizon_circle = false;

    std::println("gr-raytracer CLI");
    std::println("================");

    GRRTContext* ctx = grrt_create(&params);
    if (!ctx) {
        std::println(stderr, "Failed to create render context");
        return 1;
    }

    // Render to linear HDR
    std::vector<float> framebuffer(params.width * params.height * 4);
    int result = grrt_render(ctx, framebuffer.data());

    if (result == 0) {
        // Save linear HDR as Radiance .hdr (before tone mapping)
        // stbi_write_hdr wants RGB (3 channels), so strip alpha
        std::vector<float> hdr_rgb(params.width * params.height * 3);
        for (int i = 0; i < params.width * params.height; ++i) {
            hdr_rgb[i * 3 + 0] = framebuffer[i * 4 + 0];
            hdr_rgb[i * 3 + 1] = framebuffer[i * 4 + 1];
            hdr_rgb[i * 3 + 2] = framebuffer[i * 4 + 2];
        }
        stbi_write_hdr("output.hdr", params.width, params.height, 3, hdr_rgb.data());
        std::println("Saved HDR to output.hdr (raw linear)");

        // Tone map in-place for PNG output
        grrt_tonemap(framebuffer.data(), params.width, params.height);

        // Convert float RGBA [0,1] to uint8 RGBA [0,255]
        std::vector<unsigned char> pixels(params.width * params.height * 4);
        for (int i = 0; i < params.width * params.height * 4; ++i) {
            float v = framebuffer[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            pixels[i] = static_cast<unsigned char>(v * 255.0f);
        }

        if (draw_horizon_circle) {
            double M = params.mass;
            double a = params.spin * M;
            double r_horizon = M + std::sqrt(M * M - a * a);
            double ang_radius = r_horizon / params.observer_r;
            double pixels_per_rad = params.width / params.fov;
            double circle_radius_px = ang_radius * pixels_per_rad;
            double cx = params.width / 2.0;
            double cy = params.height / 2.0;
            draw_circle(pixels, params.width, params.height,
                        cx, cy, circle_radius_px, 255, 50, 50, 1.5);
            std::println("Horizon r_+ = {:.4f}M, {:.1f} px radius", r_horizon, circle_radius_px);
        }

        stbi_write_png("output.png", params.width, params.height, 4,
                       pixels.data(), params.width * 4);
        std::println("Saved PNG to output.png");
    } else {
        std::println(stderr, "Render failed: {}", grrt_error(ctx));
    }

    grrt_destroy(ctx);
    return result;
}
