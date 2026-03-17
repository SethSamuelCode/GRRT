#include "grrt/render/renderer.h"

namespace grrt {

Renderer::Renderer(const Camera& camera, const GeodesicTracer& tracer)
    : camera_(camera), tracer_(tracer) {}

void Renderer::render(float* framebuffer, int width, int height) const {
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            GeodesicState state = camera_.ray_for_pixel(i, j);
            RayTermination result = tracer_.trace(state);

            const int idx = (j * width + i) * 4;
            if (result == RayTermination::Escaped) {
                // White pixel
                framebuffer[idx + 0] = 1.0f;
                framebuffer[idx + 1] = 1.0f;
                framebuffer[idx + 2] = 1.0f;
                framebuffer[idx + 3] = 1.0f;
            } else {
                // Black pixel (horizon hit or max steps)
                framebuffer[idx + 0] = 0.0f;
                framebuffer[idx + 1] = 0.0f;
                framebuffer[idx + 2] = 0.0f;
                framebuffer[idx + 3] = 1.0f;
            }
        }
    }
}

} // namespace grrt
