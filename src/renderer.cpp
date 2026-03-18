#include "grrt/render/renderer.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/celestial_sphere.h"
#include "grrt/color/spectrum.h"

namespace grrt {

Renderer::Renderer(const Camera& camera, const GeodesicTracer& tracer,
                   const AccretionDisk* disk, const CelestialSphere* sphere,
                   const SpectrumLUT* spectrum, const ToneMapper& tonemapper)
    : camera_(camera), tracer_(tracer), disk_(disk), sphere_(sphere),
      spectrum_(spectrum), tonemapper_(tonemapper) {}

void Renderer::render(float* framebuffer, int width, int height) const {
    // Render to linear HDR — caller decides whether to tone map
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            GeodesicState state = camera_.ray_for_pixel(i, j);
            TraceResult result = tracer_.trace(state, disk_, spectrum_);

            Vec3 color = result.accumulated_color;

            if (result.termination == RayTermination::Escaped && sphere_) {
                color += sphere_->sample(result.final_position);
            }

            const int idx = (j * width + i) * 4;
            framebuffer[idx + 0] = static_cast<float>(color[0]);
            framebuffer[idx + 1] = static_cast<float>(color[1]);
            framebuffer[idx + 2] = static_cast<float>(color[2]);
            framebuffer[idx + 3] = 1.0f;
        }
    }
}

} // namespace grrt
