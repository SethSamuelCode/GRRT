#ifndef GRRT_RENDERER_H
#define GRRT_RENDERER_H

#include "grrt/camera/camera.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/render/tonemapper.h"
#include <functional>

namespace grrt {

class AccretionDisk;
class CelestialSphere;
class SpectrumLUT;

/// Progress callback: receives fraction in [0, 1].
using ProgressCallback = std::function<void(float)>;

class Renderer {
public:
    Renderer(const Camera& camera, const GeodesicTracer& tracer,
             const AccretionDisk* disk, const CelestialSphere* sphere,
             const SpectrumLUT* spectrum, const ToneMapper& tonemapper,
             int samples_per_pixel = 1);

    void render(float* framebuffer, int width, int height,
                ProgressCallback progress_cb = nullptr) const;

private:
    const Camera& camera_;
    const GeodesicTracer& tracer_;
    const AccretionDisk* disk_;
    const CelestialSphere* sphere_;
    const SpectrumLUT* spectrum_;
    const ToneMapper& tonemapper_;
    int spp_;
};

} // namespace grrt

#endif
