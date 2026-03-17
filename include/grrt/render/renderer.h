#ifndef GRRT_RENDERER_H
#define GRRT_RENDERER_H

#include "grrt/camera/camera.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/render/tonemapper.h"

namespace grrt {

class AccretionDisk;
class CelestialSphere;
class SpectrumLUT;

class Renderer {
public:
    Renderer(const Camera& camera, const GeodesicTracer& tracer,
             const AccretionDisk* disk, const CelestialSphere* sphere,
             const SpectrumLUT* spectrum, const ToneMapper& tonemapper);

    void render(float* framebuffer, int width, int height) const;

private:
    const Camera& camera_;
    const GeodesicTracer& tracer_;
    const AccretionDisk* disk_;
    const CelestialSphere* sphere_;
    const SpectrumLUT* spectrum_;
    const ToneMapper& tonemapper_;
};

} // namespace grrt

#endif
