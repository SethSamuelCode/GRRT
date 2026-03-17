#ifndef GRRT_RENDERER_H
#define GRRT_RENDERER_H

#include "grrt/camera/camera.h"
#include "grrt/geodesic/geodesic_tracer.h"

namespace grrt {

class Renderer {
public:
    Renderer(const Camera& camera, const GeodesicTracer& tracer);

    // Render full frame into RGBA float buffer (width * height * 4 floats)
    void render(float* framebuffer, int width, int height) const;

private:
    const Camera& camera_;
    const GeodesicTracer& tracer_;
};

} // namespace grrt

#endif
