#ifndef GRRT_API_H
#define GRRT_API_H

#include "grrt_export.h"
#include "grrt/types.h"

#ifdef __cplusplus
extern "C" {
#endif

GRRT_EXPORT GRRTContext* grrt_create(const GRRTParams* params);
GRRT_EXPORT void grrt_destroy(GRRTContext* ctx);
GRRT_EXPORT void grrt_update_params(GRRTContext* ctx, const GRRTParams* params);
GRRT_EXPORT int grrt_render(GRRTContext* ctx, float* framebuffer);
GRRT_EXPORT int grrt_render_tile(GRRTContext* ctx, float* buffer,
                                  int x, int y, int tile_width, int tile_height);
GRRT_EXPORT void grrt_cancel(GRRTContext* ctx);
GRRT_EXPORT float grrt_progress(const GRRTContext* ctx);
GRRT_EXPORT const char* grrt_error(const GRRTContext* ctx);
GRRT_EXPORT int grrt_cuda_available(void);

#ifdef __cplusplus
}
#endif

#endif
