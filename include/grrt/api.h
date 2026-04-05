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
typedef void (*grrt_progress_fn)(float fraction, void* user_data);

GRRT_EXPORT int grrt_render(GRRTContext* ctx, float* framebuffer);
GRRT_EXPORT int grrt_render_cb(GRRTContext* ctx, float* framebuffer,
                                grrt_progress_fn progress, void* user_data);
GRRT_EXPORT int grrt_render_tile(GRRTContext* ctx, float* buffer,
                                  int x, int y, int tile_width, int tile_height);
GRRT_EXPORT void grrt_cancel(GRRTContext* ctx);
GRRT_EXPORT float grrt_progress(const GRRTContext* ctx);
GRRT_EXPORT const char* grrt_error(const GRRTContext* ctx);
GRRT_EXPORT int grrt_cuda_available(void);
GRRT_EXPORT const char* grrt_last_error(void);

// Apply auto-exposure tone mapping to a linear HDR framebuffer (RGBA float, in-place)
GRRT_EXPORT void grrt_tonemap(float* framebuffer, int width, int height);

// Spectral rendering
GRRT_EXPORT void grrt_set_frequency_bins(GRRTContext* ctx,
                                          const double* frequencies_hz,
                                          int num_bins);
GRRT_EXPORT int grrt_render_spectral(GRRTContext* ctx, double* spectral_buffer,
                                      int width, int height);
GRRT_EXPORT int grrt_render_spectral_cb(GRRTContext* ctx, double* spectral_buffer,
                                         int width, int height,
                                         grrt_progress_fn progress, void* user_data);

#ifdef __cplusplus
}
#endif

#endif
