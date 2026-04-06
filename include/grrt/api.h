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

// Trace a single pixel and print per-step diagnostics to stdout.
// px, py are integer pixel coordinates (0-based, origin top-left).
GRRT_EXPORT void grrt_debug_pixel(GRRTContext* ctx, int px, int py);

// Spectral rendering
GRRT_EXPORT void grrt_set_frequency_bins(GRRTContext* ctx,
                                          const double* frequencies_hz,
                                          int num_bins);
GRRT_EXPORT int grrt_render_spectral(GRRTContext* ctx, double* spectral_buffer,
                                      int width, int height);
GRRT_EXPORT int grrt_render_spectral_cb(GRRTContext* ctx, double* spectral_buffer,
                                         int width, int height,
                                         grrt_progress_fn progress, void* user_data);

// Streaming spectral render — renders and writes directly to a FITS file
// one row at a time.  Never allocates the full cube in RAM.
// Returns 0 on success, -1 on error (call grrt_error() for details).
GRRT_EXPORT int grrt_render_spectral_to_fits_cb(GRRTContext* ctx,
                                                 const char* output_path,
                                                 int width, int height,
                                                 grrt_progress_fn progress,
                                                 void* user_data);

#ifdef __cplusplus
}
#endif

#endif
