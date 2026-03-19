#ifndef CUDA_BACKEND_H
#define CUDA_BACKEND_H

#include "grrt/types.h"
#include <cuda_runtime.h>

struct CudaRenderContext {
    float4* d_output = nullptr;
    int* d_cancel_flag = nullptr;  // Mapped pinned memory (device pointer)
    int* h_cancel_flag = nullptr;  // Host-accessible mapped pointer
    int width = 0;
    int height = 0;
};

bool cuda_available();
CudaRenderContext* cuda_context_create(const GRRTParams* params);
void cuda_context_destroy(CudaRenderContext* ctx);
int cuda_render(CudaRenderContext* ctx, const GRRTParams* params, float* framebuffer);
int cuda_render_tile(CudaRenderContext* ctx, const GRRTParams* params, float* buffer,
                     int x, int y, int tile_w, int tile_h);
void cuda_cancel(CudaRenderContext* ctx);

#endif
