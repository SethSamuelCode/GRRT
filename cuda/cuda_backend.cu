#include "cuda_backend.h"
#include <cuda_runtime.h>

bool cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}
