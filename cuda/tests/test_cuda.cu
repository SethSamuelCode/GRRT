#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_math.h"

// ---------------------------------------------------------------------------
// Device kernel: runs 7 independent math sub-tests, writing pass/fail (1/0)
// into results[0..6].
// ---------------------------------------------------------------------------
__global__ void test_math_kernel(int* results) {
    // --- results[0]: Vec4 addition ---
    {
        cuda::Vec4 a{1.0, 2.0, 3.0, 4.0};
        cuda::Vec4 b{0.5, 1.5, 2.5, 3.5};
        cuda::Vec4 c = a + b;
        bool ok = (fabs(c[0] - 1.5) < 1e-14) &&
                  (fabs(c[1] - 3.5) < 1e-14) &&
                  (fabs(c[2] - 5.5) < 1e-14) &&
                  (fabs(c[3] - 7.5) < 1e-14);
        results[0] = ok ? 1 : 0;
    }

    // --- results[1]: Vec4 scalar multiply ---
    {
        cuda::Vec4 a{1.0, 2.0, 3.0, 4.0};
        cuda::Vec4 c = a * 2.0;
        bool ok = (fabs(c[0] - 2.0) < 1e-14) &&
                  (fabs(c[3] - 8.0) < 1e-14);
        results[1] = ok ? 1 : 0;
    }

    // --- results[2]: Matrix4 diagonal construction, check diagonal + off-diagonal ---
    {
        cuda::Matrix4 mat = cuda::Matrix4::diagonal(-0.5, 2.0, 1.0, 0.25);
        bool diag_ok = (fabs(mat.m[0][0] - (-0.5)) < 1e-14) &&
                       (fabs(mat.m[1][1] - 2.0)    < 1e-14) &&
                       (fabs(mat.m[2][2] - 1.0)    < 1e-14) &&
                       (fabs(mat.m[3][3] - 0.25)   < 1e-14);
        // All off-diagonal entries must be zero
        bool off_ok = (fabs(mat.m[0][1]) < 1e-14) &&
                      (fabs(mat.m[0][3]) < 1e-14) &&
                      (fabs(mat.m[1][2]) < 1e-14) &&
                      (fabs(mat.m[2][3]) < 1e-14);
        results[2] = (diag_ok && off_ok) ? 1 : 0;
    }

    // --- results[3]: Matrix4 contract with v={1,1,1,1} on diagonal matrix ---
    // Expected result: (-0.5, 2.0, 1.0, 0.25) -- same as diagonal entries.
    {
        cuda::Matrix4 mat = cuda::Matrix4::diagonal(-0.5, 2.0, 1.0, 0.25);
        cuda::Vec4 v{1.0, 1.0, 1.0, 1.0};
        cuda::Vec4 r = mat.contract(v);
        bool ok = (fabs(r[0] - (-0.5)) < 1e-14) &&
                  (fabs(r[1] - 2.0)    < 1e-14) &&
                  (fabs(r[2] - 1.0)    < 1e-14) &&
                  (fabs(r[3] - 0.25)   < 1e-14);
        results[3] = ok ? 1 : 0;
    }

    // --- results[4]: Matrix4 inverse_diagonal ---
    // diagonal(-0.5, 2.0, 1.0, 0.25) -> inverse diagonal(-2.0, 0.5, 1.0, 4.0)
    {
        cuda::Matrix4 mat = cuda::Matrix4::diagonal(-0.5, 2.0, 1.0, 0.25);
        cuda::Matrix4 inv = mat.inverse_diagonal();
        bool ok = (fabs(inv.m[0][0] - (-2.0)) < 1e-14) &&
                  (fabs(inv.m[1][1] - 0.5)    < 1e-14) &&
                  (fabs(inv.m[2][2] - 1.0)    < 1e-14) &&
                  (fabs(inv.m[3][3] - 4.0)    < 1e-14);
        results[4] = ok ? 1 : 0;
    }

    // --- results[5]: Vec3 addition ---
    {
        cuda::Vec3 a{1.0, 2.0, 3.0};
        cuda::Vec3 b{0.5, 1.5, 2.5};
        cuda::Vec3 c = a + b;
        bool ok = (fabs(c[0] - 1.5) < 1e-14) &&
                  (fabs(c[1] - 3.5) < 1e-14) &&
                  (fabs(c[2] - 5.5) < 1e-14);
        results[5] = ok ? 1 : 0;
    }

    // --- results[6]: Matrix4 full inverse (block-diagonal Kerr-like matrix) ---
    // Input matrix (block-diagonal BL structure):
    //   m[0][0]=-1, m[0][3]=0.5, m[3][0]=0.5, m[3][3]=2.0
    //   m[1][1]=3.0, m[2][2]=4.0
    // det of (t,phi) block = (-1)(2) - (0.5)(0.5) = -2 - 0.25 = -2.25
    // inv[0][0] = 2.0 / -2.25
    // Verify M * M^-1 = I by checking [0][0] entry of product:
    //   row0 of M dotted with col0 of M^-1 should = 1.
    {
        cuda::Matrix4 mat{};
        mat.m[0][0] = -1.0;  mat.m[0][3] =  0.5;
        mat.m[3][0] =  0.5;  mat.m[3][3] =  2.0;
        mat.m[1][1] =  3.0;
        mat.m[2][2] =  4.0;

        cuda::Matrix4 inv = mat.inverse();

        // Compute (M * M^-1)[0][0]: only non-zero entries in row 0 of M are
        // m[0][0] and m[0][3], and only non-zero entries in col 0 of inv are
        // inv[0][0] and inv[3][0].
        double product_00 = mat.m[0][0] * inv.m[0][0] +
                            mat.m[0][3] * inv.m[3][0];
        // Also verify (M * M^-1)[3][3] = 1
        double product_33 = mat.m[3][0] * inv.m[0][3] +
                            mat.m[3][3] * inv.m[3][3];
        bool ok = (fabs(product_00 - 1.0) < 1e-12) &&
                  (fabs(product_33 - 1.0) < 1e-12);
        results[6] = ok ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// Host function: allocates device memory, launches kernel, reports results.
// ---------------------------------------------------------------------------
void run_math_tests() {
    const int NUM_TESTS = 7;
    const char* test_names[NUM_TESTS] = {
        "Vec4 addition",
        "Vec4 scalar multiply",
        "Matrix4 diagonal construction",
        "Matrix4 contract with {1,1,1,1}",
        "Matrix4 inverse_diagonal",
        "Vec3 addition",
        "Matrix4 full block-diagonal inverse"
    };

    int* d_results = nullptr;
    cudaMalloc(&d_results, NUM_TESTS * sizeof(int));
    cudaMemset(d_results, 0, NUM_TESTS * sizeof(int));

    test_math_kernel<<<1, 1>>>(d_results);

    int h_results[NUM_TESTS] = {};
    cudaMemcpy(h_results, d_results, NUM_TESTS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
    }

    int passed = 0;
    for (int i = 0; i < NUM_TESTS; ++i) {
        std::printf("[%s] %s\n", h_results[i] ? "PASS" : "FAIL", test_names[i]);
        passed += h_results[i];
    }
    std::printf("Math tests: %d/%d passed\n", passed, NUM_TESTS);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    std::printf("CUDA devices found: %d\n", count);
    if (count > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::printf("Device 0: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
        std::printf("Global memory: %.0f MB\n", prop.totalGlobalMem / 1048576.0);

        run_math_tests();
    }
    return (count > 0) ? 0 : 1;
}
