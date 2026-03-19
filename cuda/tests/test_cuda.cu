#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_math.h"
#include "cuda_metric.h"
#include "cuda_geodesic.h"

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
// Device kernel: runs 8 metric sub-tests, writing pass/fail (1/0)
// into results[0..7].
//
// Position used throughout: equatorial plane at r=10M, theta=pi/2.
// x = {t=0, r=10, theta=pi/2, phi=0}
// ---------------------------------------------------------------------------
__global__ void test_metric_kernel(int* results) {
    const double M   = 1.0;
    const double a   = 0.998;   // near-extremal Kerr spin
    const double r   = 10.0;
    const cuda::Vec4 x{0.0, r, 1.5707963267948966, 0.0};  // theta = pi/2

    // --- results[0]: Schwarzschild g_tt at r=10M ---
    // Expected: -(1 - 2/10) = -0.8
    {
        cuda::Matrix4 g = cuda::schwarzschild_g_lower(M, x);
        bool ok = (fabs(g.m[0][0] - (-0.8)) < 1e-12);
        results[0] = ok ? 1 : 0;
    }

    // --- results[1]: Schwarzschild g_rr at r=10M ---
    // Expected: 1/(1 - 2/10) = 1.25
    {
        cuda::Matrix4 g = cuda::schwarzschild_g_lower(M, x);
        bool ok = (fabs(g.m[1][1] - 1.25) < 1e-12);
        results[1] = ok ? 1 : 0;
    }

    // --- results[2]: Schwarzschild horizon = 2M (M=1) ---
    {
        double rh = cuda::horizon_radius(cuda::MetricType::Schwarzschild, M, 0.0);
        bool ok = (fabs(rh - 2.0) < 1e-14);
        results[2] = ok ? 1 : 0;
    }

    // --- results[3]: Schwarzschild ISCO = 6M (M=1) ---
    {
        double ri = cuda::isco_radius(cuda::MetricType::Schwarzschild, M, 0.0);
        bool ok = (fabs(ri - 6.0) < 1e-14);
        results[3] = ok ? 1 : 0;
    }

    // --- results[4]: Kerr g_tphi != 0 at r=10M, spin=0.998 (off-diagonal test) ---
    // g_tphi = -2 * M * a * r * sin²θ / Σ  which is nonzero for a > 0
    {
        cuda::Matrix4 g = cuda::kerr_g_lower(M, a, x);
        bool ok = (fabs(g.m[0][3]) > 1e-6);   // must be non-trivially nonzero
        results[4] = ok ? 1 : 0;
    }

    // --- results[5]: Kerr horizon < 2M for spin=0.998 ---
    // r_h = M + sqrt(M² - a²) < 2M since a > 0
    {
        double rh = cuda::horizon_radius(cuda::MetricType::Kerr, M, a);
        bool ok = (rh < 2.0 * M) && (rh > 0.0);
        results[5] = ok ? 1 : 0;
    }

    // --- results[6]: Schwarzschild g * g^-1 ≈ I (check [1][1] element) ---
    {
        cuda::Matrix4 g     = cuda::schwarzschild_g_lower(M, x);
        cuda::Matrix4 g_inv = cuda::schwarzschild_g_upper(M, x);
        // Product [1][1]: only m[1][1] * inv[1][1] is nonzero (diagonal metric)
        double prod = g.m[1][1] * g_inv.m[1][1];
        bool ok = (fabs(prod - 1.0) < 1e-10);
        results[6] = ok ? 1 : 0;
    }

    // --- results[7]: Kerr g * g^-1 ≈ I (check [0][0] element, tolerance 1e-10) ---
    // Row 0 of g has non-zero entries at [0][0] and [0][3].
    // Col 0 of g_inv has non-zero entries at [0][0] and [3][0].
    {
        cuda::Matrix4 g     = cuda::kerr_g_lower(M, a, x);
        cuda::Matrix4 g_inv = cuda::kerr_g_upper(M, a, x);
        double prod_00 = g.m[0][0] * g_inv.m[0][0] + g.m[0][3] * g_inv.m[3][0];
        bool ok = (fabs(prod_00 - 1.0) < 1e-10);
        results[7] = ok ? 1 : 0;
    }
}

// ---------------------------------------------------------------------------
// Host function: allocates device memory, launches metric kernel, reports.
// ---------------------------------------------------------------------------
void run_metric_tests() {
    const int NUM_TESTS = 8;
    const char* test_names[NUM_TESTS] = {
        "Schwarzschild g_tt at r=10M (expected -0.8)",
        "Schwarzschild g_rr at r=10M (expected 1.25)",
        "Schwarzschild horizon = 2M",
        "Schwarzschild ISCO = 6M",
        "Kerr g_tphi != 0 (frame-dragging off-diagonal)",
        "Kerr horizon < 2M for spin=0.998",
        "Schwarzschild g * g^-1 = I ([1][1])",
        "Kerr g * g^-1 = I ([0][0])"
    };

    int* d_results = nullptr;
    cudaMalloc(&d_results, NUM_TESTS * sizeof(int));
    cudaMemset(d_results, 0, NUM_TESTS * sizeof(int));

    test_metric_kernel<<<1, 1>>>(d_results);

    int h_results[NUM_TESTS] = {};
    cudaMemcpy(h_results, d_results, NUM_TESTS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_results);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA kernel error (metric): %s\n", cudaGetErrorString(err));
    }

    int passed = 0;
    for (int i = 0; i < NUM_TESTS; ++i) {
        std::printf("[%s] %s\n", h_results[i] ? "PASS" : "FAIL", test_names[i]);
        passed += h_results[i];
    }
    std::printf("Metric tests: %d/%d passed\n", passed, NUM_TESTS);
}

// ---------------------------------------------------------------------------
// Device kernel: radial infall geodesic in Schwarzschild spacetime.
//
// Setup: M=1, observer at r=20, equatorial plane (theta=pi/2).
// Initial momentum: p_t = -1 (E=1), p_phi=0, p_theta=0.
//   Null condition: g^{tt} p_t^2 + g^{rr} p_r^2 = 0
//   => p_r = -sqrt(-g^tt / g^rr) * |p_t|   (negative = falling inward)
//
// Integrates 100 adaptive steps with tolerance 1e-8, tracking:
//   outputs[0] = max |H|           (Hamiltonian constraint violation)
//   outputs[1] = |E_final - E_0|   (energy conservation error)
//   outputs[2] = final radius r    (should approach 2M = 2)
// ---------------------------------------------------------------------------
__global__ void test_geodesic_kernel(double* outputs) {
    const double M   = 1.0;
    const double a   = 0.0;   // Schwarzschild (no spin)
    const double r0  = 20.0;
    const double pi  = 3.14159265358979323846;

    // Initial position: equatorial plane
    cuda::Vec4 x0{0.0, r0, pi * 0.5, 0.0};

    // Compute initial p_r from null condition: g^tt p_t^2 + g^rr p_r^2 = 0
    cuda::Matrix4 g_inv0 = cuda::metric_upper(cuda::MetricType::Schwarzschild, M, a, x0);
    double g_tt_up = g_inv0.m[0][0];  // negative
    double g_rr_up = g_inv0.m[1][1];  // positive

    double p_t = -1.0;   // E = -p_t = 1
    // |p_r| = sqrt(-g^tt / g^rr) * |p_t|; negative sign = inward
    double p_r = -sqrt(-g_tt_up / g_rr_up) * fabs(p_t);

    cuda::Vec4 p0{p_t, p_r, 0.0, 0.0};

    cuda::GeodesicState state;
    state.position = x0;
    state.momentum = p0;

    double E_initial = -state.momentum[0];  // E = -p_t

    double max_H  = 0.0;
    double dlambda = 1.0;  // initial step size

    // Integrate 100 adaptive steps
    for (int step = 0; step < 100; ++step) {
        // Check Hamiltonian before advancing
        double H = cuda::hamiltonian(cuda::MetricType::Schwarzschild, M, a, state);
        double absH = fabs(H);
        if (absH > max_H) max_H = absH;

        // Stop if we've fallen into the horizon (r < 2.1 * M)
        if (state.position[1] < 2.1 * M) break;

        cuda::AdaptiveResult res = cuda::rk4_adaptive_step(
            cuda::MetricType::Schwarzschild, M, a, state, dlambda, 1e-8);
        state    = res.state;
        dlambda  = res.next_dlambda;
    }

    double E_final = -state.momentum[0];

    outputs[0] = max_H;
    outputs[1] = fabs(E_final - E_initial);
    outputs[2] = state.position[1];  // final radius
}

// ---------------------------------------------------------------------------
// Host function: allocates device memory, launches geodesic kernel, reports.
// ---------------------------------------------------------------------------
void run_geodesic_tests() {
    const int NUM_OUTPUTS = 3;

    double* d_outputs = nullptr;
    cudaMalloc(&d_outputs, NUM_OUTPUTS * sizeof(double));
    cudaMemset(d_outputs, 0, NUM_OUTPUTS * sizeof(double));

    test_geodesic_kernel<<<1, 1>>>(d_outputs);

    double h_outputs[NUM_OUTPUTS] = {};
    cudaMemcpy(h_outputs, d_outputs, NUM_OUTPUTS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_outputs);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("CUDA kernel error (geodesic): %s\n", cudaGetErrorString(err));
    }

    std::printf("\n--- Geodesic Tests (Schwarzschild radial infall from r=20M) ---\n");
    std::printf("  Max |H| (Hamiltonian violation) : %.3e\n", h_outputs[0]);
    std::printf("  |E_final - E_initial|           : %.3e\n", h_outputs[1]);
    std::printf("  Final radius                    : %.6f M\n", h_outputs[2]);

    bool pass_H  = (h_outputs[0] < 1e-8);
    bool pass_E  = (h_outputs[1] < 1e-8);
    bool pass_r  = (h_outputs[2] < 5.0);

    std::printf("[%s] Hamiltonian constraint |H| < 1e-8\n",  pass_H ? "PASS" : "FAIL");
    std::printf("[%s] Energy conservation |dE| < 1e-8\n",    pass_E ? "PASS" : "FAIL");
    std::printf("[%s] Final radius < 5.0 M (approaching horizon)\n", pass_r ? "PASS" : "FAIL");

    int passed = (pass_H ? 1 : 0) + (pass_E ? 1 : 0) + (pass_r ? 1 : 0);
    std::printf("Geodesic tests: %d/3 passed\n", passed);
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
        run_metric_tests();
        run_geodesic_tests();
    }
    return (count > 0) ? 0 : 1;
}
