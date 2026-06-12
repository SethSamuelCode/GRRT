// ===========================================================================
// SINGLE-ENTRY JACOBIAN FD STEP-SCAN  (diagnostic — DELETABLE)
// ---------------------------------------------------------------------------
// Classifies the post-Qvis-fix test-slim-jacobian marginal failure at
// (a=0.9, f_Edd=0.02, N=20): column Tc[0] (col 3), worst entry row 40
// (first radial-momentum row, sonic L'Hopital).  Central-difference step scan:
// if CD(h) -> analytic as h shrinks (O(h^2)), the analytic entry is exact and
// the test's fixed h=1e-4·Tc[0] (~2600 K) bracket straddles a kink (LUT cell
// edge / clamp); if CD(h) -> a DIFFERENT value, the analytic entry is wrong.
//
// Build:  cmake --build build --config Release --target slim-jacentry-scan
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <numbers>

using namespace grrt;
using namespace grrt::slim_detail;

static SlimDiskInputs make_inputs(double a, double f_Edd, int N) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = a; in.alpha = 0.1; in.r_g = 1.48e6;
    in.r_out = 50.0; in.n_nodes = N; in.max_iters = 100; in.tol = 1e-6;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0 / 3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

int main() {
    const int N = 20;
    const SlimDiskInputs in = make_inputs(0.9, 0.02, N);
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    std::vector<double> U = build_thin_disk_seed(in, op);
    const int n = 4 * N + 2;

    std::vector<double> Ja;
    slim_analytic_jacobian(U, in, op, Ja);

    // Scan a few (row, col) suspects: the failing (40, Tc[0]) plus energy rows
    // touching node 0 via Qvis (row 3N-1 = 59) for completeness.
    const int cols[] = {3, 3};
    const int rows[] = {40, 59};
    for (int k = 0; k < 2; ++k) {
        const int row = rows[k], col = cols[k];
        const double an = Ja[(size_t)row * n + col];
        std::printf("\n[row %d, col %d]  U[col]=%.10e  analytic=%.12e\n",
                    row, col, U[col], an);
        for (double rs : {3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8}) {
            const double h = rs * std::abs(U[col]);
            std::vector<double> Up = U, Um = U, Rp, Rm;
            Up[col] += h; Um[col] -= h;
            slim_radial_residual(Up, in, op, Rp);
            slim_radial_residual(Um, in, op, Rm);
            const double cd = (Rp[row] - Rm[row]) / (2.0 * h);
            const double rel = std::abs(cd - an) / std::max(std::abs(an), 1e-300);
            std::printf("  h_rel=%.1e (h=%.3e K)  CD=%.12e  rel_vs_analytic=%.3e\n",
                        rs, h, cd, rel);
        }
    }
    std::printf("\n[slim-jacentry-scan] done.\n");
    return 0;
}
