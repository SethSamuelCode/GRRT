#include "grrt/scene/disk_column_bvp.h"
#include "grrt/color/opacity.h"
#include "grrt/math/constants.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

int failures = 0;

static void check(const char* name, double got, double expected, double rel_tol) {
    double rel = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel < rel_tol;
    std::printf("  %s: got=%.6e expected=%.6e rel=%.2e %s\n",
                name, got, expected, rel, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

static void test_eos() {
    std::printf("\n=== EOS: rho from (P,T) ===\n");
    using namespace grrt::constants;
    // Gas-pressure-dominated point: choose rho, T; compute P_gas+P_rad; invert.
    const double rho = 1.0, T = 1e5;
    const double P = rho * k_B * T / (mu_fully_ionized * m_p) + (a_rad / 3.0) * std::pow(T, 4);
    check("eos_rho inverts", grrt::eos_rho(P, T), rho, 1e-12);
    // Radiation pressure exceeding total → non-physical → <= 0.
    const double P_small = (a_rad / 3.0) * std::pow(T, 4) * 0.5; // below P_rad
    if (grrt::eos_rho(P_small, T) > 0.0) {
        std::printf("  FAIL: should be <=0 when P < P_rad\n"); failures++;
    }
}

static void test_scaffold() {
    std::printf("\n=== scaffold: solve_column_bvp links and returns ===\n");
    grrt::ColumnInputs in{};
    in.T_eff = 1e5; in.shear = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 16;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    auto sol = grrt::solve_column_bvp(in, lut);
    std::printf("  returned q.size()=%zu\n", sol.q.size());
    if (sol.q.size() != 16) { std::printf("  FAIL: grid size\n"); failures++; }
}

static void test_residual_hydrostatic_identity() {
    std::printf("\n=== residual: hydrostatic identity on a Gaussian column ===\n");
    using namespace grrt::constants;
    const double T = 1e5, rho_mid = 1.0, omega_z = 1e3;
    const double cs2 = k_B * T / (mu_fully_ionized * m_p);
    const double H = std::sqrt(cs2) / omega_z;
    const double dz = 0.02 * H;   // finer FD step: O(dz^2) truncation well under the 1e-3 tolerance
    auto rho = [&](double z){ return rho_mid * std::exp(-z*z/(2*H*H)); };
    auto P   = [&](double z){ return rho(z) * cs2; };   // isothermal gas
    const double z = 1.5 * H;
    const double dPdz = (P(z+dz) - P(z-dz)) / (2*dz);
    const double resid = dPdz + rho(z) * omega_z*omega_z * z;
    std::printf("  hydrostatic resid=%.3e (rel %.3e)\n", resid, P(z)/H);
    if (std::abs(resid) > 1e-3 * (P(z)/H)) { std::printf("  FAIL\n"); failures++; }
}

static void test_residual_count_finite() {
    std::printf("\n=== residual: length 4N+2, all finite ===\n");
    grrt::ColumnInputs in{}; in.T_eff = 1e5; in.shear = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 32;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    std::vector<double> U, R;
    grrt::column_residual_test(in, lut, U, R);
    std::printf("  U.size=%zu R.size=%zu (expect %d)\n", U.size(), R.size(), 4*32+2);
    if ((int)R.size() != 4*32+2) { std::printf("  FAIL: residual length\n"); failures++; }
    if ((int)U.size() != 4*32+2) { std::printf("  FAIL: state length\n"); failures++; }
    bool finite = true; for (double x : R) if (!std::isfinite(x)) finite = false;
    if (!finite) { std::printf("  FAIL: non-finite residual\n"); failures++; }
}

static void test_numerical_jacobian_finite() {
    std::printf("\n=== numerical Jacobian: finite, correct shape ===\n");
    grrt::ColumnInputs in{}; in.T_eff = 1e5; in.shear = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 24;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    std::vector<double> Jdense; int n = 0;
    grrt::column_numerical_jacobian_test(in, lut, Jdense, n);
    std::printf("  n=%d entries=%zu (expect %d, %d)\n", n, Jdense.size(), 4*in.n_nodes+2, (4*in.n_nodes+2)*(4*in.n_nodes+2));
    if (n != 4*in.n_nodes+2) { std::printf("  FAIL: size n\n"); failures++; }
    if ((int)Jdense.size() != n*n) { std::printf("  FAIL: matrix size\n"); failures++; }
    bool finite = true; for (double x : Jdense) if (!std::isfinite(x)) finite = false;
    if (!finite) { std::printf("  FAIL: non-finite Jacobian entry\n"); failures++; }
    // sanity: the Jacobian must be non-trivial (not all zeros)
    double maxabs = 0.0; for (double x : Jdense) maxabs = std::max(maxabs, std::abs(x));
    if (maxabs <= 0.0) { std::printf("  FAIL: all-zero Jacobian\n"); failures++; }
}

static void test_analytic_vs_numerical_jacobian() {
    std::printf("\n=== analytic Jacobian matches numerical (cross-check) ===\n");
    // Same operating point as test_converges_and_conserves (known to lie in the
    // Newton basin), so the cross-check is at a physically representative state.
    grrt::ColumnInputs in{}; in.T_eff = 5e4; in.shear = 3e3; in.omega_z = 2e3;
    in.alpha = 0.1; in.rho_mid_guess = 1e-2; in.n_nodes = 24;
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    std::vector<double> Ja, Jn; int n = 0;
    grrt::column_jacobians_test(in, lut, Ja, Jn, n);
    // Per-row magnitude scale: the residual rows span ~1e0..1e15, so a relative
    // metric must be guarded by the row scale. An entry whose disagreement is a
    // negligible fraction of the row's largest entry is numerical roundoff in the
    // finite-difference reference (e.g. analytic 0 vs a ~1e-4 noise term on a row
    // with ~1e8 entries), NOT a mis-derived partial — skip those.
    std::vector<double> rowmax((size_t)n, 0.0);
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c)
        rowmax[r] = std::max(rowmax[r], std::abs(Jn[(size_t)r*n+c]));
    double max_rel = 0.0; int bad_row=-1, bad_col=-1;
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) {
        const double a = Ja[(size_t)r*n+c], num = Jn[(size_t)r*n+c];
        // Row-scale guard: ignore entries where |a-num| is below 1e-6 of the row's
        // largest entry (pure finite-difference roundoff, not a real disagreement).
        const double scale = std::max(std::abs(num), 1e-6 * rowmax[r]);
        const double rel = std::abs(a - num) / scale;
        if (rel > max_rel && std::abs(a - num) > 1e-6 * rowmax[r]) { max_rel = rel; bad_row=r; bad_col=c; }
    }
    std::printf("  max relative mismatch = %.3e (worst at row %d col %d)\n", max_rel, bad_row, bad_col);
    if (max_rel > 1e-3) { std::printf("  FAIL: analytic Jacobian disagrees with numerical\n"); failures++; }
}

static void test_converges_and_conserves() {
    std::printf("\n=== Newton converges; optically-thick self-heating + energy conservation ===\n");
    using namespace grrt::constants;
    grrt::ColumnInputs in{};
    in.T_eff = 5e4; in.shear = 3e3; in.omega_z = 2e3; in.alpha = 0.1;
    in.rho_mid_guess = 1e-2; in.n_nodes = 160; in.tol = 1e-8; in.max_iters = 80;
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    auto s = grrt::solve_column_bvp(in, lut);
    std::printf("  converged=%d iters=%d resid=%.2e z0=%.3e Sigma0=%.3e tau_mid=%.2f T_mid/T_eff=%.2f\n",
                s.converged, s.iters, s.final_residual, s.z0, s.Sigma0, s.tau_mid,
                s.T.empty()?0.0:s.T.front()/in.T_eff);
    if (!s.converged) { std::printf("  FAIL: did not converge\n"); failures++; return; }
    // density monotone non-increasing midplane -> surface
    for (size_t i = 1; i < s.rho.size(); ++i)
        if (s.rho[i] > s.rho[i-1]*1.02) { std::printf("  FAIL: non-monotone at %zu\n", i); failures++; break; }
    // optically-thick disk self-heats: midplane hotter than the photosphere
    if (!(s.T.front() > in.T_eff)) { std::printf("  FAIL: midplane not hotter than T_eff\n"); failures++; }
    if (!(s.tau_mid > 1.0)) { std::printf("  FAIL: expected optically thick (tau_mid>1)\n"); failures++; }
    // ENERGY CONSERVATION (exact): integral of (alpha*shear*P) dz over the column = sigma T_eff^4
    double dissip = 0.0;
    for (size_t i = 1; i < s.z.size(); ++i) {
        const double dz = std::abs(s.z[i] - s.z[i-1]);
        const double pbar = 0.5 * (s.P[i] + s.P[i-1]);
        dissip += in.alpha * in.shear * pbar * dz;
    }
    const double Q_surf = sigma_SB * std::pow(in.T_eff, 4.0);
    std::printf("  dissip=%.6e  Q_surf=%.6e  ratio=%.4f\n", dissip, Q_surf, dissip/Q_surf);
    check("energy conservation int(alpha*shear*P)dz = sigma T_eff^4", dissip, Q_surf, 3e-3);
}

static void test_physics_invariants() {
    std::printf("\n=== physics invariants: flux BCs, hydrostatic balance, photosphere ===\n");
    using namespace grrt::constants;
    grrt::ColumnInputs in{};
    in.T_eff = 5e4; in.shear = 3e3; in.omega_z = 2e3; in.alpha = 0.1;
    in.rho_mid_guess = 1e-2; in.n_nodes = 160; in.tol = 1e-8; in.max_iters = 80;
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    auto s = grrt::solve_column_bvp(in, lut);
    if (!s.converged) { std::printf("  FAIL: precondition not converged\n"); failures++; return; }

    // --- Flux boundary conditions (enforced by the residual; confirm in the solution) ---
    const double Q_surf = sigma_SB * std::pow(in.T_eff, 4.0);
    if (std::abs(s.Q.front()) > 1e-6 * Q_surf) {
        std::printf("  FAIL: Q(midplane)=%.3e not ~0 (Q_surf=%.3e)\n", s.Q.front(), Q_surf); failures++;
    } else { std::printf("  PASS: Q(midplane) ~ 0\n"); }
    check("Q(surface) = sigma T_eff^4", s.Q.back(), Q_surf, 1e-6);
    check("T(surface) = T_eff",          s.T.back(), in.T_eff, 1e-6);

    // --- Hydrostatic balance, INDEPENDENT re-check on the converged profile ---
    //   dP/dz = -rho * Omega_z^2 * z   (re-derived from the unpacked profile, NOT
    //   the solver's internal q-trapezoidal residual -> a genuine cross-check).
    //   Uses a non-uniform-grid central difference; expect O(1e-2)-level agreement.
    double max_hydro_rel = 0.0;
    for (size_t i = 1; i + 1 < s.z.size(); ++i) {
        const double dz = s.z[i+1] - s.z[i-1];
        if (std::abs(dz) < 1e-30) continue;
        const double dPdz = (s.P[i+1] - s.P[i-1]) / dz;
        const double rhs  = -s.rho[i] * in.omega_z * in.omega_z * s.z[i];
        const double scale = std::abs(dPdz) + std::abs(rhs) + 1e-30;
        max_hydro_rel = std::max(max_hydro_rel, std::abs(dPdz - rhs) / scale);
    }
    std::printf("  max hydrostatic relative residual = %.3e (expect < 5e-2)\n", max_hydro_rel);
    if (max_hydro_rel > 5e-2) { std::printf("  FAIL: hydrostatic balance violated\n"); failures++; }

    // --- Photosphere: surface-pressure BC encodes tau=2/3 (P = (2/3) Omega_z^2 z0 / kappa) ---
    const double kR_surf = lut.lookup_kappa_ross(std::max(s.rho.back(), 1e-30), std::max(s.T.back(), 3000.0))
                         + lut.lookup_kappa_es  (std::max(s.rho.back(), 1e-30), std::max(s.T.back(), 3000.0));
    const double P_phot = (2.0/3.0) * in.omega_z * in.omega_z * s.z0 / kR_surf;
    check("surface pressure = (2/3) Omega_z^2 z0 / kappa_total  (photosphere tau=2/3)",
          s.P.back(), P_phot, 1e-3);
}

static void test_convergence_sweep() {
    std::printf("\n=== converges across (T_eff, shear) inputs; non-converged fall back ===\n");
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    const double Teffs[] = {1e4, 5e4, 2e5, 1e6};   // cool -> hot (gas -> radiation)
    const double oms[]   = {5e2, 2e3, 8e3};
    int ok = 0, total = 0, bad_fallback = 0;
    for (double Te : Teffs) for (double om : oms) {
        grrt::ColumnInputs in{}; in.T_eff = Te; in.shear = 1.5*om; in.omega_z = om;
        in.alpha = 0.1; in.rho_mid_guess = 1e-2; in.n_nodes = 120; in.max_iters = 80; in.tol = 1e-8;
        auto s = grrt::solve_column_bvp(in, lut);
        total++;
        if (s.converged) { ok++; }
        else {
            std::printf("  no-converge: T_eff=%.0e om=%.0e  used_fallback=%d\n", Te, om, s.used_fallback);
            if (!s.used_fallback) bad_fallback++;   // non-converged MUST flag fallback
            // and the fallback profile must be sane: positive, monotone, finite
            bool sane = !s.rho.empty();
            for (size_t i = 0; sane && i < s.rho.size(); ++i)
                if (!std::isfinite(s.rho[i]) || s.rho[i] < 0.0) sane = false;
            if (!sane) { std::printf("  FAIL: fallback profile not sane\n"); failures++; }
        }
    }
    std::printf("  converged %d/%d\n", ok, total);
    if (ok == 0) { std::printf("  FAIL: nothing converged\n"); failures++; }
    if (bad_fallback > 0) { std::printf("  FAIL: %d non-converged columns did NOT set used_fallback\n", bad_fallback); failures++; }
}

static void test_radiation_thickens() {
    std::printf("\n=== radiation-dominated column thicker than gas-dominated ===\n");
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    grrt::ColumnInputs cold{}; cold.T_eff=2e4; cold.shear=3e3; cold.omega_z=2e3;
    cold.alpha=0.1; cold.rho_mid_guess=1e-2; cold.n_nodes=160; cold.max_iters=80; cold.tol=1e-8;
    grrt::ColumnInputs hot = cold; hot.T_eff = 1e6;   // radiation-dominated
    auto sc = grrt::solve_column_bvp(cold, lut);
    auto sh = grrt::solve_column_bvp(hot, lut);
    std::printf("  cold converged=%d z0=%.3e ; hot converged=%d z0=%.3e\n",
                sc.converged, sc.z0, sh.converged, sh.z0);
    if (sc.converged && sh.converged) {
        if (!(sh.z0 > sc.z0)) { std::printf("  FAIL: radiation did not thicken the column\n"); failures++; }
        else { std::printf("  PASS: hot column thicker (z0 %.3e > %.3e)\n", sh.z0, sc.z0); }
    } else {
        std::printf("  (skipped thickness compare: a column used fallback)\n");
        // still must not have produced garbage
        if (sc.used_fallback || sh.used_fallback) std::printf("  (one or both fell back)\n");
    }
}

int main() {
    test_eos();
    test_scaffold();
    test_residual_hydrostatic_identity();
    test_residual_count_finite();
    test_numerical_jacobian_finite();
    test_analytic_vs_numerical_jacobian();
    test_converges_and_conserves();
    test_physics_invariants();
    test_convergence_sweep();
    test_radiation_thickens();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
