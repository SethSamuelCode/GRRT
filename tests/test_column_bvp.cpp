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
    std::printf("  returned converged=%d q.size()=%zu\n", sol.converged, sol.q.size());
    // Contract: converged -> full n_nodes profile; non-converged -> empty (no fabricated profile).
    if (sol.converged) {
        if (sol.q.size() != 16) { std::printf("  FAIL: converged but grid size != 16\n"); failures++; }
    } else {
        if (!sol.q.empty()) { std::printf("  FAIL: non-converged but profile not empty\n"); failures++; }
    }
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

static void test_analytic_vs_numerical_jacobian_hot() {
    std::printf("\n=== analytic Jacobian matches numerical at a HOT (rad-pressure) point ===\n");
    // Radiation-pressure-dominated operating point (beta = P_gas/P_tot << 1), the
    // regime that ill-conditioned TOTAL-pressure state used to corrupt. With gas
    // pressure as the state variable the cancellation is gone, so this cross-check
    // is now reliable.
    grrt::ColumnInputs in{}; in.T_eff = 6e6; in.shear = 5800.0; in.omega_z = 3000.0;
    in.alpha = 0.1; in.rho_mid_guess = 1e-6; in.n_nodes = 24;
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    std::vector<double> Ja, Jn; int n = 0;
    grrt::column_jacobians_test(in, lut, Ja, Jn, n);
    std::vector<double> rowmax((size_t)n, 0.0);
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c)
        rowmax[r] = std::max(rowmax[r], std::abs(Jn[(size_t)r*n+c]));
    double max_rel = 0.0; int bad_row=-1, bad_col=-1;
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) {
        const double a = Ja[(size_t)r*n+c], num = Jn[(size_t)r*n+c];
        const double scale = std::max(std::abs(num), 1e-6 * rowmax[r]);
        const double rel = std::abs(a - num) / scale;
        if (rel > max_rel && std::abs(a - num) > 1e-6 * rowmax[r]) { max_rel = rel; bad_row=r; bad_col=c; }
    }
    std::printf("  max relative mismatch = %.3e (worst at row %d col %d)\n", max_rel, bad_row, bad_col);
    if (max_rel > 1e-3) { std::printf("  FAIL: analytic Jacobian disagrees with numerical (hot)\n"); failures++; }
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
    std::printf("\n=== converges across (T_eff, shear); non-converged return empty, never a fake profile ===\n");
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    const double Teffs[] = {1e4, 5e4, 2e5, 1e6};
    const double oms[]   = {5e2, 2e3, 8e3};
    int ok = 0, total = 0;
    for (double Te : Teffs) for (double om : oms) {
        grrt::ColumnInputs in{}; in.T_eff = Te; in.shear = 1.5*om; in.omega_z = om;
        in.alpha = 0.1; in.rho_mid_guess = 1e-2; in.n_nodes = 120; in.max_iters = 80; in.tol = 1e-8;
        auto s = grrt::solve_column_bvp(in, lut);
        total++;
        if (s.converged) {
            ok++;
            // converged solutions carry a full, sane profile
            if ((int)s.rho.size() != in.n_nodes) { std::printf("  FAIL: converged but profile size wrong\n"); failures++; }
        } else {
            // NO fallback: a non-converged solve returns EMPTY vectors, never a fabricated profile.
            if (!s.q.empty() || !s.rho.empty() || !s.T.empty()) {
                std::printf("  FAIL: non-converged solve returned a non-empty profile (T_eff=%.0e om=%.0e)\n", Te, om);
                failures++;
            }
        }
    }
    std::printf("  converged %d/%d (standalone cold-start; Plan 3 warm-starts these)\n", ok, total);
    if (ok == 0) { std::printf("  FAIL: nothing converged\n"); failures++; }
}

static void test_thickness_increases_with_teff() {
    std::printf("\n=== column thickens monotonically with T_eff (pressure supports, no collapse) ===\n");
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    // Two reliably cold-converging columns at fixed rho_mid/shear/omega_z; the hotter
    // one must be thicker (higher total c_s -> larger scale height). NOTE: at rho=1e-2
    // both columns are gas-dominated; this guards against gross collapse / a pressure-
    // structure sign error. Genuine radiation-dominated thickening is covered by Plan 3
    // integration, where the hot inner-disk columns converge via the radial warm-start march.
    grrt::ColumnInputs base{}; base.shear=3e3; base.omega_z=2e3;
    base.alpha=0.1; base.rho_mid_guess=1e-2; base.n_nodes=160; base.max_iters=80; base.tol=1e-8;
    grrt::ColumnInputs cool = base; cool.T_eff = 5e4;
    grrt::ColumnInputs hot  = base; hot.T_eff  = 2e5;
    auto sc = grrt::solve_column_bvp(cool, lut);
    auto sh = grrt::solve_column_bvp(hot, lut);
    std::printf("  cool T_eff=5e4 converged=%d z0=%.3e ; hot T_eff=2e5 converged=%d z0=%.3e\n",
                sc.converged, sc.z0, sh.converged, sh.z0);
    if (!sc.converged || !sh.converged) {
        std::printf("  FAIL: reference columns must converge from cold start (got %d, %d)\n",
                    sc.converged, sh.converged); failures++; return;
    }
    if (!(sh.z0 > sc.z0)) { std::printf("  FAIL: hotter column not thicker (z0 %.3e <= %.3e)\n", sh.z0, sc.z0); failures++; }
    else { std::printf("  PASS: hotter column thicker (z0 %.3e > %.3e)\n", sh.z0, sc.z0); }
}

static void test_hot_inner_disk_columns_converge() {
    std::printf("\n=== hot inner-disk columns converge (regime the real disk derives) ===\n");
    // Representative of a T_peak=1e7, 10 Msun, a=0.998 disk's inner columns:
    // omega_z ~ 3000/s, shear ~ 5800/s near the peak-flux radius (~2 r_g).
    // These T_eff values are what the disk DERIVES (not the easy gas-dominated
    // values the other tests pick). EXPECTED TO FAIL until the rad-pressure-regime
    // convergence barrier is fixed; do NOT relax the gate to make it pass.
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    const double Teffs[] = {1e6, 3e6, 6e6, 1e7};
    int ok = 0;
    for (double Te : Teffs) {
        grrt::ColumnInputs in{}; in.T_eff = Te; in.shear = 5800.0; in.omega_z = 3000.0;
        in.alpha = 0.1; in.rho_mid_guess = 1e-6; in.n_nodes = 200; in.max_iters = 120; in.tol = 1e-8;
        auto s = grrt::solve_column_bvp(in, lut);
        std::printf("  T_eff=%.0e converged=%d iters=%d resid=%.2e z0=%.3e\n",
                    Te, s.converged, s.iters, s.final_residual, s.z0);
        if (s.converged) ok++;
    }
    std::printf("  hot columns converged %d/4\n", ok);
    if (ok < 4) { std::printf("  FAIL: not all hot inner-disk columns converge (solver rad-pressure barrier)\n"); failures++; }
}

static void test_fadv_reduces_heating() {
    std::printf("\n=== f_adv reduces per-unit heating generation (S11 Eq 13) ===\n");
    using namespace grrt::constants;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    grrt::ColumnInputs base{};
    base.T_eff = 3e5; base.shear = 2e3; base.omega_z = 2e3;
    base.alpha = 0.1; base.rho_mid_guess = 1.0; base.n_nodes = 160;
    base.max_iters = 200; base.tol = 1e-8;
    grrt::ColumnInputs hot = base; hot.f_adv = 0.0;
    grrt::ColumnInputs adv = base; adv.f_adv = 0.5;   // 1/(1+0.5)=2/3 the generation
    auto s0 = grrt::solve_column_bvp(hot, lut);
    auto s1 = grrt::solve_column_bvp(adv, lut);
    if (!s0.converged || !s1.converged) { std::printf("  FAIL: a column did not converge\n"); failures++; return; }
    // The surface flux Q(surface)=sigma_SB T_eff^4 is a FIXED boundary condition, so
    // Q.back() is identical for both columns (it is NOT what f_adv changes). What
    // f_adv changes is the dissipation needed to DELIVER that flux: the radiated
    // generation is reduced by 1/(1+f_adv), so the column must dissipate (1+f_adv)x
    // MORE raw viscous energy ∫ alpha shear P dz to reach the same surface flux.
    // Equivalently the per-unit RADIATED generation ∫ alpha shear P/(1+f_adv) dz is
    // unchanged (= Q_surf). We assert BOTH: (a) raw dissipation rises by (1+f_adv),
    // (b) the f_adv-reduced generation matches the (identical) surface flux.
    auto raw_dissip = [&](const grrt::ColumnBVPSolution& s){
        double d = 0.0;
        for (size_t i = 1; i < s.z.size(); ++i) {
            const double dz = std::abs(s.z[i] - s.z[i-1]);
            d += base.alpha * base.shear * 0.5 * (s.P[i] + s.P[i-1]) * dz;
        }
        return d;
    };
    const double D0 = raw_dissip(s0), D1 = raw_dissip(s1);
    const double Q_surf = sigma_SB * std::pow(base.T_eff, 4.0);
    std::printf("  raw int(aSP)dz: f_adv=0 -> %.4e ; f_adv=0.5 -> %.4e ; ratio=%.4f (expect ~1.5=1+f_adv)\n",
                D0, D1, D1 / D0);
    std::printf("  reduced generation D1/(1+0.5)=%.4e vs Q_surf=%.4e (per face)\n", D1 / 1.5, Q_surf);
    // (a) raw dissipation rises by ~(1+f_adv): the column heats more to net the same flux.
    if (!(D1 > D0)) { std::printf("  FAIL: f_adv did not raise raw dissipation\n"); failures++; }
    check("raw dissipation ratio = 1+f_adv", D1 / D0, 1.5, 3e-3);
    // (b) the f_adv-reduced generation still balances the (fixed) surface flux.
    check("reduced generation = sigma T_eff^4 (per face)", D1 / 1.5, Q_surf, 5e-3);
}

static void test_analytic_vs_numerical_jacobian_fadv() {
    std::printf("\n=== analytic Jacobian matches numerical with f_adv=0.5 (cross-check) ===\n");
    // Same operating point as test_analytic_vs_numerical_jacobian, but with a nonzero
    // advection-reduction factor: the dQ partials are divided by (1+f_adv), so this
    // confirms the analytic Jacobian still matches FD after the divide.
    grrt::ColumnInputs in{}; in.T_eff = 5e4; in.shear = 3e3; in.omega_z = 2e3;
    in.alpha = 0.1; in.rho_mid_guess = 1e-2; in.n_nodes = 24; in.f_adv = 0.5;
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    std::vector<double> Ja, Jn; int n = 0;
    grrt::column_jacobians_test(in, lut, Ja, Jn, n);
    std::vector<double> rowmax((size_t)n, 0.0);
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c)
        rowmax[r] = std::max(rowmax[r], std::abs(Jn[(size_t)r*n+c]));
    double max_rel = 0.0; int bad_row=-1, bad_col=-1;
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) {
        const double a = Ja[(size_t)r*n+c], num = Jn[(size_t)r*n+c];
        const double scale = std::max(std::abs(num), 1e-6 * rowmax[r]);
        const double rel = std::abs(a - num) / scale;
        if (rel > max_rel && std::abs(a - num) > 1e-6 * rowmax[r]) { max_rel = rel; bad_row=r; bad_col=c; }
    }
    std::printf("  max relative mismatch = %.3e (worst at row %d col %d)\n", max_rel, bad_row, bad_col);
    if (max_rel > 1e-3) { std::printf("  FAIL: analytic Jacobian disagrees with numerical (f_adv=0.5)\n"); failures++; }
}

static void test_warm_start_converges_fast() {
    std::printf("\n=== warm start from a converged neighbour converges fast ===\n");
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    grrt::ColumnInputs a{}; a.T_eff = 5e4; a.shear = 3e3; a.omega_z = 2e3;
    a.alpha = 0.1; a.rho_mid_guess = 1e-2; a.n_nodes = 120; a.max_iters = 80; a.tol = 1e-8;
    auto sa = grrt::solve_column_bvp(a, lut);            // cold solve of the "neighbour"
    if (!sa.converged) { std::printf("  FAIL: neighbour did not converge\n"); failures++; return; }

    // Pack neighbour's converged state into a length-(4N+2) warm vector.
    const int N = a.n_nodes;
    std::vector<double> warm((size_t)4*N + 2, 0.0);
    for (int i = 0; i < N; ++i) {
        warm[4*i+0]=sa.P_gas[i]; warm[4*i+1]=sa.Q[i]; warm[4*i+2]=sa.T[i]; warm[4*i+3]=sa.z[i];
    }
    warm[4*N]=sa.z0; warm[4*N+1]=sa.Sigma0;

    // A nearby column (slightly hotter). Cold vs warm iteration counts.
    grrt::ColumnInputs b = a; b.T_eff = 5.2e4;
    auto cold = grrt::solve_column_bvp(b, lut);                 // no warm start
    auto warmed = grrt::solve_column_bvp(b, lut, &warm);        // warm start
    std::printf("  cold: conv=%d iters=%d ; warm: conv=%d iters=%d ; z0 cold=%.3e warm=%.3e\n",
                cold.converged, cold.iters, warmed.converged, warmed.iters, cold.z0, warmed.z0);
    if (!warmed.converged) { std::printf("  FAIL: warm start did not converge\n"); failures++; return; }
    // Newton iteration counts here are deterministic (no randomness), so the
    // cold-vs-warm gap is stable run to run. Strict < guards against a regression
    // where cold-start gets as good as warm; the absolute cap asserts the warm
    // start actually lands near the solution (a 4% T_eff perturbation converges fast).
    if (!(warmed.iters < cold.iters)) { std::printf("  FAIL: warm start not faster\n"); failures++; }
    if (!(warmed.iters <= 8)) { std::printf("  FAIL: warm start not near solution (iters=%d > 8)\n", warmed.iters); failures++; }
    // Both converged to tol=1e-8, so their half-thicknesses should match to ~1e-4 relative.
    if (cold.converged) check("warm z0 == cold z0", warmed.z0, cold.z0, 1e-4);
}

int main() {
    test_eos();
    test_scaffold();
    test_residual_hydrostatic_identity();
    test_residual_count_finite();
    test_numerical_jacobian_finite();
    test_analytic_vs_numerical_jacobian();
    test_analytic_vs_numerical_jacobian_hot();
    test_converges_and_conserves();
    test_physics_invariants();
    test_convergence_sweep();
    test_thickness_increases_with_teff();
    test_hot_inner_disk_columns_converge();
    test_fadv_reduces_heating();
    test_analytic_vs_numerical_jacobian_fadv();
    test_warm_start_converges_fast();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
