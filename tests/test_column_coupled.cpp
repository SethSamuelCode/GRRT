#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1
#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include <cstdio>
#include <cmath>
using namespace grrt;
int failures = 0;

// Forward decl (defined below) — used by the round-trip Jacobian gate.
static std::vector<double> pack_state(const ColumnClosure& c, int N);

// =============================================================================
// (1) Round-trip: drive the augmented (Σ,T_c)-driven column with the (Σ_target,
// T_c) of a converged base column built at f_adv=0, from a ROUGH T_eff seed (NOT
// the exact answer). It must recover F, z0 to <1e-3 AND back-solve f_adv ≈ 0.
// =============================================================================
static void test_coupled_repose_roundtrip() {
    std::printf("\n=== C1: (Σ,T_c)-driven augmented column recovers the f_adv=0 root ===\n");
    auto lut = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    ColumnInputs ref{}; ref.T_eff=3e5; ref.shear=2e3; ref.omega_z=2e3;
    ref.alpha=0.1; ref.f_adv=0.0; ref.rho_mid_guess=1.0; ref.n_nodes=96; ref.max_iters=300; ref.tol=1e-8;
    auto s = solve_column_bvp(ref, lut);
    if (!s.converged) { std::printf("  FAIL: reference column did not converge\n"); failures++; return; }
    const double Sigma_target = s.Sigma0, Tc_mid = s.T.front();
    ColumnCoupledInputs ci{}; ci.Sigma_target=Sigma_target; ci.Tc=Tc_mid;
    ci.shear=2e3; ci.omega_z=2e3; ci.alpha=0.1; ci.rho_mid_guess=1.0;
    ci.n_nodes=96; ci.max_iters=300; ci.tol=1e-8;
    // ROUGH seed: deliberately push the T_eff guess 40% off the true 3e5 so the
    // augmented Newton must move T_eff (and f_adv) to the root — not seeded at it.
    ci.Teff_guess = 0.6 * ref.T_eff;
    ColumnClosure c = solve_column_coupled(ci, lut, nullptr);
    const double F_expect = grrt::constants::sigma_SB*std::pow(ref.T_eff,4);
    const double relF = std::abs(c.F - F_expect)/F_expect;
    const double relz = std::abs(c.z0 - s.z0)/s.z0;
    std::printf("  conv=%d F=%.4e (expect %.4e rel=%.2e)  z0=%.4e (ref %.4e rel=%.2e)\n"
                "     Teff=%.4e (ref 3e5)  f_adv=%.4e (expect ~0)\n",
                c.converged, c.F, F_expect, relF, c.z0, s.z0, relz, c.T_eff, c.f_adv);
    if (!c.converged || relF>1e-3 || relz>1e-3) { std::printf("  FAIL: F/z0 not recovered\n"); failures++; }
    if (std::abs(c.f_adv) > 1e-3) {
        std::printf("  FAIL: back-solved f_adv not ≈0 (got %.4e)\n", c.f_adv); failures++; }
    if (!c.converged) return;   // no profile to pack — avoid reading an empty solution

    // GATE: the augmented analytic Jacobian must match the FD Jacobian at the
    // converged root (structural entries ~machine; inherited interior opacity
    // partial ~3e-4, matching the base solver's own gate).
    std::vector<double> Uc = pack_state(c, ci.n_nodes);
    const double fd_mism = coupled_jacobian_fd_mismatch(Uc, ci, lut);
    std::printf("  analytic-vs-FD Jacobian mismatch at root = %.3e (expect <4e-4)\n", fd_mism);
    if (!(fd_mism < 4.0e-4)) {
        std::printf("  FAIL: FD-Jacobian mismatch exceeds the inherited-opacity band\n"); failures++; }
}

// Helper: build a consistent (Σ_target, T_c) pair from a reference T_eff-driven column.
static bool reference_pair(const OpacityLUTs& lut, ColumnCoupledInputs& ci_out,
                           double& z0_ref, double& F_ref) {
    ColumnInputs ref{}; ref.T_eff=3e5; ref.shear=2e3; ref.omega_z=2e3;
    ref.alpha=0.1; ref.f_adv=0.0; ref.rho_mid_guess=1.0; ref.n_nodes=96; ref.max_iters=300; ref.tol=1e-8;
    auto s = solve_column_bvp(ref, lut);
    if (!s.converged) return false;
    z0_ref = s.z0;
    F_ref  = grrt::constants::sigma_SB*std::pow(ref.T_eff,4);
    ColumnCoupledInputs ci{}; ci.Sigma_target=s.Sigma0; ci.Tc=s.T.front();
    ci.shear=2e3; ci.omega_z=2e3; ci.alpha=0.1; ci.rho_mid_guess=1.0;
    ci.n_nodes=96; ci.max_iters=300; ci.tol=1e-8;
    ci_out = ci;
    return true;
}

// Pack a converged ColumnClosure into the (4N+4) augmented coupled state vector.
static std::vector<double> pack_state(const ColumnClosure& c, int N) {
    std::vector<double> U(4*N+4, 0.0);
    for (int i=0;i<N;++i){ U[4*i+0]=c.sol.P_gas[i]; U[4*i+1]=c.sol.Q[i];
                           U[4*i+2]=c.sol.T[i];     U[4*i+3]=c.sol.z[i]; }
    U[4*N]=c.z0; U[4*N+1]=c.sol.Sigma0; U[4*N+2]=c.T_eff; U[4*N+3]=c.f_adv;
    return U;
}

// Ruiz conditioning ratio (raw vs equilibrated LU pivot ratio) of the coupled
// Jacobian at a given coupled state. Uses the in-TU static helpers.
static void coupled_pivot_ratios(const std::vector<double>& U, const ColumnCoupledInputs& in,
                                 const OpacityLUTs& op, double& raw, double& eq) {
    const int n = 4*in.n_nodes + 4;
    std::vector<double> J, Jeq;
    coupled_column_jacobian(U, in, op, J);
    raw = lu_pivot_ratio(J, n);
    ruiz_scaled_copy(J, n, Jeq);
    eq  = lu_pivot_ratio(Jeq, n);
}

// =============================================================================
// (3) Basin gates: (i) Ruiz equilibration COLLAPSES the augmented Jacobian's LU
// pivot ratio. (ii) the affine-invariant Newton converges a COLD NAIVE seed and a
// 2%-off warm start to the SAME root (F & z0 to <1e-6). Reports iteration counts.
// =============================================================================
static void test_coupled_naive_seed_converges() {
    std::printf("\n=== C1: Ruiz conditioning + affine-invariant basin (naive / 2%%-off) ===\n");
    auto lut = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    ColumnCoupledInputs ci; double z0_ref=0, F_ref=0;
    if (!reference_pair(lut, ci, z0_ref, F_ref)) {
        std::printf("  FAIL: reference column did not converge\n"); failures++; return;
    }
    ColumnClosure c_sec = solve_column_coupled(ci, lut, nullptr);
    if (!c_sec.converged) { std::printf("  FAIL: default-seeded solve did not converge\n"); failures++; return; }

    // GATE (i): pivot ratio at the converged root must collapse under Ruiz equilibration.
    std::vector<double> Uroot = pack_state(c_sec, ci.n_nodes);
    double raw=0, eq=0; coupled_pivot_ratios(Uroot, ci, lut, raw, eq);
    std::printf("  LU pivot ratio at root: raw=%.3e  Ruiz-equilibrated=%.3e  (collapse x%.2e)\n",
                raw, eq, raw/std::max(eq,1e-300));
    if (!(raw > 1e8))            { std::printf("  FAIL: expected the documented stiff (~1e10) raw ratio\n"); failures++; }
    if (!(eq  < 1e6))            { std::printf("  FAIL: equilibration did not collapse the conditioning\n"); failures++; }
    if (!(raw/std::max(eq,1e-300) > 1e5)) { std::printf("  FAIL: conditioning improvement < 1e5\n"); failures++; }

    // GATE (ii-a): 2%-off-T_eff warm start MUST converge to the root (F & z0 <1e-6).
    ColumnCoupledInputs cw = ci;
    std::vector<double> Uperturb = Uroot;
    Uperturb[4*ci.n_nodes+2] *= 1.02;   // nudge T_eff 2% off-root (slot 4N+2)
    ColumnClosure c_warm = solve_column_coupled(cw, lut, &Uperturb);
    const double relF_w = std::abs(c_warm.F - F_ref)/F_ref;
    const double relz_w = std::abs(c_warm.z0 - z0_ref)/z0_ref;
    std::printf("  2%%-off warm start: converged=%d  relF=%.2e  relz=%.2e  f_adv=%.3e\n",
                c_warm.converged, relF_w, relz_w, c_warm.f_adv);
    if (!c_warm.converged || relF_w>1e-6 || relz_w>1e-6) {
        std::printf("  FAIL: 2%%-off warm start did not converge to the root\n"); failures++; }

    // GATE (ii-b): COLD NAIVE seed (rough T_eff guess, f_adv=0) MUST converge to the
    // same root WITHOUT the continuation fallback (allow_continuation=false proves the
    // primary Newton's basin alone is wide enough — the over-determination is gone).
    ColumnCoupledInputs cn = ci; cn.naive_seed = true; cn.allow_continuation = false;
    cn.Teff_guess = 0.6 * 3e5;   // rough, 40% off the true root
    ColumnClosure c_naive = solve_column_coupled(cn, lut, nullptr);
    const double relF_n = std::abs(c_naive.F - F_ref)/F_ref;
    const double relz_n = std::abs(c_naive.z0 - z0_ref)/z0_ref;
    std::printf("  cold naive seed (no continuation): converged=%d  relF=%.2e  relz=%.2e  f_adv=%.3e\n",
                c_naive.converged, relF_n, relz_n, c_naive.f_adv);
    if (!c_naive.converged || relF_n>1e-6 || relz_n>1e-6) {
        std::printf("  FAIL: cold naive seed did not converge to the root\n"); failures++; }
}

// =============================================================================
// (2) Inconsistent (Σ,T_c) pairs (Σ×1.3, ×0.7 with T_c HELD at the unperturbed
// value). With f_adv FREED these are NO LONGER folds — they MUST converge to a
// physical column. The back-solved f_adv must be finite and physical (1+f_adv>0),
// and in the right ballpark (≈+1.13 for ×1.3, ≈−0.63 for ×0.7 from the independent
// slim_fadv_freedom_probe; loose tolerance).
// =============================================================================
static void test_coupled_inconsistent_pair_converges() {
    std::printf("\n=== C1: inconsistent (Σ,T_c) pairs now CONVERGE (f_adv freed) ===\n");
    auto lut = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    ColumnCoupledInputs ci; double z0_ref=0, F_ref=0;
    if (!reference_pair(lut, ci, z0_ref, F_ref)) {
        std::printf("  FAIL: reference column did not converge\n"); failures++; return;
    }
    ColumnClosure base = solve_column_coupled(ci, lut, nullptr);
    if (!base.converged) { std::printf("  FAIL: unperturbed coupled base did not converge\n"); failures++; return; }
    std::vector<double> Uwarm = pack_state(base, ci.n_nodes);
    const double Sigma0 = ci.Sigma_target;

    // Expected back-solved f_adv from the independent probe (loose bands).
    struct Case { double mult; double fadv_lo, fadv_hi; };
    const Case cases[2] = { {1.3, 0.5, 1.8}, {0.7, -0.9, -0.3} };

    for (const Case& cs : cases) {
        ColumnCoupledInputs cp = ci;
        cp.Sigma_target = Sigma0 * cs.mult;
        std::printf("  -- Sigma x%.2f = %.4e (Tc fixed %.4e) --\n", cs.mult, cp.Sigma_target, cp.Tc);

        // GATE: the equilibrated Jacobian for the perturbed pair is well-conditioned.
        double raw=0, eq=0; coupled_pivot_ratios(Uwarm, cp, lut, raw, eq);
        std::printf("     pivot ratio (perturbed pair): raw=%.3e  Ruiz-eq=%.3e\n", raw, eq);
        if (!(eq < 1e6)) { std::printf("  FAIL: equilibration did not condition the perturbed pair\n"); failures++; }

        // The inconsistent pair MUST converge to a physical column (was a "fold").
        ColumnClosure c = solve_column_coupled(cp, lut, nullptr);
        const bool phys = c.converged && c.F>0 && c.z0>0 && c.T_eff>0 && (1.0 + c.f_adv) > 0.0;
        std::printf("     converged=%d F=%.4e z0=%.4e Teff=%.4e f_adv=%.4e (probe ~%.2f..%.2f)\n",
                    c.converged, c.F, c.z0, c.T_eff, c.f_adv, cs.fadv_lo, cs.fadv_hi);
        if (!phys) { std::printf("  FAIL: inconsistent pair did not converge to a physical column\n"); failures++; }
        if (!(c.f_adv > cs.fadv_lo && c.f_adv < cs.fadv_hi)) {
            std::printf("  FAIL: back-solved f_adv outside the probe's expected ballpark\n"); failures++; }
    }
}

int main(){
    test_coupled_repose_roundtrip();
    test_coupled_naive_seed_converges();
    test_coupled_inconsistent_pair_converges();
    std::printf("\n## %d failure(s) ##\n", failures);
    return failures?1:0;
}
