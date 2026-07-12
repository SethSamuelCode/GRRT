#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1
#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include "../src/slim_disk_radial.cpp"
#include "../src/slim_disk_coupled.cpp"
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
// (3b) WARM-START TINY-T_c-PERTURBATION regression (the C4-relevant configuration).
// Warm-start solve_column_coupled from a CONVERGED root for a tiny midplane-T_c
// nudge (relative h=1e-4 and 1e-5). This used to STALL: the monolithic augmented
// Newton cannot descend from the warm state (the inherited ~3e-4 opacity-LUT
// inexactness in the stiff interior radiative-flux/surface rows makes the step that
// closes the T(0)−T_c pin DIVERGENT; the damped line search then stalls at a
// spurious fixed point ≈ the OLD root — converged=false). The fix re-seeds via the
// well-conditioned 2-D (T_eff,f_adv) bring-up (the cold path's mechanism) on
// warm-start failure, so this now converges DIRECTLY to the TRUE perturbed root.
//
// GATE: (i) converges; (ii) reaches the TRUE perturbed root, NOT the unperturbed
// anchor — i.e. F MOVES by the expected O(h) amount (~8.6e-4 relative per the cold-
// resolve reference at h=1e-4). A "converged at the old root" masking would leave
// F unchanged (relF_move≈0) and is explicitly rejected here.
// =============================================================================
static void test_coupled_warmstart_tiny_Tc_perturbation() {
    std::printf("\n=== C1/C4: warm-start TINY-T_c-perturbation converges to the TRUE root ===\n");
    auto lut = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    ColumnCoupledInputs ci; double z0_ref=0, F_ref=0;
    if (!reference_pair(lut, ci, z0_ref, F_ref)) {
        std::printf("  FAIL: reference column did not converge\n"); failures++; return;
    }
    ColumnClosure base = solve_column_coupled(ci, lut, nullptr);
    if (!base.converged) { std::printf("  FAIL: unperturbed base did not converge\n"); failures++; return; }
    const std::vector<double> Uwarm = pack_state(base, ci.n_nodes);

    // For each tiny relative nudge: the warm-start solve must converge AND F must move
    // by ~ the cold-resolve reference amount (so it is the NEW root, not the old one).
    for (double h : { 1e-4, 1e-5 }) {
        ColumnCoupledInputs cp = ci; cp.Tc = ci.Tc * (1.0 + h);
        ColumnClosure cw = solve_column_coupled(cp, lut, &Uwarm);
        // Independent cold reference for the perturbed root (the reliable answer).
        ColumnClosure cc = solve_column_coupled(cp, lut, nullptr);
        const bool both = cw.converged && cc.converged;
        const double relF_move_warm = both ? std::abs(cw.F - base.F)/base.F : -1.0;  // should be ~8.6e-4*(h/1e-4)
        const double relF_warm_vs_cold = both ? std::abs(cw.F - cc.F)/cc.F : -1.0;    // warm vs cold root agreement
        std::printf("  h=%.0e: warm conv=%d cold conv=%d | F_move(warm)=%.3e (cold ref=%.3e) | warm-vs-cold rel=%.3e f_adv=%.3e\n",
                    h, cw.converged, cc.converged, relF_move_warm,
                    cc.converged ? std::abs(cc.F-base.F)/base.F : -1.0, relF_warm_vs_cold, cw.f_adv);
        if (!cw.converged) { std::printf("  FAIL: warm-start tiny-T_c solve STALLED (the regression)\n"); failures++; continue; }
        // (ii) the warm root must AGREE with the cold root (same true root) to <1e-4...
        if (!(relF_warm_vs_cold < 1e-4)) {
            std::printf("  FAIL: warm root disagrees with cold root (not the true perturbed root)\n"); failures++; }
        // ...and must have actually MOVED off the anchor by the expected O(h) amount
        // (a stalled "old-root" accept would leave F unmoved — masking guard).
        const double expect_move = 8.62e-4 * (h / 1e-4);
        if (!(relF_move_warm > 0.3 * expect_move)) {
            std::printf("  FAIL: F did not move off the anchor (masking: accepted the OLD root)\n"); failures++; }
    }
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

// =============================================================================
// (4) C2: column_moments η₃ = ∫E dz / ∫P dz on a SELF-CONSISTENT synthetic profile.
// Constant β ≡ P_gas/P_total reduces the moment to η₃ → 3 − 1.5β (E = (3/2)βP +
// 3(1−β)P). T is set so P_rad = (a_rad/3)T⁴ equals (1−β)P exactly at every node, so
// the profile's own radiation share matches β (the plan's constant-T draft was self-
// inconsistent; this one is not).
// =============================================================================
static void test_moments_eta3_onezone_limit() {
    std::printf("\n=== C2: eta3 = INT E/INT P reduces to 3-1.5*beta for constant beta ===\n");
    const int N=128; const double beta=0.4;
    grrt::ColumnBVPSolution s;
    s.z.resize(N); s.P.resize(N); s.P_gas.resize(N); s.T.resize(N); s.rho.resize(N);
    const double a_rad = grrt::constants::a_rad;  // SAME radiation constant the column uses
    for (int i=0;i<N;++i){
        const double zc = double(i)/(N-1);
        const double P  = std::exp(-zc*zc) + 0.1;     // total pressure, smooth & positive
        const double Prad = (1.0-beta)*P;             // radiation share => constant beta
        s.z[i]=zc; s.P[i]=P; s.P_gas[i]=beta*P;
        s.T[i]=std::pow(3.0*Prad/a_rad, 0.25);        // T consistent: (a_rad/3)T^4 = Prad
        s.rho[i]=P;                                   // (rho only used by eta4, Task 5)
    }
    double eta3=0, eta4=0; grrt::column_moments(s, eta3, eta4);
    const double expect = 3.0 - 1.5*beta;             // 2.4
    std::printf("  eta3=%.8f expect=%.8f rel=%.2e\n", eta3, expect, std::abs(eta3-expect)/expect);
    if (std::abs(eta3-expect)/expect > 1e-6) { std::printf("  FAIL\n"); failures++; }
}

// =============================================================================
// (5) C3: analytic column-output sensitivities dC/d{Σ_target,T_c} for
// C = {F, z0, η3, η4, f_adv} via the implicit-function theorem through the
// augmented column Jacobian — validated against the PERTURB-RESOLVE oracle:
// re-solve the coupled column at p·(1±h) (warm-started from the base U_c) and
// central-difference each output. Gate: each component's analytic-vs-resolve
// relative error < 1e-3 (the re-solve FD floor; the column Jacobian carries
// ~3e-4 inherited opacity-LUT inexactness, so 1e-3 is the right gate — neither
// loosen above nor tighten below it). Components whose value ≈0 are guarded with
// an absolute floor.
// =============================================================================
static void test_dC_dp_vs_resolve_oracle() {
    std::printf("\n=== C3: dC/d{Sigma,Tc} analytic (IFT) vs perturb-resolve oracle ===\n");
    auto lut = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);

    // Base coupled solve at a CONSISTENT (Σ,T_c) (taken from a reference T_eff-driven
    // column, exactly as the round-trip does) so f_adv≈0 and the base is on the root.
    ColumnCoupledInputs ci; double z0_ref=0, F_ref=0;
    if (!reference_pair(lut, ci, z0_ref, F_ref)) {
        std::printf("  FAIL: reference column did not converge\n"); failures++; return;
    }
    ColumnClosure c0 = solve_column_coupled(ci, lut, nullptr);
    if (!c0.converged) { std::printf("  FAIL: base coupled solve did not converge\n"); failures++; return; }

    // Analytic IFT sensitivity at the base.
    ColumnSensitivity sens = column_sensitivity(c0, ci, lut);

    // Warm-start vector for the perturbed re-solves (the converged base U_c).
    const std::vector<double> Uwarm = pack_state(c0, ci.n_nodes);

    // Perturb-resolve oracle for one parameter index (0 = Σ_target, 1 = T_c).
    // Returns central-difference dC/dp for C = {F, z0, η3, η4, f_adv}.
    struct Cd { double dF, dz0, deta3, deta4, dfadv; bool ok; };
    auto resolve_central = [&](int pidx, double h) -> Cd {
        const double base_p = (pidx == 0) ? ci.Sigma_target : ci.Tc;
        const double dp = h * base_p;                       // relative step
        ColumnCoupledInputs cp = ci, cm = ci;
        if (pidx == 0) { cp.Sigma_target = base_p + dp; cm.Sigma_target = base_p - dp; }
        else           { cp.Tc           = base_p + dp; cm.Tc           = base_p - dp; }
        // Re-solve the perturbed columns independently and central-difference. Try the
        // warm start (from the base U_c) first — it converges in ~1 iter and gives a
        // clean difference. If a warm-started perturbed solve STALLS (the augmented
        // Newton's basin from the unperturbed seed is narrow for a tiny T_c nudge —
        // empirically observed for the T_c branch at h≤1e-4), fall back to a COLD solve
        // (the full 2-D bring-up to the true root). The cold solve is the MORE reliable
        // reference, so this strengthens the oracle; it is not a tolerance relaxation.
        auto resolve_one = [&](const ColumnCoupledInputs& cc) -> ColumnClosure {
            ColumnClosure r = solve_column_coupled(cc, lut, &Uwarm);
            if (!r.converged) r = solve_column_coupled(cc, lut, nullptr);   // cold fallback
            return r;
        };
        ColumnClosure cpl = resolve_one(cp);
        ColumnClosure cmi = resolve_one(cm);
        Cd d{};
        d.ok = cpl.converged && cmi.converged;
        if (!d.ok) return d;
        const double inv = 1.0 / (2.0 * dp);
        d.dF    = (cpl.F     - cmi.F)     * inv;
        d.dz0   = (cpl.z0    - cmi.z0)    * inv;
        d.deta3 = (cpl.eta3  - cmi.eta3)  * inv;
        d.deta4 = (cpl.eta4  - cmi.eta4)  * inv;
        d.dfadv = (cpl.f_adv - cmi.f_adv) * inv;
        return d;
    };

    // Relative-error compare with an absolute floor for near-zero components. The
    // floor is scaled by the characteristic magnitude of each output so a component
    // that is genuinely ≈0 (e.g. dfadv at a consistent base) is not spuriously failed.
    auto rel_or_abs = [&](double analytic, double oracle, double floor) -> double {
        const double denom = std::max(std::abs(oracle), floor);
        return std::abs(analytic - oracle) / denom;
    };

    const double h = 1e-5;
    const char* pname[2] = { "Sigma_target", "T_c" };
    bool any_fail = false;
    for (int p = 0; p < 2; ++p) {
        Cd o = resolve_central(p, h);
        if (!o.ok) { std::printf("  FAIL: perturb-resolve (%s) did not converge\n", pname[p]); failures++; any_fail=true; continue; }
        const double aF    = sens.dF[p],    az0 = sens.dz0[p];
        const double ae3   = sens.deta3[p], ae4 = sens.deta4[p], afa = sens.dfadv[p];
        // Absolute floors: a small fraction of the base output magnitude, so the rel
        // metric degrades gracefully to absolute when the derivative ≈0.
        const double fF   = 1e-6 * std::abs(c0.F);
        const double fz0  = 1e-6 * std::abs(c0.z0);
        const double fe3  = 1e-6 * std::max(std::abs(c0.eta3), 1.0);
        const double fe4  = 1e-6 * std::max(std::abs(c0.eta4), 1.0);
        const double ffa  = 1e-3;   // f_adv is O(1) and ≈0 at the consistent base
        const double eF   = rel_or_abs(aF,  o.dF,    fF);
        const double ez0  = rel_or_abs(az0, o.dz0,   fz0);
        const double ee3  = rel_or_abs(ae3, o.deta3, fe3);
        const double ee4  = rel_or_abs(ae4, o.deta4, fe4);
        const double efa  = rel_or_abs(afa, o.dfadv, ffa);
        std::printf("  d/d%-12s  F: a=% .4e o=% .4e r=%.2e | z0: a=% .4e o=% .4e r=%.2e\n",
                    pname[p], aF, o.dF, eF, az0, o.dz0, ez0);
        std::printf("  %14s  eta3: a=% .4e o=% .4e r=%.2e | eta4: a=% .4e o=% .4e r=%.2e | fadv: a=% .4e o=% .4e r=%.2e\n",
                    "", ae3, o.deta3, ee3, ae4, o.deta4, ee4, afa, o.dfadv, efa);
        for (double e : { eF, ez0, ee3, ee4, efa }) {
            if (!(e < 1e-3)) { any_fail = true; }
        }
    }
    if (any_fail) { std::printf("  FAIL: a dC/dp component exceeds the 1e-3 perturb-resolve gate\n"); failures++; }
}

// =============================================================================
// (6) CONVECTIVE-STATE Jacobian oracle. The coupled Schur Jacobian's dT partials
// must switch to the MLT convective form wherever the column is convectively
// unstable (Schwarzschild), EXACTLY as the standalone analytic_jacobian does. A
// deep radiation-pressure column (T_eff=1e7, inner-disk geometry) is convective at
// nearly every interior node; its packed coupled state exposes any convection-
// blindness in coupled_column_jacobian as a large analytic-vs-FD mismatch. The
// earlier gates all used consistent / low-Ṁ (radiative) states, so this regression
// slipped through — hence a dedicated convective oracle. The FD Jacobian differences
// coupled_column_residual (which is convection-aware via node_deriv), so a convection-
// blind analytic Jacobian mismatches it by ~1e-1 on this state.
// =============================================================================
static void test_coupled_jacobian_convective_state() {
    std::printf("\n=== C1: coupled Jacobian matches FD on a CONVECTIVE column state ===\n");
    auto lut = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    // Solve the standalone (convection-aware) BVP at the deep rad-pressure geometry to
    // get a converged CONVECTIVE profile, then pack it into the augmented coupled state.
    const int N = 96;
    ColumnInputs ref{}; ref.T_eff=1e7; ref.shear=5.25e3; ref.omega_z=3.12e3;
    ref.alpha=0.1; ref.f_adv=0.0; ref.rho_mid_guess=1.0; ref.n_nodes=N; ref.max_iters=300; ref.tol=1e-8;
    auto s = solve_column_bvp(ref, lut);
    if (!s.converged) { std::printf("  FAIL: deep rad-pressure reference column did not converge\n"); failures++; return; }

    // Pack into the (4N+4) coupled state: interior nodes + z0, Σ0, T_eff(surface), f_adv=0.
    std::vector<double> U(4*N+4, 0.0);
    for (int i=0;i<N;++i){ U[4*i+0]=s.P_gas[i]; U[4*i+1]=s.Q[i]; U[4*i+2]=s.T[i]; U[4*i+3]=s.z[i]; }
    U[4*N]=s.z0; U[4*N+1]=s.Sigma0; U[4*N+2]=ref.T_eff; U[4*N+3]=0.0;

    ColumnCoupledInputs ci{}; ci.Sigma_target=s.Sigma0; ci.Tc=s.T.front();
    ci.shear=ref.shear; ci.omega_z=ref.omega_z; ci.alpha=ref.alpha; ci.rho_mid_guess=1.0;
    ci.n_nodes=N; ci.max_iters=300; ci.tol=1e-8;

    // The oracle is only meaningful if the state actually HAS convective nodes.
    int nconv=0;
    for (int i=0;i<N;++i){
        const double Pg=U[4*i+0], Q=U[4*i+1], T=U[4*i+2], z=U[4*i+3];
        const double rho=std::max(rho_from_gas(Pg,T), RHO_GHOST_FLOOR);
        const double Ptot=p_total(Pg,T);
        const double kappa=kappa_total(lut, rho, T);
        double nab; bool is_conv;
        grrt::detail_bvp::convective_gradient(rho,T,Ptot,Q,kappa,z,ci.omega_z,nab,is_conv);
        if (is_conv) ++nconv;
    }
    std::printf("  convective interior nodes: %d / %d\n", nconv, N);
    if (nconv == 0) { std::printf("  FAIL: state has no convective nodes — oracle is vacuous\n"); failures++; return; }

    // GATE: analytic coupled Jacobian must match the FD of the (convective) residual to
    // the inherited-opacity floor — the SAME <4e-4 band the radiative round-trip meets.
    const double fd_mism = coupled_jacobian_fd_mismatch(U, ci, lut);
    std::printf("  analytic-vs-FD Jacobian mismatch on convective state = %.3e (expect <4e-4)\n", fd_mism);
    if (!(fd_mism < 4.0e-4)) {
        std::printf("  FAIL: coupled Jacobian is convection-blind (mismatch exceeds FD floor)\n"); failures++; }
}

// Mass-conservation Σ<->V round-trip. The inverse must be exact.
static void test_massconsv_roundtrip() {
    std::printf("\n=== C2: mass-conservation Σ<->V round-trip ===\n");
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = 0.9; in.r_g = 1.48e6;
    in.mdot = 1.6399e16;  // f_Edd=0.001 scale
    const double rs[]  = {2.27, 5.0, 20.0, 50.0};
    const double Sig[] = {1.0e2, 5.0e3, 1.2e4, 6.0e3};
    bool any_fail = false;
    for (double r : rs) for (double S : Sig) {
        const double V  = grrt::slim_coupled_detail::V_from_sigma(in, r, S);
        const double S2 = grrt::slim_coupled_detail::sigma_from_V(in, r, V);
        const double rel = std::abs(S2 - S) / S;
        std::printf("  r=%.2f Σ=%.3e -> V=%.3e -> Σ'=%.3e  rel=%.2e\n", r, S, V, S2, rel);
        if (!(rel < 1e-10)) any_fail = true;
    }
    if (any_fail) { std::printf("  FAIL: round-trip exceeds 1e-10\n"); failures++; }
}

int main(){
    test_coupled_repose_roundtrip();
    test_coupled_naive_seed_converges();
    test_coupled_warmstart_tiny_Tc_perturbation();
    test_coupled_inconsistent_pair_converges();
    test_moments_eta3_onezone_limit();
    test_dC_dp_vs_resolve_oracle();
    test_coupled_jacobian_convective_state();
    test_massconsv_roundtrip();
    std::printf("\n## %d failure(s) ##\n", failures);
    return failures?1:0;
}
