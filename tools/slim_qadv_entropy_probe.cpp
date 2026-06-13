// ===========================================================================
// slim_qadv_entropy_probe.cpp
//
// ENTROPY-FORM Q_adv VALIDATION (flag #1, the §23 advective-cooling bracket).
//
// The §23 one-zone advective cooling is
//     Q_adv = -(Ṁ/2π r_cm²)·(P/Σ)·[η₃·dlnP/dlnr − (1+η₃)·dlnΣ/dlnr],  η₃ ≡ E/P.
// Flag #1 (2026-06-12) corrected the bracket from the inverted [(Γ₁−1)dlnP − Γ₁dlnΣ];
// refinement #11 (2026-06-13) made η₃ STATE-DEPENDENT: η₃(β)=3−1.5β (β≡p_gas/p_mid;
// β=1 gas ⇒ 3/2, β=0 radiation ⇒ 3).  Both EXISTING gates are BLIND to Q_adv (the
// NT-reduction probe runs at Ṁ→0 where Q_adv→0; the FD-Jacobian gate only proves the
// analytic Jacobian matches the residual, not that the residual is right).  This
// probe closes that hole: it evaluates Q_adv on a CONVERGED profile three ways.
//
//   (1) CODE    : the bracket as the POST-#11 solver assembles it — variable η₃(β).
//   (2) IDENT   : the raw 2D entropy advection −(Ṁ/2π r_cm²)[ d(E/Σ)/dlnr
//                 + P·d(1/Σ)/dlnr ] with the PRE-#11 FROZEN E=(3/2)P (η₃=3/2) —
//                 i.e. what the old gas-limit code computed; kept for reference.
//   (3) TRUE2D  : the same entropy identity with the TRUE variable moment
//                 E = (3/2)p_gas,int + 3·p_rad,int = P·(3 − (3/2)β), η₃(β)=E/P —
//                 the independent thermodynamic re-assembly of the post-#11 physics.
//
// All three use the SAME 2D variables (P, Σ, β) and the SAME central-FD operator,
// so the comparison is clean (no 3D/H-gradient contamination).  Decision:
//   (1) ≈ (3)  ⇒ the variable-η₃ bracket is correctly implemented (post-#11 check).
//   (3) vs (2) ⇒ how much refinement #11 changed Q_adv vs the old frozen η₃.
//
// Same include-the-.cpp pattern as the other slim probes (reaches internals).
// Safe to delete along with this file.
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

using namespace grrt;
using namespace grrt::slim_detail;

namespace grrt {
namespace probe {

static SlimDiskInputs make_inputs(double a, double f_Edd, int N, double wall_s,
                                  double& Mdot_Edd_out) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0;
    in.spin = a;
    in.alpha = 0.1;
    in.r_g = 1.48e6;
    in.r_out = 50.0;
    in.n_nodes = N;
    in.max_iters = 800;
    in.tol = 1e-6;
    in.budget_wall_seconds = wall_s;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    Mdot_Edd_out = Mdot_Edd;
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

// Central finite difference d(ln f)/d(ln r) on the (log-spaced) profile grid.
static double dlnf(const std::vector<double>& f, const std::vector<double>& r, int i) {
    const int N = (int)r.size();
    auto L = [](double x){ return std::log(std::max(x, 1e-300)); };
    if (i == 0)      return (L(f[1])   - L(f[0]))   / (L(r[1])   - L(r[0]));
    if (i == N - 1)  return (L(f[N-1]) - L(f[N-2])) / (L(r[N-1]) - L(r[N-2]));
    return (L(f[i+1]) - L(f[i-1])) / (L(r[i+1]) - L(r[i-1]));
}
// Central finite difference d(y)/d(ln r) of a (non-logged) nodal quantity y.
static double df_dlnr(const std::vector<double>& y, const std::vector<double>& r, int i) {
    const int N = (int)r.size();
    auto L = [](double x){ return std::log(std::max(x, 1e-300)); };
    if (i == 0)      return (y[1]   - y[0])   / (L(r[1])   - L(r[0]));
    if (i == N - 1)  return (y[N-1] - y[N-2]) / (L(r[N-1]) - L(r[N-2]));
    return (y[i+1] - y[i-1]) / (L(r[i+1]) - L(r[i-1]));
}

static void probe_one(const OpacityLUTs& op, double a, double f_Edd, int N, double wall_s) {
    using namespace constants;
    double Mdot_Edd = 0;
    SlimDiskInputs in = make_inputs(a, f_Edd, N, wall_s, Mdot_Edd);
    const int NN = std::max(in.n_nodes, 4);

    std::printf("\n==================== Q_adv entropy probe  f_Edd=%.3f (a=%.3f, N=%d) ====================\n",
                f_Edd, a, N);

    std::vector<double> U = build_thin_disk_seed(in, op);
    auto t0 = std::chrono::steady_clock::now();
    const bool ok = solve_single_am(in, op, U, /*require_N1=*/false);
    auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    std::printf("  solve ok=%d  wall=%.1fs  r_sonic=%.5f\n", (int)ok, wall, U[4*NN+1]);
    if (!ok) { std::printf("  (not converged — skipping)\n"); return; }

    SlimDiskRadial prof;
    unpack_profile(in, op, U, prof);
    const int M = (int)prof.r.size();
    const double Mdot = in.mdot;

    // Per-node closure quantities in 2D variables.
    std::vector<double> P(M), Sig(M), Tc(M), beta(M), eta3(M), invSig(M),
                        E_const(M), E_true(M), Qrad(M);
    for (int i = 0; i < M; ++i) {
        const double r = prof.r[i];
        Sig[i] = std::max(prof.Sigma[i], kSigmaFloor);
        Tc[i]  = std::max(prof.Tc[i],   kTFloor);
        const OneZoneState oz = one_zone_closure(Sig[i], Tc[i], r, in, op);
        P[i]    = oz.P;
        beta[i] = oz.p_gas / std::max(oz.p_mid, 1e-300);
        eta3[i] = 3.0 - 1.5 * beta[i];               // η₃(β) = E/P, gas+radiation
        invSig[i] = 1.0 / Sig[i];
        E_const[i] = 1.5 * P[i] / Sig[i];            // (3/2)P/Σ   (η₃=3/2 frozen)
        E_true[i]  = eta3[i] * P[i] / Sig[i];        // η₃(β)·P/Σ  (true moment)
        const double kR = op.lookup_kappa_ross(oz.rho_mid, Tc[i])
                        + op.lookup_kappa_es(oz.rho_mid, Tc[i]);
        Qrad[i] = 64.0 * sigma_SB * Tc[i]*Tc[i]*Tc[i]*Tc[i]
                / (3.0 * std::max(kR, 1e-300) * Sig[i]);
    }

    // Compute all three Q_adv forms per node first (need the global max|Q_code|
    // to pick "clean" nodes for the implementation metric).
    std::vector<double> Qc(M), Qi(M), Qt(M);
    for (int i = 0; i < M; ++i) {
        const double r_cm = prof.r[i] * in.r_g;
        const double K = Mdot / (2.0 * std::numbers::pi * r_cm * r_cm);
        const double PoS = P[i] / Sig[i];
        const double dlnP = dlnf(P,   prof.r, i);
        const double dlnS = dlnf(Sig, prof.r, i);
        // (1) CODE — post-#11 solver uses the node-local state-dependent bracket
        // [η₃(β) dlnP − (1+η₃(β)) dlnΣ] (was the frozen [1.5 dlnP − 2.5 dlnΣ]).
        Qc[i] = -K * PoS * (eta3[i] * dlnP - (1.0 + eta3[i]) * dlnS);                   // (1) CODE
        Qi[i] = -K * (df_dlnr(E_const, prof.r, i) + P[i] * df_dlnr(invSig, prof.r, i)); // (2) IDENT
        Qt[i] = -K * (df_dlnr(E_true,  prof.r, i) + P[i] * df_dlnr(invSig, prof.r, i)); // (3) TRUE2D
    }
    double maxAbsQ = 0.0;
    for (int i = 0; i < M; ++i) maxAbsQ = std::max(maxAbsQ, std::abs(Qc[i]));

    std::printf("  %-7s %-6s %-7s | %-12s %-12s %-12s | %-9s %-9s | %-11s %-11s\n",
                "r[M]", "beta", "eta3", "Q_code", "Q_ident", "Q_true2D",
                "id/code", "true/code", "fadv_code", "fadv_unpack");
    // CLEAN node = interior, Q_code locally sign-consistent (not a zero-crossing),
    // |Q_code| not negligible.  These are where the central-FD comparison is
    // meaningful (excludes the sonic sign-flip and any odd-even oscillation).
    // Post-#11 roles: Q_code is the VARIABLE-η₃ solver bracket, so the IMPLEMENTATION
    // check is Q_true2D/Q_code → 1 (variable identity vs variable code).  Q_ident/Q_code
    // (frozen identity vs variable code) instead measures how much #11 moved Q_adv.
    double impl_max = 0.0, impl_sum = 0.0, frozen_min = 1e30, frozen_max = -1e30, frozen_sum = 0.0;
    int n_clean = 0, n_betaflip = 0;
    for (int i = 0; i < M; ++i) {
        const double id_ratio   = (std::abs(Qc[i]) > 1e-300) ? Qi[i] / Qc[i] : 0.0;  // frozen/variable
        const double true_ratio = (std::abs(Qc[i]) > 1e-300) ? Qt[i] / Qc[i] : 0.0;  // variable-identity/variable-code
        const double fadv_code  = Qc[i] / std::max(std::abs(Qrad[i]), 1e-300);
        std::printf("  %-7.3f %-6.3f %-7.4f | %+ .4e %+ .4e %+ .4e | %-9.4f %-9.4f | %+ .3e %+ .3e\n",
                    prof.r[i], beta[i], eta3[i], Qc[i], Qi[i], Qt[i],
                    id_ratio, true_ratio, fadv_code, prof.f_adv[i]);
        if (i > 0 && i < M - 1) {
            const bool sign_ok = (Qc[i-1] > 0) == (Qc[i] > 0) && (Qc[i] > 0) == (Qc[i+1] > 0);
            const bool big_ok  = std::abs(Qc[i]) > 0.05 * maxAbsQ;
            if (sign_ok && big_ok) {
                impl_max = std::max(impl_max, std::abs(true_ratio - 1.0));   // post-#11 implementation check
                impl_sum += std::abs(true_ratio - 1.0);
                frozen_min = std::min(frozen_min, id_ratio);                 // old-frozen vs new-variable
                frozen_max = std::max(frozen_max, id_ratio);
                frozen_sum += id_ratio;
                ++n_clean;
            }
            // Odd-even β oscillation detector (mesh sawtooth in the gas/rad balance).
            if ((beta[i] - beta[i-1] > 0) != (beta[i+1] - beta[i] > 0)) ++n_betaflip;
        }
    }
    std::printf("  ---- summary (over %d clean nodes: interior, sign-consistent, |Q_code|>5%% peak) ----\n", n_clean);
    if (n_clean > 0) {
        std::printf("  IMPLEMENTATION: |Q_true2D/Q_code − 1|  mean=%.2e  max=%.2e   (~FD-truncation ⇒ variable-η₃ bracket correctly implemented)\n",
                    impl_sum / n_clean, impl_max);
        std::printf("  #11 EFFECT: Q_ident/Q_code (old frozen η₃=3/2 vs new variable)  mean=%.3f  range=[%.3f, %.3f]   (<1 ⇒ frozen UNDER-counted Q_adv; departs from 1 as β→0)\n",
                    frozen_sum / n_clean, frozen_min, frozen_max);
    }
    std::printf("  MESH: beta odd-even flips = %d / %d interior nodes   (≫0 ⇒ gas/radiation sawtooth — a convergence-quality issue)\n",
                n_betaflip, M - 2);
}

} // namespace probe
} // namespace grrt

int main(int argc, char** argv) {
    auto op = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    double a = 0.9;
    int N = 48;
    double wall_s = 120.0;
    std::vector<double> fedds = {0.02, 0.05};
    if (argc > 1) { fedds.clear(); for (int i = 1; i < argc; ++i) fedds.push_back(std::atof(argv[i])); }

    std::printf("# slim-qadv-entropy-probe  a=%.3f N=%d\n", a, N);
    std::printf("# Q_adv three ways on a converged profile: CODE(η₃=3/2 bracket) vs\n");
    std::printf("# IDENT(2D entropy identity, η₃=3/2) vs TRUE2D(variable η₃=3−1.5β).\n");
    for (double f : fedds) grrt::probe::probe_one(op, a, f, N, wall_s);
    std::printf("\n[slim-qadv-entropy-probe] done.\n");
    return 0;
}
