// ===========================================================================
// SLIM  Σ↔V  CHECK PROBE   (DIAGNOSTIC — DELETABLE)
// ---------------------------------------------------------------------------
// THE RESUME-ACTION CHECK (handoff 2026-06-30 §0): is the demanded inner
// Σ≈5e4 g/cm² PHYSICAL or a thin-disk-SEED ARTIFACT?
//
// Mass conservation (build_thin_disk_seed, slim_disk_radial.cpp:768):
//     Ṁ = -2π Σ Δ^½ (V/√(1-V²)) r_g c           (X ≡ V/√(1-V²))
// At fixed Ṁ, Σ·|V|·Γ = const.  The thin-disk seed DERIVES V from Σ via this
// law, so V_seed is the (highly subsonic, thin-disk) velocity consistent with
// Σ_seed≈5e4.  The vertical column can only hold Σ_cap≈1.3e4 (converged
// ceiling, proven by slim-coupled-nz-probe).  Carrying the SAME Ṁ through the
// smaller Σ_cap REQUIRES a faster inflow:  |V_req| ≈ (Σ_seed/Σ_cap)·|V_seed|.
//
//   * |V_req| < c AND ~transonic (|V_req| ~ c_s, physically reachable by a
//     slim disk's sonic inner flow)  ->  Σ=5e4 is a SEED ARTIFACT.
//     The thin-disk seed's V is too slow near ISCO; a transonic-aware seed
//     gives a higher inner V -> lower inner Σ within the column's capacity.
//     => Option A (transonic-aware seed).
//   * |V_req| -> c (needs |V|≳1, superluminal/forced)  ->  Σ=5e4 is FORCED;
//     the pure-radiative column genuinely cannot hold near-Eddington inner Σ.
//     => Option C (convection #13).
//
// This is the cheap, decisive check.  No full relax: per-node mass-conservation
// inversion (the EXACT code formula) + the column Σ0 ceiling (fresh, n_z=96) +
// the one-zone sound speed.
//
// Build:  cmake --build build --config Release --target slim-sigma-v-probe
// Run:    build/Release/slim-sigma-v-probe.exe
//         F_EDD=0.05 build/Release/slim-sigma-v-probe.exe   (override f_Edd)
//         NZ_CEIL=200 build/Release/slim-sigma-v-probe.exe  (ceiling n_z)
// REUSE: include-the-.cpp — opacity + disk_column_bvp + disk_column_coupled +
//        slim_disk_radial + slim_disk_coupled (mirrors the hard/walk probes),
//        so build_thin_disk_seed, node_mech, one_zone_closure, shear_cgs,
//        omega_perp_cgs, solve_column_bvp are all in scope.  Does NOT link grrt.
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include "../src/slim_disk_radial.cpp"
#include "../src/slim_disk_coupled.cpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numbers>
#include <algorithm>

using namespace grrt;
using namespace grrt::slim_coupled_detail;

static constexpr double R_G_10MSUN = 1.48e6;  // cm (GM/c² for ~10 M_sun)

static double mdot_from_fEdd(const SlimDiskInputs& in, double f_Edd) {
    using namespace constants;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    return f_Edd * Mdot_Edd;
}

int main() {
    std::setbuf(stdout, nullptr);
    using namespace constants;
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    // EXACT base operating point of the walk / hard probes.
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = 0.9; in.alpha = 0.1; in.r_g = R_G_10MSUN;
    in.r_out = 50.0; in.n_nodes = 18; in.tol = 1e-8;
    in.r_in = 0.5 * slim_detail::isco_prograde(in.mass, in.spin);

    double f_Edd = 0.01;
    if (const char* e = std::getenv("F_EDD")) f_Edd = std::atof(e);
    in.mdot = mdot_from_fEdd(in, f_Edd);

    int nz_ceil = 96;
    if (const char* e = std::getenv("NZ_CEIL")) nz_ceil = std::atoi(e);

    const double isco = slim_detail::isco_prograde(in.mass, in.spin);

    std::printf("# =====================================================================\n");
    std::printf("# slim-sigma-v-probe :  Σ↔V check  (is demanded inner Σ physical or a seed artifact?)\n");
    std::printf("#   a=%.3f alpha=%.2f f_Edd=%.4g  mdot=%.4e g/s   ISCO=%.4f M   r_g=%.3e cm\n",
                in.spin, in.alpha, f_Edd, in.mdot, isco, in.r_g);
    std::printf("#   mass conservation:  X = -mdot/(2π Σ Δ^½ r_g c),  V = X/√(1+X²)\n");
    std::printf("#   column Σ0 ceiling computed FRESH at n_z=%d (T_eff×f_adv grid)\n", nz_ceil);
    std::printf("# =====================================================================\n\n");

    // Thin-disk seed + node grid (mirrors the walk/hard/audit probes).
    std::vector<double> Uthin = build_thin_disk_seed(in, op);
    const int N = std::max(in.n_nodes, 4);
    const double r_s = Uthin[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N), ell(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i]   = std::exp(lr0 + (lr1 - lr0) * t);
        ell[i] = Uthin[4 * i + 2];
        Om[i]  = slim_detail::omega_from_ell(in.mass, in.spin, r[i], ell[i]);
    }

    // mass-conservation V from Σ (EXACTLY the build_thin_disk_seed inversion).
    // Returns V in UNITS OF c (the seed stores V dimensionless = v/c), so cs (cm/s)
    // must be divided by c_cgs to compare; v/c_s = (V/c)·c/c_s.
    auto V_from_Sigma = [&](double Sigma, double sqrtDelta) -> double {
        const double denom = 2.0 * std::numbers::pi * Sigma * sqrtDelta * in.r_g * c_cgs;
        if (!(denom > 0.0)) return 0.0;
        const double X = -in.mdot / denom;
        return X / std::sqrt(1.0 + X * X);   // = v/c
    };

    // column Σ0 capacity ceiling at a node geometry (max over a dense T_eff×f_adv grid).
    auto sigma0_ceiling = [&](double sh, double oz_, double rmg) -> double {
        double best = -1.0;
        for (int it = 0; it <= 16; ++it) {
            const double Te = 1e5 * std::pow(1e3, it / 16.0);   // 1e5 .. 1e8
            for (double fa : {2.0, 0.0, -0.5, -0.9}) {
                ColumnInputs b{}; b.T_eff = Te; b.shear = std::max(sh, 1e-300);
                b.omega_z = std::max(oz_, 1e-300); b.alpha = in.alpha; b.f_adv = fa;
                b.rho_mid_guess = rmg; b.n_nodes = nz_ceil; b.max_iters = 300; b.tol = 1e-8;
                ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
                if (s.converged && s.Sigma0 > best) best = s.Sigma0;
            }
        }
        return best;
    };

    std::printf("    %-3s %-8s %-11s %-11s %-10s %-10s %-10s %-9s %-9s %-7s\n",
                "i", "r[M]", "Sig_seed", "Sig_cap", "Vseed/c", "cs/c",
                "Vreq/c", "Vreq/cs", "Sg/cap", "verd");
    int n_inner_over = 0, n_artifact = 0, n_forced = 0;
    for (int i = 0; i < N; ++i) {
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const double sh  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double ozf = omega_perp_cgs(in, r[i]);
        const double Sg  = std::max(Uthin[4 * i + 0], 1e2);   // demanded Σ (seed)
        const double Vsd = Uthin[4 * i + 1];                  // seed V (mass-cons consistent w/ Sg)
        const double Tcs = std::max(Uthin[4 * i + 3], 1.0);

        const NodeMech mech = node_mech(in, r[i], ell[i]);
        const slim_detail::OneZoneState oz =
            slim_detail::one_zone_closure(Sg, Tcs, r[i], in, op);
        const double rmg = std::max(oz.rho_mid, 1e-30);
        const double cs  = oz.c_s;                            // cm/s

        const double cap  = sigma0_ceiling(sh, ozf, rmg);     // column capacity
        const double Vreq = V_from_Sigma(std::max(cap, 1.0), mech.sqrtDelta);

        const bool over = (cap < Sg);                         // node above ceiling
        if (over) ++n_inner_over;

        // V_from_Sigma and the stored seed V are BOTH in units of c (dimensionless v/c).
        const double aVreq = std::abs(Vreq), aVseed = std::abs(Vsd);   // v/c
        const double cs_over_c = cs / c_cgs;                            // v_s/c
        const double Vreq_over_cs = (cs_over_c > 0.0) ? aVreq / cs_over_c : 0.0;  // v/c_s
        // verdict only meaningful where the node is over the ceiling (the blocked set).
        const char* verd = "-";
        if (over) {
            // physical if the required inflow stays comfortably sub-light (and, a
            // fortiori, sub/transonic): there is no obstruction to carrying Ṁ at Σ_cap.
            const bool sublight = (aVreq < 0.5);   // v/c < 0.5
            if (sublight) { verd = "ARTI"; ++n_artifact; }
            else          { verd = "FORC"; ++n_forced; }
        }

        std::printf("    %-3d %-8.4f %-11.3e %-11.3e %-10.3e %-10.3e %-10.3e %-9.3e %-9.2f %-7s\n",
                    i, r[i], Sg, cap, aVseed, cs_over_c,
                    aVreq, Vreq_over_cs, Sg / std::max(cap, 1.0), verd);
    }

    std::printf("\n");
    std::printf("    nodes above column ceiling (the blocked set): %d/%d\n", n_inner_over, N);
    std::printf("    of those:  ARTIFACT (V_req sub-light, transonic-reachable) = %d   FORCED (V_req->c) = %d\n",
                n_artifact, n_forced);
    std::printf("\n");
    if (n_inner_over == 0) {
        std::printf("    VERDICT: no node above ceiling at this f_Edd — not the blocked regime.\n");
    } else if (n_forced == 0) {
        std::printf("    VERDICT: SEED ARTIFACT.  Every blocked node's Σ_cap is feasible at a\n");
        std::printf("             sub-light (transonic-reachable) inflow V.  The thin-disk seed's\n");
        std::printf("             V is too slow near ISCO -> too-high Σ.  => Option A (transonic-aware seed).\n");
    } else {
        std::printf("    VERDICT: FORCED LIMIT at %d node(s).  Σ_cap there needs |V|->c — the\n", n_forced);
        std::printf("             pure-radiative column genuinely cannot hold the near-Eddington inner Σ.\n");
        std::printf("             => Option C (convection #13).\n");
    }
    std::printf("DONE\n");
    return 0;
}
