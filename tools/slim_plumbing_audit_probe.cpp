// ===========================================================================
// SLIM COUPLED-COLUMN PLUMBING CONSISTENCY AUDIT PROBE  (DELETABLE)
// ---------------------------------------------------------------------------
// PURPOSE (audit only — does NOT fix anything): determine whether the two paths
// that both compute a node's column "manifold T_c" feed the column the SAME
// inputs and use the SAME manifold definition/resolution.
//
//   Path A — the calibration (calibrate_seed_to_manifold in
//            tools/slim_coupled_walk_probe.cpp): build a ColumnCoupledInputs and
//            try build_coupled_seed_2d FIRST, else build_coupled_seed; set
//            U[4i+3] = Uc[2].
//   Path B — the relax (eval_node_coupled in src/slim_disk_coupled.cpp):
//            build a ColumnCoupledInputs and call solve_column_coupled.
//
// This TU replicates BOTH input-construction sites VERBATIM at the SAME node and
// prints the inputs + the seed-builder outcome + the round-trip verdict, with NO
// full radial relax.  FAST: single f_Edd, a handful of nodes, column solves only.
//
// REUSE: same include-the-.cpp order as slim_coupled_walk_probe.cpp so all the
// radial solver's TU-local helpers + slim_coupled_detail machinery are in scope.
//
// Build:  cmake --build build --config Release --target slim-plumbing-audit-probe
// Run:    build/Release/slim-plumbing-audit-probe.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include "../src/slim_disk_radial.cpp"
#include "../src/slim_disk_coupled.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <numbers>
#include <algorithm>

using namespace grrt;
using namespace grrt::slim_coupled_detail;

static constexpr double R_G_10MSUN = 1.48e6;  // cm (GM/c² for ~10 M_sun)

// f_Edd -> Mdot [g/s], SAME convention as the coupled probes.
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

    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    // EXACT base operating point of the walk probe.
    SlimDiskInputs in_base{};
    in_base.mass = 1.0; in_base.spin = 0.9; in_base.alpha = 0.1; in_base.r_g = R_G_10MSUN;
    in_base.r_out = 50.0; in_base.n_nodes = 18; in_base.tol = 1e-8;
    in_base.r_in = 0.5 * slim_detail::isco_prograde(in_base.mass, in_base.spin);

    ColumnOpts copt;   // n_z=24, 300 iters, tol 1e-8

    const double f_Edd = 0.01;
    SlimDiskInputs in = in_base;
    in.mdot = mdot_from_fEdd(in, f_Edd);

    std::printf("# =====================================================================\n");
    std::printf("# slim-plumbing-audit-probe : Path A (calibration) vs Path B (relax)\n");
    std::printf("#   a=%.3f alpha=%.2f f_Edd=%.4g mdot=%.4e g/s  N=%d  n_z(copt)=%d\n",
                in.spin, in.alpha, f_Edd, in.mdot, in_base.n_nodes, copt.n_z);
    std::printf("# =====================================================================\n\n");

    // Build the SAME cold thin-disk seed both paths start from.
    std::vector<double> U = build_thin_disk_seed(in, op);
    const int N = std::max(in.n_nodes, 4);
    const double r_s = U[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i]  = std::exp(lr0 + (lr1 - lr0) * t);
        Om[i] = slim_detail::omega_from_ell(in.mass, in.spin, r[i], U[4 * i + 2]);
    }
    std::printf("# seed: r_s=%.5f  r_out=%.2f  (grid r[0]=r_s ... r[N-1]=r_out)\n\n", r_s, in.r_out);

    // Nodes to audit: inner (0), a couple mid, outer — a representative handful.
    const int probe_nodes[] = {0, 1, N / 3, N / 2, N - 1};

    for (int idx = 0; idx < (int)(sizeof(probe_nodes) / sizeof(int)); ++idx) {
        const int i = probe_nodes[idx];
        const int j = (i + 1 < N) ? i + 1 : i - 1;   // SAME neighbour rule both paths use

        // --------- shear / omega_z (both paths use the SAME helpers + node geometry) ----
        const double shear_A  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double omegaz_A = omega_perp_cgs(in, r[i]);
        // Path B (eval_node_coupled) recomputes these identically; recompute to compare.
        const double shear_B  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double omegaz_B = omega_perp_cgs(in, r[i]);

        const double Sigma_i = U[4 * i + 0];
        const double V_i     = U[4 * i + 1];
        const double ell_i   = U[4 * i + 2];
        const double Tc_seed = U[4 * i + 3];   // grey thin-disk seed T_c

        std::printf("=== NODE %d  (r=%.5f M, neighbour j=%d r_j=%.5f) ===\n", i, r[i], j, r[j]);
        std::printf("    seed: Sigma=%.6e  V=%.4e  ell=%.6e  Tc_seed(grey)=%.6e\n",
                    Sigma_i, V_i, ell_i, Tc_seed);
        std::printf("    shear:   A=%.10e  B=%.10e  (match=%s)\n",
                    shear_A, shear_B, (shear_A == shear_B) ? "EXACT" : "DIFFER");
        std::printf("    omega_z: A=%.10e  B=%.10e  (match=%s)\n",
                    omegaz_A, omegaz_B, (omegaz_A == omegaz_B) ? "EXACT" : "DIFFER");

        // ================= Path A: calibration's ColumnCoupledInputs (verbatim) =========
        ColumnCoupledInputs ciA{};
        ciA.Sigma_target  = std::max(U[4 * i + 0], 1e2);
        ciA.Tc            = std::max(U[4 * i + 3], 1.0);   // seeds the T_eff guess only
        ciA.shear         = std::max(shear_A, 1e-300);
        ciA.omega_z       = std::max(omegaz_A, 1e-300);
        ciA.alpha         = in.alpha;
        ciA.rho_mid_guess = 1e-3;
        ciA.n_nodes       = copt.n_z;
        ciA.max_iters     = copt.max_iter;
        ciA.tol           = copt.tol;
        ciA.Teff_guess    = 0.0;

        // ================= Path B: eval_node_coupled's ColumnCoupledInputs (verbatim) ===
        // eval_node_coupled FIRST builds a one-zone closure to get rho_mid for the seed.
        CoupledNode eB_pre;   // mirror the floors eval_node_coupled applies before the OZ call
        eB_pre.Sigma = std::max(Sigma_i, kSigmaFloor);
        eB_pre.Tc    = std::max(Tc_seed, kTFloor);
        const slim_detail::OneZoneState ozB =
            slim_detail::one_zone_closure(eB_pre.Sigma, eB_pre.Tc, r[i], in, op);
        ColumnCoupledInputs ciB{};
        ciB.Sigma_target  = eB_pre.Sigma;                       // = max(Sigma, kSigmaFloor)
        ciB.Tc            = eB_pre.Tc;                           // = max(Tc, kTFloor)
        ciB.shear         = std::max(shear_B, 1e-300);
        ciB.omega_z       = std::max(omegaz_B, 1e-300);
        ciB.alpha         = in.alpha;
        ciB.rho_mid_guess = std::max(ozB.rho_mid, 1e-30);
        ciB.n_nodes       = copt.n_z;
        ciB.max_iters     = copt.max_iter;
        ciB.tol           = copt.tol;
        ciB.Teff_guess    = 0.0;

        // -------- Check 4: field-by-field comparison of ciA vs ciB --------
        std::printf("    --- Check 4: ColumnCoupledInputs field comparison (A vs B) ---\n");
        auto cmpf = [&](const char* name, double a, double b) {
            std::printf("      %-14s A=%.6e  B=%.6e  %s\n",
                        name, a, b, (a == b) ? "match" : "*** DIFFER ***");
        };
        cmpf("Sigma_target", ciA.Sigma_target, ciB.Sigma_target);
        cmpf("Tc",           ciA.Tc,           ciB.Tc);
        cmpf("shear",        ciA.shear,        ciB.shear);
        cmpf("omega_z",      ciA.omega_z,      ciB.omega_z);
        cmpf("alpha",        ciA.alpha,        ciB.alpha);
        cmpf("rho_mid_guess",ciA.rho_mid_guess,ciB.rho_mid_guess);
        std::printf("      %-14s A=%d  B=%d  %s   <-- Check 2 (resolution)\n",
                    "n_nodes", ciA.n_nodes, ciB.n_nodes,
                    (ciA.n_nodes == ciB.n_nodes) ? "match" : "*** DIFFER ***");
        std::printf("      %-14s A=%d  B=%d\n", "max_iters", ciA.max_iters, ciB.max_iters);
        cmpf("tol",          ciA.tol,          ciB.tol);
        cmpf("Teff_guess",   ciA.Teff_guess,   ciB.Teff_guess);

        // -------- Check 1: which seed builder ACTUALLY ran in Path A, and what is Uc[2]? --
        std::printf("    --- Check 1: Path A seed builder (2-D tried FIRST, else 1-D) ---\n");
        std::vector<double> Uc2d, Uc1d;
        const bool ok2d = build_coupled_seed_2d(ciA, op, Uc2d);
        const bool ok1d_for_compare = build_coupled_seed(ciA, op, Uc1d);
        // Replicate the EXACT short-circuit the calibration uses:
        //     build_coupled_seed_2d(ci,op,Uc) || build_coupled_seed(ci,op,Uc)
        std::vector<double> UcA;
        const char* which = nullptr;
        if (ok2d) { UcA = Uc2d; which = "2-D (build_coupled_seed_2d)"; }
        else if (ok1d_for_compare) { UcA = Uc1d; which = "1-D (build_coupled_seed)"; }
        else { which = "NEITHER (calibration would skip this node)"; }

        if (!UcA.empty()) {
            const int nz = copt.n_z;
            const double TcA_assigned = UcA[2];          // what calibration writes to U[4i+3]
            const double Teff_A       = UcA[4 * nz + 2];
            const double fadv_A       = UcA[4 * nz + 3];
            std::printf("      ran: %s\n", which);
            std::printf("      Uc[2] (= midplane T(0), assigned to U[4i+3]) = %.6e\n", TcA_assigned);
            std::printf("      Uc T_eff = %.6e   Uc f_adv = %+.6e\n", Teff_A, fadv_A);
            std::printf("      ratio Uc[2]/ciA.Tc(grey input) = %.6f  -> %s\n",
                        TcA_assigned / ciA.Tc,
                        (std::abs(TcA_assigned / ciA.Tc - 1.0) < 1e-3)
                            ? "Uc[2] == INPUT grey Tc (NO calibration happened, f_adv back-solved)"
                            : "Uc[2] differs from input (1-D manifold Tc OUTPUT)");
        } else {
            std::printf("      ran: %s\n", which);
        }
        // Also independently report the 1-D manifold Tc (definitive reference value).
        double Tc_manifold_1d = std::nan("");
        if (ok1d_for_compare) {
            const int nz = copt.n_z;
            Tc_manifold_1d = Uc1d[2];
            std::printf("      [reference] 1-D (f_adv=0) manifold Tc = Uc1d[2] = %.6e"
                        "  (Teff=%.4e f_adv=%+.3e)\n",
                        Tc_manifold_1d, Uc1d[4 * nz + 2], Uc1d[4 * nz + 3]);
        } else {
            std::printf("      [reference] 1-D seed did NOT converge at this node\n");
        }

        // -------- Check 5a: is the 1-D manifold Tc a FIXED POINT of the 1-D map? --------
        // Re-run build_coupled_seed with ci.Tc already = the manifold value. The 1-D map
        // pins f_adv=0 and lets T(0) float; if the manifold is self-consistent the returned
        // Uc[2] should equal the input manifold Tc (the grey-only Teff guess shift aside).
        if (ok1d_for_compare && std::isfinite(Tc_manifold_1d)) {
            ColumnCoupledInputs ciFP = ciA;
            ciFP.Tc = Tc_manifold_1d;
            std::vector<double> UcFP;
            if (build_coupled_seed(ciFP, op, UcFP)) {
                std::printf("    --- Check 5a: 1-D fixed-point: build_coupled_seed(Tc=manifold) -> Uc[2]=%.6e"
                            "  (ratio to manifold=%.6f)\n",
                            UcFP[2], UcFP[2] / Tc_manifold_1d);
            } else {
                std::printf("    --- Check 5a: 1-D fixed-point re-solve FAILED\n");
            }
        }

        // -------- Check 5: manifold round-trip (THE definitive consistency test) --------
        // Take the 1-D manifold Tc, feed Path B's EXACT solve_column_coupled call at that
        // Tc (Σ + shear + omega_z + n_z all = Path B), report converged? f_adv? F?
        std::printf("    --- Check 5: round-trip solve_column_coupled at 1-D manifold Tc ---\n");
        if (ok1d_for_compare && std::isfinite(Tc_manifold_1d)) {
            ColumnCoupledInputs ciRT = ciB;     // Path B's exact field set ...
            ciRT.Tc = Tc_manifold_1d;           // ... but Tc pinned to the 1-D manifold value
            ColumnClosure cRT = solve_column_coupled(ciRT, op, /*warm=*/nullptr);
            std::printf("      solve_column_coupled(Sigma=%.4e, Tc_manifold=%.6e, n_z=%d): converged=%d\n",
                        ciRT.Sigma_target, ciRT.Tc, ciRT.n_nodes, (int)cRT.converged);
            if (cRT.converged) {
                std::printf("      -> f_adv=%+.6e   F=%.6e   z0=%.4e   T_eff=%.4e   Tc(out=T(0))=%.6e\n",
                            cRT.f_adv, cRT.F, cRT.z0, cRT.T_eff,
                            cRT.sol.T.empty() ? std::nan("") : cRT.sol.T.front());
                std::printf("      VERDICT(node %d): %s\n", i,
                            (std::abs(cRT.f_adv) < 1e-2)
                                ? "f_adv≈0 -> paths AGREE / manifold consistent"
                                : "f_adv NOT ≈0 -> paths DISAGREE at the 1-D manifold Tc");
            } else {
                std::printf("      VERDICT(node %d): round-trip FAILED to converge at the 1-D manifold Tc\n", i);
            }
        } else {
            std::printf("      (skipped: no 1-D manifold Tc at this node)\n");
        }

        std::printf("\n");
    }

    // ---- Also: directly reproduce the calibration's per-node assignment + then the
    //      relax's eval_node_coupled at the SAME post-calibration state, to show the
    //      manifold each path "believes in" at one representative node. ----
    std::printf("### SUMMARY TABLE (per audited node) ###\n");
    std::printf("  %-4s %-12s %-14s %-14s %-14s %-10s\n",
                "i", "Tc_seed", "A:Uc[2](written)", "ref:1D_manifold", "B:roundtrip_f_adv", "n_z A/B");
    for (int idx = 0; idx < (int)(sizeof(probe_nodes) / sizeof(int)); ++idx) {
        const int i = probe_nodes[idx];
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const double shear_i  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double omegaz_i = omega_perp_cgs(in, r[i]);

        ColumnCoupledInputs ciA{};
        ciA.Sigma_target = std::max(U[4*i+0],1e2); ciA.Tc = std::max(U[4*i+3],1.0);
        ciA.shear = std::max(shear_i,1e-300); ciA.omega_z = std::max(omegaz_i,1e-300);
        ciA.alpha = in.alpha; ciA.rho_mid_guess = 1e-3;
        ciA.n_nodes = copt.n_z; ciA.max_iters = copt.max_iter; ciA.tol = copt.tol; ciA.Teff_guess = 0.0;

        std::vector<double> Uc2d, Uc1d;
        const bool ok2d = build_coupled_seed_2d(ciA, op, Uc2d);
        const bool ok1d = build_coupled_seed(ciA, op, Uc1d);
        const double written = ok2d ? Uc2d[2] : (ok1d ? Uc1d[2] : std::nan(""));
        const double man1d   = ok1d ? Uc1d[2] : std::nan("");

        // round-trip f_adv at the 1-D manifold (Path B inputs)
        double rt_fadv = std::nan("");
        if (ok1d) {
            CoupledNode eB; eB.Sigma = std::max(U[4*i+0], kSigmaFloor); eB.Tc = std::max(U[4*i+3], kTFloor);
            const slim_detail::OneZoneState ozB = slim_detail::one_zone_closure(eB.Sigma, eB.Tc, r[i], in, op);
            ColumnCoupledInputs ciRT{};
            ciRT.Sigma_target = eB.Sigma; ciRT.Tc = man1d;
            ciRT.shear = std::max(shear_i,1e-300); ciRT.omega_z = std::max(omegaz_i,1e-300);
            ciRT.alpha = in.alpha; ciRT.rho_mid_guess = std::max(ozB.rho_mid,1e-30);
            ciRT.n_nodes = copt.n_z; ciRT.max_iters = copt.max_iter; ciRT.tol = copt.tol; ciRT.Teff_guess = 0.0;
            ColumnClosure cRT = solve_column_coupled(ciRT, op, nullptr);
            if (cRT.converged) rt_fadv = cRT.f_adv;
        }
        std::printf("  %-4d %-12.4e %-14.4e %-14.4e %-+14.4e %d/%d\n",
                    i, U[4*i+3], written, man1d, rt_fadv, ciA.n_nodes, copt.n_z);
    }

    std::printf("\nDONE\n");
    return 0;
}
