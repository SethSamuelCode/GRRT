// ===========================================================================
// SLIM COUPLED-COLUMN  n_z  RESOLUTION PROBE  (DIAGNOSTIC — DELETABLE)
// ---------------------------------------------------------------------------
// DECISIVE resolution check for the Σ0-SATURATION finding.
//
// A prior diagnostic (slim-coupled-hard-probe) measured the vertical column's
// surface-density ceiling Σ0_max ≈ 8.6e3 g/cm² at a high-Σ inner-disk node and
// concluded the column "can't hold the disk's demanded Σ≈5e4".  BUT that was
// measured at the relax bring-up default column resolution n_z=24
// (ColumnOpts::n_z), whereas the PASSING unit tests (test_column_coupled.cpp)
// validate the column at n_z=96.  Σ0 = 2∫ρ dz is a vertical quadrature, so a
// coarse n_z may UNDER-RESOLVE a sharp / optically-thick high-Σ profile and
// ARTIFICIALLY CAP Σ0.
//
// THE QUESTION: is the Σ0 saturation a RESOLUTION ARTIFACT (lifts at
// n_z=96/200, crosses Σ_target) or PHYSICAL (a real column-capacity limit that
// persists across n_z=24→400)?
//
// At the SAME failing node geometry the hard probe used (a=0.9, f_Edd=0.01;
// build_thin_disk_seed; node with Σ≈5e4, shear≈3894, omega_z≈2722), for
// n_z ∈ {24, 48, 96, 200, 400}:
//   (1) Σ0 CAPACITY SWEEP: sweep T_eff over 1e5..5e7 (~30 pts) at f_adv=0, call
//       the BASE solver solve_column_bvp, record the MAX Σ0 (the ceiling) and
//       T_eff@peak.  Table: n_z | maxΣ0 | T_eff@peak | converged_count.
//   (2) FEASIBILITY: does maxΣ0 reach Σ_target≈5e4 at each n_z?
//   (3) DIRECT COUPLED SOLVE: solve_column_coupled at (Σ_target, repr. Tc) for
//       each n_z — does the node become FEASIBLE at higher n_z?
//   (4) QUADRATURE SANITY: for one T_eff near the peak, Σ0 vs n_z (does Σ0
//       converge Richardson-like to a finite limit, or still climbing at 400?).
//
// REUSE: include-the-.cpp — opacity + disk_column_bvp + disk_column_coupled +
//        slim_disk_radial + slim_disk_coupled, in that order (mirrors the hard
//        probe) so the file-static seed builders / solve_column_coupled and the
//        radial helpers (build_thin_disk_seed, shear_cgs, omega_perp_cgs,
//        one_zone_closure) are reachable.  Does NOT link grrt.
//
// Build:  cmake --build build --config Release --target slim-coupled-nz-probe
// Run:    build/Release/slim-coupled-nz-probe.exe
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

    // EXACT base operating point of the hard probe.
    SlimDiskInputs in_base{};
    in_base.mass = 1.0; in_base.spin = 0.9; in_base.alpha = 0.1; in_base.r_g = R_G_10MSUN;
    in_base.r_out = 50.0; in_base.n_nodes = 18; in_base.tol = 1e-8;
    in_base.r_in = 0.5 * slim_detail::isco_prograde(in_base.mass, in_base.spin);

    ColumnOpts copt;   // bring-up default: n_z=24, 300 iters, tol 1e-8
    double f_Edd = 0.01;
    if (const char* e = std::getenv("F_EDD")) f_Edd = std::atof(e);
    SlimDiskInputs in = in_base;
    in.mdot = mdot_from_fEdd(in, f_Edd);

    std::printf("# =====================================================================\n");
    std::printf("# slim-coupled-nz-probe : is the Σ0 ceiling a RESOLUTION ARTIFACT (n_z)?\n");
    std::printf("#   a=%.3f alpha=%.2f f_Edd=%.4g mdot=%.4e g/s  N=%d\n",
                in.spin, in.alpha, f_Edd, in.mdot, in_base.n_nodes);
    std::printf("#   relax bring-up n_z default = %d ; unit tests validate at n_z=96\n", copt.n_z);
    std::printf("# =====================================================================\n\n");

    // Thin-disk seed + node grid (mirrors the hard/walk/audit probes EXACTLY).
    std::vector<double> Uthin = build_thin_disk_seed(in, op);
    const int N = std::max(in.n_nodes, 4);
    const double r_s = Uthin[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i]  = std::exp(lr0 + (lr1 - lr0) * t);
        Om[i] = slim_detail::omega_from_ell(in.mass, in.spin, r[i], Uthin[4 * i + 2]);
    }

    // Representative HIGH-Σ node: node 1 (Σ~5e4) — the canonical failing case the
    // hard probe used.  Allow override via env HARD_NODE.
    int node = 1;
    if (const char* e = std::getenv("HARD_NODE")) { node = std::atoi(e); node = std::clamp(node, 0, N - 1); }
    const int i = node;
    const int j = (i + 1 < N) ? i + 1 : i - 1;

    // FIXED node geometry — IDENTICAL derivation to the hard probe.
    const double shear_i  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
    const double omegaz_i = omega_perp_cgs(in, r[i]);
    const double Sigma_i  = std::max(Uthin[4 * i + 0], 1e2);   // Σ_target (demanded)
    const double Tc_seed  = std::max(Uthin[4 * i + 3], 1.0);   // grey thin-disk Tc
    const slim_detail::OneZoneState oz =
        slim_detail::one_zone_closure(Sigma_i, Tc_seed, r[i], in, op);
    const double rho_mid_guess = std::max(oz.rho_mid, 1e-30);

    std::printf("### FIXED FAILING-NODE GEOMETRY (node %d) ###\n", i);
    std::printf("    r=%.5f M  Sigma_target=%.6e  Tc_seed(grey)=%.6e\n", r[i], Sigma_i, Tc_seed);
    std::printf("    shear=%.6e  omega_z=%.6e  alpha=%.3f  rho_mid_guess=%.6e\n\n",
                shear_i, omegaz_i, in.alpha, rho_mid_guess);

    // n_z ladder.
    const std::vector<int> nz_list = {24, 48, 96, 200, 400};

    // Base ColumnInputs at a given (n_z, T_eff, f_adv) for THIS node's geometry.
    auto make_base = [&](int nz, double Te, double fa) {
        ColumnInputs b{};
        b.T_eff = Te; b.shear = std::max(shear_i, 1e-300);
        b.omega_z = std::max(omegaz_i, 1e-300); b.alpha = in.alpha; b.f_adv = fa;
        b.rho_mid_guess = rho_mid_guess; b.n_nodes = nz; b.max_iters = copt.max_iter; b.tol = copt.tol;
        return b;
    };

    // T_eff sweep grid (geometric, 1e5 .. 5e7, ~30 points). f_adv=0 per the spec.
    constexpr int NT = 30;
    auto Te_grid = [&](int k) { return 1e5 * std::pow(5e7 / 1e5, double(k) / double(NT - 1)); };

    // ======================================================================
    // (1)+(2) Σ0 CAPACITY SWEEP + FEASIBILITY: per-n_z ceiling table.
    // ======================================================================
    std::printf("### (1)+(2) Σ0 CAPACITY CEILING vs n_z  (f_adv=0, T_eff∈[1e5,5e7], %d pts) ###\n", NT);
    std::printf("    Σ_target (demanded) = %.4e g/cm²\n", Sigma_i);
    std::printf("    %-6s %-14s %-14s %-12s %-12s %-10s\n",
                "n_z", "maxΣ0", "T_eff@peak", "conv/total", "Σ0/Σ_tgt", "reaches?");
    // Record per-n_z ceiling + T_eff@peak for item (4).
    std::vector<double> ceil_nz(nz_list.size(), -1.0), Tepk_nz(nz_list.size(), 0.0);
    for (size_t ni = 0; ni < nz_list.size(); ++ni) {
        const int nz = nz_list[ni];
        double best = -1.0, bestTe = 0.0; int nconv = 0;
        for (int k = 0; k < NT; ++k) {
            const double Te = Te_grid(k);
            ColumnInputs b = make_base(nz, Te, 0.0);
            ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
            if (s.converged) {
                ++nconv;
                if (s.Sigma0 > best) { best = s.Sigma0; bestTe = Te; }
            }
        }
        ceil_nz[ni] = best; Tepk_nz[ni] = bestTe;
        const double ratio = (best > 0.0) ? best / Sigma_i : 0.0;
        std::printf("    %-6d %-14.6e %-14.4e %-3d/%-8d %-12.4f %-10s\n",
                    nz, best, bestTe, nconv, NT, ratio,
                    (best >= Sigma_i) ? "YES" : "no");
    }
    {
        // Trend verdict.
        const double c0 = ceil_nz.front(), cL = ceil_nz.back();
        const double growth = (c0 > 0.0) ? cL / c0 : 0.0;
        std::printf("    => ceiling n_z=%d -> n_z=%d : %.4e -> %.4e  (×%.3f)\n",
                    nz_list.front(), nz_list.back(), c0, cL, growth);
        bool any_reaches = false;
        for (size_t ni = 0; ni < nz_list.size(); ++ni)
            if (ceil_nz[ni] >= Sigma_i) { any_reaches = true; break; }
        std::printf("    => any n_z reaches Σ_target=%.4e ?  %s\n\n", Sigma_i,
                    any_reaches ? "YES (artifact — ceiling crosses target)"
                                : "NO  (ceiling stays below target at every n_z)");
    }

    // ======================================================================
    // (3) DIRECT COUPLED SOLVE: solve_column_coupled at (Σ_target, Tc) per n_z.
    //     Does the node become FEASIBLE (coupled Newton converges) at higher n_z?
    //     We try BOTH the grey seed Tc AND the f_adv=0 manifold T(0) (= what the
    //     relax actually pins each node at, via build_coupled_seed) so a Tc-branch
    //     mismatch can't masquerade as infeasibility.
    // ======================================================================
    std::printf("### (3) DIRECT solve_column_coupled at Σ_target vs n_z ###\n");
    std::printf("    %-6s %-10s %-14s %-10s %-14s %-12s\n",
                "n_z", "seed_ok", "Tc_manifold", "conv?", "f_adv", "T_eff");
    for (size_t ni = 0; ni < nz_list.size(); ++ni) {
        const int nz = nz_list[ni];
        ColumnCoupledInputs ci{};
        ci.Sigma_target = Sigma_i; ci.Tc = Tc_seed;
        ci.shear = std::max(shear_i, 1e-300); ci.omega_z = std::max(omegaz_i, 1e-300);
        ci.alpha = in.alpha; ci.rho_mid_guess = rho_mid_guess;
        ci.n_nodes = nz; ci.max_iters = copt.max_iter; ci.tol = copt.tol; ci.Teff_guess = 0.0;

        // The f_adv=0 manifold midplane T(0) = the Tc the relax pins each node at.
        std::vector<double> Uc;
        const bool seed_ok = build_coupled_seed(ci, op, Uc);
        const double Tc_man = seed_ok ? Uc[2] : Tc_seed;

        ColumnCoupledInputs cv = ci;
        if (seed_ok) cv.Tc = Tc_man;
        const ColumnClosure c = solve_column_coupled(cv, op, nullptr);
        std::printf("    %-6d %-10s %-14.6e %-10s %-+14.4e %-12.4e\n",
                    nz, seed_ok ? "Y" : "N", Tc_man,
                    c.converged ? "Y" : "N",
                    c.converged ? c.f_adv : std::nan(""),
                    c.converged ? c.T_eff : std::nan(""));
    }
    std::printf("\n");

    // ======================================================================
    // (4) QUADRATURE SANITY: Σ0 vs n_z at a FIXED T_eff near the ceiling peak,
    //     on a FINER n_z ladder, to see whether Σ0 converges (Richardson-like)
    //     to a finite limit as n_z→∞, or is still climbing at n_z=400 (which
    //     would mean even 400 is too coarse / a genuinely singular profile).
    // ======================================================================
    {
        // Pick the T_eff at the n_z=96 ceiling peak (the tests' resolution) as the
        // representative "near the peak" T_eff; fall back to the coarsest if needed.
        size_t i96 = 0; for (size_t ni = 0; ni < nz_list.size(); ++ni) if (nz_list[ni] == 96) i96 = ni;
        double Te_fixed = (Tepk_nz[i96] > 0.0) ? Tepk_nz[i96] : Tepk_nz.front();
        std::printf("### (4) Σ0(n_z) at FIXED T_eff=%.4e (near ceiling peak), f_adv=0 ###\n", Te_fixed);
        std::printf("    %-6s %-16s %-16s %-12s\n", "n_z", "Σ0", "ΔΣ0 vs prev", "rel-Δ");
        const std::vector<int> fine = {24, 48, 72, 96, 144, 200, 300, 400, 600};
        double prev = -1.0;
        for (int nz : fine) {
            ColumnInputs b = make_base(nz, Te_fixed, 0.0);
            ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
            const double sig = s.converged ? s.Sigma0 : std::nan("");
            if (s.converged && prev > 0.0) {
                const double d = sig - prev, rel = d / prev;
                std::printf("    %-6d %-16.8e %-+16.4e %-+12.3e\n", nz, sig, d, rel);
            } else {
                std::printf("    %-6d %-16.8e %-16s %-12s\n",
                            nz, sig, "(—)", s.converged ? "(—)" : "(fail)");
            }
            if (s.converged) prev = sig;
        }
        std::printf("    (if rel-Δ → 0 as n_z grows, Σ0 is converging; if still large at 400/600,\n");
        std::printf("     even 400 under-resolves — a genuinely sharp/singular profile)\n\n");
    }

    std::printf("DONE\n");
    return 0;
}
