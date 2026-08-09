// ===========================================================================
// S-CURVE / MULTI-ROOT probe  (DELETABLE, READ-ONLY diagnostic)
// ---------------------------------------------------------------------------
// QUESTION: the coupled slim-disk relax stalls with ~97% of its residual in the
// angular-momentum (Group-2) rows, concentrated at radial node 9, whose midplane
// T_c is an isolated ~2.3x spike above BOTH neighbours (node 8: 7.48e6, node 9:
// 1.705e7, node 10: 7.19e6).  HYPOTHESIS: this is the classic slim-disk S-CURVE —
// at fixed (r, Sigma) the vertical/thermal structure admits MULTIPLE equilibria
// (cool gas-pressure branch / hot radiation-pressure branch / advective branch),
// and the per-node bring-up's first-to-converge multi-start silently lands
// different nodes on different branches.
//
// WHAT THIS PROBE DOES (three stages, all at the LIVE checkpoint's operating point):
//   (A) BRANCH SCAN.  At each node's exact reconstructed ColumnCoupledInputs, scan
//       the base column's T_eff over a wide log bracket at fixed f_adv and record
//       Sigma0(T_eff).  The production seed builders (build_coupled_seed /
//       build_coupled_seed_advective) root-find EXACTLY this map by secant; every
//       sign change of Sigma0(T_eff) - Sigma_target is a DISTINCT f_adv-manifold
//       branch (a distinct converged column at the SAME (r, Sigma)).  Each root is
//       refined by bisection and its full thermodynamic state reported.
//   (B) PRODUCTION SEED SWEEP.  Sweep the seed midplane temperature Tc_seed
//       logarithmically and run the PRODUCTION seed path
//       (build_coupled_seed -> build_coupled_seed_advective, exactly as
//       calibrate_seed_to_manifold / build_transonic_coupled_seed do), recording
//       which branch each seed lands in.  This is what actually decides a node's
//       T_c in the relax.
//   (C) FULL COUPLED CLOSURE at each distinct root: solve_column_coupled at
//       (Sigma, T_c_root) exactly as eval_node_coupled calls it, reporting the
//       quantity the stalled Group-2 row actually consumes:
//           P = 2 * one_zone_closure(Sigma,T_c).p_mid * z0      [erg/cm^2]
//
// The node's ColumnCoupledInputs are reconstructed by MIRRORING eval_node_coupled
// (src/slim_disk_coupled.cpp) field-for-field: same shear_cgs / omega_perp_cgs /
// one_zone_closure(rho_mid_guess) / alpha / n_nodes=copt.n_z / Teff_guess=0.
//
// READ-ONLY: no solver source is modified; this file only CONSUMES them.
//
// Build:  cmake --build build --config Release --target slim-multiroot-probe
// Run:    build/Release/slim-multiroot-probe.exe 2>report.txt 1>NUL
//         (solve_column_coupled prints many unconditional lines to STDOUT, so the
//          probe report goes to STDERR.)
// REUSE: include-the-.cpp — opacity + disk_column_bvp + disk_column_coupled +
//        slim_disk_radial + slim_disk_coupled, in that order (mirrors the walk /
//        omp-gate probes), so the TU-local helpers are in scope here.
//
// ENV KNOBS (all optional; defaults are the production operating point):
//   MRP_CKPT     checkpoint path (required unless the default below exists)
//   MRP_NODES    comma list of node indices          (default "9,8,10")
//   MRP_NZ       column vertical nodes               (default 256 = production)
//   MRP_STAGES   bitmask 1=scan 2=seed-sweep 4=full  (default 7)
//   MRP_SCAN_N   T_eff scan samples per (node,f_adv) (default 48)
//   MRP_TEFF_LO / MRP_TEFF_HI   scan bracket [K]     (default 1e5 .. 3e7)
//   MRP_FADV     comma list of f_adv rungs to scan   (default "0")
//   MRP_SEED_N   Tc_seed sweep samples               (default 24)
//   MRP_TC_LO / MRP_TC_HI       seed bracket [K]     (default 1e6 .. 5e7)
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include "../src/slim_disk_radial.cpp"
#include "../src/slim_disk_coupled.cpp"

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <numbers>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace grrt;
using namespace grrt::slim_coupled_detail;

// ---- report to stderr (stdout is swamped by solve_column_coupled diagnostics) ----
static void G(const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fflush(stderr);
}

// ---- env helpers -----------------------------------------------------------
static std::string env_s(const char* k, const char* dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : std::string(dflt);
}
static double env_d(const char* k, double dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atof(v) : dflt;
}
static int env_i(const char* k, int dflt) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::atoi(v) : dflt;
}
static std::vector<double> parse_list(const std::string& s) {
    std::vector<double> out; std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::atof(tok.c_str()));
    return out;
}

// ---- checkpoint reader -----------------------------------------------------
// Format: '#'-prefixed header lines, then 4N+2 values, one per line.
struct Ckpt {
    std::vector<double> U;
    int N = 0, n_z = 0;
    double mass = 1.0, spin = 0.0, alpha = 0.1, mdot = 0.0, r_g = 0.0,
           r_in = 0.0, r_out = 0.0;
    bool ok = false;
};
static Ckpt read_ckpt(const std::string& path) {
    Ckpt c;
    std::ifstream f(path);
    if (!f) return c;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line[0] == '#') {
            std::istringstream hs(line.substr(1));
            std::string key; double v;
            if (hs >> key >> v) {
                if      (key == "N")       c.N = (int)v;
                else if (key == "n_nodes") c.N = (int)v;
                else if (key == "n_z")     c.n_z = (int)v;
                else if (key == "mass")    c.mass = v;
                else if (key == "spin")    c.spin = v;
                else if (key == "alpha")   c.alpha = v;
                else if (key == "mdot")    c.mdot = v;
                else if (key == "r_g")     c.r_g = v;
                else if (key == "r_in")    c.r_in = v;
                else if (key == "r_out")   c.r_out = v;
            }
            continue;
        }
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        c.U.push_back(std::atof(line.c_str()));
    }
    c.ok = (c.N > 0 && (int)c.U.size() == 4 * c.N + 2);
    return c;
}

// ---- one scan sample -------------------------------------------------------
struct Sample {
    double T_eff = 0.0, f_adv = 0.0;
    bool   ok = false;
    double Sigma0 = 0.0, Tc = 0.0, z0 = 0.0, tau = 0.0;
    double beta_mid = 0.0, beta_int = 0.0, P_col = 0.0;  // P_col = 2*int P dz [erg/cm^2]
    std::vector<double> U;   // packed 4N+2 base state (warm-start carrier for continuation)
};
// Run ONE base column solve at (T_eff, f_adv) through the SAME base_inputs_from()
// mapping the production seed builders use, and pack the diagnostics.
// `warm` (optional) is a packed 4N+2 state from a neighbouring T_eff — the SAME warm-start
// interface solve_column_bvp exposes — so a T_eff MARCH can cross the cold-seed basin edge.
static Sample run_base(const ColumnCoupledInputs& ci, const OpacityLUTs& op,
                       double T_eff, double f_adv,
                       const std::vector<double>* warm = nullptr) {
    Sample s; s.T_eff = T_eff; s.f_adv = f_adv;
    ColumnInputs b = base_inputs_from(ci, T_eff, f_adv);
    const int NZ = b.n_nodes;
    const bool warm_ok = (warm && (int)warm->size() == 4 * NZ + 2);
    ColumnBVPSolution sol = solve_column_bvp(b, op, warm_ok ? warm : nullptr);
    if (!sol.converged || sol.T.empty()) return s;
    s.U.assign((size_t)4 * NZ + 2, 0.0);
    for (int k = 0; k < NZ; ++k) {
        s.U[4*k+0] = sol.P_gas[k]; s.U[4*k+1] = sol.Q[k];
        s.U[4*k+2] = sol.T[k];     s.U[4*k+3] = sol.z[k];
    }
    s.U[4*NZ+0] = sol.z0; s.U[4*NZ+1] = sol.Sigma0;
    s.ok = true;
    s.Sigma0 = sol.Sigma0;
    s.Tc     = sol.T.front();
    s.z0     = sol.z0;
    s.tau    = sol.tau_mid;
    s.beta_mid = (sol.P.front() > 0.0) ? sol.P_gas.front() / sol.P.front() : 1.0;
    double iPg = 0.0, iP = 0.0;
    for (size_t k = 0; k + 1 < sol.z.size(); ++k) {
        const double dz = sol.z[k+1] - sol.z[k];
        iPg += 0.5 * (sol.P_gas[k] + sol.P_gas[k+1]) * dz;
        iP  += 0.5 * (sol.P[k]     + sol.P[k+1])     * dz;
    }
    s.beta_int = (iP > 0.0) ? iPg / iP : 1.0;
    s.P_col    = 2.0 * iP;
    return s;
}

// ---- CONVECTION characterisation of a converged column ---------------------
// Sadowski 2011 puts a large midplane convection zone at r ~ 6-40 M, which brackets
// nodes 8/9/10 (r = 9.7 / 11.7 / 14.0 M).  The vertical solver implements convection via
// MLT (detail_bvp::convective_gradient, formulas ref §24), so a RADIATIVE root and a
// CONVECTIVE root are a natural pair of distinct equilibria.  For a converged column,
// re-evaluate the SAME convective_gradient the residual calls at every z node and report
// where the MLT branch is active, plus grad_rad / grad_ad / grad_actual and beta.
//
// U is the packed base state [Pg,Q,T,z] x NZ (+ z0, Sigma0); everything below is derived
// with the SAME helpers the residual uses (rho_from_gas, p_total, kappa_total), so the
// verdict is the solver's own, not a re-derivation.
struct ConvInfo {
    int    n_conv = 0, NZ = 0;
    double z_lo = 0.0, z_hi = 0.0;      // z extent of the convective region [cm]
    double z0 = 0.0;
    double q_lo = 0.0, q_hi = 0.0;      // column-mass-fraction extent
    double max_excess = -1e300;         // max (grad_rad - grad_ad)
    double beta_mid = 0.0;
};
static ConvInfo convection_report(const std::vector<double>& U, int NZ, double omega_z,
                                  const OpacityLUTs& op, const char* ind, bool table) {
    using namespace constants;
    ConvInfo ci{}; ci.NZ = NZ; ci.z0 = U[(size_t)4*NZ+0];
    ci.z_lo = 1e300; ci.z_hi = -1e300; ci.q_lo = 1e300; ci.q_hi = -1e300;
    std::vector<double> nab_rad(NZ, 0.0), nab_ad(NZ, 0.0), nab_act(NZ, 0.0), bet(NZ, 0.0);
    std::vector<int> conv(NZ, 0);
    std::vector<double> lnT(NZ, 0.0), lnP(NZ, 0.0);
    for (int k = 0; k < NZ; ++k) {
        const double Pg = U[(size_t)4*k+0], Q = U[(size_t)4*k+1],
                     T  = U[(size_t)4*k+2], z = U[(size_t)4*k+3];
        const double rho  = std::max(rho_from_gas(Pg, T), RHO_GHOST_FLOOR);
        const double Ptot = p_total(Pg, T);
        const double kR   = kappa_total(op, rho, T);
        bet[k] = (Ptot > 0.0) ? std::clamp(Pg / Ptot, 0.0, 1.0) : 1.0;
        nab_ad[k] = detail_bvp::nabla_ad(bet[k]);
        // the SOLVER's own call (sets `convective` and the operative gradient)
        double nab_out = 0.0; bool is_conv = false;
        detail_bvp::convective_gradient(rho, T, Ptot, Q, kR, z, omega_z, nab_out, is_conv);
        // grad_rad computed explicitly (convective_gradient overwrites nab_out with
        // grad_conv on the unstable branch, so we need grad_rad separately)
        const double dTdz_rad = -3.0*kR*rho*Q/(16.0*sigma_SB*T*T*T);
        const double dPdz     = -rho*omega_z*omega_z*z;
        nab_rad[k] = (z > 0.0 && dPdz < 0.0) ? (Ptot/T)*(dTdz_rad/dPdz) : 0.0;
        conv[k] = is_conv ? 1 : 0;
        lnT[k] = std::log(std::max(T, 1e-300));
        lnP[k] = std::log(std::max(Ptot, 1e-300));
        if (is_conv) {
            ++ci.n_conv;
            const double q = (NZ > 1) ? double(k)/double(NZ-1) : 0.0;
            ci.z_lo = std::min(ci.z_lo, z); ci.z_hi = std::max(ci.z_hi, z);
            ci.q_lo = std::min(ci.q_lo, q); ci.q_hi = std::max(ci.q_hi, q);
        }
        if (z > 0.0) ci.max_excess = std::max(ci.max_excess, nab_rad[k] - nab_ad[k]);
    }
    // actual grad = dlnT/dlnP from the converged profile (central where possible)
    for (int k = 0; k < NZ; ++k) {
        const int a = std::max(k-1, 0), b = std::min(k+1, NZ-1);
        const double dP = lnP[b] - lnP[a];
        nab_act[k] = (std::abs(dP) > 1e-14) ? (lnT[b] - lnT[a]) / dP : 0.0;
    }
    ci.beta_mid = bet[0];
    if (ci.n_conv == 0) { ci.z_lo = ci.z_hi = ci.q_lo = ci.q_hi = 0.0; }

    G("%sCONVECTION: %d/%d nodes convective (%.1f%% of column mass)\n", ind,
      ci.n_conv, NZ, 100.0 * ci.n_conv / std::max(NZ, 1));
    if (ci.n_conv)
        G("%s  convective zone: z = %.4e .. %.4e cm  (z/z0 = %.4f .. %.4f;"
          " mass frac q = %.4f .. %.4f)\n", ind, ci.z_lo, ci.z_hi,
          (ci.z0 > 0.0) ? ci.z_lo/ci.z0 : 0.0, (ci.z0 > 0.0) ? ci.z_hi/ci.z0 : 0.0,
          ci.q_lo, ci.q_hi);
    G("%s  max (grad_rad - grad_ad) over the column = %+.4e  => %s\n", ind, ci.max_excess,
      (ci.max_excess > 0.0) ? "CONVECTIVELY UNSTABLE somewhere" : "radiative everywhere");
    if (table) {
        G("%s  %-6s %-6s %-11s %-9s %-9s %-9s %-9s %-5s\n", ind,
          "k", "q", "z[cm]", "beta", "grad_rad", "grad_ad", "grad_act", "conv");
        const int idx[7] = {0, 1, NZ/8, NZ/4, NZ/2, (3*NZ)/4, NZ-1};
        for (int t = 0; t < 7; ++t) {
            const int k = std::clamp(idx[t], 0, NZ-1);
            G("%s  %-6d %-6.4f %-11.4e %-9.5f %-9.4f %-9.4f %-9.4f %-5s\n", ind, k,
              (NZ > 1) ? double(k)/double(NZ-1) : 0.0, U[(size_t)4*k+3], bet[k],
              nab_rad[k], nab_ad[k], nab_act[k], conv[k] ? "YES" : "-");
        }
    }
    return ci;
}

// ---- per-node reconstruction (MIRRORS eval_node_coupled) -------------------
struct NodeSetup {
    int i = 0;
    double r = 0.0, Sigma = 0.0, V = 0.0, ell = 0.0, Tc_ckpt = 0.0;
    double shear = 0.0, omega_z = 0.0;
};
// Build the node's ColumnCoupledInputs at a GIVEN T_c, byte-for-byte as
// eval_node_coupled does (the only T_c-dependence is ci.Tc and rho_mid_guess).
static ColumnCoupledInputs make_ci(const NodeSetup& ns, const SlimDiskInputs& in,
                                   const OpacityLUTs& op, const ColumnOpts& copt,
                                   double Tc) {
    const double Sig = std::max(ns.Sigma, kSigmaFloor);
    const double Tcl = std::max(Tc, kTFloor);
    const OneZoneState oz = one_zone_closure(Sig, Tcl, ns.r, in, op);
    ColumnCoupledInputs ci{};
    ci.Sigma_target  = Sig;
    ci.Tc            = Tcl;
    ci.shear         = std::max(ns.shear,   1e-300);
    ci.omega_z       = std::max(ns.omega_z, 1e-300);
    ci.alpha         = in.alpha;
    ci.rho_mid_guess = std::max(oz.rho_mid, 1e-30);
    ci.n_nodes       = copt.n_z;
    ci.max_iters     = copt.max_iter;
    ci.tol           = copt.tol;
    ci.Teff_guess    = 0.0;
    return ci;
}

int main() {
    std::setbuf(stdout, nullptr);

    const std::string ck_default =
        "C:/Users/seth/AppData/Local/Temp/claude/C--Users-seth-projects-GRRT/"
        "e5f75106-f9ce-4bb7-a81a-99b32d8ea3ef/scratchpad/ckpt/base_it001.txt";
    const std::string ck_path = env_s("MRP_CKPT", ck_default.c_str());
    const Ckpt ck = read_ckpt(ck_path);

    G("# ==========================================================================\n");
    G("# slim-multiroot-probe : S-curve / multi-root test at the live checkpoint\n");
    G("# checkpoint: %s\n", ck_path.c_str());
#ifdef _OPENMP
    G("# _OPENMP defined; omp_get_max_threads() = %d\n", omp_get_max_threads());
#else
    G("# _OPENMP NOT defined (serial)\n");
#endif
    G("# ==========================================================================\n\n");
    if (!ck.ok) {
        G("FATAL: could not read/parse checkpoint (N=%d, values=%zu, expected %d)\n",
          ck.N, ck.U.size(), 4 * ck.N + 2);
        return 1;
    }

    const int N = ck.N;
    const std::vector<double>& U = ck.U;
    const double ell_in = U[4 * N + 0];
    const double r_s    = U[4 * N + 1];

    SlimDiskInputs in{};
    in.mass = ck.mass; in.spin = ck.spin; in.alpha = ck.alpha; in.r_g = ck.r_g;
    in.r_out = ck.r_out; in.r_in = ck.r_in; in.n_nodes = N; in.mdot = ck.mdot;
    in.tol = 1e-8;

    ColumnOpts copt;
    copt.n_z = env_i("MRP_NZ", (ck.n_z > 0) ? ck.n_z : 256);

    G("Config from checkpoint header:\n");
    G("  a=%.3f  alpha=%.3f  N=%d  n_z=%d  r_out=%.3f  r_g=%.4e  mdot=%.6e\n",
      in.spin, in.alpha, N, copt.n_z, in.r_out, in.r_g, in.mdot);
    G("  ell_in=%.6f  r_s=%.9f\n\n", ell_in, r_s);

    // Grid + Omega(ell) exactly as slim_coupled_residual builds them.
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i]  = std::exp(lr0 + (lr1 - lr0) * t);
        Om[i] = omega_from_ell(in.mass, in.spin, r[i], U[4 * i + 2]);
    }

    auto build_node = [&](int i) {
        NodeSetup ns;
        const int j = (i + 1 < N) ? i + 1 : i - 1;   // SAME neighbour choice as the residual
        ns.i = i; ns.r = r[i];
        ns.Sigma = U[4 * i + 0]; ns.V = U[4 * i + 1];
        ns.ell = U[4 * i + 2];   ns.Tc_ckpt = U[4 * i + 3];
        ns.shear   = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        ns.omega_z = omega_perp_cgs(in, r[i]);
        return ns;
    };

    G("Checkpoint node states (verification against ground truth):\n");
    G("  %-4s %-12s %-12s %-12s %-12s %-12s\n",
      "i", "r [M]", "Sigma", "T_c", "shear[1/s]", "omega_z[1/s]");
    for (int i = 7; i <= 11 && i < N; ++i) {
        NodeSetup ns = build_node(i);
        G("  %-4d %-12.5f %-12.5e %-12.5e %-12.5e %-12.5e\n",
          i, ns.r, ns.Sigma, ns.Tc_ckpt, ns.shear, ns.omega_z);
    }
    G("\n");

    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    const std::vector<double> nodes_d = parse_list(env_s("MRP_NODES", "9,8,10"));
    const std::vector<double> fadvs   = parse_list(env_s("MRP_FADV", "0"));
    const int    stages   = env_i("MRP_STAGES", 7);
    const int    scan_n   = env_i("MRP_SCAN_N", 48);
    const double teff_lo  = env_d("MRP_TEFF_LO", 1e5);
    const double teff_hi  = env_d("MRP_TEFF_HI", 3e7);
    const int    seed_n   = env_i("MRP_SEED_N", 24);
    const double tc_lo    = env_d("MRP_TC_LO", 1e6);
    const double tc_hi    = env_d("MRP_TC_HI", 5e7);

    G("Probe settings: nodes=%s stages=%d scan_n=%d T_eff[%.2e,%.2e] f_adv=%s"
      " seed_n=%d Tc[%.2e,%.2e]\n\n",
      env_s("MRP_NODES", "9,8,10").c_str(), stages, scan_n, teff_lo, teff_hi,
      env_s("MRP_FADV", "0").c_str(), seed_n, tc_lo, tc_hi);

    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    };

    for (double nd : nodes_d) {
        const int i = (int)nd;
        if (i < 0 || i >= N) { G("!! node %d out of range\n", i); continue; }
        NodeSetup ns = build_node(i);
        // The ColumnCoupledInputs the RESIDUAL builds at this node (T_c from the ckpt).
        const ColumnCoupledInputs ci_ck = make_ci(ns, in, op, copt, ns.Tc_ckpt);

        G("\n############################################################################\n");
        G("### NODE %d :  r = %.6f M   Sigma = %.6e g/cm^2   T_c(ckpt) = %.6e K\n", i,
          ns.r, ns.Sigma, ns.Tc_ckpt);
        G("###   reconstructed ColumnCoupledInputs (mirrors eval_node_coupled):\n");
        G("###   Sigma_target=%.6e  Tc=%.6e  shear=%.6e  omega_z=%.6e\n",
          ci_ck.Sigma_target, ci_ck.Tc, ci_ck.shear, ci_ck.omega_z);
        G("###   alpha=%.4f  rho_mid_guess=%.6e  n_nodes=%d  max_iters=%d  tol=%.1e"
          "  Teff_guess=%.1f\n", ci_ck.alpha, ci_ck.rho_mid_guess, ci_ck.n_nodes,
          ci_ck.max_iters, ci_ck.tol, ci_ck.Teff_guess);
        G("###   [production T_eff first guess estimate_Teff_guess = %.6e K]\n",
          estimate_Teff_guess(ci_ck, op));
        G("############################################################################\n");

        // ================= STAGE A : branch scan over T_eff ======================
        std::vector<double> root_Tc, root_Teff, root_fadv, root_z0;
        if (stages & 1) {
            for (double fa : fadvs) {
                G("\n--- (A) BRANCH SCAN: Sigma0(T_eff) at f_adv = %.3g  (target Sigma = %.5e) ---\n",
                  fa, ci_ck.Sigma_target);
                std::vector<Sample> S((size_t)scan_n);
                const double dl = (scan_n > 1) ? (std::log(teff_hi) - std::log(teff_lo)) / (scan_n - 1) : 0.0;
                int done = 0;
                #pragma omp parallel for schedule(dynamic)
                for (int k = 0; k < scan_n; ++k) {
                    const double Te = std::exp(std::log(teff_lo) + dl * k);
                    S[(size_t)k] = run_base(ci_ck, op, Te, fa);
                    #pragma omp critical
                    { ++done; G("    [scan %d/%d  t=%.0fs]\r", done, scan_n, elapsed()); }
                }
                G("\n");
                // --- CONTINUATION MARCH: the cold-start Newton basin collapses for the
                // radiation-pressure-dominated (high T_eff) columns, so a cold scan leaves
                // gaps that could HIDE a fold (the very thing we are testing for).  March
                // the T_eff grid warm-started from each converged neighbour, up and down,
                // until no new sample can be filled.  This is the same warm-start interface
                // solve_column_bvp already exposes (no solver change).
                int filled = 0;
                for (int pass = 0; pass < 4; ++pass) {
                    int gained = 0;
                    for (int k = 1; k < scan_n; ++k)
                        if (!S[(size_t)k].ok && S[(size_t)k-1].ok) {
                            const double Te = std::exp(std::log(teff_lo) + dl * k);
                            Sample t = run_base(ci_ck, op, Te, fa, &S[(size_t)k-1].U);
                            if (t.ok) { S[(size_t)k] = t; ++gained; }
                        }
                    for (int k = scan_n - 2; k >= 0; --k)
                        if (!S[(size_t)k].ok && S[(size_t)k+1].ok) {
                            const double Te = std::exp(std::log(teff_lo) + dl * k);
                            Sample t = run_base(ci_ck, op, Te, fa, &S[(size_t)k+1].U);
                            if (t.ok) { S[(size_t)k] = t; ++gained; }
                        }
                    filled += gained;
                    G("    [continuation pass %d: filled %d gap(s), t=%.0fs]\n", pass, gained, elapsed());
                    if (gained == 0) break;
                }
                G("  (continuation filled %d cold-scan gap(s))\n", filled);
                G("  %-4s %-12s %-8s %-12s %-11s %-12s %-10s %-10s %-11s\n",
                  "k", "T_eff[K]", "conv", "Sigma0", "Sig0/Sig-1", "T_c[K]", "beta_mid",
                  "beta_int", "z0[cm]");
                for (int k = 0; k < scan_n; ++k) {
                    const Sample& s = S[(size_t)k];
                    if (!s.ok) {
                        G("  %-4d %-12.5e %-8s\n", k, s.T_eff, "NO");
                    } else {
                        G("  %-4d %-12.5e %-8s %-12.5e %-11.4f %-12.5e %-10.5f %-10.5f %-11.5e\n",
                          k, s.T_eff, "yes", s.Sigma0, s.Sigma0 / ci_ck.Sigma_target - 1.0,
                          s.Tc, s.beta_mid, s.beta_int, s.z0);
                    }
                }
                // Sign changes of g(T_eff) = Sigma0 - Sigma_target between converged
                // neighbours => distinct branches.  Refine each by bisection.
                G("\n  Root brackets (sign changes of Sigma0 - Sigma_target):\n");
                int nroot = 0;
                for (int k = 0; k + 1 < scan_n; ++k) {
                    const Sample& a = S[(size_t)k];
                    const Sample& b = S[(size_t)k + 1];
                    if (!a.ok || !b.ok) continue;
                    const double ga = a.Sigma0 - ci_ck.Sigma_target;
                    const double gb = b.Sigma0 - ci_ck.Sigma_target;
                    if (!((ga < 0.0 && gb > 0.0) || (ga > 0.0 && gb < 0.0))) continue;
                    ++nroot;
                    double lo = a.T_eff, hi = b.T_eff, glo = ga;
                    Sample best = (std::abs(ga) < std::abs(gb)) ? a : b;
                    for (int it = 0; it < 26; ++it) {
                        const double mid = std::sqrt(lo * hi);   // log-bisection
                        Sample sm = run_base(ci_ck, op, mid, fa, &best.U);   // warm-marched
                        if (!sm.ok) break;
                        const double gm = sm.Sigma0 - ci_ck.Sigma_target;
                        if (std::abs(gm) < std::abs(best.Sigma0 - ci_ck.Sigma_target)) best = sm;
                        if ((glo < 0.0) == (gm < 0.0)) { lo = mid; glo = gm; } else { hi = mid; }
                        if (std::abs(gm) <= 1e-6 * ci_ck.Sigma_target) break;
                    }
                    // Everything the stalled Group-2 row consumes at this root:
                    //   P = 2 * one_zone_closure(Sigma, T_c).p_mid * z0
                    const OneZoneState oz = one_zone_closure(ci_ck.Sigma_target,
                                                             std::max(best.Tc, kTFloor), ns.r, in, op);
                    const double P_resid = 2.0 * oz.p_mid * best.z0;
                    const double beta_oz = (oz.p_mid > 0.0) ? oz.p_gas / oz.p_mid : 1.0;
                    G("   ROOT %d: T_eff=%.6e  Sigma0=%.6e (rel %.2e)\n", nroot, best.T_eff,
                      best.Sigma0, best.Sigma0 / ci_ck.Sigma_target - 1.0);
                    G("           T_c = %.6e K   z0 = %.6e cm   tau_mid = %.4e\n",
                      best.Tc, best.z0, best.tau);
                    G("           beta_mid(column) = %.6f   beta_int(column) = %.6f\n",
                      best.beta_mid, best.beta_int);
                    G("           one_zone at (Sigma,T_c): p_mid=%.6e  beta_oz=%.6f\n",
                      oz.p_mid, beta_oz);
                    G("           >>> P (Group-2 row) = 2*p_mid*z0 = %.6e erg/cm^2\n", P_resid);
                    G("           column's own 2*int(P)dz = %.6e erg/cm^2\n", best.P_col);
                    // Sadowski-2011 convection-zone character of THIS root.
                    if (!best.U.empty())
                        convection_report(best.U, ci_ck.n_nodes, ci_ck.omega_z, op,
                                          "           ", true);
                    root_Tc.push_back(best.Tc); root_Teff.push_back(best.T_eff);
                    root_fadv.push_back(fa);    root_z0.push_back(best.z0);
                }
                if (nroot == 0) G("   (none: Sigma_target lies outside the scanned Sigma0 range)\n");
                G("  => %d distinct branch root(s) at f_adv=%.3g on this bracket.\n", nroot, fa);
            }

            // Pressure ratio between roots (the quantitative S-curve confirmation).
            if (root_Tc.size() >= 2) {
                G("\n  --- Root-to-root ratios at node %d (P = 2*p_mid(Sigma,T_c)*z0) ---\n", i);
                std::vector<double> Pv(root_Tc.size());
                for (size_t a = 0; a < root_Tc.size(); ++a) {
                    const OneZoneState oz = one_zone_closure(ci_ck.Sigma_target,
                                                             std::max(root_Tc[a], kTFloor), ns.r, in, op);
                    Pv[a] = 2.0 * oz.p_mid * root_z0[a];
                }
                for (size_t a = 0; a < Pv.size(); ++a)
                    for (size_t b = a + 1; b < Pv.size(); ++b)
                        G("   root%zu(T_c=%.4e) vs root%zu(T_c=%.4e):  T_c ratio = %.4f   P ratio = %.4f\n",
                          b + 1, root_Tc[b], a + 1, root_Tc[a],
                          root_Tc[b] / root_Tc[a], (Pv[a] > 0.0) ? Pv[b] / Pv[a] : 0.0);
            }
        }

        // ============= STAGE B : production seed sweep over Tc_seed ==============
        if (stages & 2) {
            G("\n--- (B) PRODUCTION SEED SWEEP (build_coupled_seed -> _advective) ---\n");
            G("  %-4s %-12s %-12s %-8s %-12s %-12s %-9s %-12s\n",
              "k", "Tc_seed[K]", "Teff_guess", "builder", "T_c(landed)", "T_eff", "f_adv", "z0[cm]");
            struct SeedRes { double Tc_seed, Teff_guess, Tc_land, Teff, fadv, z0; int builder; };
            std::vector<SeedRes> SR((size_t)seed_n);
            const double dls = (seed_n > 1) ? (std::log(tc_hi) - std::log(tc_lo)) / (seed_n - 1) : 0.0;
            int doneB = 0;
            #pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < seed_n; ++k) {
                const double Tcs = std::exp(std::log(tc_lo) + dls * k);
                ColumnCoupledInputs ci = make_ci(ns, in, op, copt, Tcs);
                SeedRes rres{}; rres.Tc_seed = Tcs; rres.builder = 0;
                rres.Teff_guess = estimate_Teff_guess(ci, op);
                std::vector<double> Uc;
                if (build_coupled_seed(ci, op, Uc))                 rres.builder = 1;
                else if (build_coupled_seed_advective(ci, op, Uc))  rres.builder = 2;
                if (rres.builder) {
                    const int nz = ci.n_nodes;
                    rres.Tc_land = Uc[2];
                    rres.z0      = Uc[4 * nz + 0];
                    rres.Teff    = Uc[4 * nz + 2];
                    rres.fadv    = Uc[4 * nz + 3];
                }
                SR[(size_t)k] = rres;
                #pragma omp critical
                { ++doneB; G("    [seed %d/%d  t=%.0fs]\r", doneB, seed_n, elapsed()); }
            }
            G("\n");
            for (int k = 0; k < seed_n; ++k) {
                const SeedRes& s = SR[(size_t)k];
                const char* bn = (s.builder == 1) ? "1d(f=0)" : (s.builder == 2) ? "adv" : "FAIL";
                if (!s.builder)
                    G("  %-4d %-12.5e %-12.5e %-8s\n", k, s.Tc_seed, s.Teff_guess, bn);
                else
                    G("  %-4d %-12.5e %-12.5e %-8s %-12.5e %-12.5e %-9.3g %-12.5e\n",
                      k, s.Tc_seed, s.Teff_guess, bn, s.Tc_land, s.Teff, s.fadv, s.z0);
            }
            // Cluster the landed T_c within 5% relative.
            G("\n  Basin clustering of landed T_c (5%% relative grouping):\n");
            std::vector<double> reps; std::vector<int> cnt; std::vector<int> firstk;
            for (int k = 0; k < seed_n; ++k) {
                if (!SR[(size_t)k].builder) continue;
                const double t = SR[(size_t)k].Tc_land;
                bool placed = false;
                for (size_t g = 0; g < reps.size(); ++g)
                    if (std::abs(t - reps[g]) <= 0.05 * std::max(t, reps[g])) { ++cnt[g]; placed = true; break; }
                if (!placed) { reps.push_back(t); cnt.push_back(1); firstk.push_back(k); }
            }
            for (size_t g = 0; g < reps.size(); ++g)
                G("   basin %zu : T_c ~ %.6e K   (%d of %d seeds)\n", g + 1, reps[g], cnt[g], seed_n);
            if (reps.empty()) G("   (no seed converged)\n");
            // Where the production ladder's own T_eff multipliers land.
            G("\n  Production multi-start ladder check (Teff x {0.25,0.5,1,2,4} on the\n"
              "  T_eff-secant's OWN fallback ladder {0.5,2,0.25,4,0.1,10}):\n");
            for (double m : {0.25, 0.5, 1.0, 2.0, 4.0}) {
                ColumnCoupledInputs ci = make_ci(ns, in, op, copt, ns.Tc_ckpt);
                ci.Teff_guess = estimate_Teff_guess(ci, op) * m;
                std::vector<double> Uc;
                int bld = 0;
                if (build_coupled_seed(ci, op, Uc)) bld = 1;
                else if (build_coupled_seed_advective(ci, op, Uc)) bld = 2;
                if (bld) G("   Teff x%-5.3g (=%.4e) -> builder %s, T_c = %.6e, f_adv = %.3g\n",
                           m, ci.Teff_guess, (bld == 1) ? "1d" : "adv",
                           Uc[2], Uc[4 * ci.n_nodes + 3]);
                else     G("   Teff x%-5.3g (=%.4e) -> FAIL\n", m, ci.Teff_guess);
            }
        }

        // ========== STAGE C : full coupled closure at each distinct root =========
        if ((stages & 4) && !root_Tc.empty()) {
            G("\n--- (C) FULL solve_column_coupled at each distinct root (production path) ---\n");
            for (size_t a = 0; a < root_Tc.size(); ++a) {
                ColumnCoupledInputs ci = make_ci(ns, in, op, copt, root_Tc[a]);
                const auto t0 = std::chrono::steady_clock::now();
                ColumnClosure c = solve_column_coupled(ci, op, nullptr);
                const double dt = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                G("   root %zu (pinned T_c = %.6e):  converged = %d   [%.1f s]\n",
                  a + 1, ci.Tc, (int)c.converged, dt);
                if (!c.converged) continue;
                const OneZoneState oz = one_zone_closure(ci.Sigma_target, ci.Tc, ns.r, in, op);
                const double P_resid = 2.0 * oz.p_mid * c.z0;
                double iPg = 0.0, iP = 0.0;
                for (size_t k = 0; k + 1 < c.sol.z.size(); ++k) {
                    const double dz = c.sol.z[k+1] - c.sol.z[k];
                    iPg += 0.5 * (c.sol.P_gas[k] + c.sol.P_gas[k+1]) * dz;
                    iP  += 0.5 * (c.sol.P[k]     + c.sol.P[k+1])     * dz;
                }
                G("        F=%.6e  z0=%.6e  eta3=%.6f  eta4=%.6e  T_eff=%.6e  f_adv=%.6f\n",
                  c.F, c.z0, c.eta3, c.eta4, c.T_eff, c.f_adv);
                G("        beta_mid=%.6f  beta_int=%.6f  tau_mid=%.4e\n",
                  (c.sol.P.front() > 0.0) ? c.sol.P_gas.front() / c.sol.P.front() : 1.0,
                  (iP > 0.0) ? iPg / iP : 1.0, c.sol.tau_mid);
                G("        one_zone p_mid=%.6e beta_oz=%.6f\n", oz.p_mid,
                  (oz.p_mid > 0.0) ? oz.p_gas / oz.p_mid : 1.0);
                G("        >>> P (Group-2 row) = 2*p_mid*z0 = %.6e erg/cm^2\n", P_resid);
                {   // convection character of the fully-coupled converged root
                    const int nz = ci.n_nodes;
                    std::vector<double> Up((size_t)4*nz+2, 0.0);
                    for (int k = 0; k < nz; ++k) {
                        Up[4*k+0] = c.sol.P_gas[k]; Up[4*k+1] = c.sol.Q[k];
                        Up[4*k+2] = c.sol.T[k];     Up[4*k+3] = c.sol.z[k];
                    }
                    Up[4*nz+0] = c.sol.z0; Up[4*nz+1] = c.sol.Sigma0;
                    convection_report(Up, nz, ci.omega_z, op, "        ", true);
                }
            }
        }

        // ===== STAGE D : COUPLED multi-root test at the node's PINNED (Sigma, T_c) =====
        // solve_column_coupled pins BOTH Sigma and T_c and frees (T_eff, f_adv), so ITS
        // multiplicity question is: does the 2x2 map (T_eff,f_adv) -> (T_c,Sigma) have more
        // than one preimage at this node?  Drive the production bring-up
        // (build_coupled_seed_2d) + the production polish (affine_invariant_newton against
        // the ORIGINAL pins) from a spread of starting guesses — this is exactly
        // solve_column_coupled's cold multi-start try_seed_and_polish, run for EVERY combo
        // instead of stopping at the first that converges, so a second basin cannot hide.
        if (stages & 8) {
            G("\n--- (D) COUPLED multi-start at PINNED (Sigma=%.5e, T_c=%.5e) ---\n",
              ci_ck.Sigma_target, ci_ck.Tc);
            const double Te_def  = estimate_Teff_guess(ci_ck, op);
            const double rho_def = ci_ck.rho_mid_guess;
            const int    ms_n    = env_i("MRP_MS_N", 14);
            const double ms_lo   = env_d("MRP_MS_LO", 0.05);
            const double ms_hi   = env_d("MRP_MS_HI", 20.0);
            struct Combo { double te, rho; const char* tag; };
            std::vector<Combo> combos;
            for (int k = 0; k < ms_n; ++k) {
                const double m = std::exp(std::log(ms_lo)
                                 + (ms_n > 1 ? (std::log(ms_hi) - std::log(ms_lo)) / (ms_n - 1) : 0.0) * k);
                combos.push_back({m, 1.0, "sweep"});
            }
            // The EXACT production ladder (MS_COMBOS in solve_column_coupled) + its default.
            const double prod[][2] = {{1.0,1.0},{0.5,1.0},{2.0,1.0},{0.25,1.0},{4.0,1.0},
                                      {1.0,0.3},{1.0,3.0},{0.5,0.3},{2.0,3.0},
                                      {0.5,3.0},{2.0,0.3},{0.25,0.3},{4.0,3.0}};
            for (const auto& p : prod) combos.push_back({p[0], p[1], "PROD"});

            struct MRes { double te, rho; const char* tag;
                          int seed_ok, pol_ok; double Teff, fadv, z0, F, eta3, Tc_chk, Sig_chk;
                          std::vector<double> U; };
            std::vector<MRes> MR(combos.size());
            int doneD = 0;
            #pragma omp parallel for schedule(dynamic)
            for (int k = 0; k < (int)combos.size(); ++k) {
                ColumnCoupledInputs in2 = ci_ck;         // pins UNTOUCHED; only seed knobs vary
                in2.Teff_guess    = Te_def * combos[(size_t)k].te;
                in2.rho_mid_guess = std::max(rho_def * combos[(size_t)k].rho, 1e-30);
                MRes m{}; m.te = combos[(size_t)k].te; m.rho = combos[(size_t)k].rho;
                m.tag = combos[(size_t)k].tag;
                std::vector<double> Ums;
                if (build_coupled_seed_2d(in2, op, Ums)) {
                    m.seed_ok = 1;
                    const int nz = ci_ck.n_nodes;
                    int itms = 0;
                    // polish against the ORIGINAL pins, exactly as try_seed_and_polish does
                    if (affine_invariant_newton(Ums, ci_ck, op, &itms)) m.pol_ok = 1;
                    m.Teff = Ums[4*nz+2]; m.fadv = Ums[4*nz+3];
                    m.z0 = Ums[4*nz+0];   m.Sig_chk = Ums[4*nz+1];
                    m.F  = Ums[4*(nz-1)+1];
                    m.Tc_chk = Ums[2];
                    ColumnBVPSolution s{};
                    s.z.resize(nz); s.P.resize(nz); s.P_gas.resize(nz); s.T.resize(nz); s.rho.resize(nz);
                    for (int q = 0; q < nz; ++q) {
                        s.P_gas[q] = Ums[4*q+0]; s.T[q] = Ums[4*q+2]; s.z[q] = Ums[4*q+3];
                        s.P[q] = p_total(s.P_gas[q], s.T[q]);
                        s.rho[q] = std::max(rho_from_gas(s.P_gas[q], s.T[q]), 0.0);
                    }
                    double e3 = 0.0, e4 = 0.0; column_moments(s, e3, e4);
                    m.eta3 = e3;
                    m.U.assign(Ums.begin(), Ums.begin() + (4*nz + 2));  // base-layout prefix
                    m.U[4*nz+0] = Ums[4*nz+0]; m.U[4*nz+1] = Ums[4*nz+1];
                }
                MR[(size_t)k] = m;
                #pragma omp critical
                { ++doneD; G("    [multistart %d/%zu  t=%.0fs]\r", doneD, combos.size(), elapsed()); }
            }
            G("\n");
            G("  %-6s %-7s %-7s %-6s %-6s %-12s %-10s %-12s %-12s %-8s\n",
              "tag", "Teff x", "rho x", "seed", "pol", "T_eff", "f_adv", "z0[cm]", "F", "eta3");
            for (size_t k = 0; k < MR.size(); ++k) {
                const MRes& m = MR[k];
                if (!m.seed_ok) {
                    G("  %-6s %-7.4g %-7.4g %-6s %-6s\n", m.tag, m.te, m.rho, "NO", "-");
                } else {
                    G("  %-6s %-7.4g %-7.4g %-6s %-6s %-12.5e %-10.5f %-12.5e %-12.5e %-8.4f\n",
                      m.tag, m.te, m.rho, "yes", m.pol_ok ? "yes" : "no",
                      m.Teff, m.fadv, m.z0, m.F, m.eta3);
                }
            }
            // Cluster the converged coupled roots by (T_eff, f_adv) within 1% relative.
            G("\n  Distinct COUPLED roots (cluster on T_eff within 1%%, f_adv within 0.01):\n");
            std::vector<size_t> rep;
            for (size_t k = 0; k < MR.size(); ++k) {
                if (!MR[k].seed_ok || !MR[k].pol_ok) continue;
                bool dup = false;
                for (size_t g : rep)
                    if (std::abs(MR[k].Teff - MR[g].Teff) <= 0.01 * std::max(MR[k].Teff, MR[g].Teff)
                        && std::abs(MR[k].fadv - MR[g].fadv) <= 0.01) { dup = true; break; }
                if (!dup) rep.push_back(k);
            }
            for (size_t g = 0; g < rep.size(); ++g) {
                const MRes& m = MR[rep[g]];
                const OneZoneState oz = one_zone_closure(ci_ck.Sigma_target, ci_ck.Tc, ns.r, in, op);
                G("   coupled root %zu: T_eff=%.6e  f_adv=%.6f  z0=%.6e  F=%.6e  eta3=%.4f\n",
                  g + 1, m.Teff, m.fadv, m.z0, m.F, m.eta3);
                G("                   P (Group-2) = 2*p_mid*z0 = %.6e   [p_mid=%.5e beta_oz=%.5f]\n",
                  2.0 * oz.p_mid * m.z0, oz.p_mid, (oz.p_mid > 0.0) ? oz.p_gas / oz.p_mid : 1.0);
                if (!m.U.empty())
                    convection_report(m.U, ci_ck.n_nodes, ci_ck.omega_z, op, "                   ", true);
            }
            if (rep.empty()) G("   (no combo produced a POLISHED coupled root)\n");
            G("  => %zu distinct coupled root(s) at the pinned (Sigma,T_c).\n", rep.size());
        }
    }

    G("\n=== probe complete (%.0f s) ===\n", elapsed());
    return 0;
}
