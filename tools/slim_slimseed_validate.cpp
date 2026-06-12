// ===========================================================================
// SLIM-BRANCH VALIDATION PROBE  (diagnostic — DELETABLE)
// ---------------------------------------------------------------------------
// Validates + robustifies the near-Eddington slim-disk solution found by the
// slim-branch seed (slim_slimseed_probe.cpp).  Three jobs:
//
//   (1) SOUNDNESS: reproduce the converged f_Edd=0.9, a=0.9 solution and dump the
//       FULL radial profile (r, H/r, f_adv, beta, T_c, Sigma, V, Omega, Mdot_node),
//       plus conservation/regularity numbers, so we can judge whether H/r~2 is one
//       node or systemic, whether f_adv<0 is the cosmetic sonic spike or real, etc.
//
//   (2) ROBUSTNESS: sweep the slim-seed parameters (target_Hr, sigma_mult, Tc_floor)
//       and report which converge and whether converged ones land on the SAME
//       attractor (compare ell_in, r_s, peak Sigma, peak T_c, max H/r).
//
//   (3) BRANCH MAP: direct slim seed (reliable recipe) at f_Edd in {0.3,0.5,0.7,0.9}
//       at a=0.9, tabulating H/r, beta, f_adv, Sigma, merit; then a=0.998 / f_Edd=0.9.
//
// Reuses the EXACT production machinery (solve_single_am = outer ell_in bracket +
// inner relax_structure + validity gate); ONLY the seed varies.  No physics change.
//
// #includes slim_disk_radial.cpp + opacity.cpp directly (probe/test pattern).
//
// Build:  cmake --build build --config Release --target slim-slimseed-validate
// Run:    build/Release/slim-slimseed-validate.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

using namespace grrt;
using namespace grrt::slim_detail;

// ---------------------------------------------------------------------------
// inputs (mirror slim_slimseed_probe.cpp::make_inputs)
// ---------------------------------------------------------------------------
static SlimDiskInputs make_inputs(double a, double f_Edd, int N, double wall_s) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = a; in.alpha = 0.1; in.r_g = 1.48e6;
    in.r_out = 50.0; in.n_nodes = N; in.max_iters = 800; in.tol = 1e-6;
    in.budget_wall_seconds = wall_s;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

// ---------------------------------------------------------------------------
// slim-branch seed (copied verbatim from slim_slimseed_probe.cpp so this probe is
// self-contained; same construction, no physics change).
// ---------------------------------------------------------------------------
static std::vector<double> build_slim_branch_seed(const SlimDiskInputs& in,
                                                  const OpacityLUTs& op,
                                                  double target_Hr,
                                                  double sigma_mult,
                                                  double Tc_floor) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    std::vector<double> U((size_t)4 * N + 2, 0.0);

    const double r_isco = isco_prograde(in.mass, in.spin);
    const double r_s = std::max(0.98 * r_isco, in.r_in * 1.001);
    const double r_out = in.r_out;
    const double lr0 = std::log(r_s), lr1 = std::log(r_out);
    const double ell_in = ell_kepler(in.mass, in.spin, r_isco);

    auto Tc_for_H = [&](double Sig, double r, double H_target) -> double {
        auto H_of = [&](double Tc_) { return one_zone_closure(Sig, Tc_, r, in, op).H; };
        double lo = std::max(Tc_floor, 1e3), hi = 1e10;
        if (!(H_of(hi) > H_target)) return hi;
        if (H_of(lo) > H_target) return lo;
        for (int b = 0; b < 80; ++b) {
            const double mid = std::sqrt(lo * hi);
            if (H_of(mid) < H_target) lo = mid; else hi = mid;
        }
        return std::sqrt(lo * hi);
    };
    auto Vfrom = [&](double r, double Sig) -> double {
        const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
        const double dn = 2.0 * std::numbers::pi * Sig * sqrtD * in.r_g * c_cgs;
        double V = -1e-6;
        if (dn > 0.0) { const double X = -in.mdot / dn; V = X / std::sqrt(1.0 + X * X); }
        if (!(V < 0.0)) V = -1e-6;
        return std::clamp(V, -kVCap, -1e-12);
    };

    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        const double r = std::exp(lr0 + (lr1 - lr0) * t);
        const double Om_K = omega_k(in.mass, in.spin, r);
        const double Om_K_cgs = Om_K * (c_cgs / in.r_g);
        const double r_cm = r * in.r_g;
        const double Sig_slim = in.mdot / (2.0 * std::numbers::pi * r_cm * r_cm * std::max(Om_K_cgs, 1e-300));
        double Sig = std::max(sigma_mult * Sig_slim, 1.0);
        const double H_target = target_Hr * r_cm;
        double Tc = Tc_for_H(Sig, r, H_target);
        Tc = std::max(Tc, Tc_floor);

        U[4*i+0] = Sig;
        U[4*i+1] = Vfrom(r, Sig);
        U[4*i+2] = ell_kepler(in.mass, in.spin, r);
        U[4*i+3] = Tc;
    }

    {
        const double r0 = r_s;
        const double sqrtD0 = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r0), 0.0));
        const double Tc0 = U[3];
        auto mach_excess = [&](double Sig_) -> double {
            const double dn = 2.0 * std::numbers::pi * Sig_ * sqrtD0 * in.r_g * c_cgs;
            double V_ = -1e-6;
            if (dn > 0.0) { const double X = -in.mdot / dn; V_ = X / std::sqrt(1.0 + X * X); }
            V_ = std::clamp(V_, -kVCap, -1e-12);
            const OneZoneState oz = one_zone_closure(Sig_, Tc0, r0, in, op);
            const double cs2 = kGtilde1 * (oz.P / Sig_) / (c_cgs * c_cgs);
            return V_ * V_ - cs2;
        };
        double lo = 1e-2, hi = 1e12;
        if (mach_excess(lo) > 0.0 && mach_excess(hi) < 0.0) {
            for (int b = 0; b < 80; ++b) {
                const double mid = std::sqrt(lo * hi);
                if (mach_excess(mid) > 0.0) lo = mid; else hi = mid;
            }
            const double Sig0 = std::sqrt(lo * hi);
            U[0] = Sig0;
            U[1] = Vfrom(r0, Sig0);
        }
    }

    U[4*N+0] = ell_in;
    U[4*N+1] = r_s;
    return U;
}

// ---------------------------------------------------------------------------
// Solve from a seed; returns converged + leaves U in the relaxed state.
// ---------------------------------------------------------------------------
struct SolveOut {
    bool conv = false, tripped = false;
    double wall = 0.0, merit = 0.0;
    GroupMags gm{};
    ValidityResult v{};
};
static SolveOut run_solve(const SlimDiskInputs& in, const OpacityLUTs& op,
                          std::vector<double>& U, double wall_s) {
    SolveOut so;
    SolveBudget budget; budget.wall_cap_s = wall_s; budget.start = std::chrono::steady_clock::now();
    g_budget = &budget;
    auto t0 = std::chrono::steady_clock::now();
    so.conv = solve_single_am(in, op, U, /*require_N1=*/true);
    auto t1 = std::chrono::steady_clock::now();
    so.wall = std::chrono::duration<double>(t1-t0).count();
    so.tripped = budget.tripped;
    g_budget = nullptr;
    std::vector<double> R; slim_radial_residual(U, in, op, R);
    so.merit = slim_scaled_residual_norm(U, R, in);
    so.gm = slim_group_mags(U, R, in);
    so.v = slim_validity_gate(in, op, U, /*require_N1=*/true);
    return so;
}

// Per-solution fingerprint for "same attractor?" comparison.
struct Fingerprint {
    double ell_in, r_s, peakSig, peakTc, maxHr, minHr, bmin, bmax, fadv_bulk_med;
};
static Fingerprint fingerprint(const SlimDiskInputs& in, const OpacityLUTs& op,
                               const std::vector<double>& U) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    SlimDiskRadial out; unpack_profile(in, op, U, out);
    Fingerprint fp{U[4*N+0], U[4*N+1], 0, 0, 0, 1e300, 1e300, 0, 0};
    std::vector<double> fadv_bulk;
    for (int i = 0; i < N; ++i) {
        const double Hr = out.H[i] / (out.r[i] * in.r_g);
        fp.maxHr = std::max(fp.maxHr, Hr); fp.minHr = std::min(fp.minHr, Hr);
        fp.peakSig = std::max(fp.peakSig, out.Sigma[i]);
        fp.peakTc  = std::max(fp.peakTc, out.Tc[i]);
        const OneZoneState oz = one_zone_closure(std::max(out.Sigma[i], kSigmaFloor),
                                                 std::max(out.Tc[i], kTFloor), out.r[i], in, op);
        const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
        fp.bmin = std::min(fp.bmin, beta); fp.bmax = std::max(fp.bmax, beta);
        if (i >= 2 && i < N-1) fadv_bulk.push_back(out.f_adv[i]);   // skip sonic node + outer edge
    }
    if (!fadv_bulk.empty()) {
        std::sort(fadv_bulk.begin(), fadv_bulk.end());
        fp.fadv_bulk_med = fadv_bulk[fadv_bulk.size()/2];
    }
    return fp;
}

// ---------------------------------------------------------------------------
// (1) Full radial-profile dump for a converged state.
// ---------------------------------------------------------------------------
static void dump_profile(const SlimDiskInputs& in, const OpacityLUTs& op,
                         const std::vector<double>& U, const SolveOut& so) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    SlimDiskRadial out; unpack_profile(in, op, U, out);

    std::printf("\n=== FULL RADIAL PROFILE (a=%.3f, mdot=%.4e g/s) ===\n", in.spin, in.mdot);
    std::printf(" conv=%d tripped=%d wall=%.1fs merit=%.3e  r_isco=%.4f r_s=%.4f ell_in=%.6f\n",
                (int)so.conv, (int)so.tripped, so.wall, so.merit, so.v.r_isco, so.v.r_s, out.ell_in);
    std::printf(" gate: mass=%d(%.2e) sign=%d D0=%d(%.2e) N1=%d(%.2e) rs=%d smooth=%d(%.2fx)\n",
                (int)so.v.mass_ok, so.v.mass_maxrel, (int)so.v.sign_ok,
                (int)so.v.reg_D0_ok, so.v.D0_scaled, (int)so.v.reg_N1_ok, so.v.N1_scaled,
                (int)so.v.rs_ok, (int)so.v.smooth_ok, so.v.sigma_max_jump);
    std::printf(" groups: mass=%.2e ang=%.2e rad=%.2e ene=%.2e bc=%.2e reg=%.2e\n",
                so.gm.mass, so.gm.ang, so.gm.rad, so.gm.ene, so.gm.bc, so.gm.reg);

    // Mdot per node (mass conservation check) + thermodynamics per node.
    std::printf("  %-3s %-9s %-9s %-9s %-10s %-10s %-11s %-10s %-9s %-10s\n",
                "i", "r[M]", "H/r", "f_adv", "beta", "Tc[K]", "Sig[g/cm2]", "V/c", "Mdot/Mt", "rho_mid");
    for (int i = 0; i < N; ++i) {
        const double r = out.r[i];
        const double Hr = out.H[i] / (r * in.r_g);
        const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
        const double Mdot_node = mdot_of_node(in, out.Sigma[i], out.V[i], sqrtD);
        const double Mdot_rel = Mdot_node / std::max(std::abs(in.mdot), 1e-300);
        const OneZoneState oz = one_zone_closure(std::max(out.Sigma[i], kSigmaFloor),
                                                 std::max(out.Tc[i], kTFloor), r, in, op);
        const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
        std::printf("  %-3d %-9.4f %-9.4f %-+9.3e %-10.3e %-10.3e %-11.3e %-+10.3e %-9.5f %-10.3e\n",
                    i, r, Hr, out.f_adv[i], beta, out.Tc[i], out.Sigma[i], out.V[i],
                    Mdot_rel, oz.rho_mid);
    }
    std::fflush(stdout);
}

// ===========================================================================
// (1) SOUNDNESS: reproduce the f_Edd=0.9 converged solution and dump it.
//     Use the known-working variant (H/r=0.45, sig=8, Tc_floor=5e6) first; if it
//     doesn't converge, scan the others until one does, then dump that.
// ===========================================================================
static bool job1_soundness(const OpacityLUTs& op, double a, double f_Edd, double wall_s) {
    std::printf("\n############################################################\n");
    std::printf("#  (1) SOUNDNESS  a=%.3f  f_Edd=%.3f\n", a, f_Edd);
    std::printf("############################################################\n");
    SlimDiskInputs in = make_inputs(a, f_Edd, 48, wall_s);
    struct V { double Hr, sig, Tcf; };
    const std::vector<V> order = {
        {0.45, 8.0, 5e6},   // the reported working variant first
        {0.50,10.0, 1e6}, {0.40, 5.0, 1e6}, {0.60,20.0, 1e7},
        {0.50,30.0, 1e7}, {0.30, 3.0, 1e6},
    };
    for (const auto& var : order) {
        std::vector<double> U = build_slim_branch_seed(in, op, var.Hr, var.sig, var.Tcf);
        SolveOut so = run_solve(in, op, U, wall_s);
        std::printf("  variant H/r=%.2f sig=%.0f Tcf=%.0e -> conv=%d merit=%.3e wall=%.1fs\n",
                    var.Hr, var.sig, var.Tcf, (int)so.conv, so.merit, so.wall);
        if (so.conv) { dump_profile(in, op, U, so); return true; }
    }
    std::printf("  (no variant converged for soundness dump)\n");
    return false;
}

// ===========================================================================
// (2) ROBUSTNESS: basin sweep over (H/r, sigma_mult, Tc_floor).
// ===========================================================================
static void job2_basin(const OpacityLUTs& op, double a, double f_Edd, double wall_s) {
    std::printf("\n############################################################\n");
    std::printf("#  (2) ROBUSTNESS BASIN  a=%.3f  f_Edd=%.3f\n", a, f_Edd);
    std::printf("############################################################\n");
    SlimDiskInputs in = make_inputs(a, f_Edd, 48, wall_s);

    const std::vector<double> Hrs  = {0.30, 0.40, 0.45, 0.50, 0.60};
    const std::vector<double> sigs = {4.0, 8.0, 15.0, 30.0};
    const std::vector<double> tcfs = {1e6, 5e6, 1e7};

    std::printf("  %-6s %-6s %-7s %-5s %-10s %-9s %-10s %-9s %-9s %-7s %-7s\n",
                "H/r", "sig", "Tcf", "conv", "merit", "ell_in", "peakSig", "peakTc",
                "maxHr", "bmin", "wall");
    int n_conv = 0, n_total = 0;
    std::vector<Fingerprint> fps;
    for (double Hr : Hrs) for (double sig : sigs) for (double tcf : tcfs) {
        ++n_total;
        std::vector<double> U = build_slim_branch_seed(in, op, Hr, sig, tcf);
        SolveOut so = run_solve(in, op, U, wall_s);
        Fingerprint fp{};
        if (so.conv) { fp = fingerprint(in, op, U); fps.push_back(fp); ++n_conv; }
        std::printf("  %-6.2f %-6.0f %-7.0e %-5d %-10.3e %-9.5f %-10.3e %-9.3e %-9.3f %-7.1e %-7.1f%s\n",
                    Hr, sig, tcf, (int)so.conv, so.merit,
                    so.conv ? fp.ell_in : 0.0, so.conv ? fp.peakSig : 0.0,
                    so.conv ? fp.peakTc : 0.0, so.conv ? fp.maxHr : 0.0,
                    so.conv ? fp.bmin : 0.0, so.wall, so.conv ? "  *" : "");
        std::fflush(stdout);
    }
    std::printf("  --> %d / %d converged\n", n_conv, n_total);
    // "Same attractor?" — report spread of fingerprints across converged solutions.
    if (!fps.empty()) {
        double e0=1e300,e1=-1e300, r0=1e300,r1=-1e300, s0=1e300,s1=-1e300, h0=1e300,h1=-1e300;
        for (const auto& f : fps) {
            e0=std::min(e0,f.ell_in); e1=std::max(e1,f.ell_in);
            r0=std::min(r0,f.r_s);    r1=std::max(r1,f.r_s);
            s0=std::min(s0,f.peakSig);s1=std::max(s1,f.peakSig);
            h0=std::min(h0,f.maxHr);  h1=std::max(h1,f.maxHr);
        }
        std::printf("  attractor spread over %zu converged: ell_in[%.5f,%.5f] r_s[%.4f,%.4f] "
                    "peakSig[%.3e,%.3e] maxHr[%.3f,%.3f]\n",
                    fps.size(), e0,e1, r0,r1, s0,s1, h0,h1);
        std::printf("  --> ell_in spread=%.2e (rel %.2e); %s\n",
                    e1-e0, (e1-e0)/std::max(std::abs(e0),1e-300),
                    (e1-e0)/std::max(std::abs(e0),1e-300) < 1e-3 ? "UNIQUE attractor" : "MULTIPLE/spread");
    }
}

// ===========================================================================
// (3) BRANCH MAP: reliable recipe at f_Edd in {0.3,0.5,0.7,0.9} (a=0.9), then
//     a=0.998 / f_Edd=0.9.
// ===========================================================================
static bool branch_point(const OpacityLUTs& op, double a, double f_Edd,
                         double Hr, double sig, double tcf, double wall_s) {
    SlimDiskInputs in = make_inputs(a, f_Edd, 48, wall_s);
    std::vector<double> U = build_slim_branch_seed(in, op, Hr, sig, tcf);
    SolveOut so = run_solve(in, op, U, wall_s);
    Fingerprint fp{};
    if (so.conv) fp = fingerprint(in, op, U);
    std::printf("  a=%.3f f_Edd=%.2f -> conv=%d merit=%.3e | r_s=%.4f ell_in=%.5f "
                "maxHr=%.3f bmin=%.2e bmax=%.2e fadv_med=%+.3e peakSig=%.3e peakTc=%.3e wall=%.1fs\n",
                a, f_Edd, (int)so.conv, so.merit,
                so.conv?fp.r_s:0.0, so.conv?fp.ell_in:0.0, so.conv?fp.maxHr:0.0,
                so.conv?fp.bmin:0.0, so.conv?fp.bmax:0.0, so.conv?fp.fadv_bulk_med:0.0,
                so.conv?fp.peakSig:0.0, so.conv?fp.peakTc:0.0, so.wall);
    std::fflush(stdout);
    return so.conv;
}
static void job3_branch(const OpacityLUTs& op, double Hr, double sig, double tcf, double wall_s) {
    std::printf("\n############################################################\n");
    std::printf("#  (3) BRANCH MAP  reliable recipe H/r=%.2f sig=%.0f Tcf=%.0e\n", Hr, sig, tcf);
    std::printf("############################################################\n");
    std::printf("-- a=0.9 across f_Edd --\n");
    for (double f : {0.30, 0.50, 0.70, 0.90}) branch_point(op, 0.9, f, Hr, sig, tcf, wall_s);
    std::printf("-- render-target spin a=0.998 at f_Edd=0.9 --\n");
    branch_point(op, 0.998, 0.90, Hr, sig, tcf, wall_s);
}

int main(int argc, char** argv) {
    int job = 0;                       // 0 = all
    double rec_Hr = 0.45, rec_sig = 8.0, rec_tcf = 5e6;   // overridable reliable recipe
    if (argc > 1) job = std::atoi(argv[1]);
    if (argc > 4) { rec_Hr = std::atof(argv[2]); rec_sig = std::atof(argv[3]); rec_tcf = std::atof(argv[4]); }

    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    const double wall = 120.0;         // tight per-solve wall cap (kill overruns)

    if (job == 0 || job == 1) job1_soundness(op, 0.9, 0.90, wall);
    if (job == 0 || job == 2) job2_basin(op, 0.9, 0.90, wall);
    if (job == 0 || job == 3) job3_branch(op, rec_Hr, rec_sig, rec_tcf, wall);

    std::printf("\n[validate] done.\n");
    return 0;
}
