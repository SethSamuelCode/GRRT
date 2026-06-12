// ===========================================================================
// SLIM-BRANCH SEED PROBE  (diagnostic — DELETABLE)
// ---------------------------------------------------------------------------
// HYPOTHESIS: the high-Eddington (f_Edd~0.9) slim solution lives on a SEPARATE,
// stable UPPER branch of the disk S-curve, reachable by SEEDING the upper (slim)
// branch directly — a thick, advection-dominated, high-Σ, radiation-pressure-
// dominated initial guess — and relaxing onto it, WITHOUT crossing the fold near
// f_Edd≈0.11 that terminates the lower (gas/thin) branch.
//
// This probe builds build_slim_branch_seed(in, op) and runs it through the EXACT
// SAME machinery the production solver uses: solve_single_am (outer ℓ_in bracket +
// inner fixed-ℓ_in Newton relax_structure + physical validity gate).  ONLY the SEED
// differs from the thin-disk path; no residual/physics change.
//
// Experiments:
//   1. Direct slim seed at f_Edd≈0.9 (a=0.9, N=48).
//   2. If 0.9 too far: seed the slim branch at a MODERATE f_Edd just above the fold
//      (0.15, 0.2, 0.3, ...) and CONTINUE UP it warm-starting each higher rung from
//      the previous converged UPPER-branch state — never touching the fold.
//   3. Sweep seed variants (H/r, Σ multiplier, T_c floor) to find ANY thick/advective
//      seed that lands on the upper branch.
//
// #includes slim_disk_radial.cpp + opacity.cpp directly (probe/test pattern) so it
// reaches the internal helpers (relax_structure, one_zone_closure, the slim_detail::
// Kerr factors, solve_single_am, slim_validity_gate, etc.).
//
// Build:
//   cmake --build build --config Release --target slim-slimseed-probe
//   build/Release/slim-slimseed-probe.exe
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
static double mdot_edd_of(double a) {
    using namespace constants;
    SlimDiskInputs in = make_inputs(a, 1.0, 48, 1.0);
    return in.mdot;   // f_Edd=1 => this IS Mdot_Edd
}

// ---------------------------------------------------------------------------
// build_slim_branch_seed: a THICK, ADVECTION-DOMINATED, HIGH-Σ, RADIATION-PRESSURE-
// dominated initial guess characteristic of the UPPER (slim) branch.
// ---------------------------------------------------------------------------
// Distinct from build_thin_disk_seed (the LOWER-branch α-disk seed).  Physically-
// motivated; uses the verified one_zone_closure + Kerr factors, no fabricated profiles.
//
// Parameters (seed variants):
//   target_Hr  : target H/r at each node (slim disks: 0.3–0.6).
//   sigma_mult : multiplier on a slim-relation reference Σ (the UPPER branch is the
//                HIGH-Σ solution at a given Ṁ; try 3×–30×).
//   Tc_floor   : floor on T_c [K] so the seed stays radiation-pressure-dominated
//                (β≪1) even where the H/r inversion would pick a cooler T_c.
//
// Construction per node i (r_i):
//   • ℓ(r) = ℓ_K(r) (Keplerian; the residual relaxes the sub-Keplerian pressure
//     support).  ℓ_in = ℓ_K(r_isco).  r_s = 0.98·r_isco (slightly inside ISCO).
//   • Σ_i: the slim/advective reference Σ_slim ≈ Ṁ/(2π r² Ω_K) (the advection-
//     dominated surface density at radial inflow speed ~ r Ω_K · (H/r)² — i.e. a
//     thick, fast-inflow column), inflated by sigma_mult.  This puts the seed on the
//     HIGH-Σ branch.  [Σ_slim is the natural slim-disk surface density scale; the
//     exact prefactor is absorbed by sigma_mult, which the variant sweep scans.]
//   • T_c,i: chosen so the one_zone_closure scale height H(Σ_i, T_c, r_i) hits
//     target_Hr·r_i (in cm).  H increases monotonically with T_c (radiation term
//     b=2a_rad T⁴/(3Σ)), so bisect T_c for the target H.  Floored at Tc_floor to
//     guarantee radiation-pressure dominance (β≪1) — the slim regime.
//   • V_i: from EXACT mass conservation Ṁ = -2πΣΔ^½(V/√(1-V²))r_g c (fast advective
//     inflow follows from the chosen Σ).
//
// Node 0 (= r_s) gets a Mach-1 sonic override (same idea as the thin seed): at fixed
// T_c,0 bisect Σ_0 to V₀²=c_s² so 𝒟₀(r_s)=0 from the outset.
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

    // T_c that makes H(Σ,T_c,r) == H_target (cm).  H is monotone-increasing in T_c.
    auto Tc_for_H = [&](double Sig, double r, double H_target) -> double {
        auto H_of = [&](double Tc_) { return one_zone_closure(Sig, Tc_, r, in, op).H; };
        double lo = std::max(Tc_floor, 1e3), hi = 1e10;
        // Ensure the bracket spans H_target; if even hi is too cold, return hi.
        if (!(H_of(hi) > H_target)) return hi;
        if (H_of(lo) > H_target) return lo;          // already thick enough at the floor
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
        const double Om_K = omega_k(in.mass, in.spin, r);                 // [1/M]
        const double Om_K_cgs = Om_K * (c_cgs / in.r_g);                  // [1/s]
        const double r_cm = r * in.r_g;
        // Slim reference Σ: Ṁ /(2π r_cm² Ω_K) is the advection-dominated surface
        // density scale (mass law with V ~ r Ω_K, i.e. fast radial inflow).  Inflate
        // by sigma_mult to sit on the HIGH-Σ upper branch.
        const double Sig_slim = in.mdot / (2.0 * std::numbers::pi * r_cm * r_cm * std::max(Om_K_cgs, 1e-300));
        double Sig = std::max(sigma_mult * Sig_slim, 1.0);
        const double H_target = target_Hr * r_cm;                        // cm
        double Tc = Tc_for_H(Sig, r, H_target);
        Tc = std::max(Tc, Tc_floor);

        U[4*i+0] = Sig;
        U[4*i+1] = Vfrom(r, Sig);
        U[4*i+2] = ell_kepler(in.mass, in.spin, r);
        U[4*i+3] = Tc;
    }

    // Node-0 Mach-1 sonic override (so 𝒟₀(r_s)=0 from the seed).  At fixed T_c,0 both
    // |V| (∝1/Σ, mass cons.) and c_s (closure) depend on Σ; |V|²−c_s² is monotone
    // decreasing in Σ — bisect Σ_0 to the Mach-1 crossing.
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
// Report the physics of a state U (whether or not it converged).
// ---------------------------------------------------------------------------
struct Physics {
    double maxHr, minHr, bmin, bmax, fadv_min, fadv_max, pkSig, pkTc, r_s, ell_in;
    bool any_Vpos, any_Signeg;
};
static Physics physics_of(const SlimDiskInputs& in, const OpacityLUTs& op,
                          const std::vector<double>& U) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    SlimDiskRadial out;
    unpack_profile(in, op, U, out);
    Physics ph{0,1e300,1e300,0,1e300,-1e300,0,0,U[4*N+1],U[4*N+0],false,false};
    for (int i = 0; i < N; ++i) {
        const double Hr = out.H[i] / (out.r[i] * in.r_g);
        ph.maxHr = std::max(ph.maxHr, Hr);
        ph.minHr = std::min(ph.minHr, Hr);
        ph.pkSig = std::max(ph.pkSig, out.Sigma[i]);
        ph.pkTc  = std::max(ph.pkTc, out.Tc[i]);
        const OneZoneState oz = one_zone_closure(std::max(out.Sigma[i], kSigmaFloor),
                                                 std::max(out.Tc[i], kTFloor), out.r[i], in, op);
        const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
        ph.bmin = std::min(ph.bmin, beta); ph.bmax = std::max(ph.bmax, beta);
        ph.fadv_min = std::min(ph.fadv_min, out.f_adv[i]);
        ph.fadv_max = std::max(ph.fadv_max, out.f_adv[i]);
        if (U[4*i+1] >= 0.0) ph.any_Vpos = true;
        if (U[4*i+0] <= 0.0) ph.any_Signeg = true;
    }
    return ph;
}
static void print_seed_physics(const char* tag, const SlimDiskInputs& in,
                               const OpacityLUTs& op, const std::vector<double>& U) {
    const Physics ph = physics_of(in, op, U);
    std::printf("    %s: H/r=[%.3f,%.3f] beta=[%.2e,%.2e] f_adv=[%.2e,%.2e] "
                "peakSig=%.3e peakTc=%.3e r_s=%.4f ell_in=%.5f Vpos=%d Signeg=%d\n",
                tag, ph.minHr, ph.maxHr, ph.bmin, ph.bmax, ph.fadv_min, ph.fadv_max,
                ph.pkSig, ph.pkTc, ph.r_s, ph.ell_in, (int)ph.any_Vpos, (int)ph.any_Signeg);
}

// Attempt a solve from a given seed U; returns converged flag and leaves U in the
// (possibly partially-relaxed) state.  Prints gate detail + physics + group mags.
static bool try_solve(const SlimDiskInputs& in, const OpacityLUTs& op,
                      std::vector<double>& U, const char* tag, double wall_s) {
    const int N = std::max(in.n_nodes, 4);
    SolveBudget budget; budget.wall_cap_s = wall_s; budget.start = std::chrono::steady_clock::now();
    g_budget = &budget;
    auto t0 = std::chrono::steady_clock::now();
    const bool conv = solve_single_am(in, op, U, /*require_N1=*/true);
    auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1-t0).count();
    const bool tripped = budget.tripped;
    g_budget = nullptr;

    std::vector<double> R; slim_radial_residual(U, in, op, R);
    const double merit = slim_scaled_residual_norm(U, R, in);
    const GroupMags gm = slim_group_mags(U, R, in);
    const ValidityResult v = slim_validity_gate(in, op, U, /*require_N1=*/true);
    std::printf("  [%s] conv=%d tripped=%d wall=%.1fs merit=%.3e | "
                "mass=%.2e ang=%.2e rad=%.2e ene=%.2e bc=%.2e reg=%.2e\n",
                tag, (int)conv, (int)tripped, wall, merit,
                gm.mass, gm.ang, gm.rad, gm.ene, gm.bc, gm.reg);
    std::printf("        gate: mass=%d(%.2e) sign=%d D0=%d(%.2e) N1=%d(%.2e) rs=%d(%.4f<%.4f) smooth=%d(%.1fx)\n",
                (int)v.mass_ok, v.mass_maxrel, (int)v.sign_ok, (int)v.reg_D0_ok, v.D0_scaled,
                (int)v.reg_N1_ok, v.N1_scaled, (int)v.rs_ok, v.r_s, v.r_isco,
                (int)v.smooth_ok, v.sigma_max_jump);
    print_seed_physics("phys", in, op, U);
    std::fflush(stdout);
    return conv;
}

// ---------------------------------------------------------------------------
// Experiment 1: direct slim seed at f_Edd≈0.9 (a=0.9).  Sweep seed variants.
// ---------------------------------------------------------------------------
static void experiment_direct(const OpacityLUTs& op, double a, double f_Edd, double wall_s) {
    const int N = 48;
    std::printf("\n############################################################\n");
    std::printf("#  EXPERIMENT 1: DIRECT slim seed  a=%.3f  f_Edd=%.3f  N=%d\n", a, f_Edd, N);
    std::printf("############################################################\n");
    SlimDiskInputs in = make_inputs(a, f_Edd, N, wall_s);

    // Seed-variant sweep: (target_Hr, sigma_mult, Tc_floor).
    struct Variant { double Hr, sig, Tcf; };
    const std::vector<Variant> variants = {
        {0.40,  5.0, 1e6},
        {0.50, 10.0, 1e6},
        {0.30,  3.0, 1e6},
        {0.60, 20.0, 1e7},
        {0.50, 30.0, 1e7},
        {0.45,  8.0, 5e6},
    };
    for (const auto& var : variants) {
        std::printf("\n-- variant H/r=%.2f sig_mult=%.1f Tc_floor=%.1e --\n", var.Hr, var.sig, var.Tcf);
        std::vector<double> U = build_slim_branch_seed(in, op, var.Hr, var.sig, var.Tcf);
        print_seed_physics("SEED", in, op, U);
        char tag[64]; std::snprintf(tag, sizeof tag, "Hr%.2f_s%.0f", var.Hr, var.sig);
        const bool conv = try_solve(in, op, U, tag, wall_s);
        if (conv) {
            std::printf("  >>> DIRECT SLIM SEED CONVERGED at f_Edd=%.3f, a=%.3f (variant H/r=%.2f s=%.1f) <<<\n",
                        f_Edd, a, var.Hr, var.sig);
            return;
        }
    }
    std::printf("\n  (no direct slim seed variant converged at f_Edd=%.3f)\n", f_Edd);
}

// ---------------------------------------------------------------------------
// Experiment 2: seed at moderate f_Edd, then CONTINUE UP the upper branch warm-
// starting each higher rung from the previous converged state.
// ---------------------------------------------------------------------------
static void experiment_continue(const OpacityLUTs& op, double a, double wall_s) {
    const int N = 48;
    std::printf("\n############################################################\n");
    std::printf("#  EXPERIMENT 2: slim seed at moderate f_Edd + CONTINUE UP  a=%.3f  N=%d\n", a, N);
    std::printf("############################################################\n");

    // f_Edd ladder above the fold, fine near the bottom.
    const std::vector<double> ladder = {0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90};
    // Best seed variant from experiment 1 latitude (thick, high-Σ, hot).
    const double Hr = 0.50, sig = 10.0, Tcf = 1e6;

    std::vector<double> U;     // warm state across rungs
    bool have_warm = false;
    double highest = 0.0;

    for (double f : ladder) {
        SlimDiskInputs in = make_inputs(a, f, N, wall_s);
        std::printf("\n-- rung f_Edd=%.3f %s --\n", f, have_warm ? "(warm-start from previous)" : "(fresh slim seed)");
        std::vector<double> Useed;
        if (have_warm) {
            // Warm-start: carry the converged upper-branch U, but re-derive V from
            // mass conservation at the NEW Ṁ (Σ,ℓ,T_c carried; the structure is close
            // on the upper branch between adjacent rungs).  Also refresh r_s/ell_in.
            Useed = U;
            const double r_s = Useed[4*N+1];
            const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
            for (int i = 0; i < N; ++i) {
                const double t = double(i) / double(N - 1);
                const double r = std::exp(lr0 + (lr1 - lr0) * t);
                const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
                const double dn = 2.0 * std::numbers::pi * Useed[4*i+0] * sqrtD * in.r_g * constants::c_cgs;
                double V = -1e-6;
                if (dn > 0.0) { const double X = -in.mdot / dn; V = X / std::sqrt(1.0 + X*X); }
                if (!(V < 0.0)) V = -1e-6;
                Useed[4*i+1] = std::clamp(V, -kVCap, -1e-12);
            }
        } else {
            Useed = build_slim_branch_seed(in, op, Hr, sig, Tcf);
        }
        print_seed_physics("SEED", in, op, Useed);
        char tag[48]; std::snprintf(tag, sizeof tag, "f%.2f", f);
        std::vector<double> Uwork = Useed;
        const bool conv = try_solve(in, op, Uwork, tag, wall_s);
        if (conv) {
            U = Uwork; have_warm = true; highest = f;
            std::printf("  >>> upper branch reached f_Edd=%.3f (a=%.3f) <<<\n", f, a);
        } else {
            std::printf("  (rung f_Edd=%.3f did NOT converge; stop continuation)\n", f);
            break;
        }
    }
    std::printf("\n# EXP-2 RESULT (a=%.3f): highest f_Edd reached on upper branch = %.3f\n", a, highest);
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    const double a = 0.9;
    const double wall = 150.0;   // tight per-solve wall cap (kill overruns)

    std::printf("Mdot_Edd(a=%.3f) scale reference: %.4e g/s\n", a, mdot_edd_of(a));

    // 1) Direct slim seed at f_Edd≈0.9.
    experiment_direct(op, a, 0.90, wall);

    // 2) Seed the slim branch at moderate f_Edd and continue up.
    experiment_continue(op, a, wall);

    std::printf("\n[probe] done.\n");
    return 0;
}
