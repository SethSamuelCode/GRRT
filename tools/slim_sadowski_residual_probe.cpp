// ===========================================================================
// SĄDOWSKI-STRUCTURE RESIDUAL PROBE  (diagnostic — DELETABLE)
// ---------------------------------------------------------------------------
// PART-2 of the f_Edd≈0.9 closure diagnostic.  Question: when a Sądowski-SHAPE
// slim profile (NT-thin gas-dominated outer, thickening inward to a radiation-
// pressure-dominated H/r~0.3-1 inner disk, sonic point inside ISCO) is fed to OUR
// one-zone radial residual, are the group residuals SMALL (=> Sądowski's structure
// is nearly a root of our model => reachable root, category (a)) or LARGE (=> his
// real disk is NOT a root of our one-zone closure => closure inadequacy, category
// (b))?
//
// The papers publish NO numeric profile tables (confirmed: profiles are figures-
// only).  So we use the production build_slim_disk_seed — which is BY CONSTRUCTION
// the Sądowski §3/AF13 SHAPE (verified anti-torus: β→1, H/r≪1 outward; radiation-
// dominated, thick inward; f_Edd-aware r_s/ℓ_in) — as the representative Sądowski
// structure, and read slim_radial_residual + slim_group_mags AT THE SEED (no
// relaxation), then after a SHORT relaxation, at a ladder of f_Edd.  We ALSO sweep
// the seed's inner H/r peak (hr_peak proxy via a manual rebuild) to test whether
// ANY radiation-dominated thick inner structure can drive the angmom+energy groups
// toward the floor, or whether they are stuck high regardless.
//
// Build:  cmake --build build --config Release --target slim-sadowski-residual-probe
// Run:    build/Release/slim-sadowski-residual-probe.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numbers>
#include <chrono>

using namespace grrt;
using namespace grrt::slim_detail;

static SlimDiskInputs make_inputs(double a, double f_Edd, int N) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = a; in.alpha = 0.1; in.r_g = 1.48e6;
    in.r_out = 50.0; in.n_nodes = N; in.max_iters = 800; in.tol = 1e-6;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

// Summarize the seed/state shape (β, H/r, f_adv) so we can confirm it IS a
// Sądowski-shape slim profile (radiation-dominated thick inner, gas-dominated
// thin outer) before judging the residual.
static void shape_line(const char* tag, const SlimDiskInputs& in, const OpacityLUTs& op,
                       const std::vector<double>& U) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    SlimDiskRadial out; unpack_profile(in, op, U, out);
    double maxHr = 0, minHr = 1e300, bmin = 1e300, bmax = 0, fmax = -1e300, fmin = 1e300;
    double Hr_inner = 0, Hr_outer = 0, b_inner = 0, b_outer = 0;
    for (int i = 0; i < N; ++i) {
        const double Hr = out.H[i] / (out.r[i] * in.r_g);
        maxHr = std::max(maxHr, Hr); minHr = std::min(minHr, Hr);
        const OneZoneState oz = one_zone_closure(std::max(out.Sigma[i], kSigmaFloor),
                                                 std::max(out.Tc[i], kTFloor), out.r[i], in, op);
        const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
        bmin = std::min(bmin, beta); bmax = std::max(bmax, beta);
        fmax = std::max(fmax, out.f_adv[i]); fmin = std::min(fmin, out.f_adv[i]);
        if (i <= 2) { Hr_inner = std::max(Hr_inner, Hr); b_inner = beta; }
        if (i == N-1) { Hr_outer = Hr; b_outer = beta; }
    }
    std::printf("    %-10s H/r[%.3f,%.3f] (inner~%.3f,outer~%.4f)  beta[%.2e,%.2e] (inner~%.2e,outer~%.3f)  f_adv[%+.2e,%+.2e]\n",
                tag, minHr, maxHr, Hr_inner, Hr_outer, bmin, bmax, b_inner, b_outer, fmin, fmax);
}

static void resid_line(const char* tag, const SlimDiskInputs& in, const OpacityLUTs& op,
                       const std::vector<double>& U) {
    std::vector<double> R; slim_radial_residual(U, in, op, R);
    const double merit = slim_scaled_residual_norm(U, R, in);
    const GroupMags gm = slim_group_mags(U, R, in);
    std::printf("    %-10s merit=%.3e | mass=%.2e ang=%.2e rad=%.2e ene=%.2e bc=%.2e reg=%.2e\n",
                tag, merit, gm.mass, gm.ang, gm.rad, gm.ene, gm.bc, gm.reg);
}

// One f_Edd point: build the principled Sądowski-shape seed, report its shape +
// the residual groups AT THE SEED (no relaxation — the literal "is Sądowski's
// structure a root of OUR residual?" test).
static void point_seed(const OpacityLUTs& op, double a, double f_Edd, int N) {
    std::printf("\n=== a=%.3f  f_Edd=%.3f  N=%d ===\n", a, f_Edd, N);
    std::fflush(stdout);
    SlimDiskInputs in = make_inputs(a, f_Edd, N);
    std::vector<double> U = build_slim_disk_seed(in, op);
    shape_line("SEED", in, op, U);
    resid_line("SEED", in, op, U);
    std::fflush(stdout);
}

// Optional: short bounded relaxation to see if the groups DROP toward the floor
// (=> reachable) or stay pinned high (=> not a root).
static void point_relax(const OpacityLUTs& op, double a, double f_Edd, int N) {
    std::printf("\n--- RELAX a=%.3f f_Edd=%.3f N=%d ---\n", a, f_Edd, N);
    std::fflush(stdout);
    SlimDiskInputs in = make_inputs(a, f_Edd, N);
    std::vector<double> Uw = build_slim_disk_seed(in, op);
    SolveBudget budget; budget.wall_cap_s = 90.0; budget.inner_iter_cap = 30000;
    budget.start = std::chrono::steady_clock::now();
    g_budget = &budget;
    const bool conv = solve_single_am(in, op, Uw, /*require_N1=*/true);
    g_budget = nullptr;
    shape_line("RELAXED", in, op, Uw);
    resid_line("RELAXED", in, op, Uw);
    std::printf("    -> converged=%d tripped=%d\n", (int)conv, (int)budget.tripped);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    const int N = (argc > 1) ? std::atoi(argv[1]) : 48;

    std::printf("############################################################\n");
    std::printf("# SĄDOWSKI-STRUCTURE RESIDUAL TEST (Part 2)\n");
    std::printf("# Feed the Sądowski-shape slim seed to OUR residual; read groups.\n");
    std::printf("# SMALL groups => structure ~is a root (reachable, a).\n");
    std::printf("# LARGE groups => structure NOT a root (closure inadequacy, b).\n");
    std::printf("############################################################\n");

    // Ladder: sub-fold (should be near-root) -> render target. SEED residuals first
    // (fast, no relaxation) — the decisive "is Sądowski's structure a root?" read.
    std::printf("\n##### SEED-LEVEL residuals (Sądowski structure fed to our residual) #####\n");
    for (double f : {0.05, 0.10, 0.20, 0.40, 0.60, 0.90})
        point_seed(op, 0.9, f, N);

    // Then a couple of relaxations (sub-fold vs render target) to show the groups
    // drop to floor below the fold but stay pinned high at f_Edd=0.9.
    std::printf("\n##### RELAXATION (do the groups reach the floor?) #####\n");
    point_relax(op, 0.9, 0.05, N);
    point_relax(op, 0.9, 0.90, N);

    std::printf("\n[sadowski-residual] done.\n");
    return 0;
}
