// ===========================================================================
// TEMPORARY DIAGNOSTIC PROBE  (NOT a fix; safe to delete)
// ---------------------------------------------------------------------------
// Spin-homotopy continuation stall diagnosis.  Determines WHY the spin-walk
// stalls at the FIRST nonzero rung (a=0 -> 0.2) at merit ~0.2 (four orders
// above the a=0 FD floor of ~7.6e-6).
//
// Three probes:
//   1. Initial warm-start merit at a=0.2 (warm_reproject_spin of the converged
//      a=0 anchor), with full group breakdown + node-by-node mass-conservation
//      check at U_warm + V/Delta spin-correctness audit.
//   2. Fresh cold seed vs warm-start convergence at a=0.2.
//   3. Near-true seed convergence at a=0.2 (basin vs solver test).
//
// #includes slim_disk_radial.cpp + opacity.cpp directly to reach the internal
// (anonymous-namespace) helpers, exactly like tools/slim_diag_probe.cpp.
//
// Build:
//   cmake --build build --config Release --target slim-spinwalk-probe
//   build/Release/slim-spinwalk-probe.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace grrt;
using namespace grrt::slim_detail;

namespace grrt {
namespace probe {

// Per-solve wall budget (seconds) — tight guard so a stalled solve aborts honestly.
static double g_wall = 8.0;

// Build SlimDiskInputs at a given spin a, low f_Edd corner, small N for speed,
// with a tight per-solve wall budget.
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
    in.max_iters = 800;        // give the inner Newton room (matches continuation)
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

static double active_merit(const std::vector<double>& U, const SlimDiskInputs& in,
                           const OpacityLUTs& op) {
    std::vector<double> R;
    slim_radial_residual(U, in, op, R);
    return slim_scaled_residual_norm_active(U, R, in);
}

static void print_groups(const char* tag, const std::vector<double>& U,
                         const SlimDiskInputs& in, const OpacityLUTs& op) {
    std::vector<double> R;
    slim_radial_residual(U, in, op, R);
    const double merit = slim_scaled_residual_norm_active(U, R, in);
    const GroupMags g = slim_group_mags(U, R, in);
    std::printf("[%s] merit=%.4e | mass=%.4e ang=%.4e rad=%.4e ene=%.4e bc=%.4e reg=%.4e\n",
                tag, merit, g.mass, g.ang, g.rad, g.ene, g.bc, g.reg);
}

// Solve the a=0 anchor to convergence; cache it (solved once for the whole run).
static bool g_anchor_done = false, g_anchor_ok = false;
static std::vector<double> g_U0;
static SlimDiskInputs g_in0;
static bool solve_anchor(const OpacityLUTs& op, std::vector<double>& U0,
                         SlimDiskInputs& in0_out, double f_Edd, int N) {
    if (!g_anchor_done) {
        double Mdot_Edd = 0;
        g_in0 = make_inputs(0.0, f_Edd, N, 120.0, Mdot_Edd);
        g_U0 = build_thin_disk_seed(g_in0, op);
        g_anchor_ok = solve_single_am(g_in0, op, g_U0, /*require_N1=*/false);
        std::printf("[ANCHOR] a=0 f_Edd=%.4f N=%d solve_single_am=%d  ", f_Edd, N, (int)g_anchor_ok);
        print_groups("ANCHOR", g_U0, g_in0, op);
        g_anchor_done = true;
    }
    U0 = g_U0; in0_out = g_in0;
    return g_anchor_ok;
}

// =====================================================================
// PROBE 1 — initial warm-start merit at a=0.2 + mass-conservation audit.
// =====================================================================
static void probe1(const OpacityLUTs& op, double f_Edd, int N) {
    using namespace constants;
    std::printf("\n############################################################\n");
    std::printf("# PROBE 1  warm-start merit at a=0.2 (N=%d, f_Edd=%.4f)\n", N, f_Edd);
    std::printf("############################################################\n");

    std::vector<double> U0; SlimDiskInputs in0;
    if (!solve_anchor(op, U0, in0, f_Edd, N)) {
        std::printf("[PROBE1] anchor did NOT converge -> abort probe1\n");
        return;
    }

    double Mdot_Edd = 0;
    SlimDiskInputs in02 = make_inputs(0.2, f_Edd, N, g_wall, Mdot_Edd);
    // warm_reproject_spin(U_old, in_old, in_new, op).
    std::vector<double> Uw = warm_reproject_spin(U0, in0, in02, op);

    std::printf("\n[PROBE1] === warm-start state at a=0.2 BEFORE any relaxation ===\n");
    print_groups("WARM@0.2", Uw, in02, op);

    // Also show the FULL (non-active) merit and the raw N1 row.
    {
        std::vector<double> R;
        slim_radial_residual(Uw, in02, op, R);
        const int NN = std::max(in02.n_nodes, 4);
        std::printf("[PROBE1] full(incl N1) merit=%.4e  R[4N+1 raw N1]=%.4e\n",
                    slim_scaled_residual_norm(Uw, R, in02), R[4*NN+1]);
    }

    const int NN = std::max(in02.n_nodes, 4);

    // Re-derive grids for both states.
    const double r_s_old = U0[4*NN+1];
    const double r_s_new = Uw[4*NN+1];
    std::printf("[PROBE1] r_isco(a=0)=%.5f r_s_old(anchor)=%.5f | r_isco(a=0.2)=%.5f r_s_new(warm)=%.5f\n",
                isco_prograde(1.0,0.0), r_s_old, isco_prograde(1.0,0.2), r_s_new);
    std::printf("[PROBE1] ell_in: anchor=%.6f  warm=%.6f  ellK_isco(a=0.2)=%.6f\n",
                U0[4*NN+0], Uw[4*NN+0], ell_kepler(1.0,0.2,isco_prograde(1.0,0.2)));

    auto grid = [&](const std::vector<double>& U, double& rs) {
        rs = U[4*NN+1];
        const double lr0 = std::log(rs), lr1 = std::log(in02.r_out);
        std::vector<double> r(NN);
        for (int i=0;i<NN;++i){ double t=(NN==1)?0.0:double(i)/double(NN-1); r[i]=std::exp(lr0+(lr1-lr0)*t);} return r;
    };
    double rs0, rsw;
    std::vector<double> r0 = grid(U0, rs0);
    std::vector<double> rw = grid(Uw, rsw);

    // Node-by-node mass conservation at U_warm at a=0.2.
    //   Mdot_node = -2pi Sigma sqrtDelta (V/sqrt(1-V^2)) r_g c   (mdot_of_node).
    std::printf("\n[PROBE1] === node-by-node mass conservation at U_warm (a=0.2) ===\n");
    std::printf("[PROBE1]   target Mdot=%.4e g/s\n", in02.mdot);
    std::printf("  i    r        Sigma_w     V_w         Tc_w        ell_w     sqrtDelta(a=.2) Mdot_node   (Mnode-M)/M\n");
    double max_massR = 0; int imax = -1;
    for (int i=0;i<NN;++i) {
        const double Sig = Uw[4*i+0], V = Uw[4*i+1], ell = Uw[4*i+2], Tc = Uw[4*i+3];
        const double sqrtD = std::sqrt(std::max(kerr_delta(in02.mass,in02.spin,rw[i]),0.0));
        const double mdot_i = mdot_of_node(in02, Sig, V, sqrtD);
        const double massR = (mdot_i - in02.mdot)/in02.mdot;
        if (std::abs(massR) > std::abs(max_massR)) { max_massR = massR; imax = i; }
        if (i<5 || i>=NN-3 || (i%6==0))
            std::printf("  %3d %7.3f %.4e %+.4e %.4e %.5f %.6e   %.4e %+.4e\n",
                        i, rw[i], Sig, V, Tc, ell, sqrtD, mdot_i, massR);
    }
    std::printf("[PROBE1]   MAX |(Mnode-M)/M| = %.4e at node %d (r=%.3f)\n",
                max_massR, imax, imax>=0?rw[imax]:0.0);

    // V/Delta spin-correctness audit: at a representative shared-radius node, compare
    // V re-derived from mass conservation using a=0.2 Delta vs a=0 Delta. If warm_reproject
    // used the WRONG (old) spin's Delta, the node V would be inconsistent with a=0.2 mass cons.
    std::printf("\n[PROBE1] === V/Delta spin audit (is V re-derived at a=0.2?) ===\n");
    for (int i : {NN/4, NN/2, 3*NN/4, NN-1}) {
        const double Sig = Uw[4*i+0], V = Uw[4*i+1];
        const double sqrtD_new = std::sqrt(std::max(kerr_delta(1.0,0.2,rw[i]),0.0));
        const double sqrtD_old = std::sqrt(std::max(kerr_delta(1.0,0.0,rw[i]),0.0));
        // The V that EXACT mass cons at a=0.2 requires for this Sigma:
        auto Vexact = [&](double sqrtD)->double{
            const double dn = 2.0*std::numbers::pi*Sig*sqrtD*in02.r_g*c_cgs;
            double Vv=-1e-6; if(dn>0.0){const double X=-in02.mdot/dn; Vv=X/std::sqrt(1.0+X*X);} return Vv;
        };
        std::printf("  node %3d r=%7.3f: V_stored=%+.6e  V_exact(Delta@a=.2)=%+.6e  V_exact(Delta@a=0)=%+.6e  sqrtD_new=%.5e sqrtD_old=%.5e\n",
                    i, rw[i], V, Vexact(sqrtD_new), Vexact(sqrtD_old), sqrtD_new, sqrtD_old);
    }

    // WHERE is the broken 'rad' group? Dump the per-node radial-momentum ODE residual
    // R[2N+i] (scaled by s.rad=1) over the inner nodes, alongside Sigma/Tc, to locate
    // the cliff between the fresh-seed inner annulus (r<r_s_old) and the interpolated
    // old profile (r>=r_s_old).
    {
        std::vector<double> R;
        slim_radial_residual(Uw, in02, op, R);
        std::printf("\n[PROBE1] === per-node radial-ODE residual at warm start (r_s_old=%.4f) ===\n", r_s_old);
        std::printf("  interval i  r_i      Sigma_i     Tc_i        |R_rad[2N+i]|   (inner annulus iff r<r_s_old)\n");
        for (int i = 0; i < std::min(NN-1, 14); ++i) {
            std::printf("  %3d        %7.3f %.4e %.4e %.4e   %s\n",
                        i, rw[i], Uw[4*i+0], Uw[4*i+3], std::abs(R[2*NN+i]),
                        (rw[i] < r_s_old) ? "[FRESH inner]" : "[interp old]");
        }
    }

    // Compare re-projected Sigma/T_c/ell at a few SHARED radii vs the anchor's values
    // at the nearest anchor node (sanity that the interpolation tracked the old profile).
    std::printf("\n[PROBE1] === re-projected (warm) vs anchor profile at shared radii ===\n");
    std::printf("  warm_node r_w     Sig_w       Tc_w        ell_w     | nearest anchor r0   Sig0        Tc0         ell0\n");
    for (int i : {NN/4, NN/2, 3*NN/4, NN-1}) {
        // nearest anchor node by ln r
        int jb=0; double best=1e300;
        for (int j=0;j<NN;++j){ double d=std::abs(std::log(r0[j])-std::log(rw[i])); if(d<best){best=d;jb=j;} }
        std::printf("  %3d %8.3f %.4e %.4e %.5f | anchor %3d %8.3f %.4e %.4e %.5f\n",
                    i, rw[i], Uw[4*i+0], Uw[4*i+3], Uw[4*i+2],
                    jb, r0[jb], U0[4*jb+0], U0[4*jb+3], U0[4*jb+2]);
    }
}

// =====================================================================
// PROBE 2 — fresh cold seed vs warm-start convergence at a=0.2.
// =====================================================================
static void probe2(const OpacityLUTs& op, double f_Edd, int N) {
    std::printf("\n############################################################\n");
    std::printf("# PROBE 2  fresh seed vs warm-start at a=0.2 (N=%d)\n", N);
    std::printf("############################################################\n");

    std::vector<double> U0; SlimDiskInputs in0;
    if (!solve_anchor(op, U0, in0, f_Edd, N)) {
        std::printf("[PROBE2] anchor did NOT converge -> abort probe2\n");
        return;
    }
    double Mdot_Edd = 0;
    SlimDiskInputs in02 = make_inputs(0.2, f_Edd, N, g_wall, Mdot_Edd);

    // (a) FRESH cold seed at a=0.2.
    {
        std::vector<double> Uf = build_thin_disk_seed(in02, op);
        print_groups("FRESH-SEED@0.2", Uf, in02, op);
        std::vector<double> Us = Uf;
        const bool ok = solve_single_am(in02, op, Us, /*require_N1=*/false);
        std::printf("[PROBE2] FRESH solve_single_am=%d  ", (int)ok);
        print_groups("FRESH-RESULT", Us, in02, op);
    }

    // (b) WARM-START from the converged anchor.
    {
        std::vector<double> Uw = warm_reproject_spin(U0, in0, in02, op);
        std::vector<double> Us = Uw;
        const bool ok = solve_single_am(in02, op, Us, /*require_N1=*/false);
        std::printf("[PROBE2] WARM  solve_single_am=%d  ", (int)ok);
        print_groups("WARM-RESULT", Us, in02, op);
    }
}

// =====================================================================
// PROBE 3 — near-true seed at a=0.2 (basin vs solver test).
// Run relax_structure directly (fixed ell_in) from:
//   (a) the warm-reprojected state (its own ell_in),
//   (b) a fresh seed (cold).
// and report whether the inner Newton reaches the FD floor.
// =====================================================================
static void probe3(const OpacityLUTs& op, double f_Edd, int N) {
    std::printf("\n############################################################\n");
    std::printf("# PROBE 3  near-true seed -> inner relax_structure at a=0.2 (N=%d)\n", N);
    std::printf("############################################################\n");

    std::vector<double> U0; SlimDiskInputs in0;
    if (!solve_anchor(op, U0, in0, f_Edd, N)) {
        std::printf("[PROBE3] anchor did NOT converge -> abort probe3\n");
        return;
    }
    double Mdot_Edd = 0;
    SlimDiskInputs in02 = make_inputs(0.2, f_Edd, N, g_wall, Mdot_Edd);
    const int NN = std::max(in02.n_nodes, 4);

    // (a) Inner relax from the WARM state at its own ell_in (the "near-true" seed:
    // a=0.2 is close to a=0 so the reprojected converged a=0 profile SHOULD be near-true).
    {
        std::vector<double> Uw = warm_reproject_spin(U0, in0, in02, op);
        const double ell_in = Uw[4*NN+0];
        const double m0 = active_merit(Uw, in02, op);
        std::vector<double> Us = Uw;
        const bool ok = relax_structure(in02, op, ell_in, Us);
        const double m1 = active_merit(Us, in02, op);
        std::printf("[PROBE3a] WARM near-true: ell_in=%.6f  start_merit=%.4e -> inner_conv=%d end_merit=%.4e (FD floor~1e-3)\n",
                    ell_in, m0, (int)ok, m1);
        print_groups("PROBE3a-END", Us, in02, op);
    }

    // (b) Inner relax from a FRESH cold seed at its own ell_in.
    {
        std::vector<double> Uf = build_thin_disk_seed(in02, op);
        const double ell_in = Uf[4*NN+0];
        const double m0 = active_merit(Uf, in02, op);
        std::vector<double> Us = Uf;
        const bool ok = relax_structure(in02, op, ell_in, Us);
        const double m1 = active_merit(Us, in02, op);
        std::printf("[PROBE3b] FRESH cold: ell_in=%.6f  start_merit=%.4e -> inner_conv=%d end_merit=%.4e\n",
                    ell_in, m0, (int)ok, m1);
        print_groups("PROBE3b-END", Us, in02, op);
    }

    // (c) Probe a>0-specific terms directly: script_A corotation denominator,
    // omega_from_ell, calD0/calN1 at a representative node, a=0.2 vs a=0.
    std::printf("\n[PROBE3c] === a>0 term audit at a representative node (warm state) ===\n");
    std::vector<double> Uw = warm_reproject_spin(U0, in0, in02, op);
    const double lr0 = std::log(Uw[4*NN+1]), lr1 = std::log(in02.r_out);
    for (int i : {1, NN/4, NN/2}) {
        const double t = double(i)/double(NN-1);
        const double r = std::exp(lr0+(lr1-lr0)*t);
        for (double a : {0.0, 0.2}) {
            SlimDiskInputs ina = in02; ina.spin = a;
            const NodeEval e = eval_node(ina, op, r, Uw[4*i+0], Uw[4*i+1], Uw[4*i+2], Uw[4*i+3]);
            const double A = script_A(ina, r, e.mech);
            const double D0 = calD0(e);
            std::printf("  node %3d r=%7.3f a=%.1f: Omega=%+.5e A=%+.5e D0=%+.5e cs2=%.5e (Om_kp=%+.4e Om_km=%+.4e)\n",
                        i, r, a, e.mech.Omega, A, D0, e.cs2_geom, e.mech.Omega_k_plus, e.mech.Omega_k_minus);
        }
    }
}

} // namespace probe
} // namespace grrt

int main() {
    using namespace grrt;
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    const int N = 48;
    const double f_Edd = 0.02;
    grrt::probe::probe1(op, f_Edd, N);
    grrt::probe::probe3(op, f_Edd, N);   // cheap (single inner relax); most decisive
    // probe2 (full bracket warm solve) intentionally skipped: it thrashes the outer
    // bracket and only times out. probe3's fresh-vs-warm inner relax is decisive.
    (void)&grrt::probe::probe2;
    std::printf("\n[probe] done.\n");
    return 0;
}
