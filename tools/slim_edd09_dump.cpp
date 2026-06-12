// ===========================================================================
// NEAR-EDDINGTON (f_Edd=0.9) PROFILE DUMP + MESH CHECK  (diagnostic — DELETABLE)
// ---------------------------------------------------------------------------
// Post-Qvis-correction re-test of the near-Eddington slim disk (a=0.9):
//   1. Cold thin-disk anchor at f_Edd=0.03 (the post-correction cold basin edge
//      is in (0.03, 0.04); the true arclength fold is at f_Edd~0.071), then a
//      warm-start f_Edd ladder upward with step halving (floor 0.005).
//   2. At f_Edd=0.90 / N=48: FULL radial profile dump (H/r, beta, f_adv, T_c,
//      Sigma, V, per-node mass-conservation error) + validity gate + group mags.
//   3. MESH CHECK: interpolate the converged N=48 state onto N=96 (in ln r),
//      re-solve, dump, and report a mid-disk sawtooth metric (zig-zag amplitude
//      of ln Sigma / ln T_c, r in [8,30]) at both N to see if it amplifies.
//
// Exact production machinery (solve_single_am + budget + validity gate); seeds/
// continuation only.  NO physics change.  Safety budget ON, per-solve wall caps.
//
// Build:  cmake --build build --config Release --target slim-edd09-dump
// Run:    build/Release/slim-edd09-dump.exe
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
#include <numbers>

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

struct SolveOut {
    bool conv = false, tripped = false;
    double wall = 0.0, merit = 0.0;
    GroupMags gm{};
    ValidityResult v{};
};
static SolveOut run_solve(const SlimDiskInputs& in, const OpacityLUTs& op,
                          std::vector<double>& U, double wall_s, bool require_N1) {
    SolveOut so;
    SolveBudget budget; budget.wall_cap_s = wall_s;
    budget.start = std::chrono::steady_clock::now();
    g_budget = &budget;
    auto t0 = std::chrono::steady_clock::now();
    so.conv = solve_single_am(in, op, U, require_N1);
    auto t1 = std::chrono::steady_clock::now();
    so.wall = std::chrono::duration<double>(t1 - t0).count();
    so.tripped = budget.tripped;
    g_budget = nullptr;
    std::vector<double> R; slim_radial_residual(U, in, op, R);
    so.merit = slim_scaled_residual_norm(U, R, in);
    so.gm = slim_group_mags(U, R, in);
    so.v = slim_validity_gate(in, op, U, require_N1);
    return so;
}

// Full profile dump + shape summary + mid-disk sawtooth metric.
static void dump_profile(const SlimDiskInputs& in, const OpacityLUTs& op,
                         const std::vector<double>& U, const SolveOut& so,
                         const char* tag) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    SlimDiskRadial out; unpack_profile(in, op, U, out);
    std::printf("\n  ---- PROFILE [%s]  N=%d  conv=%d merit=%.3e  ell_in=%.6f r_s=%.4f (isco=%.4f) ----\n",
                tag, N, (int)so.conv, so.merit, out.ell_in, out.r_sonic, so.v.r_isco);
    std::printf("  validity: mass_ok=%d sign_ok=%d reg_D0_ok=%d reg_N1_ok=%d rs_ok=%d smooth_ok=%d"
                "  (mass_maxrel=%.2e D0=%.2e N1=%.2e sigjump=%.2e)\n",
                (int)so.v.mass_ok, (int)so.v.sign_ok, (int)so.v.reg_D0_ok, (int)so.v.reg_N1_ok,
                (int)so.v.rs_ok, (int)so.v.smooth_ok,
                so.v.mass_maxrel, so.v.D0_scaled, so.v.N1_scaled, so.v.sigma_max_jump);
    std::printf("  groups: mass=%.1e ang=%.1e rad=%.1e ene=%.1e bc=%.1e reg=%.1e\n",
                so.gm.mass, so.gm.ang, so.gm.rad, so.gm.ene, so.gm.bc, so.gm.reg);
    std::printf("  %-3s %-8s %-8s %-10s %-+10s %-11s %-11s %-+11s %-+9s\n",
                "i", "r[M]", "H/r", "beta", "f_adv", "Tc[K]", "Sigma", "V[c]", "mdot/Md-1");
    const int stride = (N > 48) ? 2 : 1;   // keep the table readable at N=96
    for (int i = 0; i < N; ++i) {
        const double r = out.r[i];
        const double Sig = std::max(out.Sigma[i], kSigmaFloor);
        const double Tc  = std::max(out.Tc[i], kTFloor);
        const OneZoneState oz = one_zone_closure(Sig, Tc, r, in, op);
        const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
        const double Hr = out.H[i] / (r * in.r_g);
        const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
        const double mrel = mdot_of_node(in, out.Sigma[i], out.V[i], sqrtD) / in.mdot - 1.0;
        if (i % stride == 0 || i == N-1)
            std::printf("  %-3d %-8.4f %-8.4f %-10.3e %-+10.2e %-11.4e %-11.4e %-+11.3e %-+9.1e\n",
                        i, r, Hr, beta, out.f_adv[i], Tc, out.Sigma[i], out.V[i], mrel);
    }
    // Shape summary: H/r peak location, beta trend, f_adv trend, monotonicity.
    int iHmax = 0; double Hmax = 0;
    for (int i = 0; i < N; ++i) {
        const double Hr = out.H[i] / (out.r[i] * in.r_g);
        if (Hr > Hmax) { Hmax = Hr; iHmax = i; }
    }
    bool Tc_mono = true;
    for (int i = 2; i < N - 1; ++i) if (out.Tc[i+1] > out.Tc[i] * 1.001) Tc_mono = false;
    std::printf("  >> H/r peak %.4f at i=%d (r=%.3f); H/r(outer)=%.4f | Tc monotone-dec(bulk)=%d\n",
                Hmax, iHmax, out.r[iHmax], out.H[N-2]/(out.r[N-2]*in.r_g), (int)Tc_mono);
    // Mid-disk sawtooth metric on ln Sigma and ln Tc, r in [8,30]:
    //   zig = max over interior i of |0.5(x_{i-1}+x_{i+1}) - x_i| (curvature spike),
    //   alt = max run-length of strictly alternating signs of successive diffs.
    auto zigzag = [&](const std::vector<double>& q, const char* nm) {
        double zig = 0; int izig = -1;
        int alt_run = 0, alt_max = 0; double prev_d = 0;
        for (int i = 1; i < N - 1; ++i) {
            if (out.r[i] < 8.0 || out.r[i] > 30.0) continue;
            const double x0 = std::log(std::max(q[i-1], 1e-300));
            const double x1 = std::log(std::max(q[i],   1e-300));
            const double x2 = std::log(std::max(q[i+1], 1e-300));
            const double z = std::abs(0.5*(x0 + x2) - x1);
            if (z > zig) { zig = z; izig = i; }
            const double d = x1 - x0;
            if (i > 1 && d * prev_d < 0.0) { alt_run++; alt_max = std::max(alt_max, alt_run); }
            else alt_run = 0;
            prev_d = d;
        }
        std::printf("  >> sawtooth[%s, r=8..30]: max zigzag=%.3e (at i=%d, r=%.3f), max alt-run=%d\n",
                    nm, zig, izig, (izig>=0)?out.r[izig]:0.0, alt_max);
    };
    zigzag(out.Sigma, "lnSigma");
    zigzag(out.Tc,    "lnTc");
}

int main() {
    using namespace constants;
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    const double a = 0.9;
    const int N = 48;
    const double wall_rung = 240.0, wall_fine = 360.0;
    const double total_guard_s = 35.0 * 60.0;
    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count(); };

    // ---- 1. cold anchor (post-Qvis-fix basin: fold at f_Edd~0.07, cold basin
    //         edge in (0.03, 0.04)) — anchor 0.03, fallback 0.02 ----
    double f_cur = 0.03;
    SlimDiskInputs in = make_inputs(a, f_cur, N, wall_rung);
    std::vector<double> U = build_thin_disk_seed(in, op);
    SolveOut so = run_solve(in, op, U, wall_rung, /*require_N1=*/true);
    std::printf("[anchor] f_Edd=%.3f conv=%d merit=%.3e wall=%.1fs%s\n",
                f_cur, (int)so.conv, so.merit, so.wall, so.tripped ? " <<TRIP>>" : "");
    if (!so.conv) {
        f_cur = 0.02;
        in = make_inputs(a, f_cur, N, wall_rung);
        U = build_thin_disk_seed(in, op);
        so = run_solve(in, op, U, wall_rung, true);
        std::printf("[anchor] f_Edd=%.3f conv=%d merit=%.3e wall=%.1fs%s\n",
                    f_cur, (int)so.conv, so.merit, so.wall, so.tripped ? " <<TRIP>>" : "");
        if (!so.conv) { std::printf("[edd09] anchor failed; abort.\n"); return 1; }
    }

    // ---- 2. warm-start ladder upward with halving (floor 0.005) ----
    const double targets[] = {0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.25, 0.50, 0.90};
    std::vector<double> U90;
    SlimDiskInputs in90{};
    SolveOut so90{};
    bool have90 = false;
    for (double f_t : targets) {
        while (f_cur < f_t - 1e-12) {
            if (elapsed() > total_guard_s) { std::printf("[edd09] total guard hit; stop ladder.\n"); break; }
            double f_try = f_t;
            bool ok = false;
            std::vector<double> Utry;
            while (true) {
                Utry = U;
                SlimDiskInputs int_ = make_inputs(a, f_try, N, wall_rung);
                SolveOut st = run_solve(int_, op, Utry, wall_rung, true);
                std::printf("[rung] f_Edd=%.4f conv=%d merit=%.3e wall=%.1fs%s\n",
                            f_try, (int)st.conv, st.merit, st.wall, st.tripped ? " <<TRIP>>" : "");
                if (st.conv) { ok = true; U = Utry; f_cur = f_try; in = int_; so = st; break; }
                const double step = f_try - f_cur;
                if (step <= 0.005 + 1e-12) break;         // step floor; give up this target
                f_try = f_cur + 0.5 * step;               // halve
            }
            if (!ok) break;
        }
        if (std::abs(f_cur - 0.90) < 1e-9) {
            have90 = true; U90 = U; in90 = in; so90 = so; break;
        }
        if (f_cur < f_t - 1e-12) break;                  // ladder stuck below this target
    }
    std::printf("\n[ladder] reached f_Edd=%.4f (target 0.90 %s)\n",
                f_cur, have90 ? "REACHED" : "NOT reached");

    // ---- 3. dump N=48 at the highest converged rung ----
    dump_profile(in, op, U, so, have90 ? "f_Edd=0.90 N=48" : "highest rung N=48");

    // ---- 4. mesh check at N=96 (interp in ln r, re-solve) ----
    if (elapsed() < total_guard_s) {
        const int N2 = 96;
        const double r_s48 = U[4*N+1], ell_in48 = U[4*N+0];
        std::vector<double> r48(N), r96(N2);
        const double lr0 = std::log(r_s48), lr1 = std::log(in.r_out);
        for (int i = 0; i < N;  ++i) r48[i] = std::exp(lr0 + (lr1-lr0)*double(i)/double(N-1));
        for (int i = 0; i < N2; ++i) r96[i] = std::exp(lr0 + (lr1-lr0)*double(i)/double(N2-1));
        std::vector<double> U2((size_t)4*N2 + 2, 0.0);
        for (int i = 0; i < N2; ++i) {
            const double lr = std::log(r96[i]);
            int j = 0; while (j < N-2 && std::log(r48[j+1]) < lr) ++j;
            const double l0 = std::log(r48[j]), l1 = std::log(r48[j+1]);
            const double w = std::clamp((lr - l0)/(l1 - l0), 0.0, 1.0);
            for (int k = 0; k < 4; ++k) {
                double q0 = U[4*j+k], q1 = U[4*(j+1)+k];
                if (k == 0 || k == 3) {       // Sigma, Tc: interp in log
                    U2[4*i+k] = std::exp((1.0-w)*std::log(std::max(q0,1e-300))
                                       + w*std::log(std::max(q1,1e-300)));
                } else {
                    U2[4*i+k] = (1.0-w)*q0 + w*q1;
                }
            }
        }
        U2[4*N2+0] = ell_in48; U2[4*N2+1] = r_s48;
        SlimDiskInputs in2 = make_inputs(a, f_cur, N2, wall_fine);
        SolveOut so2 = run_solve(in2, op, U2, wall_fine, true);
        std::printf("\n[mesh] N=96 warm re-solve at f_Edd=%.4f: conv=%d merit=%.3e wall=%.1fs%s\n",
                    f_cur, (int)so2.conv, so2.merit, so2.wall, so2.tripped ? " <<TRIP>>" : "");
        dump_profile(in2, op, U2, so2, "mesh-check N=96");
    }

    std::printf("\n[slim-edd09-dump] done (%.0fs total).\n", elapsed());
    return 0;
}
