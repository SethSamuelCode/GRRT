// ===========================================================================
// SLIM COUPLED-COLUMN SMOKE PROBE (C4 bring-up gate — DELETABLE)
// ---------------------------------------------------------------------------
// Verifies that solve_slim_disk_coupled RUNS and returns a sane result (or an
// honest converged=false) at a MODEST operating point, and that a degenerate
// input (mdot=0) returns converged=false without crashing.
//
// This is NOT the f_Edd=0.9 target (that is a later task).  PASS = both calls
// return cleanly (no UB) — a converged result with sane per-node F/z0/H/r/β if
// it converges, or an honest converged=false otherwise.
//
// Also prints a one-shot NORMALIZATION CHECK comparing a standalone column F
// (per face) to the radial one-zone Q_rad = 64σT_c⁴/(3κΣ) (both faces) at the
// same (Σ,T_c) — confirming the reroute factor Q_rad -> 2F.
//
// Build:  cmake --build build --config Release --target slim-coupled-smoke-probe
// Run:    build/Release/slim-coupled-smoke-probe.exe
// REUSE: include-the-.cpp — opacity + column-bvp + column-coupled + slim-radial +
//        slim-coupled, in that order, so slim_disk_coupled.cpp's TU-local helpers
//        see the radial solver's anonymous-namespace machinery.
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
#include <limits>

using namespace grrt;

// r_g for a 10 M_sun black hole (matches the slim-disk tests).
static constexpr double R_G_10MSUN = 1.48e6;  // cm (GM/c² for ~10 M_sun)

// f_Edd -> Mdot [g/s] using the SAME textbook convention as solve_slim_disk_radial.
static double mdot_from_fEdd(const SlimDiskInputs& in, double f_Edd) {
    using namespace constants;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    return f_Edd * Mdot_Edd;
}

// FACE-CONVENTION check for the Q_rad -> 2F reroute.
//
// The radial Q_rad = 64σT_c⁴/(3κΣ) is the BOTH-FACES emergent flux (disk-physics §23,
// "one face = 32σT_c⁴/3κΣ").  The column F = Q(z0) = σT_eff⁴ is the ONE-FACE flux: the
// column integrates the viscous heating q+ = α·shear·P_tot/(1+f_adv) over the HALF
// column [0,z0] and sets Q(z0)=F.  So (1+f_adv)·F = ∫_half α·shear·P_tot dz, and the
// BOTH-FACES viscous dissipation is Q_vis = 2·∫_half α·shear·P_tot dz = 2(1+f_adv)F.
// With Sądowski Q_rad = Q_vis/(1+f_adv), that gives Q_rad = 2F (both faces).  This test
// confirms the bookkeeping by recomputing ∫_half α·shear·P_tot dz from the converged
// column and checking it equals (1+f_adv)·F.  (Comparing F to the ONE-ZONE 64σT_c⁴/3κΣ
// directly is NOT a face test — the column and the one-zone are different COOLING models
// and agree only at a mutually-consistent (Σ,T_c); their ratio is ~unconstrained.)
static void normalization_check(const OpacityLUTs& lut) {
    std::printf("\n--- FACE-CONVENTION CHECK (reroute Q_rad -> 2F): column heating bookkeeping ---\n");
    ColumnInputs ref{}; ref.T_eff=3e5; ref.shear=2e3; ref.omega_z=2e3;
    ref.alpha=0.1; ref.f_adv=0.0; ref.rho_mid_guess=1.0; ref.n_nodes=96; ref.max_iters=300; ref.tol=1e-8;
    auto s = solve_column_bvp(ref, lut);
    if (!s.converged) { std::printf("  (reference column did not converge; skipping)\n"); return; }
    const double F = grrt::constants::sigma_SB*std::pow(ref.T_eff,4);   // one-face emergent flux
    const double f_adv = 0.0;   // this reference column was built at f_adv=0
    // Recompute ∫_half α·shear·P_tot dz (trapezoidal over the stored half-profile).
    double heat_half = 0.0;
    for (size_t i = 0; i + 1 < s.z.size(); ++i) {
        const double dz = s.z[i+1] - s.z[i];
        const double q0 = ref.alpha * ref.shear * s.P[i];
        const double q1 = ref.alpha * ref.shear * s.P[i+1];
        heat_half += 0.5 * (q0 + q1) * dz;
    }
    const double expect = (1.0 + f_adv) * F;
    std::printf("  one-face F = sT_eff^4         = %.6e\n", F);
    std::printf("  int_half (alpha shear P) dz   = %.6e\n", heat_half);
    std::printf("  (1+f_adv)*F                   = %.6e\n", expect);
    std::printf("  ratio int/(1+f_adv)F          = %.4f   (== 1 => Q_vis=2(1+f_adv)F => Q_rad->2F OK)\n",
                heat_half/std::max(expect,1e-300));
}

// Print per-node diagnostics for a converged coupled profile (F, z0, H/r, β) by
// re-solving each node's column at its converged (Σ, T_c).
static void dump_converged(const SlimDiskInputs& in, const OpacityLUTs& op,
                           const SlimDiskRadial& prof) {
    using namespace grrt::slim_coupled_detail;
    std::printf("  converged=%d  ell_in=%.5f  r_sonic=%.5f  N=%zu\n",
                (int)prof.converged, prof.ell_in, prof.r_sonic, prof.r.size());
    std::printf("  %-4s %-10s %-10s %-12s %-12s %-10s %-10s %-8s\n",
                "i", "r[M]", "Sigma", "Tc[K]", "F[erg/cm2/s]", "z0[cm]", "H/r", "beta");
    const int N = (int)prof.r.size();
    const int step = std::max(N / 12, 1);   // sample ~12 nodes
    for (int i = 0; i < N; i += step) {
        const double r = prof.r[i];
        const double Sigma = prof.Sigma[i];
        const double Tc = prof.Tc[i];
        // geometry for the column
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const double Om_i = slim_detail::omega_from_ell(in.mass, in.spin, r, /*ell*/0.0); // unused fallback
        (void)Om_i;
        // Use the radial-stored Omega for the shear FD (CGS Omega in prof.Omega).
        const double dOmega_dr_cgs = (prof.Omega[j] - prof.Omega[i]) / ((prof.r[j]-prof.r[i])*in.r_g); // [1/s/cm]
        const double shear_i = std::abs(r*in.r_g * dOmega_dr_cgs);  // |r dOmega/dr| [1/s]
        const double omegaz_i = std::sqrt(std::max(slim_detail::omega_perp2(in.mass,in.spin,r),0.0))
                              * (constants::c_cgs/in.r_g);
        ColumnCoupledInputs ci{};
        ci.Sigma_target=Sigma; ci.Tc=Tc; ci.shear=std::max(shear_i,1e-300);
        ci.omega_z=std::max(omegaz_i,1e-300); ci.alpha=in.alpha;
        ci.rho_mid_guess=1e-3; ci.n_nodes=24; ci.max_iters=300; ci.tol=1e-8; ci.Teff_guess=0.8*Tc;
        ColumnClosure c = solve_column_coupled(ci, op, nullptr);
        const double Hr = (c.converged && r>0) ? c.z0/(r*in.r_g) : std::nan("");
        // beta from a one-zone closure at (Sigma,Tc) (diagnostic only).
        const auto oz = slim_detail::one_zone_closure(std::max(Sigma,1e-30),std::max(Tc,1.0),r,in,op);
        const double beta = oz.p_gas/std::max(oz.p_mid,1e-300);
        std::printf("  %-4d %-10.4f %-10.3e %-12.4e %-12.4e %-10.4e %-10.4e %-8.4f\n",
                    i, r, Sigma, Tc, c.F, c.z0, Hr, beta);
    }
}

int main() {
    std::setbuf(stdout, nullptr);
    auto lut = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    int pass = 1;

    normalization_check(lut);

    // -----------------------------------------------------------------------
    // (1) Modest operating point: a=0.9, f_Edd=0.02, small N, short budget.
    // -----------------------------------------------------------------------
    std::printf("\n=== (1) coupled solve @ a=0.9, f_Edd=0.02, N=12 (bring-up) ===\n");
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = 0.9; in.alpha = 0.1; in.r_g = R_G_10MSUN;
    in.r_out = 50.0; in.n_nodes = 12; in.max_iters = 120; in.tol = 1e-8;
    in.r_in = 0.5 * slim_detail::isco_prograde(in.mass, in.spin);   // hard inner floor
    in.mdot = mdot_from_fEdd(in, 0.02);
    // Short-ish budget so a non-convergent run fails honestly rather than grinding.
    in.budget_wall_seconds = 240.0;
    in.budget_inner_iter_cap = 4000;
    std::printf("  mdot=%.4e g/s  r_in=%.4f  r_isco=%.4f\n",
                in.mdot, in.r_in, slim_detail::isco_prograde(in.mass,in.spin));

    SlimDiskRadial prof;
    bool threw = false;
    try {
        prof = solve_slim_disk_coupled(in, lut);
    } catch (const std::exception& ex) {
        threw = true; std::printf("  EXCEPTION: %s\n", ex.what());
    } catch (...) {
        threw = true; std::printf("  EXCEPTION (unknown)\n");
    }
    if (threw) { pass = 0; std::printf("  FAIL: solve threw\n"); }
    else if (prof.converged) {
        std::printf("  RESULT: converged=true\n");
        dump_converged(in, lut, prof);
    } else {
        std::printf("  RESULT: converged=false (honest fallback — acceptable for this gate)\n");
    }

    // -----------------------------------------------------------------------
    // (2) Degenerate input: mdot=0 must return converged=false, no crash.
    // -----------------------------------------------------------------------
    std::printf("\n=== (2) degenerate input: mdot=0 ===\n");
    SlimDiskInputs ind = in;
    ind.mdot = 0.0;
    bool threw2 = false;
    SlimDiskRadial profd;
    try {
        profd = solve_slim_disk_coupled(ind, lut);
    } catch (...) { threw2 = true; std::printf("  EXCEPTION on degenerate input\n"); }
    if (threw2) { pass = 0; std::printf("  FAIL: degenerate input threw\n"); }
    else if (profd.converged) {
        pass = 0; std::printf("  FAIL: degenerate (mdot=0) returned converged=true\n");
    } else {
        std::printf("  OK: mdot=0 -> converged=false (no crash)\n");
    }

    std::printf("\n%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
