// ===========================================================================
// SLIM RICHARDSON CAPACITY PROBE  (DIAGNOSTIC — DELETABLE)
// ---------------------------------------------------------------------------
// THE PIVOT-DECIDING TEST.  For the two "resistant" inner nodes (3 and 9) at the
// f_Edd=0.001, a=0.9 base rung, measure the column Σ0 capacity ceiling at
// n_z = 96 / 128 / 192 / 256 and Richardson-extrapolate the n_z→∞ limit.
//   limit ABOVE demand  -> feasibility is a RESOLUTION artifact (curable: higher
//                          n_z or a stretched vertical grid). Split stays viable.
//   limit PLATEAUS BELOW demand -> the Σ demand is PHYSICAL, unsupportable by a
//                          resolved column -> pivot to the monolithic Sądowski build.
//
// Capacity ceiling = max converged Σ0 over a T_eff×f_adv envelope (Σ0 is an OUTPUT
// of the base column solver, so the max over T_eff is the robust capacity), same
// method/envelope as slim_nz_refine_probe Part 2, so numbers are comparable.
//
// Build:  cmake --build build --config Release --target slim-richardson-probe
// Run:    build/Release/slim-richardson-probe.exe
// REUSE: same include-the-.cpp order as the other coupled probes; does NOT link grrt.
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
#include <chrono>

using namespace grrt;
using namespace grrt::slim_coupled_detail;

static constexpr double R_G_10MSUN = 1.48e6;

static double mdot_from_fEdd(const SlimDiskInputs& in, double f_Edd) {
    using namespace constants;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / 0.34;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    return f_Edd * Mdot_Edd;
}

// Max converged Σ0 over a T_eff×f_adv envelope at resolution nz (robust capacity).
static double sigma0_ceiling(const SlimDiskInputs& in, const OpacityLUTs& op, int nz,
                             double shear, double omega_z, double rho_mid_guess,
                             int NT, double Te_lo, double Te_hi,
                             const std::vector<double>& fadv, int& nconv) {
    double best = -1.0; nconv = 0;
    const int denom = std::max(NT - 1, 1);
    for (int it = 0; it < NT; ++it) {
        const double Te = Te_lo * std::pow(Te_hi / Te_lo, double(it) / double(denom));
        for (double fa : fadv) {
            ColumnInputs b{};
            b.T_eff = Te; b.shear = std::max(shear, 1e-300); b.omega_z = std::max(omega_z, 1e-300);
            b.alpha = in.alpha; b.f_adv = fa; b.rho_mid_guess = rho_mid_guess;
            b.n_nodes = nz; b.max_iters = 300; b.tol = 1e-8;
            ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
            if (s.converged) { ++nconv; if (s.Sigma0 > best) best = s.Sigma0; }
        }
    }
    return best;
}

int main() {
    std::setbuf(stdout, nullptr);
    const auto t0 = std::chrono::steady_clock::now();
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    SlimDiskInputs in{};
    in.mass=1.0; in.spin=0.9; in.alpha=0.1; in.r_g=R_G_10MSUN;
    in.r_out=50.0; in.n_nodes=18; in.tol=1e-8;
    in.r_in = 0.5 * slim_detail::isco_prograde(in.mass, in.spin);
    in.mdot = mdot_from_fEdd(in, 1e-3);

    // Thin seed = the demanded (Σ, T_c) at the base rung (calibration only sets T_c,
    // not Σ, so the demanded Σ is the thin-seed Σ directly).
    std::vector<double> U = build_thin_disk_seed(in, op);
    const int N = std::max(in.n_nodes, 4);
    const double r_s = U[4*N+1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);

    std::printf("# =====================================================================\n");
    std::printf("# slim-richardson-probe : Σ0 capacity vs n_z for the resistant nodes\n");
    std::printf("#   a=%.3f f_Edd=0.001  r_s=%.4f  envelope T_eff[1e5,5e7]x20 x f_adv{2,0,-.5,-.9}\n", in.spin, r_s);
    std::printf("# =====================================================================\n\n");

    const std::vector<double> fadv = {2.0, 0.0, -0.5, -0.9};
    const std::vector<int> nz_ladder = {96, 128, 192, 256};
    const int study[] = {3, 9};

    for (int idx = 0; idx < 2; ++idx) {
        const int i = study[idx];
        const double t = double(i)/double(N-1);
        const double ri = std::exp(lr0 + (lr1-lr0)*t);
        const int j = (i+1<N)?i+1:i-1;
        const double tj = double(j)/double(N-1);
        const double rj = std::exp(lr0 + (lr1-lr0)*tj);
        const double Omi = slim_detail::omega_from_ell(in.mass,in.spin,ri,U[4*i+2]);
        const double Omj = slim_detail::omega_from_ell(in.mass,in.spin,rj,U[4*j+2]);
        const double shear = shear_cgs(in, ri, Omi, rj, Omj);
        const double omz   = omega_perp_cgs(in, ri);
        const double Sig_demand = U[4*i+0];
        const double Tc = std::max(U[4*i+3], 1.0);
        const double rho_mid = std::max(slim_detail::one_zone_closure(Sig_demand, Tc, ri, in, op).rho_mid, 1e-30);

        std::printf("### NODE %d  r=%.4f M  Σ_demand=%.4e  shear=%.3e  omega_z=%.3e ###\n",
                    i, ri, Sig_demand, shear, omz);
        std::printf("    %-6s %-13s %-11s %-10s %-8s\n", "n_z", "Sigma0_max", "cap/demand", "feasible?", "nconv");
        std::vector<double> caps;
        for (int nz : nz_ladder) {
            int nconv=0;
            const double cap = sigma0_ceiling(in, op, nz, shear, omz, rho_mid, 20, 1e5, 5e7, fadv, nconv);
            caps.push_back(cap);
            const double ratio = cap / std::max(Sig_demand, 1.0);
            std::printf("    %-6d %-13.4e %-11.4f %-10s %-8d\n",
                        nz, cap, ratio, (cap >= Sig_demand) ? "YES" : "no", nconv);
        }
        // Richardson geometric-tail extrapolation from the last three points.
        if (caps.size() >= 3) {
            const int m = (int)caps.size();
            const double d1 = caps[m-2] - caps[m-3];
            const double d2 = caps[m-1] - caps[m-2];
            const double q = (std::abs(d1) > 0) ? d2/d1 : 1.0;
            if (q > 0.0 && q < 1.0) {
                const double limit = caps[m-1] + d2 * (q/(1.0-q));
                std::printf("    -> geometric-tail ratio q=%.3f  Σ0_max(n_z→∞)≈%.4e  limit/demand=%.3f  => %s\n",
                            q, limit, limit/Sig_demand,
                            (limit >= Sig_demand) ? "RESOLUTION (curable)" : "PHYSICAL WALL (rebuild)");
            } else {
                std::printf("    -> increments not geometrically shrinking (q=%.3f) — capacity not clearly converging; inspect rows\n", q);
            }
        }
        std::printf("\n");
    }
    std::printf("wall %.1f s\nDONE\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());
    return 0;
}
