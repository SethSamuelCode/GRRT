// ===========================================================================
// SLIM f_adv FREEDOM PROBE  (decisive numerical validation — DELETABLE)
// ---------------------------------------------------------------------------
// QUESTION: The coupled column re-pose pins (Sigma, T_c) AND fixes f_adv as an
// input. But the vertical structure is a TWO-parameter family (Sadowski 2011:
// (T_c, f_adv) <=> (T_c, Sigma) with f_adv DETERMINED by (T_c, Sigma)). If so,
// fixing all three (Sigma, T_c, f_adv) is over-determined, and the "folds" seen
// when perturbing Sigma at fixed (T_c, f_adv) are ARTIFACTS of the held-fixed
// f_adv — not physical turning points.
//
// TEST: Does a column solution exist at the "folded" targets (T_c, 1.3*Sigma)
// and (T_c, 0.7*Sigma) when f_adv is allowed to VARY? Solve the 2x2 system
//   eval(T_eff, f_adv) = (T_c_target, Sigma_target)
// with the BASE solver as a black box (no new column physics).
//
// Build: cmake --build build --config Release --target slim-fadv-freedom-probe
// Run:   build/Release/slim-fadv-freedom-probe.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <array>

using namespace grrt;

// Base inputs as specified by the task (reference uses T_eff=3e5, f_adv=0).
static ColumnInputs make_base() {
    ColumnInputs in{};
    in.T_eff = 3e5;
    in.shear = 2e3;
    in.omega_z = 2e3;
    in.alpha = 0.1;
    in.rho_mid_guess = 1.0;
    in.n_nodes = 96;
    in.max_iters = 300;
    in.tol = 1e-8;
    in.f_adv = 0.0;
    return in;
}

struct EvalResult {
    bool   converged = false;
    double Tc = 0.0;     // midplane temperature = T.front()
    double Sigma = 0.0;  // Sigma0
};

static EvalResult eval(const OpacityLUTs& op, double Teff, double fadv) {
    ColumnInputs in = make_base();
    in.T_eff = Teff;
    in.f_adv = fadv;
    ColumnBVPSolution s = solve_column_bvp(in, op);
    EvalResult r;
    r.converged = s.converged;
    if (s.converged && !s.T.empty()) {
        r.Tc = s.T.front();
        r.Sigma = s.Sigma0;
    }
    return r;
}

int main() {
    auto op = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);

    std::printf("# slim-fadv-freedom-probe : does freeing f_adv dissolve the Sigma-folds?\n");
    std::printf("# base: T_eff=3e5 shear=2e3 omega_z=2e3 alpha=0.1 rho_guess=1 n=96\n\n");

    // ---- 1. Reference solve at f_adv = 0 ------------------------------------
    EvalResult ref = eval(op, 3e5, 0.0);
    if (!ref.converged) {
        std::printf("ABORT: reference solve (T_eff=3e5, f_adv=0) did not converge.\n");
        return 1;
    }
    const double Sigma0_ref = ref.Sigma;
    const double Tc_ref = ref.Tc;
    std::printf("REFERENCE (f_adv=0): Sigma0 = %.6e g/cm^2   T_c = %.6e K\n\n",
                Sigma0_ref, Tc_ref);

    // ---- 4. Newton-solve the 2x2 system at each scaled Sigma target ---------
    auto rms_inf = [](double a, double b) { return std::max(std::abs(a), std::abs(b)); };

    for (double scale : {1.3, 0.7}) {
        const double Tc_target = Tc_ref;
        const double Sigma_target = scale * Sigma0_ref;
        std::printf("==== TARGET: (T_c=%.6e, Sigma=%.6e = %.2f x ref) ====\n",
                    Tc_target, Sigma_target, scale);

        double Teff = 3e5, fadv = 0.0;
        bool   solved = false;
        double res_inf = 1e300;
        double ach_Tc = 0.0, ach_Sig = 0.0;

        // residual r = (Tc/Tc_target - 1, Sigma/Sigma_target - 1)
        auto residual = [&](double Te, double fa, bool& ok) -> std::array<double,2> {
            EvalResult e = eval(op, Te, fa);
            ok = e.converged;
            if (!ok) return {1e300, 1e300};
            return { e.Tc / Tc_target - 1.0, e.Sigma / Sigma_target - 1.0 };
        };

        bool ok0 = false;
        std::array<double,2> r0 = residual(Teff, fadv, ok0);
        if (!ok0) {
            std::printf("  start point did not converge (unexpected); skipping.\n\n");
            continue;
        }

        for (int iter = 0; iter < 40; ++iter) {
            res_inf = rms_inf(r0[0], r0[1]);
            if (res_inf < 1e-6) { solved = true; break; }

            // 2x2 forward-difference Jacobian.
            const double dTe = 1e-3 * Teff;
            const double dfa = (std::abs(fadv) < 1e-9) ? 1e-3 : 1e-3;  // absolute 1e-3
            bool okT = false, okF = false;
            std::array<double,2> rT = residual(Teff + dTe, fadv, okT);
            std::array<double,2> rF = residual(Teff, fadv + dfa, okF);
            if (!okT || !okF) {
                std::printf("  iter %2d: Jacobian probe non-converged (Te+:%d fa+:%d) — abort Newton.\n",
                            iter, (int)okT, (int)okF);
                break;
            }
            // J columns: d r / d Teff , d r / d fadv
            const double J00 = (rT[0] - r0[0]) / dTe, J10 = (rT[1] - r0[1]) / dTe;
            const double J01 = (rF[0] - r0[0]) / dfa, J11 = (rF[1] - r0[1]) / dfa;
            const double det = J00 * J11 - J01 * J10;
            if (std::abs(det) < 1e-300) {
                std::printf("  iter %2d: singular Jacobian (det=%.3e) — abort.\n", iter, det);
                break;
            }
            // solve J * step = -r0
            const double sTe = -( J11 * r0[0] - J01 * r0[1]) / det;
            const double sFa = -(-J10 * r0[0] + J00 * r0[1]) / det;

            // line search: halve until ||r|| decreases AND base solve converges.
            double lambda = 1.0;
            bool   accepted = false;
            std::array<double,2> rNew = r0;
            double newTe = Teff, newFa = fadv;
            for (int bt = 0; bt < 20; ++bt) {
                double tTe = Teff + lambda * sTe;
                double tFa = fadv + lambda * sFa;
                // physicality cap: 1 + f_adv > 0
                if (tFa <= -0.999) { lambda *= 0.5; continue; }
                bool okN = false;
                std::array<double,2> rt = residual(tTe, tFa, okN);
                if (okN && rms_inf(rt[0], rt[1]) < res_inf) {
                    rNew = rt; newTe = tTe; newFa = tFa; accepted = true; break;
                }
                lambda *= 0.5;
            }
            if (!accepted) {
                std::printf("  iter %2d: line search failed (||r||_inf=%.3e); stopping.\n",
                            iter, res_inf);
                break;
            }
            Teff = newTe; fadv = newFa; r0 = rNew;
            std::printf("  iter %2d: Teff=%.6e fadv=%+.6e  ||r||_inf=%.3e\n",
                        iter, Teff, fadv, rms_inf(r0[0], r0[1]));
        }

        // achieved
        {
            bool okf = false;
            std::array<double,2> rf = residual(Teff, fadv, okf);
            res_inf = okf ? rms_inf(rf[0], rf[1]) : 1e300;
            EvalResult e = eval(op, Teff, fadv);
            ach_Tc = e.Tc; ach_Sig = e.Sigma;
            solved = okf && res_inf < 1e-6;
        }

        std::printf("  RESULT: %s\n", solved ? "CONVERGED" : "NOT converged");
        std::printf("    final (T_eff, f_adv) = (%.6e, %+.6e)\n", Teff, fadv);
        std::printf("    achieved (T_c, Sigma) = (%.6e, %.6e)\n", ach_Tc, ach_Sig);
        std::printf("    target   (T_c, Sigma) = (%.6e, %.6e)\n", Tc_target, Sigma_target);
        std::printf("    residual ||r||_inf = %.3e\n", res_inf);
        std::printf("    f_adv physical (1+f_adv>0)? %s  (1+f_adv = %.6e)\n\n",
                    (fadv > -1.0) ? "YES" : "NO", 1.0 + fadv);
    }

    // ---- 5. Context sweep: Sigma vs f_adv at fixed T_eff=3e5 ----------------
    std::printf("==== SWEEP: f_adv -> (T_c, Sigma) at fixed T_eff=3e5 ====\n");
    std::printf("  %-10s %-14s %-14s %-6s\n", "f_adv", "T_c [K]", "Sigma [g/cm^2]", "conv");
    double sig_at_fadv[5]; double fadv_grid[5] = {-0.5, -0.25, 0.0, 0.25, 0.5};
    for (int i = 0; i < 5; ++i) {
        EvalResult e = eval(op, 3e5, fadv_grid[i]);
        sig_at_fadv[i] = e.converged ? e.Sigma : NAN;
        std::printf("  %-+10.3f %-14.6e %-14.6e %-6d\n",
                    fadv_grid[i], e.Tc, e.Sigma, (int)e.converged);
    }
    // dSigma/dfadv at fadv=0 by central difference (+/-0.25 grid points).
    if (std::isfinite(sig_at_fadv[1]) && std::isfinite(sig_at_fadv[3])) {
        const double dSig = (sig_at_fadv[3] - sig_at_fadv[1]) / (0.25 - (-0.25));
        std::printf("\n  dSigma/df_adv at f_adv=0 (central, +/-0.25) = %.6e g/cm^2 per unit f_adv\n", dSig);
        if (std::isfinite(sig_at_fadv[2]) && sig_at_fadv[2] != 0.0)
            std::printf("  relative: (dSigma/df_adv)/Sigma(0) = %.4f per unit f_adv\n",
                        dSig / sig_at_fadv[2]);
    }

    std::printf("\n[slim-fadv-freedom-probe] done.\n");
    return 0;
}
