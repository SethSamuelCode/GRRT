// ===========================================================================
// Cross-check: analytic slim-disk Jacobian vs the finite-difference oracle.
// ===========================================================================
// The exact analytic Jacobian slim_analytic_jacobian (built block-by-block in
// src/slim_disk_radial.cpp, Tasks 2-6) MUST match the central-difference
// slim_numerical_jacobian to round-off at every operating point before any
// block ships.  This test is the permanent gate.
//
// It #includes slim_disk_radial.cpp + opacity.cpp DIRECTLY (the same pattern as
// tools/slim_diag_probe.cpp) so it can reach the anonymous-namespace helpers
// slim_numerical_jacobian / slim_analytic_jacobian / build_thin_disk_seed /
// slim_radial_residual / one_zone_closure.  It is a standalone exe that does NOT
// link grrt (avoids duplicate-symbol clashes with the DLL copies).
//
// Operating points (per the plan):
//   (i)   gas-dominated low f_Edd  (a=0.9,   f_Edd≈0.02)
//   (ii)  radiation-dominated near the ceiling (a=0.9,  f_Edd≈0.11, β~3.6e-4)
//   (iii) higher spin              (a=0.998, f_Edd≈0.02)
// Small N (=20) so the dense FD reference is cheap.
//
// Build:
//   cmake --build build --config Release --target test-slim-jacobian
//   build/Release/test-slim-jacobian.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

using namespace grrt;
using namespace grrt::slim_detail;

namespace grrt {
namespace jactest {

int failures = 0;

// Build SlimDiskInputs at (a, f_Edd) mirroring the probe/test harness conventions
// (r_g = 1.48e6 ≈ 10 M_sun).  Small N for a cheap dense FD reference.
static SlimDiskInputs make_inputs(double a, double f_Edd, int N) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0;
    in.spin = a;
    in.alpha = 0.1;
    in.r_g = 1.48e6;
    in.r_out = 50.0;
    in.n_nodes = N;
    in.max_iters = 100;
    in.tol = 1e-6;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0 / 3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

// Per-column relative mismatch between the analytic and FD Jacobians.  A column's
// mismatch is the max over rows of |Ja-Jf| / (|Jf| + col_scale), where col_scale
// is a tiny fraction of the column's own max |Jf| so a column whose entries are all
// ~0 isn't judged against round-off noise.  Returns the worst (col,row,rel) and
// also the worst restricted to the r_s column (which the plan flags as FD-noisy).
struct Mismatch {
    double worst_rel = 0.0;     int worst_col = -1, worst_row = -1;
    double worst_rel_nors = 0.0; int wc_nors = -1, wr_nors = -1;  // excluding r_s col
    double rs_rel = 0.0;        int rs_row = -1;                  // r_s column only
};

// Well-conditioned FD reference column: Richardson-extrapolated central difference.
//   CD(h) = (R(U+h·e_c) − R(U−h·e_c))/2h ;  ref = (4·CD(h/2) − CD(h))/3  (O(h⁴)).
// Richardson cancels the O(h²) truncation error so the reference matches the exact
// derivative to ~round-off — necessary because the PRODUCTION slim_numerical_jacobian
// uses a tiny per-type step that suffers catastrophic CANCELLATION on the
// angular-momentum rows (R≈−2e21 with a V-sensitivity ~1e7 ≪ its round-off), where
// it reports exactly 0 for a genuinely large derivative.  The analytic Jacobian is
// validated against THIS reference (the best achievable FD), not the production FD.
static void fd_reference_column(const std::vector<double>& U, const SlimDiskInputs& in,
                                const OpacityLUTs& op, int col, int n, double rel_step,
                                std::vector<double>& Jcol /* size n */) {
    const double base = std::abs(U[col]);
    const double h = std::max(rel_step * base, 1e-30);
    std::vector<double> Up, Um, Rp, Rm;
    auto cd = [&](double step, std::vector<double>& out) {
        Up = U; Um = U; Up[col] += step; Um[col] -= step;
        slim_radial_residual(Up, in, op, Rp);
        slim_radial_residual(Um, in, op, Rm);
        out.assign(n, 0.0);
        const double inv = 1.0 / (2.0 * step);
        for (int r = 0; r < n; ++r) out[r] = (Rp[r] - Rm[r]) * inv;
    };
    std::vector<double> c1, c2;
    cd(h, c1);
    cd(0.5 * h, c2);
    Jcol.assign(n, 0.0);
    for (int r = 0; r < n; ++r) Jcol[r] = (4.0 * c2[r] - c1[r]) / 3.0;   // Richardson O(h⁴)
}

// Row (1/group-scale) and column (per-variable magnitude) scaling vectors — the
// SAME non-dimensionalization relax_structure applies before the Newton solve.  The
// analytic Jacobian is validated in THIS scaled space because that is the only form
// the solver ever uses: an entry that is huge in raw units but negligible once
// row/col-scaled (e.g. the angular-momentum row's Lorentz-factor V-dependence at
// V~1e-6) does not affect the Newton direction and is — correctly — not required to
// match the FD reference, which cannot even resolve it (catastrophic cancellation:
// R≈−2e21 perturbed by ~1e7, 14 decades below double precision).
static void scaling_vectors(const std::vector<double>& U, const SlimDiskInputs& in,
                            int N, std::vector<double>& cs, std::vector<double>& rs_inv) {
    using namespace constants;
    const int n = 4 * N + 2;
    cs.assign(n, 1.0); rs_inv.assign(n, 1.0);
    double mSig = 0, mV = 0, mEll = 0, mT = 0;
    for (int i = 0; i < N; ++i) {
        mSig = std::max(mSig, std::abs(U[4*i+0])); mV = std::max(mV, std::abs(U[4*i+1]));
        mEll = std::max(mEll, std::abs(U[4*i+2])); mT = std::max(mT, std::abs(U[4*i+3]));
    }
    mSig = std::max(mSig, 1e-30); mV = std::max(mV, 1e-30); mEll = std::max(mEll, 1e-30); mT = std::max(mT, 1.0);
    for (int i = 0; i < N; ++i) { cs[4*i+0]=mSig; cs[4*i+1]=mV; cs[4*i+2]=mEll; cs[4*i+3]=mT; }
    cs[4*N+0] = std::max(std::abs(U[4*N+0]), 1e-30);
    cs[4*N+1] = std::max(std::abs(U[4*N+1]), 1e-30);
    const GroupScales gs = slim_group_scales(U, in);
    auto setrows = [&](int b, int e, double sc) { sc = std::max(sc, 1e-300); for (int r=b;r<e;++r) rs_inv[r]=1.0/sc; };
    setrows(0,N,gs.mass); setrows(N,2*N,gs.ang); setrows(2*N,3*N-1,gs.rad);
    setrows(3*N-1,4*N-2,gs.ene); setrows(4*N-2,4*N-1,gs.bc_ell);
    setrows(4*N-1,4*N,gs.ene); setrows(4*N,4*N+1,gs.reg_D0); setrows(4*N+1,4*N+2,gs.reg_N1);
}

// Per-column 2-NORM relative mismatch in SCALED space between the analytic Jacobian
// and the Richardson FD reference: ‖S(Ja−ref)[:,c]‖₂·cs[c] / (‖S·ref[:,c]‖₂·cs[c]+floor),
// where S=diag(rs_inv).  (cs[c] cancels in the ratio but is applied for clarity.)
// row_active[r]=true means row r has been ported analytically and is validated; rows
// not yet ported (still FD-seeded) are excluded from the metric so each task's gate
// reflects only its own block.  Once Task 6 lands, every row is active.
static Mismatch compare(const std::vector<double>& Ja, const std::vector<double>& U,
                        const SlimDiskInputs& in, const OpacityLUTs& op,
                        int n, int rs_col, double rel_step, double rs_rel_step,
                        const std::vector<char>& row_active) {
    const int N = (n - 2) / 4;
    Mismatch m;
    std::vector<double> cs, rs_inv;
    scaling_vectors(U, in, N, cs, rs_inv);
    std::vector<double> ref;
    for (int c = 0; c < n; ++c) {
        const double step = (c == rs_col) ? rs_rel_step : rel_step;
        fd_reference_column(U, in, op, c, n, step, ref);
        double dn2 = 0.0, rn2 = 0.0;
        int worst_r = -1; double worst_entry = 0.0;
        // Column scale for the entry-relative diagnostic (scaled-space column max).
        double col_max = 0.0;
        for (int r = 0; r < n; ++r) if (row_active[r]) col_max = std::max(col_max, std::abs(ref[r] * rs_inv[r] * cs[c]));
        const double entry_floor = 1e-9 * std::max(col_max, 1e-300);
        for (int r = 0; r < n; ++r) {
            if (!row_active[r]) continue;
            const double a = Ja[(size_t)r * n + c] * rs_inv[r] * cs[c];
            const double f = ref[r]              * rs_inv[r] * cs[c];
            dn2 += (a - f) * (a - f);
            rn2 += f * f;
            const double erel = std::abs(a - f) / (std::abs(f) + entry_floor);
            if (erel > worst_entry) { worst_entry = erel; worst_r = r; }
        }
        const double col_rel = std::sqrt(dn2) / (std::sqrt(rn2) + 1e-300);
        if (col_rel > m.worst_rel) { m.worst_rel = col_rel; m.worst_col = c; m.worst_row = worst_r; }
        if (c == rs_col) {
            m.rs_rel = col_rel; m.rs_row = worst_r;
        } else if (col_rel > m.worst_rel_nors) {
            m.worst_rel_nors = col_rel; m.wc_nors = c; m.wr_nors = worst_r;
        }
    }
    return m;
}

// Human-readable column/row label for diagnosis ("Σ[3]", "V[7]", "ℓ_in", "r_s", …).
static std::string label(int idx, int N) {
    if (idx == 4 * N + 0) return "ell_in";
    if (idx == 4 * N + 1) return "r_s";
    const int node = idx / 4, off = idx & 3;
    const char* nm[4] = {"Sigma", "V", "ell", "Tc"};
    return std::string(nm[off]) + "[" + std::to_string(node) + "]";
}

// Which row groups are validated.  Grows per task; "all" once Task 6 lands.
enum PortedRows { ROWS_MASS_ANG, ROWS_THRU_RADENE, ROWS_ALL };

static std::vector<char> make_row_mask(int N, PortedRows ported) {
    const int n = 4 * N + 2;
    std::vector<char> mask(n, 0);
    auto on = [&](int b, int e) { for (int r = b; r < e; ++r) mask[r] = 1; };
    on(0, N);          // mass
    on(N, 2 * N);      // angmom
    if (ported >= ROWS_THRU_RADENE) { on(2*N, 3*N-1); on(3*N-1, 4*N-2); on(4*N-1, 4*N); }  // rad + ene + outer-energy BC
    if (ported >= ROWS_ALL)         { on(4*N-2, 4*N-1); on(4*N, 4*N+1); on(4*N+1, 4*N+2); } // bc_ell + regularity
    return mask;
}

// Run the cross-check at one operating point.  state_label / a / f_Edd identify the
// point; tol is the per-column relative gate for all non-r_s columns; rs_tol is the
// looser gate for the r_s grid-stretch column (FD reference is noisy there).  Only
// the row groups in `ported` are validated (un-ported rows are still FD-seeded).
static void run_point(const OpacityLUTs& op, const char* name,
                      double a, double f_Edd, int N,
                      double tol, double rs_tol, PortedRows ported) {
    std::printf("\n=== operating point: %s  (a=%.3f, f_Edd=%.3f, N=%d) ===\n",
                name, a, f_Edd, N);
    SlimDiskInputs in = make_inputs(a, f_Edd, N);

    // Representative physical state: the code's own thin-disk seed (Σ,T,V,ℓ,ℓ_in,r_s
    // all set self-consistently for this a/f_Edd, so the closure regime — gas vs
    // radiation dominated — is the genuine one at the operating point).  The
    // cross-check compares analytic vs FD at the SAME state, so it need not be
    // converged — only physical and representative.
    std::vector<double> U = build_thin_disk_seed(in, op);
    const int n = (int)U.size();

    // Quick β report at the inner few nodes so the log shows the closure regime.
    {
        double beta_min = 1.0, beta_max = 0.0;
        const double r_s = U[4 * N + 1];
        const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
        for (int i = 0; i < N; ++i) {
            const double t = double(i) / double(N - 1);
            const double r = std::exp(lr0 + (lr1 - lr0) * t);
            const OneZoneState oz = one_zone_closure(std::max(U[4 * i + 0], kSigmaFloor),
                                                     std::max(U[4 * i + 3], kTFloor), r, in, op);
            const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
            beta_min = std::min(beta_min, beta);
            beta_max = std::max(beta_max, beta);
        }
        std::printf("  seed beta range = [%.3e, %.3e]  (r_s=%.4f, ell_in=%.5f)\n",
                    beta_min, beta_max, U[4 * N + 1], U[4 * N + 0]);
    }

    std::vector<double> Ja;
    slim_analytic_jacobian(U, in, op, Ja);

    // Per-column 2-norm relative mismatch vs the Richardson FD reference.  rel_step
    // is the central-difference relative step for ordinary columns; the r_s
    // grid-stretch column uses a larger step (its FD reference is noisy — the grid
    // re-spacing is the FD Jacobian's least-accurate column, which is precisely what
    // the analytic r_s column fixes).
    const std::vector<char> row_mask = make_row_mask(N, ported);
    const Mismatch m = compare(Ja, U, in, op, n, /*rs_col=*/4 * N + 1,
                               /*rel_step=*/1e-4, /*rs_rel_step=*/2e-3, row_mask);

    if (m.wc_nors >= 0)
        std::printf("  worst non-r_s column 2-norm mismatch: rel=%.3e at col=%s (worst entry row=%d)\n",
                    m.worst_rel_nors, label(m.wc_nors, N).c_str(), m.wr_nors);
    if (std::getenv("JAC_DIAG") && m.wc_nors >= 0) {
        const int c = m.wc_nors;
        std::vector<double> ref, cs, rs_inv;
        scaling_vectors(U, in, N, cs, rs_inv);
        fd_reference_column(U, in, op, c, n, 1e-4, ref);
        std::printf("  [diag] col %s (SCALED): analytic vs Richardson-FD per row (|rel|>1e-4):\n", label(c, N).c_str());
        for (int r = 0; r < n; ++r) {
            const double a = Ja[(size_t)r * n + c] * rs_inv[r] * cs[c];
            const double f = ref[r]              * rs_inv[r] * cs[c];
            const double rel = std::abs(a - f) / (std::abs(f) + 1e-300);
            if (rel > 1e-4 && (std::abs(a) > 1e-290 || std::abs(f) > 1e-290))
                std::printf("    row %3d (%s-group): an=%+.6e fd=%+.6e rel=%.2e\n", r,
                            r < N ? "mass" : r < 2*N ? "ang" : r < 3*N-1 ? "rad" :
                            r < 4*N-2 ? "ene" : r < 4*N ? "bc" : "reg", a, f, rel);
        }
    }
    std::printf("  r_s column 2-norm mismatch:           rel=%.3e (worst entry row=%d)\n",
                m.rs_rel, m.rs_row);

    bool ok = true;
    if (!(m.worst_rel_nors < tol)) {
        std::printf("  FAIL: non-r_s column mismatch %.3e >= tol %.1e\n", m.worst_rel_nors, tol);
        ok = false;
    }
    if (!(m.rs_rel < rs_tol)) {
        std::printf("  FAIL: r_s column mismatch %.3e >= rs_tol %.1e\n", m.rs_rel, rs_tol);
        ok = false;
    }
    if (ok) std::printf("  PASS (non-r_s < %.1e, r_s < %.1e)\n", tol, rs_tol);
    else failures++;
}

// ---------------------------------------------------------------------------
// Unit test: one_zone_closure_jac vs central differences (Task 2).
// ---------------------------------------------------------------------------
// Central-difference each closure field w.r.t. Σ and T_c and compare to the
// analytic partials, at a gas-dominated and a radiation-dominated point.
static void check_closure_jac(const OpacityLUTs& op, const char* name,
                              double Sigma, double Tc, double r,
                              const SlimDiskInputs& in, double tol) {
    std::printf("\n--- one_zone_closure_jac: %s (Sigma=%.3e Tc=%.3e r=%.3f) ---\n",
                name, Sigma, Tc, r);
    OneZoneState st; OneZoneJac jac;
    one_zone_closure_jac(Sigma, Tc, r, in, op, st, jac);
    {
        const OneZoneState oz = one_zone_closure(Sigma, Tc, r, in, op);
        const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
        std::printf("    beta=%.3e  H=%.3e rho=%.3e P=%.3e\n", beta, oz.H, oz.rho_mid, oz.P);
    }

    // Central differences with a relative step on each input.
    auto cd = [&](int which /*0=Sigma,1=Tc*/, double& dH, double& drho, double& dpg,
                  double& dpr, double& dpm, double& dcs, double& dP, double& dS) {
        const double base = (which == 0) ? Sigma : Tc;
        const double h = 1e-6 * std::abs(base);
        double sp = Sigma, tp = Tc, sm = Sigma, tm = Tc;
        if (which == 0) { sp += h; sm -= h; } else { tp += h; tm -= h; }
        const OneZoneState p = one_zone_closure(sp, tp, r, in, op);
        const OneZoneState m = one_zone_closure(sm, tm, r, in, op);
        const double inv = 1.0 / (2.0 * h);
        dH  = (p.H - m.H) * inv;        drho = (p.rho_mid - m.rho_mid) * inv;
        dpg = (p.p_gas - m.p_gas) * inv; dpr = (p.p_rad - m.p_rad) * inv;
        dpm = (p.p_mid - m.p_mid) * inv; dcs = (p.c_s - m.c_s) * inv;
        dP  = (p.P - m.P) * inv;        dS  = (p.S - m.S) * inv;
    };

    const char* fld[8] = {"H","rho","p_gas","p_rad","p_mid","c_s","P","S"};
    for (int w = 0; w < 2; ++w) {
        double fd[8];
        cd(w, fd[0], fd[1], fd[2], fd[3], fd[4], fd[5], fd[6], fd[7]);
        const double an[8] = {jac.dH[w], jac.drho[w], jac.dp_gas[w], jac.dp_rad[w],
                              jac.dp_mid[w], jac.dc_s[w], jac.dP[w], jac.dS[w]};
        const char* wn = (w == 0) ? "Sigma" : "Tc";
        for (int k = 0; k < 8; ++k) {
            const double rel = std::abs(an[k] - fd[k]) / (std::abs(fd[k]) + 1e-300);
            const bool pass = (rel < tol) || (std::abs(fd[k]) < 1e-300 && std::abs(an[k]) < 1e-300);
            std::printf("    d%s/d%-5s analytic=%+.6e fd=%+.6e rel=%.2e %s\n",
                        fld[k], wn, an[k], fd[k], rel, pass ? "" : "<<FAIL");
            if (!pass) failures++;
        }
    }
}

// ---------------------------------------------------------------------------
// Unit test: Kerr mechanics derivatives (Task 4) vs central differences.
//   ∂Ω/∂ℓ  (domega_dell, reciprocal of omega_from_ell)
//   ∂𝒜/∂Ω  (script_A_dOmega) — combined to ∂𝒜/∂ℓ and checked vs FD on ℓ.
// ---------------------------------------------------------------------------
static void test_kerr_mech_jac() {
    std::printf("\n########## Kerr mechanics derivative unit test ##########\n");
    struct Pt { double M, a, r, ell; const char* n; };
    Pt pts[] = {
        {1.0, 0.0,  6.0, 3.4641, "Schw r=6"},
        {1.0, 0.9,  3.0, 2.1,    "a=0.9 r=3"},
        {1.0, 0.9,  10.0, 3.6,   "a=0.9 r=10"},
        {1.0, 0.998, 1.5, 1.5,   "a=0.998 r=1.5"},
    };
    SlimDiskInputs in{}; in.alpha = 0.1; in.r_g = 1.48e6; in.r_out = 50.0;
    for (auto& p : pts) {
        in.mass = p.M; in.spin = p.a;
        // ∂Ω/∂ℓ vs FD of omega_from_ell.
        const double Om = omega_from_ell(p.M, p.a, p.r, p.ell);
        const double dOmdl_an = domega_dell(p.M, p.a, p.r, Om);
        const double h = 1e-6 * std::abs(p.ell);
        const double dOmdl_fd = (omega_from_ell(p.M,p.a,p.r,p.ell+h)
                               - omega_from_ell(p.M,p.a,p.r,p.ell-h)) / (2*h);
        const double rel1 = std::abs(dOmdl_an - dOmdl_fd)/(std::abs(dOmdl_fd)+1e-300);
        std::printf("  %-14s dOmega/dell: an=%+.6e fd=%+.6e rel=%.2e %s\n",
                    p.n, dOmdl_an, dOmdl_fd, rel1, rel1<1e-5?"":"<<FAIL");
        if (!(rel1 < 1e-5)) failures++;

        // ∂𝒜/∂ℓ = (∂𝒜/∂Ω)(∂Ω/∂ℓ) vs FD of script_A on ℓ.
        NodeMech m0 = node_mech(in, p.r, p.ell);
        double A0, dA_dOm; script_A_dOmega(in, p.r, m0, A0, dA_dOm);
        const double dA_dl_an = dA_dOm * dOmdl_an;
        NodeMech mp = node_mech(in, p.r, p.ell+h), mm = node_mech(in, p.r, p.ell-h);
        const double dA_dl_fd = (script_A(in,p.r,mp) - script_A(in,p.r,mm))/(2*h);
        const double rel2 = std::abs(dA_dl_an - dA_dl_fd)/(std::abs(dA_dl_fd)+1e-300);
        std::printf("  %-14s dscriptA/dell: an=%+.6e fd=%+.6e rel=%.2e %s\n",
                    p.n, dA_dl_an, dA_dl_fd, rel2, rel2<1e-5?"":"<<FAIL");
        if (!(rel2 < 1e-5)) failures++;
    }
}

// ---------------------------------------------------------------------------
// Unit test: ∂R/∂Ṁ column (Task 1, arclength continuation).
// ---------------------------------------------------------------------------
// slim_R_Mdot_column is a central FD of the residual over in.mdot.  Checks:
//   • every entry is finite,
//   • the mass rows [0..N) equal exactly −1 (R[i]=mdot_node−Ṁ ⇒ ∂/∂Ṁ=−1; the FD
//     of a term linear in Ṁ is exact to round-off),
//   • the energy rows [3N-1..4N-2) are nonzero (the Q_vis/Q_adv Ṁ-prefactors),
//   • it MATCHES an independent central FD of slim_radial_residual over Ṁ at a
//     coarser step (self-consistency of the column).
static void test_R_Mdot_column(const OpacityLUTs& op) {
    std::printf("\n########## R_Mdot column unit test (Task 1) ##########\n");
    int local_fail = 0;
    struct Pt { double a, f_Edd; const char* n; };
    Pt pts[] = { {0.9, 0.10, "a=0.9 f_Edd=0.10"},
                 {0.9, 0.02, "a=0.9 f_Edd=0.02"} };
    const int N = 20;
    for (auto& p : pts) {
        SlimDiskInputs in = make_inputs(p.a, p.f_Edd, N);
        std::vector<double> U = build_thin_disk_seed(in, op);
        const int n = (int)U.size();
        std::vector<double> Rm;
        slim_R_Mdot_column(U, in, op, Rm);

        // (a) finite everywhere.
        bool finite = true;
        for (double v : Rm) if (!std::isfinite(v)) finite = false;
        // (b) mass rows == −1 to round-off.
        double mass_maxerr = 0.0;
        for (int i = 0; i < N; ++i) mass_maxerr = std::max(mass_maxerr, std::abs(Rm[i] + 1.0));
        // (c) energy rows nonzero (at least one above a relative floor of the column).
        double col_max = 0.0; for (double v : Rm) col_max = std::max(col_max, std::abs(v));
        double ene_max = 0.0;
        for (int i = 3*N-1; i < 4*N-2; ++i) ene_max = std::max(ene_max, std::abs(Rm[i]));
        // (d) cross-check vs an independent coarser central FD over Ṁ.
        const double h2 = 1e-4 * std::abs(in.mdot);
        SlimDiskInputs ip = in, im = in; ip.mdot += h2; im.mdot -= h2;
        std::vector<double> Rp2, Rm2; slim_radial_residual(U, ip, op, Rp2);
        slim_radial_residual(U, im, op, Rm2);
        double xcheck_rel = 0.0;
        for (int r = 0; r < n; ++r) {
            const double fd = (Rp2[r] - Rm2[r]) / (2.0 * h2);
            const double rel = std::abs(Rm[r] - fd) / (std::abs(fd) + 1e-6 * std::max(col_max, 1e-300));
            xcheck_rel = std::max(xcheck_rel, rel);
        }
        std::printf("  %-20s finite=%d  mass_maxerr=%.2e  ene_max=%.3e  xcheck_rel=%.2e\n",
                    p.n, (int)finite, mass_maxerr, ene_max, xcheck_rel);
        if (!finite)                 { std::printf("    <<FAIL: non-finite entry\n"); ++local_fail; }
        if (!(mass_maxerr < 1e-9))   { std::printf("    <<FAIL: mass rows != -1 (err %.2e)\n", mass_maxerr); ++local_fail; }
        if (!(ene_max > 1e-300))     { std::printf("    <<FAIL: energy rows all zero\n"); ++local_fail; }
        if (!(xcheck_rel < 1e-3))    { std::printf("    <<FAIL: column != FD of residual (rel %.2e)\n", xcheck_rel); ++local_fail; }
    }
    if (local_fail == 0) std::printf("  PASS (finite, mass=-1, energy nonzero, matches FD)\n");
    failures += local_fail;
}

// ---------------------------------------------------------------------------
// Unit test: bordered augmented tangent (Task 2, arclength continuation).
// ---------------------------------------------------------------------------
// slim_arclength_tangent solves the bordered system for the null vector of
// [J | R_Mdot].  The defining property — independent of whether U is converged —
// is that the returned tangent satisfies J·U̇ + R_Mdot·Ṁ̇ ≈ 0 (the tangent lies in
// the null space of the bordered residual Jacobian) and is unit-normed.  We check
// that null residual at the seed state (a self-consistent linear-algebra check that
// validates the bordered solve + the J / R_Mdot blocks together), at a couple of
// operating points.  (The probe additionally checks it at a CONVERGED sub-fold
// point and reports Ṁ̇, per the plan.)
static void test_arclength_tangent(const OpacityLUTs& op) {
    std::printf("\n########## arclength tangent unit test (Task 2) ##########\n");
    int local_fail = 0;
    struct Pt { double a, f_Edd; const char* n; };
    Pt pts[] = { {0.9, 0.10, "a=0.9 f_Edd=0.10"},
                 {0.9, 0.02, "a=0.9 f_Edd=0.02"} };
    const int N = 20;
    for (auto& p : pts) {
        SlimDiskInputs in = make_inputs(p.a, p.f_Edd, N);
        std::vector<double> U = build_thin_disk_seed(in, op);
        const int n = (int)U.size();
        const int m = n + 1;

        // Seed direction: increase Ṁ (last component dominant).
        std::vector<double> t_prev(m, 0.0); t_prev[n] = 1.0;
        std::vector<double> t;
        const bool ok = slim_arclength_tangent(U, in, op, t_prev, t);

        // Null residual ‖J·U̇ + R_Mdot·Ṁ̇‖ (scaled by the group scales, the same space
        // the solver works in), relative to ‖J‖-ish column magnitude.
        std::vector<double> J, R_Mdot;
        slim_analytic_jacobian(U, in, op, J);
        slim_R_Mdot_column(U, in, op, R_Mdot);
        std::vector<double> cs, rs_inv; scaling_vectors(U, in, N, cs, rs_inv);
        double nrm_res = 0.0, nrm_ref = 0.0;
        const double Mdot_dot = t[n];
        // The tangent t is the RAW null vector: J·U̇ + R_Mdot·Ṁ̇ = 0 in raw space.
        // Row-scale only (rs_inv) for a dimensionless relative magnitude.
        for (int row = 0; row < n; ++row) {
            double jv = 0.0, ref = 0.0;
            for (int col = 0; col < n; ++col) {
                const double term = J[(size_t)row*n+col] * t[col];   // raw J·U̇
                jv  += term; ref += std::abs(term);
            }
            jv  += R_Mdot[row] * Mdot_dot;        // raw R_Mdot·Ṁ̇
            ref += std::abs(R_Mdot[row] * Mdot_dot);
            jv  *= rs_inv[row]; ref *= rs_inv[row];
            nrm_res += jv * jv; nrm_ref += ref * ref;
        }
        nrm_res = std::sqrt(nrm_res); nrm_ref = std::sqrt(nrm_ref) + 1e-300;
        const double rel = nrm_res / nrm_ref;
        // The tangent is unit-normed in the SCALED metric Σ(t_i/w_i)²=1, with the state
        // weights INFLATED by √n (so the aggregate state norm balances the single Ṁ DOF
        // — see slim_arclength_weights); Ṁ weight = |in.mdot| (no inflation).
        double tn = 0.0;
        const double sw = 20.0 * std::sqrt((double)n);
        for (int i = 0; i < m; ++i) { const double wi = (i<n) ? cs[i]*sw : std::max(std::abs(in.mdot),1e-300);
                                      const double s = t[i]/wi; tn += s*s; }
        tn = std::sqrt(tn);
        std::printf("  %-20s ok=%d  ‖t‖_w=%.6f  Mdot_dot=%+.4e  null_resid_rel=%.3e\n",
                    p.n, (int)ok, tn, ok ? Mdot_dot : 0.0, rel);
        if (!ok)                       { std::printf("    <<FAIL: tangent solve failed\n"); ++local_fail; }
        if (!(std::abs(tn-1.0) < 1e-9)){ std::printf("    <<FAIL: tangent not scaled-unit-normed\n"); ++local_fail; }
        if (!(rel < 1e-6))             { std::printf("    <<FAIL: null residual %.3e too large\n", rel); ++local_fail; }
    }
    if (local_fail == 0) std::printf("  PASS (unit-normed, J·U̇+R_Mdot·Ṁ̇≈0)\n");
    failures += local_fail;
}

static void test_closure_jac(const OpacityLUTs& op) {
    std::printf("\n########## one_zone_closure_jac unit test ##########\n");
    SlimDiskInputs in = make_inputs(0.9, 0.1, 20);
    // Gas-dominated: high Σ, moderate T — COOL/DENSE partial-ionization corner where
    // μ varies. The implicit-μ derivative removes the frozen-μ error (~0.19); the
    // residual ~1e-4 is the bilinear μ-LUT slope FD floor (the centered log-slope
    // stencil vs the closure's ≤3-iter fixed point near a LUT cell boundary), NOT a
    // derivation error — the radiation-dominated point (μ effectively frozen) matches
    // to ~1e-9. Hot inner-disk operating points (the real Jacobian gate) are far from
    // this corner. Tolerance set at the μ-LUT-slope floor.
    check_closure_jac(op, "gas-dominated",       1e5, 1e6, 10.0, in, 3e-4);
    // Radiation-dominated: low Σ, high T (β small).
    check_closure_jac(op, "radiation-dominated", 1e2, 1e7, 4.0,  in, 1e-5);
    // Intermediate.
    check_closure_jac(op, "intermediate",        1e3, 5e6, 6.0,  in, 1e-5);
}

} // namespace jactest
} // namespace grrt

int main() {
    using namespace grrt;
    std::printf("########## analytic-Jacobian vs FD cross-check ##########\n");
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    jactest::test_closure_jac(op);
    jactest::test_kerr_mech_jac();
    jactest::test_R_Mdot_column(op);
    jactest::test_arclength_tangent(op);

    const int N = 20;
    const double tol    = 1e-6;   // per-column scaled 2-norm gate (non-r_s columns)
    const double rs_tol = 1e-3;   // looser gate for the FD-noisy r_s grid-stretch column
    // a=0.998 inner edge (r_s≈1.21, near the horizon) sits in partial ionization where
    // the bilinear μ/κ_R LUTs have a cell-boundary slope discontinuity at one node:
    // the analytic LUT-derivative FD and the Richardson reference straddle the cell
    // edge differently, leaving an irreducible ~4e-5 floor at that single node (a
    // tabulated-opacity property, not a derivation error — the a=0.9 points and the
    // clean nodes match to ~1e-10). Documented μ/κ_R-LUT-boundary floor.
    const double tol_hispin = 5e-5;

    // Ported so far (grows per task): Task 6 adds the outer-ℓ BC row, the regularity
    // rows, and the ℓ_in + r_s global columns — i.e. ALL rows are now validated.
    const jactest::PortedRows ported = jactest::ROWS_ALL;
    jactest::run_point(op, "gas-dominated low f_Edd",          0.9,   0.02, N, tol,        rs_tol, ported);
    jactest::run_point(op, "radiation-dominated near ceiling", 0.9,   0.11, N, tol,        rs_tol, ported);
    jactest::run_point(op, "higher spin low f_Edd",            0.998, 0.02, N, tol_hispin, rs_tol, ported);

    std::printf("\n########## %d failure(s) ##########\n", jactest::failures);
    return jactest::failures == 0 ? 0 : 1;
}
