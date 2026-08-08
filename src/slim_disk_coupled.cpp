// ===========================================================================
// C4 + C5: nested coupled-column slim-disk radial driver (bring-up).
// ---------------------------------------------------------------------------
// REUSE MECHANISM (include-the-.cpp consumer — do NOT add to the grrt library):
//   This TU is #included into the test/probe TU AFTER, in this order,
//       opacity.cpp + disk_column_bvp.cpp + disk_column_coupled.cpp + slim_disk_radial.cpp
//   so that ALL of the radial solver's TU-local helpers (anonymous-namespace /
//   file-static) are in scope here and can be called DIRECTLY:
//       eval_node, node_mech, script_A, calD0, calN1, omega_perp2, omega_from_ell,
//       beta_of, eta3_of_beta, gtilde1_of_beta, mdot_of_node, kerr_delta, kerr_A,
//       dense_solve, deglitch_sigma_outliers, slim_validity_gate, slim_fadv_ok,
//       slim_radial_residual, slim_group_scales, slim_scaled_residual_norm,
//       slim_group_mags, GroupScales, NodeEval, NodeMech, OneZoneState, g_budget,
//       kSigmaFloor, kTFloor, kVCap.
//   Plus the column closure (solve_column_coupled / ColumnCoupledInputs /
//   ColumnClosure) from disk_column_coupled.cpp.
//
// WHAT THIS DOES — the one-zone closure is REPLACED per node by the in-tree vertical
// BVP column (the (Σ,T_c)→(F,z0,η3,η4,f_adv) map).  The radial residual is reassembled
// with three reroutings + the C5 𝒩₁ restoration (see the per-node block below).  The
// driver mirrors relax_structure / arclength_corrector's LM-damped Newton loop but uses
// a NUMERICAL central-difference Jacobian of the coupled residual (re-solving every
// column per perturbation — slow but correct for bring-up; the analytic Schur Jacobian
// is the next task).  Honest converged=false on failure (no fabricated profile).
//
// SCOPE: this is a fresh full-system solve (all 4N+2 rows, all unknowns incl. ℓ_in,
// r_s — like the arclength corrector, NOT the inner/outer split), driven directly from
// the cold thin-disk seed.  It is intentionally the SIMPLEST robust mirror.
// ===========================================================================

namespace grrt {

namespace slim_coupled_detail {

using slim_detail::OneZoneState;
using slim_detail::one_zone_closure;
using slim_detail::omega_from_ell;
using slim_detail::omega_perp2;
using slim_detail::kerr_delta;
using slim_detail::kerr_A;
using slim_detail::ell_kepler;
using slim_detail::isco_prograde;

// Mass conservation:  Ṁ = 2πΣ|V|√Δ·r_g·c / √(1−V²).
// Forward (Σ→V): mirrors build_slim_disk_seed's Vfrom (returns V<0 inflow).
static double V_from_sigma(const SlimDiskInputs& in, double r, double Sigma) {
    using namespace constants;
    const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
    const double dn = 2.0 * std::numbers::pi * Sigma * sqrtD * in.r_g * c_cgs;
    if (!(dn > 0.0)) return -1e-12;
    const double X = -in.mdot / dn;
    double V = X / std::sqrt(1.0 + X*X);
    if (!(V < 0.0)) V = -1e-12;
    return std::clamp(V, -0.9999, -1e-12);
}
// Inverse (V→Σ):  Σ = Ṁ√(1−V²) / (2π|V|√Δ·r_g·c).   (verified vs Vfrom)
static double sigma_from_V(const SlimDiskInputs& in, double r, double V) {
    using namespace constants;
    const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
    const double aV = std::abs(V);
    if (!(aV > 0.0) || !(sqrtD > 0.0)) return 0.0;
    return in.mdot * std::sqrt(std::max(1.0 - V*V, 0.0))
         / (2.0 * std::numbers::pi * aV * sqrtD * in.r_g * c_cgs);
}

// Per-node column closure cache entry: the converged augmented column state U_c
// (length 4*n_z+4), warm-started across radial iterations + FD perturbations so each
// node's column converges in a few polish steps.  Keyed by node index.
struct ColumnCache {
    std::vector<std::vector<double>> Uc;  // [node] -> converged augmented column state
    // NOTE: std::vector<char> (NOT std::vector<bool>) is REQUIRED for thread safety.
    // The per-node column loops below are OpenMP-parallelized; each node writes only its
    // OWN slot valid[i]. std::vector<bool> is a bit-packed proxy, so writes to distinct
    // indices sharing a word would DATA-RACE. char gives each element its own byte.
    std::vector<char> valid;              // [node] -> has a converged Uc to warm-start from
    int n_nodes = 0;                      // radial node count (size of the cache)
    int n_z = 0;                          // column node count (for the expected Uc length)
    void resize(int N, int nz) {
        if ((int)Uc.size() != N) { Uc.assign(N, {}); valid.assign(N, false); }
        n_nodes = N; n_z = nz;
    }
};

// Column-closure tuning for the COUPLED driver.  n_z small (bring-up: speed over the
// fine vertical resolution the standalone gates use), and a generous iter budget so a
// cold node still converges.  These are SOLVER-EFFORT knobs only.
struct ColumnOpts {
    int    n_z      = 24;     // column vertical nodes (small for bring-up)
    int    max_iter = 300;    // column Newton iters
    double tol      = 1e-8;   // column tolerance
};

// Result of one coupled node closure: the rerouted thermodynamic quantities the radial
// residual needs, with H REPLACED by the column z0 and (F, η3, η4, f_adv) from the
// column.  Mirrors the radial NodeEval fields the assembly consumes, plus the column
// outputs.  `ok` is false if the column failed to converge (⇒ infeasible step).
struct CoupledNode {
    bool   ok = false;
    double r = 0.0, Sigma = 0.0, V = 0.0, ell = 0.0, Tc = 0.0;
    NodeMech mech;   // anonymous-namespace type from slim_disk_radial.cpp (same TU)
    double Gamma = 0.0;
    // Closure quantities with H -> z0 (column photosphere half-thickness):
    double z0 = 0.0;              // = H_i  [cm]
    double P = 0.0;               // vertically-integrated pressure 2·p_mid·z0 [erg/cm²]
    double rho_mid = 0.0;         // Σ/(2 z0) [g/cm³]
    double P_over_Sigma_geom = 0.0;  // (P/Σ)/c²  [dimensionless]
    double cs2_geom = 0.0;        // Γ̃₁(η3)·(P/Σ)/c²  [dimensionless]
    double gtilde1 = 0.0;         // Γ̃₁ = 1 + 1/η3  (node-local, from column η3)
    // Column outputs:
    double F = 0.0;               // emergent flux per FACE = σT_eff⁴ [erg/cm²/s]
    double eta3 = 0.0, eta4 = 0.0, f_adv = 0.0;
};

// Local dense partial-pivot Gaussian solve A·x=b (x written into b).  DUPLICATED (a
// ~10-line primitive) ONLY because both slim_disk_radial.cpp and disk_column_bvp.cpp
// define a file-static `dense_solve` with identical signature in the same merged TU, so
// an unqualified call is AMBIGUOUS (the anonymous-namespace one cannot be qualified).
// Byte-identical algorithm to both; avoids touching either source.
static bool coupled_dense_solve(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int k = 0; k < n; ++k) {
        int piv = k; double maxv = std::abs(A[(size_t)k*n+k]);
        for (int i = k+1; i < n; ++i) { double v = std::abs(A[(size_t)i*n+k]); if (v>maxv){maxv=v;piv=i;} }
        if (maxv < 1e-300) return false;
        if (piv != k) { for (int j=0;j<n;++j) std::swap(A[(size_t)k*n+j],A[(size_t)piv*n+j]); std::swap(b[k],b[piv]); }
        const double akk = A[(size_t)k*n+k];
        for (int i = k+1; i < n; ++i) {
            const double f = A[(size_t)i*n+k]/akk;
            if (f != 0.0) { for (int j=k;j<n;++j) A[(size_t)i*n+j]-=f*A[(size_t)k*n+j]; b[i]-=f*b[k]; }
        }
    }
    for (int i = n-1; i >= 0; --i) { double sgi=b[i]; for (int j=i+1;j<n;++j) sgi-=A[(size_t)i*n+j]*b[j]; b[i]=sgi/A[(size_t)i*n+i]; }
    return true;
}

// Geometric vertical epicyclic Ω_⊥ in CGS [1/s] at radius r (GRRT Ω_⊥²=Ω_K²·ℋ).
static inline double omega_perp_cgs(const SlimDiskInputs& in, double r) {
    const double conv = constants::c_cgs / in.r_g;          // [1/s per 1/M]
    return std::sqrt(std::max(omega_perp2(in.mass, in.spin, r), 0.0)) * conv;
}

// Local orbital shear |r dΩ/dr| in CGS [1/s] at node i, using a forward/back FD of the
// LOCAL orbital Ω(ℓ) across the (i, j=i±1) node pair — the SAME Ω the radial heating
// law differentiates (Gbalance's dOmega_dr).  Ω is geometric [1/M]; r dΩ/dr is
// dimensionless in M, then ×(c/r_g) gives [1/s].  (shear = |r dΩ/dr|; >0.)
static inline double shear_cgs(const SlimDiskInputs& in, double r_i, double Om_i,
                               double r_j, double Om_j) {
    const double dOmega_geom = (Om_j - Om_i) / (r_j - r_i);  // [1/M²]
    const double r_dOmega_dr = r_i * dOmega_geom;            // [1/M] (dimensionless in M)
    return std::abs(r_dOmega_dr) * (constants::c_cgs / in.r_g);  // [1/s]
}

// Lever C — transonic-Σ seed for the coupled relax. Drop-in for
// build_thin_disk_seed(in,op) in the coupled seed path. Returns the standard
// 4N+2 packed radial state ([Σ,V,ℓ,T_c]×N, tail ℓ_in, r_s).
std::vector<double> build_transonic_coupled_seed(const SlimDiskInputs& in,
                                                 const OpacityLUTs& op,
                                                 const ColumnOpts& copt) {
    using namespace constants;
    using grrt::slim_detail::one_zone_closure;
    using grrt::slim_detail::omega_from_ell;
    // 1) Grid, ℓ(r), r_s from the thin seed.
    std::vector<double> U = build_thin_disk_seed(in, op);
    const int N = std::max(in.n_nodes, 4);
    const double r_s = U[4*N+1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N==1)?0.0:double(i)/double(N-1);
        r[i]  = std::exp(lr0 + (lr1-lr0)*t);
        Om[i] = omega_from_ell(in.mass, in.spin, r[i], U[4*i+2]);
    }
    // 2) Sonic anchor: |V(r_s)| = c_s/c from the thin-seed (Σ,T_c) at node 0.
    const double cs0 = one_zone_closure(std::max(U[0],1e2), std::max(U[3],1.0), r_s, in, op).c_s;
    const double Vsonic = -std::clamp(cs0 / c_cgs, 1e-6, 0.9999);
    // 3) Far-field subsonic target |V| from the thin seed at the outer node.
    const double Vout = -std::max(std::abs(U[4*(N-1)+1]), 1e-12);
    // 4) Monotone ln|V| interp r_s->r_out; Σ from mass conservation.
    for (int i = 0; i < N; ++i) {
        const double t = (N==1)?0.0:double(i)/double(N-1);
        const double lnV = std::log(std::abs(Vsonic))
                         + (std::log(std::abs(Vout)) - std::log(std::abs(Vsonic))) * t;
        double Vi = std::clamp(-std::exp(lnV), -0.9999, -1e-12);
        double Sig = std::max(sigma_from_V(in, r[i], Vi), 1e2);
        U[4*i+0] = Sig;
        U[4*i+1] = V_from_sigma(in, r[i], Sig);   // re-derive V from the clamped Σ (consistency)
    }
    // 5) T_c on the f_adv≈0 manifold at each node's (Σ, geometry).
    for (int i = 0; i < N; ++i) {
        const int j = (i+1<N)?i+1:i-1;
        const double shear_i  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double omegaz_i = omega_perp_cgs(in, r[i]);
        const OneZoneState oz = one_zone_closure(U[4*i+0], std::max(U[4*i+3],1.0), r[i], in, op);
        ColumnCoupledInputs ci{};
        ci.Sigma_target=U[4*i+0]; ci.Tc=std::max(U[4*i+3],1.0);
        ci.shear=std::max(shear_i,1e-300); ci.omega_z=std::max(omegaz_i,1e-300);
        ci.alpha=in.alpha; ci.rho_mid_guess=std::max(oz.rho_mid,1e-30);
        ci.n_nodes=copt.n_z; ci.max_iters=copt.max_iter; ci.tol=copt.tol; ci.Teff_guess=0.0;
        std::vector<double> Uc;
        if (build_coupled_seed(ci, op, Uc)) U[4*i+3] = std::max(Uc[2], 1.0);
    }
    return U;
}

// Solve the per-node column closure at (Σ_i, T_c,i, r_i) and pack a CoupledNode with H
// rerouted to z0 and (F, η3, η4, f_adv) from the column.  Warm-starts from the cached
// converged column state for node i (if any), then refreshes the cache on success.
// shear_i / omega_z_i are passed in (computed by the caller from the node geometry).
//
// `c_out` / `ci_out` (default null) are PURELY ADDITIVE outputs the analytic reduced
// Jacobian (Task 10) needs: the converged ColumnClosure and the exact ColumnCoupledInputs
// the column was solved at, so column_sensitivity (C3) can be formed at this node WITHOUT
// re-deriving the inputs.  When both are null the behaviour is byte-identical to before.
static CoupledNode eval_node_coupled(const SlimDiskInputs& in, const OpacityLUTs& op,
                                     const ColumnOpts& copt, ColumnCache& cache,
                                     int i, double r, double Sigma, double V, double ell,
                                     double Tc, double shear_i, double omega_z_i,
                                     ColumnClosure* c_out = nullptr,
                                     ColumnCoupledInputs* ci_out = nullptr) {
    using namespace constants;
    CoupledNode e;
    e.r = r;
    e.Sigma = std::max(Sigma, kSigmaFloor);
    e.V   = std::clamp(V, -kVCap, kVCap);
    e.ell = ell;
    e.Tc  = std::max(Tc, kTFloor);
    e.mech = node_mech(in, r, ell);
    {
        const double A = std::max(e.mech.A, 1e-300);
        e.Gamma = std::sqrt(1.0 / (1.0 - e.V * e.V) + e.ell * e.ell * e.r * e.r / A);
    }

    // A one-zone closure at (Σ,T_c) supplies the midplane pressure p_mid and a rho_mid
    // guess for the column seed; H from it is DISCARDED (the column z0 replaces it).
    const OneZoneState oz = one_zone_closure(e.Sigma, e.Tc, r, in, op);

    // ----- column closure: (Σ_i, T_c,i) -> (F, z0, η3, η4, f_adv) -----
    ColumnCoupledInputs ci{};
    ci.Sigma_target = e.Sigma;
    ci.Tc           = e.Tc;
    ci.shear        = std::max(shear_i, 1e-300);     // |r dΩ/dr| [1/s] (>0)
    ci.omega_z      = std::max(omega_z_i, 1e-300);   // Ω_⊥ [1/s] (>0)
    ci.alpha        = in.alpha;
    ci.rho_mid_guess = std::max(oz.rho_mid, 1e-30);
    ci.n_nodes      = copt.n_z;
    ci.max_iters    = copt.max_iter;
    ci.tol          = copt.tol;
    // Surface-T guess: leave 0 so the column uses its OWN grey-diffusion estimate
    // T_eff = T_c/(0.75 τ)^{1/4} (estimate_Teff_guess) — for an optically-thick inner
    // node T_eff ≪ T_c, so a naive T_eff≈T_c guess is wildly off and pushes the column
    // bring-up out of basin.  (When a warm column is cached, the full warm state carries
    // its converged T_eff directly, so this guess is only used on the cold first solve.)
    ci.Teff_guess   = 0.0;
    if (ci_out) *ci_out = ci;   // (additive) hand the exact inputs to the Jacobian builder

    const int n_c = 4 * copt.n_z + 4;
    const std::vector<double>* warm = nullptr;
    if (i >= 0 && i < (int)cache.valid.size() && cache.valid[i]
        && (int)cache.Uc[i].size() == n_c) {
        warm = &cache.Uc[i];
    }
    ColumnClosure c = solve_column_coupled(ci, op, warm);
    if (c_out) *c_out = c;       // (additive) hand the converged closure to the Jacobian builder
    if (!c.converged) { e.ok = false; return e; }

    // Cache the converged augmented column state for the next warm-start.  Reconstruct
    // the augmented U_c from the converged ColumnBVPSolution (same packing the column
    // uses internally: [Pg,Q,T,z]×n_z + (z0, Σ0, T_eff, f_adv)).
    if (i >= 0 && i < (int)cache.Uc.size()) {
        const int nz = copt.n_z;
        std::vector<double>& U = cache.Uc[i];
        U.assign(n_c, 0.0);
        for (int k = 0; k < nz; ++k) {
            U[4*k+0] = c.sol.P_gas[k];
            U[4*k+1] = c.sol.Q[k];
            U[4*k+2] = c.sol.T[k];
            U[4*k+3] = c.sol.z[k];
        }
        U[4*nz+0] = c.sol.z0;
        U[4*nz+1] = c.sol.Sigma0;
        U[4*nz+2] = c.T_eff;
        U[4*nz+3] = c.f_adv;
        cache.valid[i] = true;
    }

    e.ok    = true;
    e.F     = c.F;       // per-FACE emergent flux σT_eff⁴
    e.z0    = c.z0;      // = H_i
    e.eta3  = c.eta3;
    e.eta4  = c.eta4;
    e.f_adv = c.f_adv;

    // ----- reroute the H-dependent EOS: H -> z0 -----
    // The one-zone closure builds H from hydrostatic balance and P = 2·p_mid·H.  Here
    // H is the column photosphere z0, so rho_mid = Σ/(2 z0) and P = 2·p_mid·z0 with the
    // SAME midplane p_mid(Σ,T_c) the one-zone EOS gives at (Σ,T_c).  (The column's own
    // vertically-integrated P is a Phase-later refinement; C4 reroutes H only, per the
    // 06-20 design "H = z₀ from hydrostatic".)
    const double z0_s = std::max(e.z0, 1e-300);
    e.rho_mid = e.Sigma / (2.0 * z0_s);
    e.P       = 2.0 * oz.p_mid * z0_s;                      // [erg/cm²]
    const double P_over_Sigma = e.P / e.Sigma;             // [cm²/s²]
    e.P_over_Sigma_geom = P_over_Sigma / (c_cgs * c_cgs);  // dimensionless
    // Γ̃₁ from the COLUMN η3 (η3∈[1.5,3] is bounded away from 0 ⇒ safe):
    const double eta3_safe = std::max(e.eta3, 1e-6);
    e.gtilde1 = 1.0 + 1.0 / eta3_safe;
    e.cs2_geom = e.gtilde1 * e.P_over_Sigma_geom;          // Γ̃₁(η3)·(P/Σ)/c²
    return e;
}

// 𝒟₀ = V² − Γ̃₁(η3)·(P/Σ)  (dimensionless; mirrors calD0 but with coupled cs2_geom).
static inline double calD0_coupled(const CoupledNode& e) {
    return e.V * e.V - e.cs2_geom;
}

// 𝒩₁ (coupled + C5).  Mirrors the one-zone calN1's three baseline terms but with the
// coupled (P/Σ), Γ̃₁(η3), and Q_adv, THEN RESTORES the S11 η-gradient terms the one-zone
// drops (constant-η ⇒ zero gradient):
//   𝒩₁ = 𝒜 + (2πr²/(Ṁ η3))·Q_adv
//        + (P/Σ)·[ r(r−M)/Δ·Γ̃₁ + dln η3/dln r ]            <-- C5 dlnη3 term
//        + Ω_⊥²·(η4/η3)·dln η4/dln r                          <-- C5 η4 term
// All dimensionless (V in c).  Qadv_geom is the (2πr²/(Ṁη3))·Q_adv already reduced to
// dimensionless by the caller (CGS / c², matching the one-zone path).  dlnη3_dlnr and
// dlnη4_dlnr are FD radial log-gradients across the node pair (caller-supplied).
// Ω_⊥² is the geometric vertical epicyclic ×(c/r_g)² → CGS, then the whole
// Ω_⊥²·η4·(...) /c² renders dimensionless (η4 is in cm²; same /c² as (P/Σ)).
static double calN1_coupled(const SlimDiskInputs& in, const CoupledNode& e,
                            double Qadv_geom, double dlneta3_dlnr, double dlneta4_dlnr) {
    using namespace constants;
    const double A_term = script_A(in, e.r, e.mech);
    const double M = in.mass, r = e.r, Delta = std::max(e.mech.Delta, 1e-30);
    // baseline pressure term, with C5 dlnη3 added INSIDE the (P/Σ)·[...] bracket:
    const double press_term = e.P_over_Sigma_geom
                            * (r * (r - M) / Delta * e.gtilde1 + dlneta3_dlnr);
    // C5 η4 term: Ω_⊥²·(η4/η3)·dlnη4/dlnr, rendered dimensionless (CGS Ω_⊥²·η4 / c²).
    const double omega_perp2_cgs = std::max(omega_perp2(in.mass, in.spin, r), 0.0)
                                 * (c_cgs / in.r_g) * (c_cgs / in.r_g);   // [1/s²]
    const double eta3_safe = std::max(e.eta3, 1e-6);
    const double eta4_term = omega_perp2_cgs * (e.eta4 / eta3_safe) * dlneta4_dlnr
                           / (c_cgs * c_cgs);                            // dimensionless
    return A_term + Qadv_geom + press_term + eta4_term;
}

// ===========================================================================
// Row assembly (4N+2) from a CLOSED node array — shared by the live residual and
// the analytic Jacobian's frozen-column residual.
// ===========================================================================
// Given the per-node closed state e[] (Σ,V,ℓ,T_c + the column-rerouted P, cs2_geom,
// gtilde1, F, η3, η4 already packed), the geometric Ω(ℓ) array Om[], the grid r[],
// and the globals (ℓ_in, Ṁ), assemble the SIX row groups EXACTLY as the live residual
// did (this is a verbatim code-motion extraction — no physics change).  Keeping this
// in one place lets the Jacobian build a "columns-frozen" residual (the e[] column
// outputs held fixed while U_r is perturbed) by feeding a frozen-closed e[] here.
static void assemble_coupled_rows(const SlimDiskInputs& in,
                                  const std::vector<CoupledNode>& e,
                                  const std::vector<double>& Om,
                                  const std::vector<double>& r,
                                  double ell_in, double Mdot,
                                  std::vector<double>& R) {
    using namespace constants;
    const int N = (int)e.size();
    R.assign((size_t)4 * N + 2, 0.0);

    // FD radial log-gradient helper (mirrors slim_radial_residual's dln).
    auto dln = [&](double f_lo, double f_hi, double r_lo, double r_hi) {
        return (std::log(std::max(f_hi, 1e-300)) - std::log(std::max(f_lo, 1e-300)))
             / (std::log(r_hi) - std::log(r_lo));
    };

    // -------- Group 1: mass conservation (N rows) — UNCHANGED (no closure dep). -----
    for (int i = 0; i < N; ++i) {
        const double mdot_i = mdot_of_node(in, e[i].Sigma, e[i].V, e[i].mech.sqrtDelta);
        R[i] = mdot_i - Mdot;
    }

    // -------- Group 2: angular momentum (N rows) — uses the coupled P. --------------
    for (int i = 0; i < N; ++i) {
        const CoupledNode& ei = e[i];
        const double dl_cgs = (ei.ell - ell_in) * in.r_g * c_cgs;                 // [cm²/s]
        const double lhs = (Mdot / (2.0 * std::numbers::pi)) * dl_cgs;            // erg
        const double geomlen = ei.mech.sqrtA * ei.mech.sqrtDelta / ei.r;          // [M²]
        const double rhs = geomlen * in.r_g * in.r_g * ei.Gamma * in.alpha * ei.P; // erg
        R[N + i] = lhs - rhs;
    }

    // -------- Group 3: radial-momentum transonic ODE (N-1 rows) — coupled 𝒩₁/𝒟₀. ---
    // Qadv_geom(i,neighbor): (2πr²/(Ṁη3))·Q_adv, dimensionless (CGS/c²), η3 = column η3_i.
    auto qadv_term_geom = [&](int i, int j) -> double {
        const CoupledNode& a = e[i];
        const CoupledNode& b = e[j];
        const double dlnP = dln(a.P, b.P, a.r, b.r);
        const double dlnS = dln(a.Sigma, b.Sigma, a.r, b.r);
        const double eta3_i = std::max(a.eta3, 1e-6);
        const double r_cm = a.r * in.r_g;
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (a.P / a.Sigma)
                          * (eta3_i * dlnP - (1.0 + eta3_i) * dlnS);              // [erg/cm²/s]
        const double term = (2.0 * std::numbers::pi * r_cm * r_cm / (Mdot * eta3_i)) * Qadv;
        return term / (c_cgs * c_cgs);
    };
    // FD η3 / η4 radial log-gradients across the node pair (C5).  η3,η4>0 ⇒ log-FD safe.
    auto dlneta3 = [&](int i, int j) { return dln(e[i].eta3, e[j].eta3, e[i].r, e[j].r); };
    auto dlneta4 = [&](int i, int j) { return dln(e[i].eta4, e[j].eta4, e[i].r, e[j].r); };

    auto rhs_radial = [&](int i, int neighbor) -> double {
        const CoupledNode& ei = e[i];
        const double D0 = calD0_coupled(ei);
        const double D0g = (std::abs(D0) > 1e-30) ? D0 : std::copysign(1e-30, D0 == 0 ? 1.0 : D0);
        const double Qadv_g = qadv_term_geom(i, neighbor);
        const double N1 = calN1_coupled(in, ei, Qadv_g, dlneta3(i, neighbor), dlneta4(i, neighbor));
        return (N1 / D0g) * (1.0 - ei.V * ei.V);
    };
    // L'Hôpital rhs at the sonic node 0 (= r_s): both 𝒩₁(r_s),𝒟₀(r_s)→0 at convergence.
    auto rhs_radial_sonic_node0 = [&]() -> double {
        const double Qadv0 = qadv_term_geom(0, 1);
        const double Qadv1 = qadv_term_geom(1, 0);
        const double N1_0 = calN1_coupled(in, e[0], Qadv0, dlneta3(0, 1), dlneta4(0, 1));
        const double N1_1 = calN1_coupled(in, e[1], Qadv1, dlneta3(1, 0), dlneta4(1, 0));
        const double D0_0 = calD0_coupled(e[0]);
        const double D0_1 = calD0_coupled(e[1]);
        const double dlnr = std::log(r[1]) - std::log(r[0]);
        const double dN1 = (N1_1 - N1_0) / dlnr;
        double dD0 = (D0_1 - D0_0) / dlnr;
        if (std::abs(dD0) < 1e-30) dD0 = std::copysign(1e-30, dD0 == 0 ? -1.0 : dD0);
        return (dN1 / dD0) * (1.0 - e[0].V * e[0].V);
    };
    for (int i = 0; i < N - 1; ++i) {
        const double lnVi  = std::log(std::max(-e[i].V,   1e-300));
        const double lnVi1 = std::log(std::max(-e[i+1].V, 1e-300));
        const double dlnr  = std::log(r[i+1]) - std::log(r[i]);
        const double rhs_i  = (i == 0) ? rhs_radial_sonic_node0() : rhs_radial(i, i + 1);
        const double rhs_i1 = rhs_radial(i+1, i);
        R[2 * N + i] = (lnVi1 - lnVi) - 0.5 * dlnr * (rhs_i + rhs_i1);
    }

    // -------- Group 4: energy ODE Q_vis = 2F + Q_adv (N-1 rows). ---------------------
    // Q_rad (one-zone, both faces) -> 2·F_i (column F is one face).  Q_vis + Q_adv keep
    // their §23 form, Q_adv with the column η3_i.
    auto Gbalance = [&](int i, int j) -> double {
        const CoupledNode& a = e[i];
        const CoupledNode& b = e[j];
        const double r_cm  = a.r * in.r_g;
        const double dOmega_geom = (Om[j] - Om[i]) / (b.r - a.r);                 // [1/M²]
        const double dOmega_dr = dOmega_geom * (c_cgs / in.r_g) / in.r_g;         // [1/s/cm]
        const double geomfac = a.mech.sqrtA
                             / (std::max(a.mech.sqrtDelta, 1e-30) * a.r);         // dimensionless
        const double dl_cgs = (a.ell - ell_in) * in.r_g * c_cgs;                  // [cm²/s]
        const double Qvis = -(Mdot / (2.0 * std::numbers::pi)) * dl_cgs * dOmega_dr
                          * a.Gamma * (geomfac / r_cm);                           // [erg/cm²/s]
        // Q_rad -> 2F (both faces):
        const double Qrad = 2.0 * a.F;                                           // [erg/cm²/s]
        // Q_adv (raw CGS, column η3_i):
        const double dlnP = dln(a.P, b.P, a.r, b.r);
        const double dlnS = dln(a.Sigma, b.Sigma, a.r, b.r);
        const double eta3_a = std::max(a.eta3, 1e-6);
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (a.P / a.Sigma)
                          * (eta3_a * dlnP - (1.0 + eta3_a) * dlnS);             // [erg/cm²/s]
        return Qvis - Qrad - Qadv;
    };
    for (int i = 0; i < N - 1; ++i) {
        const double Gi  = Gbalance(i,   i + 1);
        const double Gi1 = Gbalance(i+1, i);
        R[3 * N - 1 + i] = 0.5 * (Gi + Gi1);
    }

    // -------- Group 5: outer BCs (2 rows). ℓ matched-slope (cubic), outer energy. ----
    const int last = N - 1;
    {
        const double x0 = std::log(r[last - 1]);
        const double x1 = std::log(r[last - 2]);
        const double x2 = std::log(r[last - 3]);
        const double x3 = std::log(r[last - 4]);
        const double x  = std::log(r[last]);
        const double f0 = e[last - 1].ell, f1 = e[last - 2].ell,
                     f2 = e[last - 3].ell, f3 = e[last - 4].ell;
        const double d01  = (f0 - f1) / (x0 - x1);
        const double d12  = (f1 - f2) / (x1 - x2);
        const double d23  = (f2 - f3) / (x2 - x3);
        const double d012 = (d01 - d12) / (x0 - x2);
        const double d123 = (d12 - d23) / (x1 - x3);
        const double d0123 = (d012 - d123) / (x0 - x3);
        const double ell_extrap = f0 + (x - x0) * d01
                                + (x - x0) * (x - x1) * d012
                                + (x - x0) * (x - x1) * (x - x2) * d0123;
        R[4 * N - 2] = e[last].ell - ell_extrap;
    }
    R[4 * N - 1] = Gbalance(last, last - 1);

    // -------- Group 6: sonic-point regularity at node 0 (= r_s). --------------------
    {
        const double Qadv_g0 = qadv_term_geom(0, 1);
        R[4 * N + 0] = calD0_coupled(e[0]);
        R[4 * N + 1] = calN1_coupled(in, e[0], Qadv_g0, dlneta3(0, 1), dlneta4(0, 1));
    }
}

// ===========================================================================
// Coupled radial residual (4N+2), with the column rerouting + C5.
// ===========================================================================
// Mirrors slim_radial_residual EXACTLY for the grid + the six row groups (mass,
// angular momentum, radial-momentum transonic ODE, energy ODE, outer BCs, sonic
// regularity), but:
//   • every node's closure is the column closure (eval_node_coupled), so H = z0, the
//     EOS pressure P = 2 p_mid z0, and η3/η4/F come from the column;
//   • energy row: Q_rad = 64σT_c⁴/(3κΣ)  →  2·F_i (column F is one face, radial Q_rad
//     is both faces — disk-physics §23);
//   • Q_adv uses the column η3_i wherever eta3_of_beta(beta) appeared;
//   • 𝒩₁ uses calN1_coupled (C5 η-gradient terms restored).
//
// `infeasible` is set true (and the residual filled with a large sentinel) if ANY
// node's column fails to converge, so the driver's feasibility line search rejects the
// step.  The per-node column cache is warm-started + refreshed in eval_node_coupled.
static void slim_coupled_residual(const std::vector<double>& U, const SlimDiskInputs& in,
                                  const OpacityLUTs& op, const ColumnOpts& copt,
                                  ColumnCache& cache, std::vector<double>& R,
                                  bool& infeasible) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    R.assign((size_t)4 * N + 2, 0.0);
    infeasible = false;

    const double ell_in = U[4 * N + 0];
    const double r_s    = U[4 * N + 1];
    const double Mdot   = in.mdot;

    // Free-inner-node grid [r_s, r_out], r[0] == r_s (node 0 is the sonic point) — the
    // SAME log grid slim_radial_residual builds.
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i] = std::exp(lr0 + (lr1 - lr0) * t);
    }
    cache.resize(N, copt.n_z);

    // Per-node orbital Ω (geometric) for shear FD (LOCAL Ω(ℓ), like Gbalance).
    std::vector<double> Om(N);
    for (int i = 0; i < N; ++i)
        Om[i] = omega_from_ell(in.mass, in.spin, r[i], U[4 * i + 2]);

    // Closure every node (column).  shear_i / Ω_z,i from the node geometry.
    // PARALLEL: the per-node column solves are INDEPENDENT — node i reads/writes only
    // its own ColumnCache slot i (Uc[i], valid[i]) and its own e[i]; no cross-node shared
    // mutable state (g_budget is untouched here; op is read-only).  So the loop is
    // embarrassingly parallel and its result is order-independent (pure per-node scatter).
    // any_fail is an int OR-reduction (MSVC OpenMP 2.0-compatible).
    std::vector<CoupledNode> e(N);
    int any_fail = 0;
    #pragma omp parallel for schedule(dynamic) reduction(||:any_fail)
    for (int i = 0; i < N; ++i) {
        const int j = (i + 1 < N) ? i + 1 : i - 1;   // neighbour for the shear FD
        const double shear_i  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double omegaz_i = omega_perp_cgs(in, r[i]);
        e[i] = eval_node_coupled(in, op, copt, cache, i,
                                 r[i], U[4*i+0], U[4*i+1], U[4*i+2], U[4*i+3],
                                 shear_i, omegaz_i);
        if (!e[i].ok) any_fail = 1;
    }
    if (any_fail) {
        // Column infeasible at this iterate: fill a large finite sentinel so the merit
        // is huge and the line search rejects the step (NEVER a fabricated profile).
        infeasible = true;
        for (double& x : R) x = 1e300;
        return;
    }

    // All columns converged ⇒ assemble the 4N+2 rows from the closed node array.
    assemble_coupled_rows(in, e, Om, r, ell_in, Mdot, R);
}

// ===========================================================================
// Columns-FROZEN residual + analytic reduced (Schur) Jacobian (Task 10).
// ===========================================================================

// FrozenCol — the per-node column OUTPUTS the radial residual consumes (the ONLY column
// dependence the rows carry: F, z0, η3, η4).  Holding these fixed while U_r is perturbed
// isolates the DIRECT (columns-frozen) part of ∂R_r/∂U_r; perturbing them isolates ∂R_r/∂C.
struct FrozenCol { double F = 0.0, z0 = 0.0, eta3 = 0.0, eta4 = 0.0; };

// eval_node_frozen — eval_node_coupled with the column SOLVE replaced by frozen outputs.
// It recomputes every U_r-direct quantity (Σ,V,ℓ,T_c, mech, Γ, the one-zone p_mid, and the
// z0/η3-derived P, ρ_mid, P/Σ, Γ̃₁, c_s²) byte-for-byte as eval_node_coupled does, so a
// frozen-column residual differs from the live one ONLY by holding (F,z0,η3,η4) fixed.
// (f_adv is consumed by NO row ⇒ left 0 — see assemble_coupled_rows.)
static CoupledNode eval_node_frozen(const SlimDiskInputs& in, const OpacityLUTs& op,
                                    double r, double Sigma, double V, double ell, double Tc,
                                    const FrozenCol& fc) {
    using namespace constants;
    CoupledNode e;
    e.r = r;
    e.Sigma = std::max(Sigma, kSigmaFloor);
    e.V   = std::clamp(V, -kVCap, kVCap);
    e.ell = ell;
    e.Tc  = std::max(Tc, kTFloor);
    e.mech = node_mech(in, r, ell);
    {
        const double A = std::max(e.mech.A, 1e-300);
        e.Gamma = std::sqrt(1.0 / (1.0 - e.V * e.V) + e.ell * e.ell * e.r * e.r / A);
    }
    const OneZoneState oz = one_zone_closure(e.Sigma, e.Tc, r, in, op);
    e.ok    = true;
    e.F     = fc.F;
    e.z0    = fc.z0;
    e.eta3  = fc.eta3;
    e.eta4  = fc.eta4;
    e.f_adv = 0.0;
    const double z0_s = std::max(e.z0, 1e-300);
    e.rho_mid = e.Sigma / (2.0 * z0_s);
    e.P       = 2.0 * oz.p_mid * z0_s;
    const double P_over_Sigma = e.P / e.Sigma;
    e.P_over_Sigma_geom = P_over_Sigma / (c_cgs * c_cgs);
    const double eta3_safe = std::max(e.eta3, 1e-6);
    e.gtilde1 = 1.0 + 1.0 / eta3_safe;
    e.cs2_geom = e.gtilde1 * e.P_over_Sigma_geom;
    return e;
}

// Columns-frozen residual: build e[] from U_r with each node's column outputs HELD at fc[]
// (NO column solve), then run the shared assembly.  Always evaluable (no infeasible path).
// At fc[] = the live converged outputs this reproduces slim_coupled_residual EXACTLY.
static void slim_coupled_residual_frozen(const std::vector<double>& U, const SlimDiskInputs& in,
                                         const OpacityLUTs& op, const std::vector<FrozenCol>& fc,
                                         std::vector<double>& R) {
    const int N = std::max(in.n_nodes, 4);
    const double ell_in = U[4 * N + 0];
    const double r_s    = U[4 * N + 1];
    const double Mdot   = in.mdot;
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i] = std::exp(lr0 + (lr1 - lr0) * t);
    }
    std::vector<double> Om(N);
    for (int i = 0; i < N; ++i)
        Om[i] = omega_from_ell(in.mass, in.spin, r[i], U[4 * i + 2]);
    std::vector<CoupledNode> e(N);
    for (int i = 0; i < N; ++i)
        e[i] = eval_node_frozen(in, op, r[i], U[4*i+0], U[4*i+1], U[4*i+2], U[4*i+3], fc[i]);
    assemble_coupled_rows(in, e, Om, r, ell_in, Mdot, R);
}

// ---------------------------------------------------------------------------
// Analytic reduced (Schur) Jacobian J_red of the coupled residual.
// ---------------------------------------------------------------------------
//   J_red[row][col] = dR_r[row]/dU_r[col]
//                   = ∂R_r/∂U_r|_C            (direct, columns frozen)
//                   + Σ_i (∂R_r/∂C_i)(dC_i/dU_r[col]).
//
// HYBRID assembly (06-20 design + Task-10 brief):
//   • Σ_i, V_i, T_c,i columns — DIRECT part by central-FD of the COLUMNS-FROZEN residual
//     (cheap residual algebra, NO column solves).  V carries no column dependence ⇒ its
//     frozen-FD already IS the total derivative.
//   • Σ_i, T_c,i columns — PLUS the analytic column-mediated Schur term  B_i·dC_i/d{Σ,T_c},
//     where B_i = ∂R_r/∂{F_i,z0_i,η3_i,η4_i} (central-FD of the frozen residual w.r.t. the
//     four HELD outputs — the explicit residual derivatives) and dC_i/d{Σ,T_c} is C3's
//     analytic IFT sensitivity (column_sensitivity).  Σ_i (resp. T_c,i) enters ONLY column
//     i (not its neighbours'), so the Schur sum over nodes collapses to the single node i.
//   • ℓ_i, ℓ_in, r_s columns — FULL FD of the LIVE residual (re-solving columns), one-sided
//     (forward) by default / central with SLIM_FD_ONESIDED=0 — see the block below:
//     captures the shear→Ω(ℓ) and grid→(shear,Ω_⊥) mediated column response WITHOUT a
//     bespoke dC/dshear chain (06-20 "option (b)").  ℓ_in carries no column dependence; ℓ_i
//     and r_s do (via the geometry the columns are solved at).
//
// Returns false if the BASE point is column-infeasible (a node failed) — the driver then
// handles it exactly as an infeasible residual (no fabricated profile).
static bool slim_coupled_reduced_jacobian(const std::vector<double>& U, const SlimDiskInputs& in,
                                          const OpacityLUTs& op, const ColumnOpts& copt,
                                          ColumnCache& cache, std::vector<double>& J) {
    const int N = std::max(in.n_nodes, 4);
    const int n = 4 * N + 2;
    J.assign((size_t)n * n, 0.0);

    const double r_s = U[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i] = std::exp(lr0 + (lr1 - lr0) * t);
    }
    std::vector<double> Om(N);
    for (int i = 0; i < N; ++i)
        Om[i] = omega_from_ell(in.mass, in.spin, r[i], U[4 * i + 2]);
    cache.resize(N, copt.n_z);

    // --- Base column solve at U: capture the frozen outputs fc[] + the C3 sensitivities S[]. ---
    std::vector<FrozenCol> fc(N);
    std::vector<ColumnSensitivity> S(N);
    // PARALLEL: identical independence to the residual node loop — each node solves its
    // own column (own cache slot i) and scatters into its own fc[i]/S[i].  A `return` is
    // illegal from an OpenMP region, so an infeasible node sets a shared flag (int
    // OR-reduction) and the function returns false AFTER the loop (same outcome: J is
    // discarded on an infeasible base point).  The base loop stays pure per-node scatter.
    int base_infeasible = 0;
    #pragma omp parallel for schedule(dynamic) reduction(||:base_infeasible)
    for (int i = 0; i < N; ++i) {
        const int jn = (i + 1 < N) ? i + 1 : i - 1;
        const double shear_i  = shear_cgs(in, r[i], Om[i], r[jn], Om[jn]);
        const double omegaz_i = omega_perp_cgs(in, r[i]);
        ColumnClosure c; ColumnCoupledInputs ci;
        const CoupledNode ei = eval_node_coupled(in, op, copt, cache, i,
                                                 r[i], U[4*i+0], U[4*i+1], U[4*i+2], U[4*i+3],
                                                 shear_i, omegaz_i, &c, &ci);
        if (!ei.ok || !c.converged) { base_infeasible = 1; continue; }   // infeasible node
        fc[i] = FrozenCol{ c.F, c.z0, c.eta3, c.eta4 };
        S[i]  = column_sensitivity(c, ci, op);
    }
    if (base_infeasible) return false;                     // infeasible base point

    // Snapshot the converged base columns. Every full-FD column below warm-starts from
    // THIS pristine snapshot (not from the previous column's mutated cache), so each
    // column's result is independent of evaluation order — the precondition for a
    // bit-identical parallelization (Task 2). See spec 2026-07-26-jacobian-column-parallel.
    const ColumnCache base_snap = cache;

    // Per-variable FD step — identical keying to slim_coupled_numerical_jacobian.
    auto step_for = [&](int col) -> double {
        const double u = U[col];
        double floor;
        if (col >= 4 * N) { floor = (col == 4*N) ? 1e-6 : 1e-5; }   // ℓ_in : r_s
        else {
            switch (col % 4) {
                case 0: floor = 1e-3 * std::max(std::abs(u), 1e2); break; // Σ
                case 1: floor = 1e-9;                              break; // V
                case 2: floor = 1e-6;                              break; // ℓ
                default: floor = 1.0;                             break; // T_c
            }
        }
        return std::max(1e-6 * std::abs(u), floor);
    };

    std::vector<double> Up, Um, Rp, Rm;

    // ---------- (1) DIRECT part for the Σ, V, T_c columns (frozen-column FD). ----------
    auto frozen_fd_col = [&](int col) {
        const double h = step_for(col);
        Up = U; Um = U; Up[col] += h; Um[col] -= h;
        slim_coupled_residual_frozen(Up, in, op, fc, Rp);
        slim_coupled_residual_frozen(Um, in, op, fc, Rm);
        const double inv = 1.0 / (2.0 * h);
        for (int row = 0; row < n; ++row) {
            double d = (Rp[row] - Rm[row]) * inv;
            if (!std::isfinite(d)) d = 0.0;
            J[(size_t)row * n + col] = d;
        }
    };
    for (int i = 0; i < N; ++i) { frozen_fd_col(4*i+0); frozen_fd_col(4*i+1); frozen_fd_col(4*i+3); }

    // ---------- (2) FULL-FD for the ℓ_i, ℓ_in, r_s columns (option (b): re-solve cols). ----------
    // Base residual at the (feasibility-checked) point U — the anchor for one-sided
    // differencing when a perturbed side goes infeasible. U is feasible (checked above),
    // so inf0 is FALSE on the normal path.
    std::vector<double> R0; bool inf0 = true;
    { ColumnCache anchor_cache = base_snap;
      slim_coupled_residual(U, in, op, copt, anchor_cache, R0, inf0); }
    // One-sided (forward) differencing for the full-FD columns HALVES the dominant cost
    // (2 residual evals per column -> 1; each eval re-solves ALL N columns, ~95% of the
    // iteration's column-solve work). Accuracy cost is buried far below the existing error
    // floor: the FD budget here is dominated by column-solver NOISE (copt.tol/h ~ 1e-8/1e-6
    // ~ 1e-2), which is IDENTICAL for both schemes (central merely divides by 2h), while
    // truncation only rises O(h^2)~1e-12 -> O(h)~1e-6 — still a million-fold below the
    // noise floor. So `h` must NOT change: step_for(col) stays exactly as it is.
    // Default ON; SLIM_FD_ONESIDED=0 restores central differencing (A/B without a rebuild).
    static const bool kOneSidedFD = [] {
        const char* e = std::getenv("SLIM_FD_ONESIDED");
        return !(e && e[0] == '0');
    }();
    auto full_fd_col = [&](int col) {
        const double h = step_for(col);
        ColumnCache col_cache = base_snap;                 // warm-start from the pristine base
        std::vector<double> Upc = U, Umc = U, Rpc, Rmc;
        // Flags are INFEASIBILITY flags (true = that side has no usable residual).
        // Initialized true so a SKIPPED side (one-sided fast path) never reads as usable —
        // Rmc is empty then and must never be touched by the central branch.
        bool infp = true, infm = true;
        Upc[col] += h;
        slim_coupled_residual(Upc, in, op, copt, col_cache, Rpc, infp);
        // One-sided: solve the −h side ONLY if the +h side went INFEASIBLE (rare), so the
        // common path costs ONE residual eval. Central mode always solves both. The −h solve
        // intentionally reuses col_cache (warm-started from the +h solve) — this is per-call
        // state, so it does NOT affect order-independence ACROSS columns (each column gets
        // its own fresh copy of base_snap); do not split into two copies.
        if (!kOneSidedFD || infp) {
            Umc[col] -= h;
            slim_coupled_residual(Umc, in, op, copt, col_cache, Rmc, infm);
        }
        // A perturbed side that goes infeasible is sentinel-filled (1e300); the central
        // difference (1e300 − finite) is FINITE (~1e305), so the isfinite guard would NOT
        // catch it and a garbage entry would land in J. Fall back to a one-sided difference
        // from the feasible base; zero the entry only if BOTH sides are infeasible (no usable
        // gradient — the LM damping + feasibility line search bound the held unknown).
        for (int row = 0; row < n; ++row) {
            double d;
            if      (!infp && !infm) d = (Rpc[row] - Rmc[row]) / (2.0 * h);  // central
            else if (!infp && !inf0) d = (Rpc[row] - R0[row]) / h;           // forward
            else if (!infm && !inf0) d = (R0[row] - Rmc[row]) / h;           // backward
            else                     d = 0.0;                                // no usable gradient
            if (!std::isfinite(d)) d = 0.0;
            J[(size_t)row * n + col] = d;
        }
    };
    // Parallelize over the ℓ_i / ℓ_in / r_s columns. Each column is INDEPENDENT: full_fd_col
    // seeds its own col_cache from base_snap and writes only its own J column, so there is
    // no shared mutable state and the result is identical to the serial order (Task 1
    // base-seeding is the precondition). schedule(dynamic): every column task solves ALL
    // nodes incl. the one dominant high-Σ node, so tasks are near-equal-cost and dynamic
    // scheduling keeps threads full. The inner per-node parallel-for inside slim_coupled_
    // residual runs serially here (OpenMP nesting is off by default) — intended.
    std::vector<int> fd_cols;
    fd_cols.reserve(N + 2);
    for (int i = 0; i < N; ++i) fd_cols.push_back(4*i + 2);  // ℓ_i
    fd_cols.push_back(4*N + 0);                              // ℓ_in
    fd_cols.push_back(4*N + 1);                              // r_s
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < (int)fd_cols.size(); ++k) full_fd_col(fd_cols[k]);

    // ---------- (3) Column-mediated Schur term for Σ_i, T_c,i (analytic via C3). ----------
    // B_i columns = ∂R_r/∂{F_i,z0_i,η3_i,η4_i} via central-FD of the FROZEN residual w.r.t.
    // the four held outputs of node i (cheap — no column solves).  Then
    //   J[:, Σ_i ] += B_i·(dF/dΣ,  dz0/dΣ,  dη3/dΣ,  dη4/dΣ)    [C3 index 0]
    //   J[:, T_c,i] += B_i·(dF/dT_c, dz0/dT_c, dη3/dT_c, dη4/dT_c) [C3 index 1]
    std::vector<double> bcol[4];   // bF, bz0, be3, be4
    auto frozen_fd_output = [&](int i, int which, std::vector<double>& out) {
        double v0, floor;
        switch (which) {
            case 0:  v0 = fc[i].F;    floor = 1e-300; break;
            case 1:  v0 = fc[i].z0;   floor = 1e-300; break;
            case 2:  v0 = fc[i].eta3; floor = 1e-9;   break;
            default: v0 = fc[i].eta4; floor = 1e-300; break;
        }
        const double h = std::max(1e-6 * std::abs(v0), floor);
        std::vector<FrozenCol> fcp = fc, fcm = fc;
        auto setv = [&](std::vector<FrozenCol>& f, double val) {
            switch (which) { case 0: f[i].F=val; break; case 1: f[i].z0=val; break;
                             case 2: f[i].eta3=val; break; default: f[i].eta4=val; }
        };
        setv(fcp, v0 + h); setv(fcm, v0 - h);
        slim_coupled_residual_frozen(U, in, op, fcp, Rp);
        slim_coupled_residual_frozen(U, in, op, fcm, Rm);
        const double inv = 1.0 / (2.0 * h);
        out.assign(n, 0.0);
        for (int row = 0; row < n; ++row) out[row] = (Rp[row] - Rm[row]) * inv;
    };
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < 4; ++k) frozen_fd_output(i, k, bcol[k]);
        const int cS = 4*i+0, cT = 4*i+3;
        const double dC_dS[4] = { S[i].dF[0], S[i].dz0[0], S[i].deta3[0], S[i].deta4[0] };
        const double dC_dT[4] = { S[i].dF[1], S[i].dz0[1], S[i].deta3[1], S[i].deta4[1] };
        for (int row = 0; row < n; ++row) {
            double schur_S = 0.0, schur_T = 0.0;
            for (int k = 0; k < 4; ++k) {
                schur_S += bcol[k][row] * dC_dS[k];
                schur_T += bcol[k][row] * dC_dT[k];
            }
            if (std::isfinite(schur_S)) J[(size_t)row*n+cS] += schur_S;
            if (std::isfinite(schur_T)) J[(size_t)row*n+cT] += schur_T;
        }
    }
    return true;
}

// ===========================================================================
// Numerical (central-difference) Jacobian of the coupled residual.
// ===========================================================================
// J[row*n + col] = ∂R_row/∂U_col, n = 4N+2.  Perturbs each unknown and RE-ASSEMBLES
// the full coupled residual (re-solving every column per perturbation — slow but
// correct; the analytic Schur Jacobian is the next task).  Per-variable absolute step
// floors keyed to the state-variable TYPE (mirrors slim_numerical_jacobian) so a state
// entry that is ~0 at the seed does not collapse the FD step to round-off.  If a perturbed
// residual side is infeasible (a column failed → sentinel-filled), that side is dropped and
// a one-sided difference from the feasible base is used; if BOTH sides fail the column entry
// is set to 0 (the LM damping + line search handle the resulting noise).
static void slim_coupled_numerical_jacobian(const std::vector<double>& U,
                                            const SlimDiskInputs& in, const OpacityLUTs& op,
                                            const ColumnOpts& copt, ColumnCache& cache,
                                            std::vector<double>& J) {
    const int N = std::max(in.n_nodes, 4);
    const int n = 4 * N + 2;
    J.assign((size_t)n * n, 0.0);

    // Per-variable FD step (relative with a type-keyed absolute floor).
    auto step_for = [&](int col) -> double {
        const double u = U[col];
        double floor;
        if (col >= 4 * N) {                       // globals
            floor = (col == 4*N) ? 1e-6 : 1e-5;   // ℓ_in : r_s
        } else {
            const int off = col % 4;
            switch (off) {
                case 0: floor = 1e-3 * std::max(std::abs(u), 1e2); break; // Σ
                case 1: floor = 1e-9;                                break; // V (small-neg)
                case 2: floor = 1e-6;                                break; // ℓ
                default: floor = 1.0;                               break; // T_c
            }
        }
        return std::max(1e-6 * std::abs(u), floor);
    };

    // Base residual at U — the anchor for one-sided differencing when a perturbed side
    // goes infeasible (a finite ~1e305 sentinel difference would otherwise slip past the
    // isfinite guard and inject garbage).
    // Flags are INFEASIBILITY flags (true = that side has no usable residual); every side
    // here is ALWAYS solved (this oracle stays strictly central — no one-sided fast path),
    // so each flag is overwritten with a real value before it is read.
    std::vector<double> Up, Um, Rp, Rm, R0; bool inf0 = true;
    slim_coupled_residual(U, in, op, copt, cache, R0, inf0);
    for (int col = 0; col < n; ++col) {
        const double h = step_for(col);
        Up = U; Um = U;
        Up[col] += h; Um[col] -= h;
        bool infp = true, infm = true;
        slim_coupled_residual(Up, in, op, copt, cache, Rp, infp);
        slim_coupled_residual(Um, in, op, copt, cache, Rm, infm);
        for (int row = 0; row < n; ++row) {
            double d;
            if      (!infp && !infm) d = (Rp[row] - Rm[row]) / (2.0 * h);   // central
            else if (!infp && !inf0) d = (Rp[row] - R0[row]) / h;           // forward
            else if (!infm && !inf0) d = (R0[row] - Rm[row]) / h;           // backward
            else                     d = 0.0;                               // no usable gradient
            if (!std::isfinite(d)) d = 0.0;
            J[(size_t)row * n + col] = d;
        }
    }
}

// ===========================================================================
// The coupled driver: LM-damped Newton over the FULL 4N+2 system.
// ===========================================================================
// Mirrors arclength_corrector / relax_structure's loop (group + column scaling, the
// Nielsen gain-ratio LM, the feasibility line search) but (a) uses slim_coupled_residual
// for R, (b) builds J by slim_coupled_numerical_jacobian, (c) the feasibility line
// search ALSO rejects any step whose coupled residual is infeasible (a column failed).
// Convergence: merit floored AND step small AND the validity gate passes (require_N1
// = true — this IS a fully-regular solution; ℓ_in and r_s are solved jointly here).
// Returns true iff converged; the caller unpacks the profile.
static bool relax_coupled(const SlimDiskInputs& in, const OpacityLUTs& op,
                          const ColumnOpts& copt, std::vector<double>& U, int max_iters) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    const int n = 4 * N + 2;
    const bool kDiag = std::getenv("SLIM_DIAG") != nullptr;
    // Jacobian selector: analytic reduced/Schur J_red by default (Task 10); the pure
    // numerical (re-solve-every-column) Jacobian stays available as a fallback + oracle
    // behind SLIM_COUPLED_NUMJAC=1 (set it to force the old path for A/B comparison).
    const bool kNumJac = std::getenv("SLIM_COUPLED_NUMJAC") != nullptr;
    // OBSERVABILITY ONLY (no effect on the solve): SLIM_CHECKPOINT=<path-prefix> dumps
    // the state vector U once per outer iteration to "<prefix>_it%03d.txt".  Unset (the
    // default) => nothing is opened, nothing is written.  See the write site below for
    // the file format and the U layout.
    const char* const kCkptPrefix = std::getenv("SLIM_CHECKPOINT");

    ColumnCache cache; cache.resize(N, copt.n_z);

    // Helper: full coupled scaled merit (mirrors slim_scaled_residual_norm — INCLUDES
    // the 𝒩₁ row, since this full solve drives 𝒩₁→0).  Returns +inf if infeasible.
    auto eval_merit = [&](const std::vector<double>& Uw, std::vector<double>& Rw,
                          bool& infeas) -> double {
        slim_coupled_residual(Uw, in, op, copt, cache, Rw, infeas);
        if (infeas) return 1e300;   // huge finite sentinel (a column failed) -> reject
        return slim_scaled_residual_norm(Uw, Rw, in);
    };

    std::vector<double> R;
    bool infeas = false;
    double merit = eval_merit(U, R, infeas);
    if (infeas || !std::isfinite(merit)) {
        if (kDiag) std::printf("[COUPLED] seed INFEASIBLE (a column failed) -> abort\n");
        return false;
    }
    double merit_prev = merit;

    // LM state (mirrors relax_structure).
    double lm_mu = 1e-3, lm_nu = 2.0;
    constexpr double kMuMax = 1e12, kMuMin = 1e-9;
    constexpr double kStepCap   = 0.5;
    // The coupled residual's column FD inexactness + the numerical radial Jacobian set
    // the achievable merit floor; accept a few × above the standalone analytic-J inner
    // floor, guarded by the validity gate (same philosophy as relax_structure/arc-corr).
    constexpr double kMeritFloor = 5e-3;
    constexpr double kStepFloor  = 5e-3;
    constexpr double kPlateauRel = 5e-3;

    const double cnt_full = (double)n;
    auto merit_to_F = [cnt_full](double m) { return 0.5 * cnt_full * m * m; };

    bool converged = false;
    for (int it = 0; it < max_iters; ++it) {
        if (g_budget) { ++g_budget->inner_iters; if (g_budget->check()) {
            if (kDiag) std::printf("[COUPLED] it=%d BUDGET EXCEEDED -> abort\n", it);
            break; } }

        // Jacobian of the coupled residual.  Analytic reduced/Schur J_red by default
        // (Σ,T_c columns analytic via C3; ℓ/ℓ_in/r_s by re-solve FD); numerical fallback
        // if forced (SLIM_COUPLED_NUMJAC) or if the analytic base point is infeasible.
        std::vector<double> J;
        bool jac_ok = false;
        if (!kNumJac) jac_ok = slim_coupled_reduced_jacobian(U, in, op, copt, cache, J);
        if (!jac_ok)  slim_coupled_numerical_jacobian(U, in, op, copt, cache, J);

        // Row + column scaling (full system; same scheme as relax_structure).
        std::vector<double> cs(n), rs_inv(n);
        {
            double mSig=0,mV=0,mEll=0,mT=0;
            for (int i=0;i<N;++i){ mSig=std::max(mSig,std::abs(U[4*i+0])); mV=std::max(mV,std::abs(U[4*i+1]));
                                   mEll=std::max(mEll,std::abs(U[4*i+2])); mT=std::max(mT,std::abs(U[4*i+3])); }
            mSig=std::max(mSig,1e-30); mV=std::max(mV,1e-30); mEll=std::max(mEll,1e-30); mT=std::max(mT,1.0);
            for (int i=0;i<N;++i){ cs[4*i+0]=mSig; cs[4*i+1]=mV; cs[4*i+2]=mEll; cs[4*i+3]=mT; }
            cs[4*N+0]=std::max(std::abs(U[4*N+0]),1e-30);   // ℓ_in
            cs[4*N+1]=std::max(std::abs(U[4*N+1]),1e-30);   // r_s
            const GroupScales gs = slim_group_scales(U, in);
            auto setrows=[&](int b,int e,double sc){ sc=std::max(sc,1e-300); for(int rr=b;rr<e;++rr) rs_inv[rr]=1.0/sc; };
            setrows(0,N,gs.mass); setrows(N,2*N,gs.ang); setrows(2*N,3*N-1,gs.rad);
            setrows(3*N-1,4*N-2,gs.ene); setrows(4*N-2,4*N-1,gs.bc_ell);
            setrows(4*N-1,4*N,gs.ene); setrows(4*N,4*N+1,gs.reg_D0); setrows(4*N+1,4*N+2,gs.reg_N1);
        }

        // Scaled full Jacobian Js (n×n) and residual Rs (n).
        std::vector<double> Js((size_t)n*n,0.0), Rs(n,0.0);
        for (int a = 0; a < n; ++a) {
            Rs[a] = R[a] * rs_inv[a];
            for (int b = 0; b < n; ++b)
                Js[(size_t)a*n+b] = J[(size_t)a*n+b] * rs_inv[a] * cs[b];
        }
        // Normal equations.
        std::vector<double> JtJ((size_t)n*n,0.0), Jtr(n,0.0);
        for (int i=0;i<n;++i) for (int k=0;k<n;++k){
            const double jik = Js[(size_t)k*n+i]; if (jik==0.0) continue;
            Jtr[i]+=jik*Rs[k];
            for (int j=0;j<n;++j) JtJ[(size_t)i*n+j]+=jik*Js[(size_t)k*n+j];
        }

        const double F_old = merit_to_F(merit);
        std::vector<double> Adamp((size_t)n*n), bdamp(n);
        bool step_taken=false, bail=false;
        double merit_try=merit;
        std::vector<double> Utry, Rtry;

        while (true) {
            bool solved=false;
            for (int tries=0; tries<12 && !solved; ++tries) {
                Adamp=JtJ;
                for (int i=0;i<n;++i) Adamp[(size_t)i*n+i]+=lm_mu*std::max(JtJ[(size_t)i*n+i],1e-300);
                for (int i=0;i<n;++i) bdamp[i]=-Jtr[i];
                if (coupled_dense_solve(Adamp,bdamp,n)) { solved=true; break; }
                lm_mu=std::min(lm_mu*10.0,kMuMax);
                if (lm_mu>=kMuMax) break;
            }
            if (!solved) { if (kDiag) std::printf("[COUPLED] it=%d SINGULAR -> bail\n", it); bail=true; break; }

            double pred=0.0;
            for (int i=0;i<n;++i){ const double Dii=std::max(JtJ[(size_t)i*n+i],1e-300);
                                   pred+=lm_mu*Dii*bdamp[i]*bdamp[i]-bdamp[i]*Jtr[i]; }
            pred*=0.5;

            // Unscale the step.
            std::vector<double> dU(n,0.0);
            for (int b=0;b<n;++b) dU[b]=bdamp[b]*cs[b];

            // Trust-region cap on Σ,T_c.
            double lam=1.0;
            for (int i=0;i<N;++i) for (int c : {0,3}) {
                const double u=U[4*i+c], d=dU[4*i+c];
                if (u!=0.0 && d!=0.0){ const double f=std::abs(d/u); if (f*lam>kStepCap) lam=kStepCap/f; }
            }

            // Feasibility line search: largest λ giving a PHYSICAL iterate (Σ>0,T_c>0,
            // V<0, r_s∈(r_in,r_out)) AND a FEASIBLE coupled residual (all columns
            // converged) AND no f_adv blow-up.  Reject (ρ≤0) if none found.
            bool physical=false; double F_new=F_old; bool infeas_try=false;
            for (int ls=0; ls<40; ++ls) {
                Utry.assign(U.begin(),U.end());
                for (int i=0;i<n;++i) Utry[i]+=lam*dU[i];
                physical=true;
                for (int i=0;i<N&&physical;++i){ const double S=Utry[4*i+0],V=Utry[4*i+1],T=Utry[4*i+3];
                    if (S<=0.0||T<=0.0||!(V<0.0)) physical=false; }
                if (physical){ const double rs=Utry[4*N+1]; if(!(rs>in.r_in&&rs<in.r_out)) physical=false; }
                if (physical && !slim_fadv_ok(in, op, Utry, 50.0)) physical=false;
                if (physical){
                    merit_try=eval_merit(Utry,Rtry,infeas_try);
                    if (infeas_try || !std::isfinite(merit_try)) { physical=false; }  // a column failed
                    else { F_new=merit_to_F(merit_try); break; }
                }
                lam*=0.5;
            }

            const double act = physical ? (F_old-F_new) : -1.0;
            const double rho = act/std::max(pred,1e-300);
            if (rho>0.0) {
                const double t=2.0*rho-1.0;
                lm_mu=std::max(lm_mu*std::max(1.0/3.0,1.0-t*t*t),kMuMin);
                lm_nu=2.0; step_taken=true; break;
            }
            if (lm_mu>=kMuMax){ if (kDiag) std::printf("[COUPLED] it=%d GAIN-RATIO STALL merit=%.3e\n", it, merit); bail=true; break; }
            lm_mu=std::min(lm_mu*lm_nu,kMuMax); lm_nu*=2.0;
        }

        if (bail || !step_taken) break;

        double maxrel=0.0;
        for (int i=0;i<n;++i) maxrel=std::max(maxrel,std::abs(Utry[i]-U[i])/std::max(std::abs(U[i]),1e-300));

        U.swap(Utry); R.swap(Rtry); merit=merit_try;

        // De-glitch Σ-outliers introduced by the step (same source fix as the one-zone
        // path); refresh R/merit if anything was repaired.
        {
            const int nrep = deglitch_sigma_outliers(in, U);
            if (nrep > 0) {
                bool inf2=false; merit = eval_merit(U, R, inf2);
                if (inf2 || !std::isfinite(merit)) { if (kDiag) std::printf("[COUPLED] deglitch -> infeasible\n"); break; }
            }
        }

        if (kDiag) {  // EVERY outer iteration (merit trajectory: converging vs stuck)
            std::printf("[COUPLED] it=%d merit=%.3e maxrel=%.2e mu=%.1e r_s=%.4f\n",
                        it, merit, maxrel, lm_mu, U[4*N+1]);
            // --- PURELY OBSERVATIONAL: per-row-group breakdown of the SAME scaled
            // residual the merit is built from.  merit = sqrt(Σ_rows (R[row]/s_grp)²/n)
            // (slim_scaled_residual_norm), so these group L2s satisfy
            //     merit = sqrt(Σ_groups L2_group² / n).
            // Reuses the residual R ALREADY computed for this accepted iterate (no extra
            // residual evaluation, no column solves); the group scales are the same cheap
            // O(N) reduction over U that slim_scaled_residual_norm does internally.  Note
            // the row-scale vector rs_inv built above belongs to the PRE-step iterate, so
            // it is deliberately NOT reused here — the scales must match the U/R pair that
            // produced the merit being printed.  Group partition is exactly the setrows
            // partition above (mass|ang|rad|ene|bc_ell|bc_ene|reg_D0|reg_N1).
            const GroupScales gsd = slim_group_scales(U, in);
            struct GrpStat { double l2 = 0.0, mx = 0.0; int at = 0; };
            auto grp = [&](int b, int e, double sc) -> GrpStat {
                GrpStat g; g.at = b;
                const double s = std::max(sc, 1e-300);
                for (int rr = b; rr < e; ++rr) {
                    const double v = R[rr] / s;
                    g.l2 += v * v;
                    if (std::abs(v) > g.mx) { g.mx = std::abs(v); g.at = rr; }
                }
                g.l2 = std::sqrt(g.l2);
                return g;
            };
            const GrpStat g_mass = grp(0,       N,       gsd.mass);
            const GrpStat g_ang  = grp(N,       2*N,     gsd.ang);
            const GrpStat g_rad  = grp(2*N,     3*N-1,   gsd.rad);
            const GrpStat g_ene  = grp(3*N-1,   4*N-2,   gsd.ene);
            const GrpStat g_bcl  = grp(4*N-2,   4*N-1,   gsd.bc_ell);
            const GrpStat g_bce  = grp(4*N-1,   4*N,     gsd.ene);
            const GrpStat g_rD0  = grp(4*N,     4*N+1,   gsd.reg_D0);
            const GrpStat g_rN1  = grp(4*N+1,   4*N+2,   gsd.reg_N1);
            std::printf("[COUPLED] it=%d RESID mass=%.2e(max %.2e@%d) ang=%.2e(max %.2e@%d)"
                        " rad=%.2e(max %.2e@%d) ene=%.2e(max %.2e@%d)\n",
                        it, g_mass.l2, g_mass.mx, g_mass.at, g_ang.l2, g_ang.mx, g_ang.at,
                        g_rad.l2, g_rad.mx, g_rad.at, g_ene.l2, g_ene.mx, g_ene.at);
            std::printf("[COUPLED] it=%d RESID bc_ell=%.2e(max %.2e@%d) bc_ene=%.2e(max %.2e@%d)"
                        " reg_D0=%.2e(max %.2e@%d) reg_N1=%.2e(max %.2e@%d)\n",
                        it, g_bcl.l2, g_bcl.mx, g_bcl.at, g_bce.l2, g_bce.mx, g_bce.at,
                        g_rD0.l2, g_rD0.mx, g_rD0.at, g_rN1.l2, g_rN1.mx, g_rN1.at);
        }

        // --- PURELY OBSERVATIONAL: state checkpoint (OFF unless SLIM_CHECKPOINT is set).
        // HOW TO USE:
        //   set SLIM_CHECKPOINT=C:\path\prefix   (unset => nothing is written, zero I/O)
        // One file PER OUTER ITERATION: "<prefix>_it%03d.txt" — per-iteration (not a
        // single overwritten file) precisely so states can be DIFFED across iterations,
        // which is the point of checkpointing a stalling relax.
        // FILE FORMAT: '#'-prefixed header (iteration, N, merit, maxrel, lm_mu, and the
        // SlimDiskInputs needed to interpret/reload the state), then exactly 4N+2 lines,
        // one value of U per line, "%.17g" (round-trip exact).
        // U LAYOUT (verified against this file: cs[] packing above, the physicality
        // check in the line search, and unpack_profile):
        //   for i in [0,N):  U[4*i+0] = Sigma_i [g/cm^2]
        //                    U[4*i+1] = V_i     (radial velocity, <0 = inflow)
        //                    U[4*i+2] = ell_i   (specific angular momentum)
        //                    U[4*i+3] = T_c,i   [K]
        //   then             U[4*N+0] = ell_in  (eigenvalue)
        //                    U[4*N+1] = r_s     (sonic radius [M])
        // NODE RADII are NOT stored (they are implied): the grid is log-uniform on
        //   [r_s, r_out] with r_i = exp(ln r_s + (ln r_out - ln r_s)*i/(N-1)),
        //   r_s = U[4*N+1], r_out = the r_out recorded in the header.
        // A failed open is a NON-EVENT for the solve: warn under kDiag and continue.
        if (kCkptPrefix != nullptr && kCkptPrefix[0] != '\0') {
            char path[1024];
            std::snprintf(path, sizeof(path), "%s_it%03d.txt", kCkptPrefix, it);
            std::FILE* f = std::fopen(path, "w");
            if (f == nullptr) {
                if (kDiag) std::printf("[COUPLED] it=%d CHECKPOINT open failed: %s\n", it, path);
            } else {
                std::fprintf(f, "# GRRT slim-disk COUPLED state checkpoint (SLIM_CHECKPOINT)\n");
                std::fprintf(f, "# iter %d\n", it);
                std::fprintf(f, "# N %d\n", N);
                std::fprintf(f, "# n_unknowns %d\n", n);
                std::fprintf(f, "# merit %.17g\n", merit);
                std::fprintf(f, "# maxrel %.17g\n", maxrel);
                std::fprintf(f, "# lm_mu %.17g\n", lm_mu);
                std::fprintf(f, "# mass %.17g\n", in.mass);
                std::fprintf(f, "# spin %.17g\n", in.spin);
                std::fprintf(f, "# alpha %.17g\n", in.alpha);
                std::fprintf(f, "# mdot %.17g\n", in.mdot);
                std::fprintf(f, "# r_g %.17g\n", in.r_g);
                std::fprintf(f, "# r_in %.17g\n", in.r_in);
                std::fprintf(f, "# r_out %.17g\n", in.r_out);
                std::fprintf(f, "# n_nodes %d\n", in.n_nodes);
                std::fprintf(f, "# n_z %d\n", copt.n_z);
                std::fprintf(f, "# layout [Sigma_i,V_i,ell_i,T_c_i] x N, then ell_in, r_s\n");
                std::fprintf(f, "# grid r_i = exp(ln(r_s) + (ln(r_out)-ln(r_s))*i/(N-1)), r_s = U[4N+1]\n");
                std::fprintf(f, "# %d values follow, one per line\n", n);
                for (int i = 0; i < n; ++i) std::fprintf(f, "%.17g\n", U[i]);
                std::fclose(f);
            }
        }

        const bool merit_floored = (merit < kMeritFloor);
        const bool step_ideal    = (maxrel < in.tol);
        const bool step_plateau  = (maxrel < kStepFloor)
                                && ((merit_prev - merit) <= kPlateauRel * std::max(merit, 1e-300));
        if (merit_floored && (step_ideal || step_plateau)) {
            const ValidityResult v = slim_validity_gate(in, op, U, /*require_N1=*/true);
            if (kDiag)
                std::printf("[COUPLED] it=%d ACCEPT-CHECK merit=%.3e maxrel=%.2e gate.all=%d\n",
                            it, merit, maxrel, (int)v.all(true));
            if (v.all(/*require_N1=*/true)) { converged = true; break; }
        }
        merit_prev = merit;
    }
    return converged;
}

} // namespace slim_coupled_detail

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------
SlimDiskRadial solve_slim_disk_coupled(const SlimDiskInputs& in, const OpacityLUTs& op) {
    using namespace slim_coupled_detail;
    const bool kDiag = std::getenv("SLIM_DIAG") != nullptr;

    // Honest degenerate-input guard: no accretion (Ṁ≤0) ⇒ no transonic structure.
    if (!(in.mdot > 0.0)) {
        if (kDiag) std::printf("[COUPLED] mdot<=0 -> converged=false (degenerate)\n");
        return SlimDiskRadial{};
    }

    // Install the runaway safety budget for THIS solve (RAII-cleared on return).  The
    // coupled solve is far slower per iteration (a full column solve per node per FD
    // column), so respect the caller's caps; defaults are the generous SolveBudget ones.
    SolveBudget budget;
    if (in.budget_inner_iter_cap > 0) budget.inner_iter_cap = in.budget_inner_iter_cap;
    if (in.budget_wall_seconds   > 0) budget.wall_cap_s     = in.budget_wall_seconds;
    struct BudgetGuard { ~BudgetGuard() { g_budget = nullptr; } } budget_guard;
    g_budget = &budget;

    ColumnOpts copt;   // bring-up column resolution (small n_z, generous iters)

    // Cold thin-disk seed on the [r_s, r_out] grid (the SAME seed the one-zone driver
    // starts from).  build_thin_disk_seed pins r_s≈0.98·ISCO, ℓ_in≈ℓ_K(ISCO), the
    // node-0 Mach-1 override, and a §23-consistent inner annulus.
    std::vector<double> U = build_thin_disk_seed(in, op);

    const bool ok = relax_coupled(in, op, copt, U, std::max(in.max_iters, 1));
    if (g_budget && g_budget->tripped) {
        std::fprintf(stderr, "[COUPLED] BUDGET EXCEEDED (%s) -> honest fallback\n",
                     budget.what ? budget.what : "?");
        return SlimDiskRadial{};
    }
    if (!ok) {
        if (kDiag) std::printf("[COUPLED] relax_coupled did not converge -> converged=false\n");
        return SlimDiskRadial{};
    }

    // Unpack the converged state into the output profile.  Reuse the one-zone
    // unpack_profile machinery via a fresh SlimDiskRadial: it rebuilds H/f_adv from the
    // one-zone closure for the OUTPUT fields (the coupled solve is on (Σ,V,ℓ,T_c); the
    // reported H/f_adv here are the one-zone diagnostics — the converged column F/z0/η
    // can be re-derived from (Σ,T_c) by the caller via solve_column_coupled).  The
    // state (Σ,V,Ω,T_c,r_s,ℓ_in) IS the coupled solution.
    SlimDiskRadial out;
    unpack_profile(in, op, U, out);
    out.converged = true;
    out.iters     = std::max(in.max_iters, 1);
    return out;
}

} // namespace grrt
