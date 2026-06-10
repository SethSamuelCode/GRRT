#include "grrt/scene/slim_disk_radial.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>
#include <numbers>
#include <cstdlib>
#include <cstdio>

namespace grrt {

// ---------------------------------------------------------------------------
// One-zone vertical closure
// ---------------------------------------------------------------------------
namespace slim_detail {

/// Solve the scale-height quadratic for a fixed μ.
///
/// Hydrostatic balance H = c_s/Ω_⊥ with c_s² = p_mid/ρ_mid and
/// ρ_mid = Σ/(2H) gives:
///   Ω_⊥² H² − b H − c_s_gas² = 0 ,  b = 2 a_rad T_c⁴ / (3 Σ)
/// Positive root: H = (b + sqrt(b² + 4 Ω_⊥² c_s_gas²)) / (2 Ω_⊥²)
static double solve_H(double Sigma, double Tc, double mu, double Omega_perp2_cgs) {
    using namespace constants;
    const double cs_gas2 = k_B * Tc / (mu * m_p);                    // [cm²/s²]
    const double b       = 2.0 * a_rad * Tc*Tc*Tc*Tc / (3.0 * Sigma); // radiation term [cm²/s² per cm]
    // H = (b + sqrt(b² + 4 Ω_⊥² c_s_gas²)) / (2 Ω_⊥²)
    const double disc = b * b + 4.0 * Omega_perp2_cgs * cs_gas2;
    return (b + std::sqrt(disc)) / (2.0 * Omega_perp2_cgs);          // [cm]
}

OneZoneState one_zone_closure(double Sigma, double Tc, double r,
                              const SlimDiskInputs& in, const OpacityLUTs& op) {
    using namespace constants;

    // Vertical epicyclic frequency in CGS.
    //   ω_⊥² (geometric, 1/M²) → CGS (1/s²) by multiplying by (c_cgs/r_g)²
    const double conv = c_cgs / in.r_g;                               // [1/s per 1/M]
    const double Omega_perp2_geom = omega_perp2(in.mass, in.spin, r); // [1/M²]
    const double Omega_perp2_cgs  = Omega_perp2_geom * conv * conv;   // [1/s²]
    const double Omega_perp       = std::sqrt(Omega_perp2_cgs);       // [1/s]

    // Numerical guards for transient non-physical Newton iterates (Tasks 4-5).
    constexpr double SIGMA_FLOOR = 1e-30;  // g/cm^2 — guards b=2aT^4/(3Sigma) and rho_mid sign
    constexpr double RHO_FLOOR   = 1e-30;  // g/cm^3 — guards log(rho), 1/rho in the entropy
    constexpr double T_FLOOR     = 1.0;    // K — guards c_s_gas^2 and T^4
    if (!(Omega_perp2_cgs > 0.0)) return OneZoneState{};   // unsolvable here; caller checks st.H==0
    const double Sigma_s = std::max(Sigma, SIGMA_FLOOR);
    const double Tc_s    = std::max(Tc, T_FLOOR);

    // -----------------------------------------------------------------------
    // ≤3-iteration μ fixed point; μ is ~constant in the fully-ionized inner
    // disk but varies at low T where partial ionization matters.
    // -----------------------------------------------------------------------
    double mu = mu_fully_ionized;
    for (int it = 0; it < 3; ++it) {
        const double H_it   = solve_H(Sigma_s, Tc_s, mu, Omega_perp2_cgs);
        const double rho_it = Sigma_s / (2.0 * H_it);
        double mu_new = op.lookup_mu(rho_it, Tc_s);
        if (!(mu_new > 0.0) || !std::isfinite(mu_new)) mu_new = mu_fully_ionized;
        const bool conv = std::abs(mu_new - mu) <= 1e-6 * mu;
        mu = mu_new;
        if (conv) break;
    }
    const double H       = solve_H(Sigma_s, Tc_s, mu, Omega_perp2_cgs);
    const double rho_mid = Sigma_s / (2.0 * H);

    // -----------------------------------------------------------------------
    // Thermodynamic quantities
    // -----------------------------------------------------------------------

    const double p_gas = rho_mid * k_B * Tc_s / (mu * m_p);          // [erg/cm³]
    const double p_rad = (a_rad / 3.0) * Tc_s*Tc_s*Tc_s*Tc_s;        // [erg/cm³]
    const double p_mid = p_gas + p_rad;                               // [erg/cm³]

    // Total sound speed: by construction equals H·Ω_⊥.
    const double c_s = Omega_perp * H;                                // [cm/s]

    // Vertically-integrated pressure (α-stress, trap #9 — use P, not p_mid).
    const double P = 2.0 * p_mid * H;                                 // [erg/cm²]

    // Specific entropy per unit mass (gas + radiation; additive constant irrelevant).
    // S = (k_B/(μ m_p)) ln(T_c^{3/2}/ρ_mid) + (4 a_rad T_c³)/(3 ρ_mid)
    // Only dS/dr enters Q_adv in the radial solver.
    const double rho_e  = std::max(rho_mid, RHO_FLOOR);
    const double S_gas = (k_B / (mu * m_p)) * std::log(std::pow(Tc_s, 1.5) / rho_e);
    const double S_rad = (4.0 * a_rad * Tc_s*Tc_s*Tc_s) / (3.0 * rho_e);
    const double S     = S_gas + S_rad;                               // [erg/(g·K)] (up to additive const)

    OneZoneState st;
    st.H       = H;
    st.rho_mid = rho_mid;
    st.c_s     = c_s;
    st.p_mid   = p_mid;
    st.p_gas   = p_gas;
    st.p_rad   = p_rad;
    st.P       = P;
    st.S       = S;
    st.mu      = mu;
    return st;
}

// ---------------------------------------------------------------------------
// Equatorial Kerr orbital mechanics helpers (geometric units, G=c=1)
// ---------------------------------------------------------------------------
//
// Equatorial Kerr metric components (match VolumetricDisk::circular_velocity):
//   g_tt   = -(1 - 2M/r)
//   g_tφ   = -2 M a / r
//   g_φφ   =  r² + a² + 2 M a²/r
//
// For a circular orbit u^φ = Ω u^t, normalization u^μu_μ=-1 gives
//   (u^t)² · D(Ω) = 1,  D(Ω) = -(g_tt + 2 g_tφ Ω + g_φφ Ω²)
// and the covariant specific angular momentum is
//   ℓ(Ω) = u_φ = (g_tφ + g_φφ Ω) · u^t = (g_tφ + g_φφ Ω) / √D(Ω).
//
// The seed sets ℓ = ℓ_K (Keplerian); the residual needs the inverse Ω(ℓ).
// ℓ(Ω) is smooth and monotone over the physically relevant prograde branch,
// so we invert with a short damped Newton iteration seeded from Ω_K. This is a
// well-defined local solve (not a guess); documented per the task latitude note.

static inline void eq_metric(double M, double a, double r,
                             double& g_tt, double& g_tphi, double& g_phph) {
    g_tt   = -(1.0 - 2.0 * M / r);
    g_tphi = -2.0 * M * a / r;
    g_phph = r * r + a * a + 2.0 * M * a * a / r;
}

double ell_kepler(double M, double a, double r) {
    double g_tt, g_tphi, g_phph;
    eq_metric(M, a, r, g_tt, g_tphi, g_phph);
    const double Om = omega_k(M, a, r);                       // prograde Ω_K
    const double D  = -(g_tt + 2.0 * g_tphi * Om + g_phph * Om * Om);
    const double ut = 1.0 / std::sqrt(std::max(D, 1e-30));
    return (g_tphi + g_phph * Om) * ut;                       // ℓ_K = u_φ
}

double isco_prograde(double M, double a) {
    // BPT72 marginally-stable circular orbit (prograde). The expression is
    // dimensionless in a_* = a/M and scales linearly with M, so evaluate at
    // a_* and multiply by M. For M=1 this reduces to the bare BPT72 formula.
    const double as = a / M;
    const double Z1 = 1.0 + std::cbrt(1.0 - as * as)
                          * (std::cbrt(1.0 + as) + std::cbrt(1.0 - as));
    const double Z2 = std::sqrt(3.0 * as * as + Z1 * Z1);
    const double r_star = 3.0 + Z2 - std::sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2));
    return M * r_star;
}

double omega_from_ell(double M, double a, double r, double ell) {
    double g_tt, g_tphi, g_phph;
    eq_metric(M, a, r, g_tt, g_tphi, g_phph);
    // ℓ(Ω) = (g_tφ + g_φφ Ω)/√D,  D = -(g_tt + 2 g_tφ Ω + g_φφ Ω²)
    // dℓ/dΩ = g_φφ/√D + (g_tφ + g_φφ Ω)·(g_tφ + g_φφ Ω)/D^{3/2}
    //       = g_φφ/√D + (numer)²/D^{3/2}   (numer ≡ g_tφ + g_φφ Ω)
    double Om = omega_k(M, a, r);                             // seed from Keplerian
    for (int it = 0; it < 40; ++it) {
        const double D = -(g_tt + 2.0 * g_tphi * Om + g_phph * Om * Om);
        if (!(D > 0.0)) { Om *= 0.5; continue; }              // stay timelike
        const double sqrtD = std::sqrt(D);
        const double numer = g_tphi + g_phph * Om;
        const double f  = numer / sqrtD - ell;
        const double df = g_phph / sqrtD + numer * numer / (D * sqrtD);
        if (!(std::abs(df) > 0.0)) break;
        double step = f / df;
        // damp to keep D>0
        double Om_new = Om - step;
        double D_new  = -(g_tt + 2.0 * g_tphi * Om_new + g_phph * Om_new * Om_new);
        int guard = 0;
        while (!(D_new > 0.0) && guard++ < 30) {
            step *= 0.5;
            Om_new = Om - step;
            D_new  = -(g_tt + 2.0 * g_tphi * Om_new + g_phph * Om_new * Om_new);
        }
        Om = Om_new;
        if (std::abs(step) <= 1e-14 * (std::abs(Om) + 1e-30)) break;
    }
    return Om;
}

} // namespace slim_detail

// ---------------------------------------------------------------------------
// Transonic radial residual (Phase 1, one-zone)
// ---------------------------------------------------------------------------
//
// State vector U (length 4N+2), index 0 = inner edge, N-1 = outer edge:
//   per node i:  U[4i+0]=Σ_i [g/cm²]   U[4i+1]=V_i (corotating radial vel, <0 inflow)
//                U[4i+2]=ℓ_i [u_φ,geom] U[4i+3]=T_{c,i} [K]
//   globals:     U[4N]=ℓ_in (eigenvalue)   U[4N+1]=r_s (sonic radius, [M])
//
// GRID: logarithmic in r over [in.r_in, in.r_out]. Log spacing concentrates
//       nodes near the hot inner disk / sonic point. r_i = r_in·(r_out/r_in)^{i/(N-1)}.
//
// CONVENTIONS / SIMPLIFICATIONS (Phase 1, one-zone):
//   * Geometric mechanics; CGS thermodynamics. The one-zone closure
//     one_zone_closure() returns the vertically-integrated P [erg/cm²] and the
//     midplane p_mid, rho_mid, c_s used below.
//   * Γ₁ fixed = 5/3 (ideal monatomic gas) ⇒ η₃ = 1/(Γ₁-1) = 1.5, Γ̃₁ = 1+1/η₃ = 5/3.
//     One-zone ⇒ η₃,η₄ are CONSTANTS, so dlnη₃/dlnr = dlnη₄/dlnr = 0; the η-gradient
//     term in 𝒩₁ and the Ω_⊥²(η₄/η₃)dlnη₄/dlnr term BOTH vanish (§22/§23).
//   * Ṁ cgs↔geometric: §23 mass law Ṁ = -2π Σ Δ^½ V/√(1-V²) is geometric in length.
//     Σ is CGS [g/cm²], Δ^½ is geometric [M]. The physical mass rate [g/s] is
//       Ṁ = -2π Σ Δ^½ (V/√(1-V²)) · r_g · c_cgs
//     ( Δ^½·r_g → cm gives g/cm; the 4-velocity component ·c_cgs → cm/s gives g/s ).
//     We compare against in.mdot [g/s] directly.  (Trap #5/§19: lengths × r_g.)
//   * P [erg/cm²] in the angular-momentum and pressure-gradient terms is the
//     vertically-integrated pressure; P/Σ has units erg/g = cm²/s², i.e. (c_s)².
//     Mechanics terms multiply P/Σ by 1/c² to make it the geometric (dimensionless)
//     specific-pressure that the §23 𝒟₀/𝒩₁ forms expect (where V is in units of c).
//
// ROW LAYOUT (square 4N+2; see header):
//   N (mass) + N (ang.mom.) + (N-1) (radial ODE) + (N-1) (energy ODE)
//     + 2 (outer BC) + 2 (regularity) = 4N+2.
// ---------------------------------------------------------------------------
namespace {

using slim_detail::OneZoneState;
using slim_detail::one_zone_closure;
using slim_detail::omega_k;
using slim_detail::kerr_delta;
using slim_detail::kerr_A;
using slim_detail::omega_perp2;
using slim_detail::ell_kepler;
using slim_detail::omega_from_ell;
using slim_detail::isco_prograde;

// Fixed Phase-1 adiabatic indices (documented simplification).
constexpr double kGamma1 = 5.0 / 3.0;          // ideal monatomic gas
constexpr double kEta3   = 1.0 / (kGamma1 - 1.0); // = 1.5
constexpr double kGtilde1 = 1.0 + 1.0 / kEta3;  // = Γ̃₁ = 5/3

// State guards for transient Newton iterates.
constexpr double kSigmaFloor = 1e-30;
constexpr double kTFloor     = 1.0;
constexpr double kVCap       = 0.999999;        // keep |V|<1 (timelike)

struct NodeMech {
    double Delta, sqrtDelta, A, sqrtA;
    double Omega;          // orbital Ω [geometric 1/M] from ℓ
    double Omega_k_plus;   // prograde Keplerian Ω_K⁺
    double Omega_k_minus;  // retrograde Keplerian Ω_K⁻
};

static NodeMech node_mech(const SlimDiskInputs& in, double r, double ell) {
    NodeMech m;
    m.Delta     = kerr_delta(in.mass, in.spin, r);
    m.sqrtDelta = std::sqrt(std::max(m.Delta, 0.0));
    m.A         = kerr_A(in.mass, in.spin, r);
    m.sqrtA     = std::sqrt(std::max(m.A, 0.0));
    m.Omega     = omega_from_ell(in.mass, in.spin, r, ell);
    const double sqM = std::sqrt(in.mass);
    const double r32 = r * std::sqrt(r);
    m.Omega_k_plus  =  sqM / (r32 + in.spin * sqM);   // Ω_K⁺ = +√M/(r^{3/2}+a√M)
    // Ω_K⁻ denominator → 0 near r ≈ (a√M)^{2/3} just outside the horizon for near-extremal a.
    // Guard: floor |denom| to 1e-30 so we never divide by ~0 if r_in is pushed toward the horizon.
    const double denom_minus = r32 - in.spin * sqM;
    m.Omega_k_minus = -sqM / (std::abs(denom_minus) > 1e-30 ? denom_minus
                                                             : std::copysign(1e-30, denom_minus));
    return m;
}

// Gravitational/centrifugal term 𝒜 (S09 Eq 3, §23):
//   𝒜 = -M·A/(r³ Δ Ω_K⁺ Ω_K⁻) · (Ω-Ω_K⁺)(Ω-Ω_K⁻)/(1 - Ω̃²R̃²)
//   Ω̃ = Ω - ω,  ω = 2Mar/A,  R̃ = A/(r²Δ^½)
static double script_A(const SlimDiskInputs& in, double r, const NodeMech& m) {
    const double M = in.mass, a = in.spin;
    const double omega   = 2.0 * M * a * r / m.A;
    const double Om_tilde = m.Omega - omega;
    const double R_tilde  = m.A / (r * r * std::max(m.sqrtDelta, 1e-30));
    const double denom_rel = 1.0 - Om_tilde * Om_tilde * R_tilde * R_tilde;
    const double pref = -M * m.A
                      / (r * r * r * std::max(m.Delta, 1e-30)
                         * m.Omega_k_plus * m.Omega_k_minus);
    const double num  = (m.Omega - m.Omega_k_plus) * (m.Omega - m.Omega_k_minus);
    // guard the (1 - Ω̃²R̃²) denominator (corotation singularity); floor magnitude.
    const double dr = (std::abs(denom_rel) > 1e-12) ? denom_rel
                                                    : std::copysign(1e-12, denom_rel);
    return pref * num / dr;
}

// Mdot from a node (CGS, [g/s]) per the mass law; sign(V<0)->Mdot>0 for inflow.
static double mdot_of_node(const SlimDiskInputs& in, double Sigma, double V,
                           double sqrtDelta) {
    using namespace constants;
    const double Vc = std::clamp(V, -kVCap, kVCap);
    const double Gamma = 1.0 / std::sqrt(1.0 - Vc * Vc);
    return -2.0 * std::numbers::pi * Sigma * sqrtDelta * Vc * Gamma * in.r_g * c_cgs;
}

// Bundle of per-node derived quantities used across rows.
struct NodeEval {
    double r, Sigma, V, ell, Tc;
    OneZoneState oz;
    NodeMech mech;
    double Gamma;           // Lorentz factor 1/√(1-V²)
    double P_over_Sigma_geom; // (P/Σ)/c²  [dimensionless, = (c_s/c)²-ish]
    double cs2_geom;        // Γ̃₁·(P/Σ)/c²  (geometric specific c_s²)
};

static NodeEval eval_node(const SlimDiskInputs& in, const OpacityLUTs& op,
                          double r, double Sigma, double V, double ell, double Tc) {
    using namespace constants;
    NodeEval e;
    e.r = r;
    e.Sigma = std::max(Sigma, kSigmaFloor);
    e.V   = std::clamp(V, -kVCap, kVCap);
    e.ell = ell;
    e.Tc  = std::max(Tc, kTFloor);
    e.oz  = one_zone_closure(e.Sigma, e.Tc, r, in, op);
    e.mech = node_mech(in, r, ell);
    e.Gamma = 1.0 / std::sqrt(1.0 - e.V * e.V);
    // P/Σ has units erg/g = cm²/s²; divide by c² to get the geometric specific
    // pressure that the §23 𝒟₀/𝒩₁ forms (V in units of c) expect.
    const double P_over_Sigma = e.oz.P / e.Sigma;                 // [cm²/s²]
    e.P_over_Sigma_geom = P_over_Sigma / (c_cgs * c_cgs);         // dimensionless
    e.cs2_geom = kGtilde1 * e.P_over_Sigma_geom;
    return e;
}

// One-zone transonic denominator/numerator (§23). Both dimensionless (V in c).
static double calD0(const NodeEval& e) {
    return e.V * e.V - e.cs2_geom;       // 𝒟₀ = V² - Γ̃₁(P/Σ)
}

// 𝒩₁ = 𝒜 + (2πr²/(Ṁ η₃))·Q_adv + (P/Σ)·r(r-M)/Δ·Γ̃₁   (one-zone; η-grad & Ω_⊥² drop)
// Q_adv passed in geometric (dimensionless) form already; see caller.
static double calN1(const SlimDiskInputs& in, const NodeEval& e, double Qadv_geom) {
    const double A_term = script_A(in, e.r, e.mech);
    const double M = in.mass, r = e.r, Delta = std::max(e.mech.Delta, 1e-30);
    // Qadv_geom = (2π r²/(Ṁ η₃))·Q_adv already reduced to dimensionless by the
    // caller (CGS form / c²); the pressure term is the geometric (P/Σ)/c² · r(r-M)/Δ · Γ̃₁.
    const double press_term = e.P_over_Sigma_geom * r * (r - M) / Delta * kGtilde1;
    return A_term + Qadv_geom + press_term;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Crude thin-disk seed builder
// ---------------------------------------------------------------------------

std::vector<double> build_thin_disk_seed(const SlimDiskInputs& in,
                                         const OpacityLUTs& op) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);
    std::vector<double> U((size_t)4 * N + 2, 0.0);

    // Free-inner-node grid (Task 5, option B): node 0 IS the sonic point.
    // Seed the sonic radius just inside the ISCO (it relaxes inward at high Ṁ),
    // clamped above the horizon-floor guard in.r_in. The grid spans [r_s, r_out].
    const double r_isco = isco_prograde(in.mass, in.spin);
    const double r_s = std::max(0.98 * r_isco, in.r_in * 1.001);
    const double r_out = in.r_out;
    const double lr0 = std::log(r_s), lr1 = std::log(r_out);

    // Novikov-Thorne thin-disk seed (no magic factors). At the thin low rung
    // advection is negligible, so the seed satisfies the residual's OWN
    // relativistic ANGULAR-MOMENTUM and ENERGY balances simultaneously, so those
    // two groups start near zero:
    //   • angular momentum (Group 2): (Ṁ/2π)(ℓ_K−ℓ_in)·r_g·c
    //         = (A^½Δ^½/r)·r_g²·Γ·α·P   ⇒  P_target (the α-stress pins P, hence Σ).
    //   • energy (Group 4, advection≈0): Q_vis = Q_rad = 64σT_c⁴/(3κΣ)
    //         ⇒  T_c⁴ = 3κΣ·Q_vis/(64σ),  with the relativistic Q_vis below.
    // P(Σ,T_c) is the one-zone closure's integrated pressure; Σ and T_c are coupled
    // (angular momentum binds Σ at fixed T_c; energy binds T_c at fixed Σ), closed
    // by a 1D bisection on T_c (with Σ on the angular-momentum branch each step).
    // The opacity κ_R used in the energy balance is the residual's own LUT value.
    const double ell_in = ell_kepler(in.mass, in.spin, r_isco);

    // Grid radii + per-node Keplerian Ω_K, ℓ_K (for dΩ/dr and Q_vis).
    std::vector<double> rg(N), OmK(N), ellK(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        rg[i]   = std::exp(lr0 + (lr1 - lr0) * t);
        OmK[i]  = omega_k(in.mass, in.spin, rg[i]);                  // [1/M]
        ellK[i] = ell_kepler(in.mass, in.spin, rg[i]);
    }

    for (int i = 0; i < N; ++i) {
        const double r = rg[i];
        // dΩ_K/dr in CGS (geometric → 1/s via c/r_g; r → cm via r_g), FD on the grid.
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const double dOmega_geom = (OmK[j] - OmK[i]) / (rg[j] - rg[i]);          // [1/M²]
        const double dOmega_dr   = dOmega_geom * (c_cgs / in.r_g) / in.r_g;      // [1/s/cm]
        const double sqrtDelta = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
        const double sqrtA     = std::sqrt(std::max(kerr_A(in.mass, in.spin, r), 0.0));
        // Relativistic Q_vis (Group 4 form), Γ≈1 at the thin seed.
        const double geomfac3  = sqrtA * sqrtDelta / (r * r * r);               // dimensionless
        const double dl_cgs    = (ellK[i] - ell_in) * in.r_g * c_cgs;           // [cm²/s]
        const double Qvis = -(in.mdot / (2.0 * std::numbers::pi)) * dl_cgs * dOmega_dr
                          * (geomfac3 / in.r_g);                                 // [erg/cm²/s]
        const double Qvis_pos = std::max(Qvis, 0.0);                            // ≥0 (heating)
        // Angular-momentum P_target (Group 2, Γ≈1):
        //   (Ṁ/2π)·dl_cgs = (A^½Δ^½/r)·r_g²·α·P  ⇒  P = LHS / [(A^½Δ^½/r)·r_g²·α].
        const double geomlen   = sqrtA * sqrtDelta / r;                         // [M²]
        const double angm_rhs_coef = geomlen * in.r_g * in.r_g * in.alpha;      // [cm²]
        const double P_target = (in.mdot / (2.0 * std::numbers::pi)) * dl_cgs
                              / std::max(angm_rhs_coef, 1e-300);                 // [erg/cm²]

        // Σ(T_c): bisect Σ so the closure's P(Σ,T_c) == P_target (P increases
        // monotonically with Σ at fixed T_c) — pins the angular-momentum balance.
        auto sigma_for_Tc = [&](double Tc_) -> double {
            double lo = 1e-2, hi = 1e12;
            for (int b = 0; b < 70; ++b) {
                const double mid = std::sqrt(lo * hi);            // geometric bisection
                if (one_zone_closure(mid, Tc_, r, in, op).P < P_target) lo = mid; else hi = mid;
            }
            const double s = std::sqrt(lo * hi);
            return (std::isfinite(s) && s > 0.0) ? s : 1e4;
        };
        // Energy-balanced T_c: bisect on the residual's OWN node energy imbalance
        //   g(T_c) = Q_rad(T_c) − Q_vis(T_c),
        // with Σ on the angular-momentum branch (sigma_for_Tc) and BOTH Q's built
        // through eval_node so the Lorentz factor Γ, Ω(ℓ) and κ_R match the residual
        // exactly. g is monotone increasing in T_c (Q_rad ∝ T⁴/(κΣ): T⁴↑, Σ↓, Kramers
        // κ↓ all push it up; Q_vis is ~flat), so the GAS-supported root is the unique
        // sign change — we bracket only up to the radiation-pressure ceiling on T_c
        // (above which p_rad alone exceeds P_target and no Σ exists) so the bisection
        // stays on the physical gas branch.  A naive T_c⁴=3κΣQ_vis/(64σ) fixed point
        // is contractive toward a spurious COLD root (Kramers κ(T) inverts the slope).
        auto V_for_sigma = [&](double Sig_) -> double {
            const double dn = 2.0 * std::numbers::pi * Sig_ * sqrtDelta * in.r_g * c_cgs;
            double Vv = -1e-6;
            if (dn > 0.0) { const double X = -in.mdot / dn; Vv = X / std::sqrt(1.0 + X * X); }
            if (!(Vv < 0.0)) Vv = -1e-6;
            return std::clamp(Vv, -kVCap, -1e-12);
        };
        // Radiation-pressure T_c ceiling: P (integrated) ≈ 2·(aT⁴/3)·H even at Σ→0;
        // cap T_c where p_rad·(scale height) would already overshoot P_target. Use
        // the gas-free closure to find the largest T_c that still admits a Σ>floor.
        const double dOmega_dr_node = dOmega_geom * (c_cgs / in.r_g) / in.r_g;       // [1/s/cm]
        auto energy_imbalance = [&](double Tc_, double& Sig_out) -> double {
            const double Sig_ = sigma_for_Tc(Tc_);
            Sig_out = Sig_;
            const NodeEval ev = eval_node(in, op, r, Sig_, V_for_sigma(Sig_), ellK[i], Tc_);
            const double geomfac_e = ev.mech.sqrtA * ev.mech.sqrtDelta / (r * r * r);
            const double dl_e = (ellK[i] - ell_in) * in.r_g * c_cgs;
            const double Qvis_e = -(in.mdot / (2.0 * std::numbers::pi)) * dl_e * dOmega_dr_node
                                * ev.Gamma * (geomfac_e / in.r_g);
            const double kR = op.lookup_kappa_ross(ev.oz.rho_mid, Tc_)
                            + op.lookup_kappa_es(ev.oz.rho_mid, Tc_);
            const double Qrad_e = 64.0 * sigma_SB * Tc_ * Tc_ * Tc_ * Tc_
                                / (3.0 * std::max(kR, 1e-300) * std::max(Sig_, 1e-30));
            return Qrad_e - std::max(Qvis_e, 0.0);
        };
        // Upper T_c bracket = largest T_c for which a gas-supported Σ still exists
        // (Σ above the bisection floor); scan upward geometrically.
        double Thi = 1e5;
        {
            double Sig_probe = 1e4;
            for (int s = 0; s < 60; ++s) {
                const double Sg = sigma_for_Tc(Thi * 2.0);
                if (!(Sg > 1.0)) break;               // Σ has collapsed → past the ceiling
                Sig_probe = Sg;
                Thi *= 2.0;
            }
            (void)Sig_probe;
        }
        double Tc = 1e6, Sigma = 1e4;
        {
            double Tlo = 1e4;
            double glo = 0.0, ghi = 0.0; double Sd = 1e4;
            glo = energy_imbalance(Tlo, Sd);
            ghi = energy_imbalance(Thi, Sd);
            if (glo * ghi > 0.0) {
                // No bracketed root in the gas window: pick the end with smaller |g|.
                Tc = (std::abs(glo) < std::abs(ghi)) ? Tlo : Thi;
            } else {
                for (int b = 0; b < 80; ++b) {
                    const double Tm = std::sqrt(Tlo * Thi);
                    double Sm = 1e4;
                    if (energy_imbalance(Tm, Sm) < 0.0) Tlo = Tm; else Thi = Tm;
                }
                Tc = std::sqrt(Tlo * Thi);
            }
            Sigma = sigma_for_Tc(Tc);       // Σ consistent with the balanced T_c
        }

        // ℓ(r) = Keplerian ℓ_K.
        const double ell = ellK[i];

        // V(r) from mass conservation:  Ṁ = -2π Σ Δ^½ (V/√(1-V²)) r_g c.
        // Let X ≡ V/√(1-V²) = -Ṁ / (2π Σ Δ^½ r_g c)  (X<0 inflow).
        const double denom = 2.0 * std::numbers::pi * Sigma * sqrtDelta * in.r_g * c_cgs;
        double V = -1e-6;
        if (denom > 0.0) {
            const double X = -in.mdot / denom;                 // V/√(1-V²)
            V = X / std::sqrt(1.0 + X * X);                    // invert (|V|<1)
        }
        if (!(V < 0.0)) V = -1e-6;                             // enforce inflow
        V = std::clamp(V, -kVCap, -1e-12);

        U[4 * i + 0] = Sigma;
        U[4 * i + 1] = V;
        U[4 * i + 2] = ell;
        U[4 * i + 3] = Tc;
    }

    // Radial-smoothness repair (numerical robustness, not a physics change). The
    // per-node energy/Σ bisections above are INDEPENDENT, so a node near r_s where
    // ℓ_K→ℓ_in (P_target→0, the NT zero-torque collapse) can land on a disconnected
    // cold/low-Σ root while its neighbours stay on the warm branch — producing a
    // Σ "cliff" (a single node 100×+ off both neighbours). That cliff makes the
    // node's mass-conservation V huge and wrecks the radial-momentum/regularity
    // FD stencils, stranding the inner relaxation. Replace any such interior
    // outlier by log-interpolating Σ and T_c from its neighbours (the smooth NT
    // branch), then re-deriving V from mass conservation. Pure de-glitching of the
    // seed; the relaxation refines from there.
    {
        auto Vfrom = [&](int i, double Sig_) -> double {
            const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, rg[i]), 0.0));
            const double dn = 2.0 * std::numbers::pi * Sig_ * sqrtD * in.r_g * c_cgs;
            double V = -1e-6;
            if (dn > 0.0) { const double X = -in.mdot / dn; V = X / std::sqrt(1.0 + X * X); }
            if (!(V < 0.0)) V = -1e-6;
            return std::clamp(V, -kVCap, -1e-12);
        };
        for (int i = 1; i < N - 1; ++i) {
            const double Sm = U[4*(i-1)+0], Sc = U[4*i+0], Sp = U[4*(i+1)+0];
            const double lo = std::min(Sm, Sp), hi = std::max(Sm, Sp);
            // Outlier iff Σ_i is >8× outside the [neighbour-min, neighbour-max] band.
            if (Sc > 8.0 * hi || Sc < lo / 8.0) {
                const double Snew = std::sqrt(std::max(Sm, kSigmaFloor) * std::max(Sp, kSigmaFloor));
                const double Tnew = std::sqrt(std::max(U[4*(i-1)+3], kTFloor)
                                            * std::max(U[4*(i+1)+3], kTFloor));
                U[4*i+0] = Snew;
                U[4*i+3] = Tnew;
                U[4*i+1] = Vfrom(i, Snew);
            }
        }
    }

    // Node 0 (= r_s) sonic override: make the seed START at Mach 1 so the
    // regularity 𝒟₀(r_s)=V₀²−c_s²=0 is satisfied from the outset (otherwise the
    // free-boundary relaxation has to discover the sonic transition from a fully
    // subsonic seed, which strands r_s and stalls the radial-momentum block).
    // At fixed T₀ both |V| (mass conservation, ∝1/Σ) and c_s (closure) depend on
    // Σ; |V| decreases and c_s increases with Σ, so |V|²−c_s² is monotone
    // decreasing in Σ — bisect Σ₀ to the Mach-1 crossing.
    {
        const double r0 = rg[0];
        const double sqrtD0 = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r0), 0.0));
        const double Tc0 = U[3];                                   // node-0 T_c from above
        auto mach_excess = [&](double Sig_) -> double {            // V² − c_s²  (geometric, c-units)
            const double dn = 2.0 * std::numbers::pi * Sig_ * sqrtD0 * in.r_g * c_cgs;
            double V_ = -1e-6;
            if (dn > 0.0) { const double X = -in.mdot / dn; V_ = X / std::sqrt(1.0 + X * X); }
            V_ = std::clamp(V_, -kVCap, -1e-12);
            const OneZoneState oz = one_zone_closure(Sig_, Tc0, r0, in, op);
            const double cs2 = kGtilde1 * (oz.P / Sig_) / (c_cgs * c_cgs);  // Γ̃₁(P/Σ)/c²
            return V_ * V_ - cs2;
        };
        double lo = 1e-2, hi = 1e12;
        // mach_excess(lo) > 0 (tiny Σ → |V|→1 ≫ c_s); mach_excess(hi) < 0 (huge Σ).
        if (mach_excess(lo) > 0.0 && mach_excess(hi) < 0.0) {
            for (int b = 0; b < 80; ++b) {
                const double mid = std::sqrt(lo * hi);
                if (mach_excess(mid) > 0.0) lo = mid; else hi = mid;
            }
            const double Sig0 = std::sqrt(lo * hi);
            const double dn = 2.0 * std::numbers::pi * Sig0 * sqrtD0 * in.r_g * c_cgs;
            double V0 = -1e-6;
            if (dn > 0.0) { const double X = -in.mdot / dn; V0 = X / std::sqrt(1.0 + X * X); }
            V0 = std::clamp(V0, -kVCap, -1e-12);
            U[0] = Sig0;
            U[1] = V0;
        }
    }

    // Globals: ℓ_in = ℓ_K(r_isco), r_s = the seeded sonic radius (node 0).
    U[4 * N + 0] = ell_kepler(in.mass, in.spin, r_isco);
    U[4 * N + 1] = r_s;
    return U;
}

// ---------------------------------------------------------------------------
// Radial residual evaluation
// ---------------------------------------------------------------------------
void slim_radial_residual(const std::vector<double>& U, const SlimDiskInputs& in,
                          const OpacityLUTs& op, std::vector<double>& R) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);
    R.assign((size_t)4 * N + 2, 0.0);

    const double ell_in = U[4 * N + 0];
    const double r_s    = U[4 * N + 1];
    const double Mdot   = in.mdot;                              // [g/s]

    // FREE-INNER-NODE grid (Task 5, option B): the grid spans [r_s, r_out] with
    // the CURRENT sonic radius r_s = U[4N+1] as the innermost node, so r[0] == r_s
    // EXACTLY and node 0 IS the sonic point. in.r_in is only a hard floor/guard.
    // Log spacing concentrates nodes near the hot inner disk / sonic point.
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i] = std::exp(lr0 + (lr1 - lr0) * t);
    }

    // Unpack + evaluate every node once.
    std::vector<NodeEval> e(N);
    for (int i = 0; i < N; ++i) {
        e[i] = eval_node(in, op, r[i],
                         U[4 * i + 0], U[4 * i + 1], U[4 * i + 2], U[4 * i + 3]);
    }

    // -----------------------------------------------------------------------
    // Group 1: mass conservation (N algebraic rows).  R = Ṁ_node - Ṁ_target.
    // -----------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        const double mdot_i = mdot_of_node(in, e[i].Sigma, e[i].V, e[i].mech.sqrtDelta);
        R[i] = mdot_i - Mdot;
    }

    // -----------------------------------------------------------------------
    // Group 2: angular momentum (N algebraic rows, §23):
    //   R = (Ṁ/2π)(ℓ_i - ℓ_in) - (A^½ Δ^½ Γ / r)·α·P
    //
    // ℓ convention: stored in GEOMETRIC units (u_φ, dimensionless ~ [M]).
    //   ℓ_geom × r_g × c_cgs → [cm²/s]  (specific angular momentum in CGS).
    // LHS: (Ṁ/2π)(ℓ-ℓ_in): Ṁ [g/s] × [cm²/s] = [g·cm²/s²] = erg.
    // RHS dimensional count: A^½~[M²], Δ^½~[M], /r~[1/M] → geomlen [M²].
    //   geomlen × r_g² → [cm²];  × Γ(1) × α(1) × P[erg/cm²] = erg.
    // Both sides in erg. ✓
    // -----------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        const NodeEval& ei = e[i];
        // LHS: (Ṁ/2π)(ℓ-ℓ_in), ℓ geometric [M] → CGS [cm²/s] via r_g·c_cgs.
        const double dl_cgs = (ei.ell - ell_in) * in.r_g * c_cgs; // [cm²/s]
        const double lhs = (Mdot / (2.0 * std::numbers::pi)) * dl_cgs;  // [g·cm²/s²] = erg
        // RHS: (A^½Δ^½/r) is [M^2] (A^{1/2}~M^2, Delta^{1/2}~M, /r~M); × r_g² → [cm²].
        const double geomlen = ei.mech.sqrtA * ei.mech.sqrtDelta / ei.r;  // [M^2]
        const double rhs = geomlen * in.r_g * in.r_g * ei.Gamma * in.alpha * ei.oz.P;  // [cm^2]*[erg/cm^2] = erg
        R[N + i] = lhs - rhs;
    }

    // -----------------------------------------------------------------------
    // Group 3: radial-momentum transonic ODE (N-1 trapezoidal rows, §23):
    //   dlnV/dlnr = (𝒩₁/𝒟₀)(1-V²)
    //   R = (lnV_{i+1}-lnV_i) - 0.5(lnr_{i+1}-lnr_i)(rhs_i + rhs_{i+1})
    // 𝒩₁ needs Q_adv (geometric).  Build Q_adv per node here (uses FD gradients
    // for the energy group too — but the radial-momentum term needs the geometric
    // scaled (2πr²/(Ṁ η₃))Q_adv, which we assemble inline).
    // -----------------------------------------------------------------------
    // Precompute geometric Q_adv-in-𝒩₁ contribution per node using FD gradients
    // of lnP and lnΣ.  Use one-sided/centred FD on the log grid.
    auto dln = [&](double f_lo, double f_hi, double r_lo, double r_hi) {
        return (std::log(std::max(f_hi, 1e-300)) - std::log(std::max(f_lo, 1e-300)))
             / (std::log(r_hi) - std::log(r_lo));
    };

    // Geometric Ṁ for the (2πr²/(Ṁη₃)) factor: invert the cgs conversion.
    //   Ṁ_geom (the bare -2πΣΔ^½V/√(1-V²) with Σ,length geometric) relates to the
    //   physical Ṁ [g/s] by Ṁ = Ṁ_geom · (Σ_unit · r_g · c).  But the §23 𝒩₁ uses
    //   the SAME geometric Ṁ that appears in its geometric Q_adv, so we keep BOTH
    //   in geometric form and let the (Σ_unit) factor cancel: i.e. evaluate
    //   (2πr²/(Ṁ_g η₃))·Q_adv,g with Q_adv,g = Q_adv,phys/(Σ_unit r_g c) ... this is
    //   fiddly.  Phase-1 expedient: assemble (2πr²/(Ṁ_phys η₃))·Q_adv_phys with BOTH
    //   in CGS, then divide by c² to render dimensionless (matching 𝒜 / 𝒟₀).
    //   Documented: only structural finiteness is required in Phase 1; Tasks 5-8
    //   validate the exact normalization.
    auto qadv_term_geom = [&](int i, int j) -> double {
        // FD gradients between nodes i,j (j=i±1) at node i's values for evaluation.
        const NodeEval& a = e[i];
        const NodeEval& b = e[j];
        const double dlnP = dln(a.oz.P, b.oz.P, a.r, b.r);
        const double dlnS = dln(a.Sigma, b.Sigma, a.r, b.r);
        // Q_adv (S11 Eq 29, CGS) = -(Ṁ/2π r_cm²)(P/Σ)[(Γ₁-1)dlnP - Γ₁ dlnΣ]
        const double r_cm = a.r * in.r_g;
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (a.oz.P / a.Sigma)
                          * ((kGamma1 - 1.0) * dlnP - kGamma1 * dlnS);   // [erg/cm²/s]
        // (2π r²/(Ṁ η₃))·Q_adv, rendered geometric/dimensionless:
        //   2π r_cm²/(Ṁ[g/s] η₃) [s·cm²/g] · Q_adv[erg/cm²/s=g/s³] = [cm²/s²];  /c² → dimensionless
        const double term = (2.0 * std::numbers::pi * r_cm * r_cm / (Mdot * kEta3)) * Qadv;
        return term / (c_cgs * c_cgs);
    };

    auto rhs_radial = [&](int i, int neighbor) -> double {
        const NodeEval& ei = e[i];
        const double D0 = calD0(ei);
        const double D0g = (std::abs(D0) > 1e-30) ? D0 : std::copysign(1e-30, D0 == 0 ? 1.0 : D0);
        const double Qadv_g = qadv_term_geom(i, neighbor);
        const double N1 = calN1(in, ei, Qadv_g);
        return (N1 / D0g) * (1.0 - ei.V * ei.V);
    };

    // L'Hôpital rhs at the sonic node 0 (= r_s). At convergence both 𝒩₁(r_s) and
    // 𝒟₀(r_s) → 0, so (𝒩₁/𝒟₀) is 0/0; the finite transonic slope is the ratio of
    // their radial derivatives. FD across nodes 0,1 on the log grid:
    //   dlnV/dlnr|_0 = (d𝒩₁/dlnr)/(d𝒟₀/dlnr)·(1−V0²).
    // Used ONLY for the node-0 rhs of the [0,1] trapezoidal row; node 1 keeps the
    // direct 𝒩₁/𝒟₀ (𝒟₀(1)<0, non-singular).
    auto rhs_radial_sonic_node0 = [&]() -> double {
        const double Qadv0 = qadv_term_geom(0, 1);
        const double Qadv1 = qadv_term_geom(1, 0);
        const double N1_0 = calN1(in, e[0], Qadv0);
        const double N1_1 = calN1(in, e[1], Qadv1);
        const double D0_0 = calD0(e[0]);
        const double D0_1 = calD0(e[1]);
        const double dlnr = std::log(r[1]) - std::log(r[0]);
        const double dN1 = (N1_1 - N1_0) / dlnr;
        double dD0 = (D0_1 - D0_0) / dlnr;
        // d𝒟₀/dlnr is generically nonzero (𝒟₀: 0 at node 0 → negative at node 1),
        // but floor its magnitude to avoid a divide-by-zero on a transient iterate.
        if (std::abs(dD0) < 1e-30) dD0 = std::copysign(1e-30, dD0 == 0 ? -1.0 : dD0);
        return (dN1 / dD0) * (1.0 - e[0].V * e[0].V);
    };

    for (int i = 0; i < N - 1; ++i) {
        const double lnVi  = std::log(std::max(-e[i].V,   1e-300));   // V<0; use ln|V|
        const double lnVi1 = std::log(std::max(-e[i+1].V, 1e-300));
        const double dlnr  = std::log(r[i+1]) - std::log(r[i]);
        // Node-0 rhs uses the L'Hôpital limit on the sonic interval [0,1]; all
        // other interval endpoints use the direct 𝒩₁/𝒟₀.
        const double rhs_i  = (i == 0) ? rhs_radial_sonic_node0() : rhs_radial(i, i + 1);
        const double rhs_i1 = rhs_radial(i+1, i);
        R[2 * N + i] = (lnVi1 - lnVi) - 0.5 * dlnr * (rhs_i + rhs_i1);
    }

    // -----------------------------------------------------------------------
    // Group 4: energy ODE Q_vis = Q_rad + Q_adv (N-1 trapezoidal rows, §23).
    //   Q_vis = -(Ṁ/2π)(ℓ-ℓ_in)(dΩ/dr)(A^½Δ^½Γ/r³)
    //   Q_rad = 64 σ T_c⁴/(3 κ_R Σ)
    //   Q_adv = -(Ṁ/2π r²)(P/Σ)[(Γ₁-1)dlnP/dlnr - Γ₁ dlnΣ/dlnr]
    // Evaluate the residual at the interval midpoint trapezoidally: R = (G_i+G_{i+1})/2
    // where G = Q_vis - Q_rad - Q_adv, with dΩ/dr, dlnP, dlnΣ as FD across i,i+1.
    // All CGS [erg/cm²/s].
    // -----------------------------------------------------------------------
    auto Gbalance = [&](int i, int j) -> double {
        const NodeEval& a = e[i];
        const NodeEval& b = e[j];
        const double r_cm  = a.r * in.r_g;
        const double r_cm3 = r_cm * r_cm * r_cm;
        // dΩ/dr in CGS: Ω geometric → 1/s via c/r_g; r geometric → cm via r_g.
        const double dOmega_geom = (b.mech.Omega - a.mech.Omega) / (b.r - a.r); // [1/M²]
        const double dOmega_dr = dOmega_geom * (c_cgs / in.r_g) / in.r_g;        // [1/s/cm]
        // Q_vis = -(Ṁ/2π)(ℓ-ℓ_in)(dΩ/dr)(A^½Δ^½Γ/r³)  [erg/cm²/s]
        // Dimensional bookkeeping (all quantities below are in CGS unless noted):
        //   A^½~[M²], Δ^½~[M], r³~[M³]  →  geomfac ≡ A^½Δ^½/r³ is DIMENSIONLESS.
        //   Ṁ [g/s] × dl_cgs [cm²/s] × dOmega_dr [1/(s·cm)] × geomfac [1] × geomfac/r_g
        //   needs an extra [1/cm] to land at erg/cm²/s = g/s³.
        //   That [1/cm] comes from dividing the dimensionless geomfac by r_g [cm]:
        //     [g/s] × [cm²/s] × [1/(s·cm)] × (1/r_g)[1/cm] = g/s³ = erg/cm²/s.  ✓
        // Assembly: use geomfac/r_g as the net geometric factor.
        const double geomfac = a.mech.sqrtA * a.mech.sqrtDelta / (a.r * a.r * a.r); // dimensionless (A^{1/2}~M^2, Delta^{1/2}~M, /r^3~1/M^3)
        // (Ṁ/2π)(ℓ-ℓ_in): ℓ geometric → cm²/s via r_g·c.
        const double dl_cgs = (a.ell - ell_in) * in.r_g * c_cgs;                    // [cm²/s]
        const double Qvis = -(Mdot / (2.0 * std::numbers::pi)) * dl_cgs * dOmega_dr
                          * a.Gamma * (geomfac / in.r_g);  // [g/s]*[cm²/s]*[1/(s·cm)]*[1/cm] = erg/cm²/s
        // Q_rad:
        const double rho_mid = a.oz.rho_mid;
        const double kR = op.lookup_kappa_ross(rho_mid, a.Tc) + op.lookup_kappa_es(rho_mid, a.Tc);
        const double Qrad = 64.0 * sigma_SB * a.Tc * a.Tc * a.Tc * a.Tc
                          / (3.0 * std::max(kR, 1e-300) * a.Sigma);                  // [erg/cm²/s]
        // Q_adv:
        const double dlnP = dln(a.oz.P, b.oz.P, a.r, b.r);
        const double dlnS = dln(a.Sigma, b.Sigma, a.r, b.r);
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (a.oz.P / a.Sigma)
                          * ((kGamma1 - 1.0) * dlnP - kGamma1 * dlnS);               // [erg/cm²/s]
        return Qvis - Qrad - Qadv;
    };

    for (int i = 0; i < N - 1; ++i) {
        const double Gi  = Gbalance(i,   i + 1);
        const double Gi1 = Gbalance(i+1, i);
        R[3 * N - 1 + i] = 0.5 * (Gi + Gi1);
    }

    // -----------------------------------------------------------------------
    // Group 5: outer boundary conditions (2 rows).  The two ODE variables that
    // need an outer IC are ℓ (angular-momentum equation, treated as ODE in relaxation)
    // and T_c (energy ODE):
    //   row 4N-2:  RADIAL-EQUILIBRIUM ℓ via a matched-slope (zero-curvature) BC:
    //                ℓ(r_out) − ℓ_extrap = 0,
    //              where ℓ_extrap is the linear-in-ln r extrapolation of ℓ from the
    //              two inward neighbours (nodes N-2, N-3); equivalently the
    //              d²ℓ/d(ln r)² = 0 condition at the outer node.  See the physics
    //              note below.
    //   row 4N-1:  LOCAL ENERGY BALANCE at the outer node:
    //                Q_vis(r_out) − Q_rad(r_out) − Q_adv(r_out) = 0
    //              i.e. the SAME §23 G-balance the interior energy ODE enforces,
    //              evaluated AT the boundary (FD toward the inward neighbour N-2).
    //              This determines the outer-node T_c CONSISTENTLY with the interior
    //              §23 energy physics, and — being the identical residual form as the
    //              bulk energy rows — the Newton drives it to the floor (a separate
    //              T_c-pinning row anchored to an externally-solved value is an
    //              implicit/moving target that stalls the Newton ~3 decades above floor).
    // (V_out and Σ_out are determined by the algebraic mass & angular-momentum rows;
    //  ℓ_out and T_c,out are the two ODE outer ICs.  Together these yield a
    //  well-posed square system.)
    //
    // WHY NOT the vacuum-Keplerian pin ℓ(r_out)=ℓ_K (the previous BC): a real disk is
    // slightly SUB-Keplerian at the outer edge because the radial PRESSURE GRADIENT
    // helps support it against gravity — an ~(H/r)² effect carried by the §22 radial-
    // momentum balance (V/(1−V²))dV/dr = 𝒜/r − (1/Σ)dP/dr.  The angular-momentum
    // group (Group 2: (Ṁ/2π)(ℓ−ℓ_in)=(A^½Δ^½Γ/r)αP) ALREADY encodes this balance and
    // determines ℓ(r) pointwise INCLUDING the pressure support, so the physical ℓ(r_out)
    // sits ~0.15% BELOW ℓ_K (verified: ℓ(r_out)=7.282 vs ℓ_K=7.293 at a=0,f_Edd=0.02).
    // Pinning ℓ(r_out)=ℓ_K over-constrains the system by exactly that physical offset, so
    // the Newton can never zero the row (it floored bc_ell at ~2.3e-3).  The matched-slope
    // BC instead requires only that ℓ(r_out) be the SMOOTH CONTINUATION of the interior
    // equilibrium profile Group 2 produces — anchoring the level without dictating its
    // (sub-Keplerian) value.  It is satisfiable to the FD floor (verified: the converged
    // ℓ(r_out) equals the linear-in-ln r extrapolation to 1e-5, vs 0.06% for the
    // 𝒜=(r/Σ)dP/dr first-principles balance which drops the inertial term).  At large
    // r_out, (H/r)²→0 ⇒ ℓ→ℓ_K, so this BC reduces to Keplerian asymptotically.
    // -----------------------------------------------------------------------
    const int last = N - 1;
    {
        // Matched-slope outer ℓ BC: quadratic (parabola-in-ln r) extrapolation of ℓ
        // from the THREE inward neighbours (nodes N-2, N-3, N-4), i.e. the
        // d³ℓ/d(ln r)³ = 0 condition.  The disk's ℓ(r) has a small but genuine
        // (negative) curvature at the outer edge as it relaxes toward ℓ_K from below;
        // a purely linear (d²ℓ/dln r²=0) extrapolation leaves a curvature model-error
        // that re-floors the row near ~8e-4, whereas the quadratic fit captures that
        // curvature and drives the row to the FD floor.  Newton's-divided-difference
        // form on the (generally non-uniform, here log-uniform) ln r grid:
        //   ℓ_extrap = ℓ[c] + (x−x_c)·ℓ[c,b] + (x−x_c)(x−x_b)·ℓ[c,b,a]
        // with x=ln r_{N-1}, nodes a=N-4, b=N-3, c=N-2 (c the nearest inward).
        // Use a CUBIC (4-point, Newton-divided-difference) extrapolation in ln r from
        // nodes N-2,N-3,N-4,N-5 (d⁴ℓ/dln r⁴ = 0).  The extra order drops the
        // extrapolation truncation error well below the FD floor (a linear fit left
        // ~8e-4, quadratic ~4e-4; cubic reaches the ~1e-4 band).  Divided differences:
        const double x0 = std::log(r[last - 1]);   // nearest inward (Newton base point)
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
    // §23-consistent outer T_c: local energy balance G(last; N-2)=Q_vis−Q_rad−Q_adv=0.
    R[4 * N - 1] = Gbalance(last, last - 1);

    // -----------------------------------------------------------------------
    // Group 6: sonic-point regularity AT node 0 (= r_s, the free inner node).
    // 𝒟₀(r_s)=0 (Mach 1: V²=c_s²) and 𝒩₁(r_s)=0 (regularity → finite dV/dr).
    // No interpolation: node 0 IS the sonic point, so evaluate directly at e[0].
    // On [r_s, r_out] the subsonic branch has 𝒟₀ ≤ 0 with equality only here.
    // -----------------------------------------------------------------------
    {
        const double Qadv_g0 = qadv_term_geom(0, 1);   // FD across nodes 0,1
        R[4 * N + 0] = calD0(e[0]);
        R[4 * N + 1] = calN1(in, e[0], Qadv_g0);
    }
}

// ---------------------------------------------------------------------------
// Newton relaxation machinery (mirrors src/disk_column_bvp.cpp)
// ---------------------------------------------------------------------------
namespace {

/// Dense central-difference Jacobian J[row*n + col] = ∂R_row/∂U_col of the
/// slim-disk radial residual. Mirrors disk_column_bvp::numerical_jacobian:
/// per-variable absolute step floors keyed to each state-variable TYPE, so that
/// state entries that are ~0 at the seed (e.g. V is small-negative; r_s, ℓ_in
/// are O(1)) do not collapse the finite-difference step to roundoff.
///
/// Variable types in the 4N+2 layout: per node {Σ (off 0), V (off 1),
/// ℓ (off 2), T_c (off 3)}; two globals {ℓ_in (4N), r_s (4N+1)}.
/// NOTE: the r_s column may be slightly noisy when a perturbation moves the
/// sonic-point interpolation bracket; the per-variable floor here plus the
/// line search downstream absorb that.
static void slim_numerical_jacobian(const std::vector<double>& U,
                                    const SlimDiskInputs& in,
                                    const OpacityLUTs& op, std::vector<double>& J) {
    const int n = (int)U.size();
    const int N = std::max(in.n_nodes, 4);
    J.assign((size_t)n * n, 0.0);

    // Per-variable-TYPE step floors: the largest |value| of that variable type
    // across all nodes sets the scale; floor at 1e-7 of it (never below 1e-30).
    double sSig = 0, sV = 0, sEll = 0, sT = 0;
    for (int i = 0; i < N; ++i) {
        sSig = std::max(sSig, std::abs(U[4*i+0])); sV   = std::max(sV,   std::abs(U[4*i+1]));
        sEll = std::max(sEll, std::abs(U[4*i+2])); sT   = std::max(sT,   std::abs(U[4*i+3]));
    }
    const double floorSig = 1e-7 * std::max(sSig, 1e-30);
    const double floorV   = 1e-7 * std::max(sV,   1e-30);
    const double floorEll = 1e-7 * std::max(sEll, 1e-30);
    const double floorT   = 1e-7 * std::max(sT,   1e-30);
    const double floorLin = 1e-7 * std::max(std::abs(U[4*N+0]), 1e-30);   // ℓ_in
    const double floorRs  = 1e-7 * std::max(std::abs(U[4*N+1]), 1e-30);   // r_s

    std::vector<double> Up, Um, Rp, Rm;
    for (int j = 0; j < n; ++j) {
        double absfloor;
        if (j < 4*N) {
            switch (j & 3) { case 0: absfloor = floorSig; break; case 1: absfloor = floorV; break;
                             case 2: absfloor = floorEll; break; default: absfloor = floorT; }
        } else {
            absfloor = (j == 4*N) ? floorLin : floorRs;
        }
        const double delta = std::max(1e-7 * std::abs(U[j]), absfloor);
        Up = U; Um = U;
        Up[j] += delta; Um[j] -= delta;
        slim_radial_residual(Up, in, op, Rp);
        slim_radial_residual(Um, in, op, Rm);
        for (int row = 0; row < n; ++row)
            J[(size_t)row * n + j] = (Rp[row] - Rm[row]) / (2.0 * delta);
    }
}

/// Dense Gaussian elimination with partial pivoting (adapted from
/// disk_column_bvp::dense_solve). Solves A x = b; A is row-major (n×n), modified
/// in place; the solution is returned in b. Returns false if (numerically) singular.
static bool dense_solve(std::vector<double>& A, std::vector<double>& b, int n) {
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

/// Characteristic per-group residual scales, derived from the STATE and INPUTS
/// (never from the residual itself), so the merit genuinely → 0 as R → 0.
struct GroupScales { double mass, ang, rad, ene, bc_ell, bc_T, reg_D0, reg_N1; };
static GroupScales slim_group_scales(const std::vector<double>& U, const SlimDiskInputs& in) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    const double Mdot = std::max(std::abs(in.mdot), 1e-300);
    // Mean |ℓ|, |T_c|, |Σ|, and V² over the nodes (typical magnitudes of the
    // state vars; V² sets the 𝒟₀ = V²−c_s² regularity scale, since V²≈c_s² near
    // the sonic point).
    double mEll = 0, mT = 0, mSig = 0, mV2 = 0;
    for (int i = 0; i < N; ++i) {
        mEll += std::abs(U[4*i+2]); mT += std::abs(U[4*i+3]); mSig += std::abs(U[4*i+0]);
        mV2  += U[4*i+1] * U[4*i+1];
    }
    mEll = std::max(mEll / N, 1e-30); mT = std::max(mT / N, 1.0); mSig = std::max(mSig / N, 1e-30);
    mV2  = std::max(mV2 / N, 1e-300); (void)mV2;
    // Node-0 (= r_s) radial speed² — the natural magnitude of BOTH terms of the
    // 𝒟₀ = V²−c_s² regularity row at the SONIC node (where V₀²≈c_s²). Using the
    // bulk mean(V²) instead would under-resolve 𝒟₀: away from r_s the disk is far
    // subsonic (V²≪c_s²(r_s)), so mean(V²) can be orders of magnitude below the
    // sonic-node scale and the scaled 𝒟₀ residual blows up spuriously.
    const double V0sq = std::max(U[1] * U[1], 1e-300);

    GroupScales s{};
    // mass [g/s]:   row = Ṁ_node - Ṁ ;  scale = Ṁ.
    s.mass = Mdot;
    // angmom [erg]: row LHS = (Ṁ/2π)(ℓ-ℓ_in)·r_g·c ;  scale = (Ṁ/2π)·ℓ̄·r_g·c.
    s.ang  = (Mdot / (2.0 * std::numbers::pi)) * mEll * in.r_g * c_cgs;
    // radmom: dlnV-difference ODE, intrinsically O(1) dimensionless.
    s.rad  = 1.0;
    // energy [erg/cm²/s]: scale by the characteristic VISCOUS dissipation (the
    // heating side, which dominates the Group-4 row) — the Novikov-Thorne flux
    // F_NT = 3GMṀ/(8π r³) evaluated at a representative inner radius. This is the
    // genuine row magnitude (the radiative Q_rad can be Kramers-throttled far below
    // it, so a Q_rad-based scale grossly under-weights the row and stalls the line
    // search). State/input-derived only, so the merit still → 0 as R → 0.
    {
        const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;     // M [g]
        // Representative radius: a few r_g into the disk (inner edge dominates the
        // heating). Use 3× the inner-grid guard (≈ inner disk) in cm.
        const double r_rep_cm = std::max(3.0 * std::max(in.r_in, 1.0), 1.0) * in.r_g;
        s.ene = std::max(3.0 * G_cgs * M_cgs * Mdot
                         / (8.0 * std::numbers::pi * r_rep_cm * r_rep_cm * r_rep_cm), 1e-300);
    }
    // outer BCs: row 4N-2 = ℓ-ℓ_K (scale ℓ̄), row 4N-1 = T_c-T_eff (scale T̄).
    s.bc_ell = mEll;
    s.bc_T   = mT;
    // regularity: the two rows have DIFFERENT magnitudes and need SEPARATE scales.
    //   • 𝒟₀ = V²−c_s²  at the SONIC node ~ V₀²(=c_s²(r_s)) — scale by the node-0
    //     radial speed². (mean(V²) over ALL nodes is dominated by the bulk-subsonic
    //     disk where V²≪c_s²(r_s); using it over-amplifies 𝒟₀ by orders of
    //     magnitude and the inner merit can never reach the floor.)
    //   • 𝒩₁ = 𝒜 + Q_adv-term + pressure-term  ~ O(1) (the gravitational 𝒜 term is
    //     O(1)+) — scale by 1. (Lumping it with the tiny 𝒟₀ scale would over-
    //     weight the 𝒩₁ eigenvalue row and make the merit blow up.)
    s.reg_D0 = V0sq;
    s.reg_N1 = 1.0;
    return s;
}

/// Scale-balanced merit for the slim-disk radial residual (RMS of per-row
/// dimensionless residuals). Each of the six row GROUPS (mass, angmom, radmom,
/// energy, outer BC, regularity) is normalized by a STATE-derived characteristic
/// magnitude (slim_group_scales) — NOT by its own residual norm — so the merit
/// decreases monotonically as the residual shrinks. The groups span ~1e17 (mass,
/// g/s) to O(1) (the dimensionless transonic ODE / regularity rows); without this
/// the line search makes no progress on the small-magnitude rows.
/// Mirrors disk_column_bvp::scaled_residual_norm (variable-magnitude row scaling).
static double slim_scaled_residual_norm(const std::vector<double>& U,
                                        const std::vector<double>& R,
                                        const SlimDiskInputs& in) {
    const int N = std::max(in.n_nodes, 4);
    const GroupScales s = slim_group_scales(U, in);

    double sum = 0.0; int cnt = 0;
    auto accum = [&](int begin, int end, double scale) {
        const double sc = std::max(scale, 1e-300);
        for (int i = begin; i < end; ++i) { double v = R[i]/sc; sum += v*v; ++cnt; }
    };
    accum(0,       N,     s.mass);
    accum(N,       2*N,   s.ang);
    accum(2*N,     3*N-1, s.rad);
    accum(3*N-1,   4*N-2, s.ene);
    accum(4*N-2,   4*N-1, s.bc_ell);   // ℓ(r_out)-ℓ_K
    accum(4*N-1,   4*N,   s.ene);      // outer energy balance Q_vis-Q_rad-Q_adv=0
    accum(4*N,     4*N+1, s.reg_D0);   // 𝒟₀(r_s)=0
    accum(4*N+1,   4*N+2, s.reg_N1);   // 𝒩₁(r_s)=0
    return std::sqrt(sum / (double)std::max(cnt, 1));
}

/// Reduced (INNER-solve) scaled merit: identical to slim_scaled_residual_norm but
/// EXCLUDES the 𝒩₁ regularity row R[4N+1].  In the two-level hybrid (spec §7) the
/// inner Newton holds ℓ_in fixed and does NOT impose 𝒩₁(r_s)=0 — that row is the
/// OUTER bracket root function g(ℓ_in).  𝒩₁ is generically ≠0 until the outer loop
/// converges it, so including it would pin the inner merit above the floor forever.
/// The inner convergence test must measure ONLY the 4N+1 active rows.
static double slim_scaled_residual_norm_active(const std::vector<double>& U,
                                               const std::vector<double>& R,
                                               const SlimDiskInputs& in) {
    const int N = std::max(in.n_nodes, 4);
    const GroupScales s = slim_group_scales(U, in);

    double sum = 0.0; int cnt = 0;
    auto accum = [&](int begin, int end, double scale) {
        const double sc = std::max(scale, 1e-300);
        for (int i = begin; i < end; ++i) { double v = R[i]/sc; sum += v*v; ++cnt; }
    };
    accum(0,       N,     s.mass);
    accum(N,       2*N,   s.ang);
    accum(2*N,     3*N-1, s.rad);
    accum(3*N-1,   4*N-2, s.ene);
    accum(4*N-2,   4*N-1, s.bc_ell);   // ℓ(r_out)-ℓ_K
    accum(4*N-1,   4*N,   s.ene);      // outer energy balance Q_vis-Q_rad-Q_adv=0
    accum(4*N,     4*N+1, s.reg_D0);   // 𝒟₀(r_s)=0  (the only inner regularity row)
    // NOTE: row 4N+1 (𝒩₁) deliberately omitted — it is the OUTER bracket residual.
    return std::sqrt(sum / (double)std::max(cnt, 1));
}

// Per-group SCALED rms magnitudes (R/scale), for the non-convergence diagnostic dump.
struct GroupMags { double mass, ang, rad, ene, bc, reg; };
static GroupMags slim_group_mags(const std::vector<double>& U, const std::vector<double>& R,
                                 const SlimDiskInputs& in) {
    const int N = std::max(in.n_nodes, 4);
    const GroupScales s = slim_group_scales(U, in);
    auto rms = [&](int begin, int end, double sc) {
        double t = 0.0; int c = 0; sc = std::max(sc, 1e-300);
        for (int i = begin; i < end; ++i) { double v = R[i]/sc; t += v*v; ++c; }
        return c ? std::sqrt(t/c) : 0.0;
    };
    return { rms(0,N,s.mass), rms(N,2*N,s.ang), rms(2*N,3*N-1,s.rad),
             rms(3*N-1,4*N-2,s.ene),
             std::max(rms(4*N-2,4*N-1,s.bc_ell), rms(4*N-1,4*N,s.ene)),
             std::max(rms(4*N,4*N+1,s.reg_D0), rms(4*N+1,4*N+2,s.reg_N1)) };
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// INNER solve: fixed-ℓ_in Newton relaxation of the radial structure (spec §7)
// ---------------------------------------------------------------------------
// Two-level hybrid (Sądowski / Muchotrzeb-Czerny): the OUTER loop brackets the
// eigenvalue ℓ_in; this INNER solve relaxes the structure for a FIXED ℓ_in.
//
// relax_structure() Newton-relaxes the 4N+1 unknowns = ALL of U EXCEPT U[4N]=ℓ_in
// (held fixed at the passed value), over the 4N+1 residual rows = all rows EXCEPT
// R[4N+1] (the 𝒩₁ regularity row, which is the OUTER bracket's root function).
// The reduced system is well-posed: with ℓ_in fixed there is no eigenvalue
// degeneracy — r_s is pinned by 𝒟₀(r_s)=0 plus the radial-momentum chain back to
// the outer BC. ALL the Newton machinery (numerical Jacobian, row+column scaling,
// Levenberg-Marquardt, state-derived scaled merit, trust region, damped line
// search, SLIM_DIAG) is preserved — only the variable/row INDEX SETS are reduced
// (skip column 4N, skip row 4N+1) and the merit/convergence test uses the active
// (𝒩₁-excluded) reduced norm. U is in-out (warm-started by the caller).
//
// Returns true iff the reduced merit < floor AND the max relative step < tol.
namespace {

// ---------------------------------------------------------------------------
// Interior Σ-outlier de-glitch (SOURCE fix for the V-collapse Σ-runaway).
// ---------------------------------------------------------------------------
// ROOT CAUSE (confirmed by tools/slim_diag_probe.cpp task C): at the stall a
// handful of interior nodes slide onto the DISCONNECTED high-Σ / low-V branch of
// the mass-conservation hyperbola Σ·(V/√(1-V²))·Δ^½ = const. The pair
//   (V≈-4e-5, Σ≈1e3)   and   (V≈-3e-10, Σ≈2e8)
// both conserve mass EXACTLY (the flux invariant is identical to ~0.1%), so the
// Newton step is free to collapse V→0 (hence Σ→∞ by mass conservation). Once Σ
// blows up the cooling law Q_rad=64σT_c⁴/(3κ_RΣ) is throttled to ~0, the node
// decouples thermally (no T_c balances energy: Q_rad<Q_vis ∀T_c) and parks there,
// flooring the energy group at ~9%. The huge dlnΣ across the resulting cliff
// (probe shows ±850) feeds the Q_adv stencil and keeps the neighbours pinned.
//
// FIX: the physical disk lives on the WARM (gas-pressure-supported) branch where
// Σ varies SMOOTHLY in r. A node whose Σ is >kOutlierFac× outside the band of its
// two neighbours is, by construction, on the wrong branch (a smooth profile cannot
// jump 5 decades between adjacent log-grid nodes). Project it back: log-interpolate
// Σ and T_c from the neighbours (the smooth branch) and RE-DERIVE V from mass
// conservation Ṁ=-2πΣΔ^½(V/√(1-V²))r_g c. This is the SAME proven repair the seed
// builder applies (build_thin_disk_seed ~lines 519-531); here it runs DURING the
// solve (the cliff re-forms in relaxation where the seed-only repair can't reach).
// No magic profile — only neighbour interpolation + the exact mass law. The Newton
// then refines the repaired node on the warm branch.
//
// Returns the number of nodes repaired this pass (0 ⇒ clean).
static int deglitch_sigma_outliers(const SlimDiskInputs& in, std::vector<double>& U) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);
    // Outlier factor: a node whose Σ is more than this × off the LOCAL SMOOTH TREND
    // (the median of a ±kHalf window) is on the disconnected high-Σ / low-V branch.
    // A window median (not the two immediate neighbours) is essential: the runaway
    // can form as a CONNECTED PAIR (probe: nodes 147-148 both blow up together near
    // the outer BC), which a 3-point neighbour test misses because each spiked node
    // sits "between" a normal node and its spiked partner. The window median is
    // robust to a contiguous minority of spiked nodes. 8× matches the seed-builder
    // band; a genuine warm-branch gradient across a few log-grid nodes is well under it.
    constexpr double kOutlierFac = 8.0;
    constexpr int    kHalf       = 3;     // window half-width (median over 2·kHalf+1)

    // Rebuild the grid from r_s = U[4N+1] (same as the residual / unpack).
    const double r_s = U[4*N+1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    auto rofi = [&](int i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        return std::exp(lr0 + (lr1 - lr0) * t);
    };
    auto Vfrom = [&](int i, double Sig_) -> double {
        const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, rofi(i)), 0.0));
        const double dn = 2.0 * std::numbers::pi * Sig_ * sqrtD * in.r_g * c_cgs;
        double V = -1e-6;
        if (dn > 0.0) { const double X = -in.mdot / dn; V = X / std::sqrt(1.0 + X * X); }
        if (!(V < 0.0)) V = -1e-6;
        return std::clamp(V, -kVCap, -1e-12);
    };
    // Local log-Σ median over [i-kHalf, i+kHalf] EXCLUDING node i itself (so a spiked
    // node never inflates its own reference). Window clamped to the grid; the pinned
    // outer node N-1 contributes to its neighbours' windows but is never repaired here.
    auto local_median = [&](int i, int off) -> double {
        std::vector<double> w;
        for (int k = std::max(0, i - kHalf); k <= std::min(N - 1, i + kHalf); ++k) {
            if (k == i) continue;
            w.push_back(std::log(std::max(U[4*k+off], kSigmaFloor)));
        }
        std::sort(w.begin(), w.end());
        const size_t m = w.size();
        return (m == 0) ? std::log(kSigmaFloor)
                        : (m & 1 ? w[m/2] : 0.5 * (w[m/2 - 1] + w[m/2]));
    };

    int nrepaired = 0;
    for (int i = 1; i < N - 1; ++i) {
        const double Sc      = std::max(U[4*i+0], kSigmaFloor);
        const double med_lnS = local_median(i, 0);          // smooth-trend log-Σ
        const double lnfac   = std::log(kOutlierFac);
        if (std::abs(std::log(Sc) - med_lnS) > lnfac) {
            // On the wrong branch: project Σ and T_c back to the local smooth trend,
            // then re-derive V from exact mass conservation. No magic profile.
            const double Snew = std::exp(med_lnS);
            const double Tnew = std::exp(local_median(i, 3));
            U[4*i+0] = std::max(Snew, kSigmaFloor);
            U[4*i+3] = std::max(Tnew, kTFloor);
            U[4*i+1] = Vfrom(i, U[4*i+0]);
            ++nrepaired;
        }
    }
    return nrepaired;
}

// ---------------------------------------------------------------------------
// Physical-validity gate (Task 2): "converged" must mean PHYSICALLY VALID at the
// achievable FD precision, not merely "the scaled merit got small".  Because the
// FD Jacobian limits the merit floor to ~1e-3 (the bc_ell matched-slope row and the
// r_s FD-wandering keep it there), we additionally PROVE the accepted profile is a
// genuine slim-disk solution before declaring convergence.  Each check has a
// PHYSICAL tolerance (not the merit floor): a profile that satisfies the structure
// laws to these tolerances is a valid disk regardless of the residual RMS.
//
// Checks (all must pass):
//   • mass conservation: Ṁ_node is constant in r to kMassTol (relative) — Ṁ truly
//     flux-invariant, no V-collapse / Σ-runaway node survived.
//   • V<0 (inflow) and Σ>0 everywhere — physical inflow, positive density.
//   • sonic regularity 𝒟₀(r_s)≈0 (Mach 1 at the inner node) to kRegTol (scaled).
//     [𝒩₁(r_s)≈0 is the OUTER bracket's job — checked by the FINAL gate, not here.]
//   • r_s < r_isco — sonic point inside the ISCO (slim-disk requirement).
//   • profile monotone/smooth: no residual Σ-cliff (adjacent-node Σ ratio bounded),
//     i.e. the de-glitch left a smooth warm-branch profile.
// `require_N1` adds the 𝒩₁(r_s)≈0 sonic-regularity check (the FINAL, post-outer-
// bracket gate; the inner gate leaves it to the outer loop).
struct ValidityResult {
    bool mass_ok = false, sign_ok = false, reg_D0_ok = false, reg_N1_ok = false,
         rs_ok = false, smooth_ok = false;
    double mass_maxrel = 0.0, D0_scaled = 0.0, N1_scaled = 0.0, sigma_max_jump = 0.0;
    double r_s = 0.0, r_isco = 0.0;
    bool all(bool require_N1) const {
        return mass_ok && sign_ok && reg_D0_ok && rs_ok && smooth_ok
            && (!require_N1 || reg_N1_ok);
    }
};
static ValidityResult slim_validity_gate(const SlimDiskInputs& in,
                                         const OpacityLUTs& opacity,
                                         const std::vector<double>& U, bool require_N1) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);
    ValidityResult v;

    // Physical tolerances. Mass-conservation is checked on Ṁ_node/Ṁ_target so the
    // tolerance is RELATIVE; the regularity rows are checked on the SAME scaled
    // residuals the merit uses (reg_D0). These are PHYSICAL acceptance bands a few ×
    // above the FD floor (the FD Jacobian resolves the structure to ~7e-5; we accept
    // a valid disk at ~1e-3 relative on the conservation laws, ~1e-2 scaled on the
    // single regularity rows whose FD step-wandering is the merit's tightest floor).
    constexpr double kMassTol  = 1e-3;   // |Ṁ_node/Ṁ − 1| everywhere
    constexpr double kRegTol   = 1e-2;   // |𝒟₀|/reg_D0 and |𝒩₁|/reg_N1 (scaled)
    constexpr double kSigJump  = 8.0;    // adjacent-node Σ ratio cap (de-glitch band)

    std::vector<double> R;
    slim_radial_residual(U, in, opacity, R);
    const GroupScales gs = slim_group_scales(U, in);
    const double Mdot = std::max(std::abs(in.mdot), 1e-300);

    // mass conservation (Group 1 rows are Ṁ_node − Ṁ): relative everywhere.
    v.mass_ok = true; v.mass_maxrel = 0.0;
    for (int i = 0; i < N; ++i) {
        const double rel = std::abs(R[i]) / Mdot;
        v.mass_maxrel = std::max(v.mass_maxrel, rel);
        if (!(rel < kMassTol)) v.mass_ok = false;
    }
    // V<0 (inflow) and Σ>0 everywhere.
    v.sign_ok = true;
    for (int i = 0; i < N; ++i) {
        if (!(U[4*i+1] < 0.0) || !(U[4*i+0] > 0.0)) v.sign_ok = false;
    }
    // sonic regularity (scaled residuals of the two regularity rows).
    v.D0_scaled = std::abs(R[4*N+0]) / std::max(gs.reg_D0, 1e-300);
    v.N1_scaled = std::abs(R[4*N+1]) / std::max(gs.reg_N1, 1e-300);
    v.reg_D0_ok = v.D0_scaled < kRegTol;
    v.reg_N1_ok = v.N1_scaled < kRegTol;
    // r_s < r_isco.
    v.r_s = U[4*N+1];
    v.r_isco = isco_prograde(in.mass, in.spin);
    v.rs_ok = (v.r_s > in.r_in) && (v.r_s < v.r_isco);
    // profile smoothness: no residual Σ-cliff (adjacent-node ratio bounded).
    v.smooth_ok = true; v.sigma_max_jump = 1.0;
    for (int i = 1; i < N; ++i) {
        const double s0 = std::max(U[4*(i-1)+0], kSigmaFloor);
        const double s1 = std::max(U[4*i+0],     kSigmaFloor);
        const double ratio = std::max(s0 / s1, s1 / s0);
        v.sigma_max_jump = std::max(v.sigma_max_jump, ratio);
        if (!(ratio < kSigJump)) v.smooth_ok = false;
    }
    return v;
}

static bool relax_structure(const SlimDiskInputs& in, const OpacityLUTs& opacity,
                            double ell_in, std::vector<double>& U) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);
    const int n = 4*N + 2;            // full residual / state length

    // Hold ℓ_in fixed: write it once, never step it.
    U[4*N+0] = ell_in;

    // De-glitch the (possibly warm-started) seed before the first residual eval, so
    // a glitch inherited from a prior trial / rung cannot seed the Newton off-branch.
    deglitch_sigma_outliers(in, U);
    U[4*N+0] = ell_in;

    // Active variable indices (Newton unknowns): all columns EXCEPT 4N (ℓ_in).
    // Active row indices (residual rows):        all rows    EXCEPT 4N+1 (𝒩₁).
    // Both lists have length na = 4N+1; the reduced Newton system is na×na.
    const int na = n - 1;
    std::vector<int> var(na), row(na);
    {
        int p = 0; for (int j = 0; j < n; ++j) if (j != 4*N)   var[p++] = j;   // skip col ℓ_in
        p = 0;     for (int j = 0; j < n; ++j) if (j != 4*N+1) row[p++] = j;   // skip row 𝒩₁
    }

    const bool kDiag = std::getenv("SLIM_DIAG") != nullptr;

    std::vector<double> R, J, rhs, Utry, Rtry;
    slim_radial_residual(U, in, opacity, R);
    double merit = slim_scaled_residual_norm_active(U, R, in);
    double merit_prev = merit;        // for the FD-plateau detector (Task 2)

    if (kDiag) {
        const GroupMags g = slim_group_mags(U, R, in);
        std::printf("[INNER] ell_in=%.5f seed merit=%.3e  mass=%.2e ang=%.2e rad=%.2e ene=%.2e bc=%.2e reg=%.2e | r_s=%.4f g(N1raw)=%.3e\n",
                    ell_in, merit, g.mass, g.ang, g.rad, g.ene, g.bc, g.reg, U[4*N+1], R[4*N+1]);
    }

    bool converged = false;
    int iters = 0; (void)iters;

    // FINITE-DIFFERENCE RESIDUAL FLOOR (Task 2 — honest, validity-gated).
    // The inner Newton uses a CENTRAL-DIFFERENCE Jacobian; that FD Jacobian resolves
    // the Newton direction to only ~7e-5 (the documented FD-Jacobian precision of the
    // bulk Q_adv dlnP/dlnΣ gradients), so the achievable scaled-merit floor is set by
    // FD noise, NOT by the physics.  MEASURED at the easiest corner (a=0,f_Edd≈0.02,
    // matched-slope ℓ BC, 800 iters): the active RMS merit plateaus at ~7.6e-6 with
    // ALL conservation/regularity groups at the FD floor (mass~1e-6, ang~9e-7,
    // rad~6e-7, ene~6e-6, reg~2e-5) EXCEPT the single bc_ell matched-slope row, whose
    // cubic-extrapolation truncation floors it at ~1.7e-4; the r_s FD-wandering also
    // keeps the max relative step near ~2e-3.  Demanding the old 1e-6 from an
    // FD-Jacobian solve asks for more precision than the discretization carries (the
    // solve hits max_iters still crawling and NEVER returns true).
    //
    // We therefore set the floor to a principled small multiple above the measured
    // FD-noise plateau and PAIR it with a physical-validity gate (slim_validity_gate)
    // so "converged" means "physically valid at the achievable precision", not "merit
    // got small".  1e-3 sits ~130× above the ~7.6e-6 plateau and ~6× above the
    // bc_ell single-row floor — tight enough to reject a genuinely unconverged solve,
    // loose enough to accept the FD-limited true solution.  The RIGOROUS route to a
    // tighter tolerance is an ANALYTIC Jacobian (DEFERRED by user decision; not built
    // here).  Acceptance additionally requires the validity gate to pass (below).
    constexpr double kMeritFloor = 1e-3;
    // Step-size floor: the FD-noise wandering of r_s / the near-sonic node keeps the
    // max relative Newton step from ever reaching the ideal in.tol (~1e-6) — it
    // plateaus near ~2e-3.  We keep the IDEAL maxrel<in.tol as a fast early-exit, but
    // also accept when the merit has reached its FD floor AND stopped improving (the
    // method has delivered its precision) AND the validity gate passes.  kStepFloor
    // bounds the "stopped improving" plateau detector below.
    constexpr double kStepFloor    = 5e-3;   // FD-noise step-wandering band
    constexpr double kPlateauRel   = 5e-3;   // per-step merit rel. improvement floor
    // Step cap on the strictly-positive variables (Σ off 0, T_c off 3) so a single
    // Newton step cannot drive them negative (the closure / EOS need Σ,T_c>0).
    constexpr double kStepCap = 0.5;
    // Levenberg-Marquardt damping, adapted by a Nielsen/Marquardt GAIN-RATIO rule
    // (see the solve below): μ rises whenever the realized merit drop is poor
    // relative to the model's prediction (even on an accepted step), and falls
    // only when the step genuinely tracks the local model. lm_nu is the geometric
    // bump factor used on rejection. Bounds: a μ ceiling (bail above it) and a sane
    // floor so an over-Newtonized step can never park μ at ≈0 forever.
    double lm_mu = 1e-3;
    double lm_nu = 2.0;
    constexpr double kMuMax = 1e12;
    constexpr double kMuMin = 1e-9;

    // F = ½‖Rs‖² over the ACTIVE rows (the SAME scaled sum-of-squares the active
    // merit RMS measures), so the gain ratio's act = F_old−F_new is monotonically
    // consistent with the accept/reject test. cnt = number of active rows (4N−1:
    // mass N + ang N + rad N−1 + ene N−1 + bc_ell 1 + bc_T 1 + reg_D0 1). The
    // active merit is sqrt(Σ Rs²/cnt), hence F = ½·cnt·merit².
    const double cnt_active = (double)(4 * N - 1);
    auto merit_to_F = [cnt_active](double m) { return 0.5 * cnt_active * m * m; };

    for (int it = 0; it < in.max_iters; ++it) {
        // 2a) Numerical Jacobian (full n×n) — we gather the ACTIVE submatrix below.
        slim_numerical_jacobian(U, in, opacity, J);

        // 2a') Row + column scaling (non-dimensionalize the reduced Newton system).
        // Same scaling as the original monolithic solver (Dc = per-variable
        // magnitude, Dr = 1/per-group scale), restricted to the active index sets.
        // The raw Jacobian columns span ~33 orders of magnitude; this makes the
        // linear solve numerically well-posed (an exact, physics-neutral rescale).
        std::vector<double> cs(n), rs_inv(n);
        {
            double mSig=0, mV=0, mEll=0, mT=0;
            for (int i = 0; i < N; ++i) {
                mSig=std::max(mSig,std::abs(U[4*i+0])); mV =std::max(mV ,std::abs(U[4*i+1]));
                mEll=std::max(mEll,std::abs(U[4*i+2])); mT =std::max(mT ,std::abs(U[4*i+3]));
            }
            mSig=std::max(mSig,1e-30); mV=std::max(mV,1e-30); mEll=std::max(mEll,1e-30); mT=std::max(mT,1.0);
            for (int i = 0; i < N; ++i) { cs[4*i+0]=mSig; cs[4*i+1]=mV; cs[4*i+2]=mEll; cs[4*i+3]=mT; }
            cs[4*N+0]=std::max(std::abs(U[4*N+0]),1e-30);   // ℓ_in (unused column, kept for indexing)
            cs[4*N+1]=std::max(std::abs(U[4*N+1]),1e-30);   // r_s
            const GroupScales gs = slim_group_scales(U, in);
            auto setrows = [&](int b,int e,double sc){ sc=std::max(sc,1e-300); for(int r=b;r<e;++r) rs_inv[r]=1.0/sc; };
            setrows(0,N,gs.mass); setrows(N,2*N,gs.ang); setrows(2*N,3*N-1,gs.rad);
            setrows(3*N-1,4*N-2,gs.ene); setrows(4*N-2,4*N-1,gs.bc_ell);
            setrows(4*N-1,4*N,gs.ene);   // outer energy-balance BC row
            setrows(4*N,4*N+1,gs.reg_D0); setrows(4*N+1,4*N+2,gs.reg_N1);  // 4N+1 unused (inactive row)
        }
        // Reduced scaled Jacobian Js (na×na) and residual Rs (na) over the active sets:
        //   Js[a][b] = Dr[row[a]] · J[row[a]][var[b]] · Dc[var[b]],  Rs[a] = Dr[row[a]]·R[row[a]]
        std::vector<double> Js((size_t)na*na, 0.0), Rs(na, 0.0);
        for (int a = 0; a < na; ++a) {
            const int ra = row[a];
            Rs[a] = R[ra] * rs_inv[ra];
            for (int b = 0; b < na; ++b) {
                const int vb = var[b];
                Js[(size_t)a*na+b] = J[(size_t)ra*n+vb] * rs_inv[ra] * cs[vb];
            }
        }

        // Levenberg-Marquardt on the scaled normal equations:
        //   (Js^T Js + μ·diag(Js^T Js)) y = -Js^T Rs.
        // LM damping rotates the step toward scaled gradient descent in the near-null
        // (Σ↑,V↓ at fixed Ṁ) subspace while staying Newton-like elsewhere; μ adapts
        // (decrease on success, increase on stall). Physics-neutral regularization.
        std::vector<double> JtJ((size_t)na*na, 0.0), Jtr(na, 0.0);
        for (int i = 0; i < na; ++i) {
            for (int k = 0; k < na; ++k) {
                const double jik = Js[(size_t)k*na+i];   // Js[k][i]
                if (jik == 0.0) continue;
                Jtr[i] += jik * Rs[k];
                for (int j = 0; j < na; ++j) JtJ[(size_t)i*na+j] += jik * Js[(size_t)k*na+j];
            }
        }
        // 2b) Gain-ratio (trust-region) Levenberg-Marquardt step. We RE-SOLVE the
        //     damped normal equations at the CURRENT μ; if the realized merit drop
        //     is poor versus the model prediction (gain ratio ρ≤0) we REJECT, raise
        //     μ geometrically (Nielsen bump), and re-solve WITHOUT advancing U. We
        //     accept only a genuine decrease (ρ>0), then lower μ in proportion to
        //     how well the step tracked the model. This replaces the old "decay on
        //     accept / raise only on line-search stall" heuristic, which ratcheted
        //     μ to its floor (near-pure-Newton) and stalled in a microscopic-λ step.
        const double F_old = merit_to_F(merit);
        std::vector<double> Adamp((size_t)na*na), bdamp(na);
        bool step_taken = false;        // did we ACCEPT a step this iteration?
        bool bail = false;              // hard failure (singular at μ_max)
        double lambda = 1.0;            // line-search scale of the accepted step
        double maxrel = 0.0;
        double merit_try = merit;
        int reject_count = 0;

        while (true) {
            // Solve (JtJ + μ·diag(JtJ)) y = -Jtr at the current μ. If singular even
            // when damped, stiffen μ and re-solve (the inner LM stabilizer).
            bool solved = false;
            for (int tries = 0; tries < 12 && !solved; ++tries) {
                Adamp = JtJ;
                for (int i = 0; i < na; ++i)
                    Adamp[(size_t)i*na+i] += lm_mu * std::max(JtJ[(size_t)i*na+i], 1e-300);
                for (int i = 0; i < na; ++i) bdamp[i] = -Jtr[i];
                if (dense_solve(Adamp, bdamp, na)) { solved = true; break; }
                lm_mu = std::min(lm_mu * 10.0, kMuMax);
                if (lm_mu >= kMuMax) break;
            }
            if (!solved) {
                if (kDiag) std::printf("[INNER] it=%d SINGULAR (LM) at mu=%.1e -> bail\n", it, lm_mu);
                bail = true; break;
            }

            // y = bdamp (scaled step). Predicted reduction of F=½‖Rs‖² for the LM
            // step:  pred = ½ yᵀ(μ·D·y − Jtr),  D=diag(JtJ),  Jtr = Js^T Rs (scaled
            // gradient).  ≥0 for the LM step by construction.
            double pred = 0.0;
            for (int i = 0; i < na; ++i) {
                const double Dii = std::max(JtJ[(size_t)i*na+i], 1e-300);
                pred += lm_mu * Dii * bdamp[i] * bdamp[i] - bdamp[i] * Jtr[i];
            }
            pred *= 0.5;

            // Unscale + scatter into the active variables only: dU[var[b]] = Dc·y.
            rhs.assign(n, 0.0);
            for (int b = 0; b < na; ++b) rhs[var[b]] = bdamp[b] * cs[var[b]];
            rhs[4*N+0] = 0.0;                        // ℓ_in column: never step it
            const std::vector<double>& dU = rhs;

            // Trust-region cap on the strictly-positive vars (Σ, T_c): the largest
            // λ that keeps |Δ/u| ≤ kStepCap. Prefer λ=1; only the feasibility line
            // search below shrinks λ further (just enough to regain physicality).
            double lam = 1.0;
            for (int i = 0; i < N; ++i) {
                for (int c : {0, 3}) {                  // Σ (off 0), T_c (off 3)
                    const double u = U[4*i+c], d = dU[4*i+c];
                    if (u != 0.0 && d != 0.0) {
                        const double frac = std::abs(d / u);
                        if (frac * lam > kStepCap) lam = kStepCap / frac;
                    }
                }
            }

            // Feasibility line search: take the LARGEST λ (≤ the trust-region cap)
            // that yields a PHYSICAL iterate (Σ>0, T_c>0, |V|<1, r_s∈(r_in,r_out)),
            // i.e. reduce λ only to regain feasibility — do NOT shrink it to chase
            // a merit decrease (the gain ratio governs accept/reject instead).
            bool physical = false;
            double F_new = F_old;
            for (int ls = 0; ls < 40; ++ls) {
                Utry.assign(U.begin(), U.end());
                for (int i = 0; i < n; ++i) Utry[i] += lam * dU[i];
                Utry[4*N+0] = ell_in;               // keep ℓ_in pinned exactly
                physical = true;
                for (int i = 0; i < N && physical; ++i) {
                    const double Sig = Utry[4*i+0], Vv = Utry[4*i+1], Tc = Utry[4*i+3];
                    if (Sig <= 0.0 || Tc <= 0.0 || std::abs(Vv) >= 1.0) physical = false;
                }
                if (physical) {
                    const double rs = Utry[4*N+1];
                    if (!(rs > in.r_in && rs < in.r_out)) physical = false;
                }
                if (physical) {
                    slim_radial_residual(Utry, in, opacity, Rtry);
                    merit_try = slim_scaled_residual_norm_active(Utry, Rtry, in);
                    F_new = merit_to_F(merit_try);
                    break;
                }
                lam *= 0.5;
            }

            // Gain ratio ρ = act/pred. A non-physical full step (no feasible λ
            // found) counts as ρ≤0 (reject). pred is clamped away from 0.
            const double act = physical ? (F_old - F_new) : -1.0;
            const double rho = act / std::max(pred, 1e-300);

            if (rho > 0.0) {
                // Genuine decrease -> ACCEPT. Lower μ in proportion to fit quality
                // (Nielsen): μ *= max(1/3, 1 − (2ρ−1)³); reset the bump factor.
                const double t = 2.0 * rho - 1.0;
                lm_mu = std::max(lm_mu * std::max(1.0/3.0, 1.0 - t*t*t), kMuMin);
                lm_nu = 2.0;
                lambda = lam;
                step_taken = true;
                break;
            }
            // No real decrease (incl. an infeasible full step) -> REJECT. Raise μ
            // (μ *= ν; ν *= 2) and re-solve WITHOUT advancing U. Bail honestly if μ
            // exceeds the ceiling with no acceptable step.
            ++reject_count;
            if (lm_mu >= kMuMax) {
                if (kDiag) {
                    const GroupMags g = slim_group_mags(U, R, in);
                    std::printf("[INNER] it=%d GAIN-RATIO STALL merit=%.3e (mu=%.1e maxed, rejects=%d)  "
                                "mass=%.2e ang=%.2e rad=%.2e ene=%.2e bc=%.2e reg=%.2e | r_s=%.4f\n",
                                it, merit, lm_mu, reject_count, g.mass, g.ang, g.rad, g.ene, g.bc, g.reg, U[4*N+1]);
                }
                bail = true; break;
            }
            lm_mu = std::min(lm_mu * lm_nu, kMuMax);
            lm_nu *= 2.0;
        }

        if (bail) break;                            // stuck / singular -> non-converged
        if (!step_taken) break;                     // defensive (shouldn't happen)

        // 2d) Convergence on the (capped) relative step size (active variables).
        // The accepted increment is (Utry - U) = lambda·dU over the active vars.
        maxrel = 0.0;
        for (int b = 0; b < na; ++b) {
            const int j = var[b];
            const double rel = std::abs(Utry[j] - U[j]) / std::max(std::abs(U[j]), 1e-300);
            maxrel = std::max(maxrel, rel);
        }

        U.swap(Utry);
        R.swap(Rtry);
        merit = merit_try;
        iters = it + 1;

        // SOURCE fix / safety net: de-glitch any node that the accepted Newton step
        // pushed onto the disconnected high-Σ / low-V mass-conservation branch (the
        // confirmed cause of the irreducible ~9% energy floor). Project it back to
        // the smooth warm branch (neighbour log-interpolation + exact mass-law V).
        // If anything was repaired, refresh R/merit so the next iteration's Jacobian,
        // line search and convergence test see the repaired (in-basin) state. Pure
        // de-glitching — never touches a node already on the warm branch.
        {
            const int nrep = deglitch_sigma_outliers(in, U);
            if (nrep > 0) {
                U[4*N+0] = ell_in;                 // keep ℓ_in pinned exactly
                slim_radial_residual(U, in, opacity, R);
                merit = slim_scaled_residual_norm_active(U, R, in);
                if (kDiag)
                    std::printf("[INNER] it=%d DEGLITCH repaired %d Σ-outlier node(s) -> merit=%.3e\n",
                                it, nrep, merit);
            }
        }

        if (kDiag) {
            const GroupMags g = slim_group_mags(U, R, in);
            std::printf("[INNER] it=%d lambda=%.2e mu=%.1e merit=%.3e maxrel=%.2e  "
                        "mass=%.2e ang=%.2e rad=%.2e ene=%.2e bc=%.2e reg=%.2e | r_s=%.4f g(N1raw)=%.3e\n",
                        it, lambda, lm_mu, merit, maxrel, g.mass, g.ang, g.rad, g.ene, g.bc, g.reg,
                        U[4*N+1], R[4*N+1]);
        }

        // ------------------------------------------------------------------
        // Convergence test (Task 2): HONEST, validity-gated, FD-floor-aware.
        // Accept iff:
        //   (1) the reduced merit is at/below its FD floor (merit < kMeritFloor),
        //   (2) EITHER the ideal step condition (maxrel < in.tol) holds, OR the merit
        //       has reached its FD plateau — the step is in the FD-noise band
        //       (maxrel < kStepFloor) AND the last step no longer materially improves
        //       the merit ((merit_prev−merit) ≤ kPlateauRel·merit) — i.e. the method
        //       has delivered all the precision the FD Jacobian carries, and
        //   (3) the physical-validity gate passes (V<0, Σ>0, mass conserved, sonic
        //       𝒟₀(r_s)≈0, r_s<r_isco, profile smooth).  𝒩₁(r_s) is the OUTER
        //       bracket's root, so it is NOT gated here (require_N1=false).
        // This makes "converged" mean "physically valid at the achievable FD
        // precision", never "the residual RMS happened to be small".
        const bool merit_floored = (merit < kMeritFloor);
        const bool step_ideal    = (maxrel < in.tol);
        const bool step_plateau  = (maxrel < kStepFloor)
                                && ((merit_prev - merit) <= kPlateauRel * std::max(merit, 1e-300));
        if (merit_floored && (step_ideal || step_plateau)) {
            const ValidityResult v = slim_validity_gate(in, opacity, U, /*require_N1=*/false);
            if (kDiag)
                std::printf("[INNER] it=%d ACCEPT-CHECK merit=%.3e maxrel=%.2e | gate: mass=%d(%.2e) sign=%d D0=%d(%.2e) rs=%d(%.4f<%.4f) smooth=%d(%.2fx) -> %s\n",
                            it, merit, maxrel, (int)v.mass_ok, v.mass_maxrel, (int)v.sign_ok,
                            (int)v.reg_D0_ok, v.D0_scaled, (int)v.rs_ok, v.r_s, v.r_isco,
                            (int)v.smooth_ok, v.sigma_max_jump, v.all(false) ? "VALID" : "INVALID");
            if (v.all(/*require_N1=*/false)) { converged = true; break; }
        }
        merit_prev = merit;
    }

    return converged;
}

// ---------------------------------------------------------------------------
// Unpack a converged state U into the SlimDiskRadial output profile.
// ---------------------------------------------------------------------------
static void unpack_profile(const SlimDiskInputs& in, const OpacityLUTs& opacity,
                           const std::vector<double>& U, SlimDiskRadial& out) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);

    // Rebuild the free-inner-node grid from the converged r_s = U[4N+1] (node 0 =
    // sonic point), spanning [r_s, r_out].
    std::vector<double> rgrid(N);
    {
        const double lr0u = std::log(U[4*N+1]), lr1u = std::log(in.r_out);
        for (int i = 0; i < N; ++i) {
            const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
            rgrid[i] = std::exp(lr0u + (lr1u - lr0u) * t);
        }
    }
    out.r.resize(N); out.Sigma.resize(N); out.V.resize(N); out.Omega.resize(N);
    out.Tc.resize(N); out.H.resize(N); out.f_adv.resize(N);
    const double ell_in = U[4*N+0];
    const double Mdot   = in.mdot;
    auto dln = [&](double f_lo, double f_hi, double r_lo, double r_hi) {
        return (std::log(std::max(f_hi, 1e-300)) - std::log(std::max(f_lo, 1e-300)))
             / (std::log(r_hi) - std::log(r_lo));
    };
    for (int i = 0; i < N; ++i) {
        const double r   = rgrid[i];
        const double Sig = U[4*i+0], V = U[4*i+1], ell = U[4*i+2], Tc = U[4*i+3];
        const OneZoneState oz = one_zone_closure(std::max(Sig, kSigmaFloor),
                                                 std::max(Tc, kTFloor), r, in, opacity);
        out.r[i]     = r;
        out.Sigma[i] = Sig;
        out.V[i]     = V;
        out.Omega[i] = omega_from_ell(in.mass, in.spin, r, ell) * (c_cgs / in.r_g);
        out.Tc[i]    = Tc;
        out.H[i]     = oz.H;
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const OneZoneState ozj = one_zone_closure(std::max(U[4*j+0], kSigmaFloor),
                                                  std::max(U[4*j+3], kTFloor), rgrid[j], in, opacity);
        const double dlnP = dln(oz.P, ozj.P, r, rgrid[j]);
        const double dlnS = dln(Sig,  U[4*j+0], r, rgrid[j]);
        const double r_cm = r * in.r_g;
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (oz.P / std::max(Sig, kSigmaFloor))
                          * ((kGamma1 - 1.0) * dlnP - kGamma1 * dlnS);    // [erg/cm²/s]
        const double rho_mid = oz.rho_mid;
        const double kR = opacity.lookup_kappa_ross(rho_mid, Tc) + opacity.lookup_kappa_es(rho_mid, Tc);
        const double Qrad = 64.0 * sigma_SB * Tc*Tc*Tc*Tc
                          / (3.0 * std::max(kR, 1e-300) * std::max(Sig, kSigmaFloor));  // [erg/cm²/s]
        out.f_adv[i] = Qadv / std::max(std::abs(Qrad), 1e-300);
    }
    out.ell_in  = ell_in;
    out.r_sonic = U[4*N+1];
}

// ---------------------------------------------------------------------------
// OUTER bracket: find ℓ_in such that g(ℓ_in) = 𝒩₁(r_s; ℓ_in) = R[4N+1] = 0
// ---------------------------------------------------------------------------
// For a trial ℓ_in, relax_structure converges the inner BVP; the FULL residual's
// 𝒩₁ regularity row (scaled by reg_N1) is the outer root function g(ℓ_in). We
// scan a physical window around ℓ_K(r_isco), find a sign change in g, then bisect.
// A failed inner solve is treated as one topology side (flow can't reach a regular
// sonic point) and the bracket steps away from it. Warm-starts U across trials.
//
// On success: leaves the converged state in U (with U[4N]=ℓ_in, U[4N+1]=r_s) and
// returns true. On failure to bracket at all: returns false (honest fallback).
static bool solve_outer_bracket(const SlimDiskInputs& in, const OpacityLUTs& opacity,
                                std::vector<double>& U) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);
    const bool kDiag = std::getenv("SLIM_DIAG") != nullptr;

    const double r_isco = isco_prograde(in.mass, in.spin);
    const double ellK_isco = ell_kepler(in.mass, in.spin, r_isco);

    // Outer-bracket tolerance on the scaled 𝒩₁(r_s) regularity root.
    constexpr double kGtol = 1e-4;       // |g| (already scaled by reg_N1)
    constexpr int    kMaxBisect = 40;

    // Scaled outer-root function: g(ℓ_in) = R[4N+1]/reg_N1 after the inner converges.
    // Returns {ok, g, U_at_trial}. ok=false => inner did not converge for this ℓ_in.
    std::vector<double> Ubase = U;     // warm-start template for each trial
    auto eval_g = [&](double ell_in, std::vector<double>& Uwork, double& g) -> bool {
        Uwork = Ubase;
        const bool ok = relax_structure(in, opacity, ell_in, Uwork);
        if (!ok) { g = std::nan(""); return false; }
        std::vector<double> Rfull;
        slim_radial_residual(Uwork, in, opacity, Rfull);
        const GroupScales gs = slim_group_scales(Uwork, in);
        g = Rfull[4*N+1] / std::max(gs.reg_N1, 1e-300);
        return true;
    };

    // --- Scan a window [lo_frac, hi_frac]·ℓ_K(r_isco) for a sign change in g. ---
    // ℓ_in lies near and slightly below ℓ_K(r_isco) physically. Start tight, widen
    // the floor toward 0.5· (and the ceiling slightly above 1.0·) if no bracket.
    struct Sample { double ell, g; std::vector<double> U; bool ok; };

    // direct_accept: set true (with U_accept/ell_accept) if a scan sample converges
    // with the regularity root |g| already below kGtol — then that ℓ_in IS the
    // eigenvalue and no bracketing/bisection is needed.  This is the physically-thin
    // (≈Novikov-Thorne) corner: the seed sits at ℓ_in≈ℓ_K(r_isco), r_s≈r_isco, and
    // the inner solve already meets 𝒩₁(r_s)≈0 to the FD floor.  Honest: g IS the
    // 𝒩₁(r_s) regularity residual, so |g|<kGtol means regularity is satisfied.
    bool direct_accept = false;
    double ell_accept = 0.0;
    std::vector<double> U_accept;

    auto scan = [&](double lo_frac, double hi_frac, int nsamp,
                    double& ell_a, double& g_a, std::vector<double>& Ua,
                    double& ell_b, double& g_b, std::vector<double>& Ub) -> bool {
        std::vector<Sample> S;
        S.reserve(nsamp);
        for (int k = 0; k < nsamp; ++k) {
            const double f = lo_frac + (hi_frac - lo_frac) * double(k) / double(nsamp - 1);
            const double ell = f * ellK_isco;
            Sample s; s.ell = ell;
            std::vector<double> Uw;
            s.ok = eval_g(ell, Uw, s.g);
            s.U.swap(Uw);
            if (kDiag)
                std::printf("[OUTER]   scan ell_in=%.5f (%.3f·ellK_isco) inner_ok=%d g=%.4e\n",
                            ell, f, (int)s.ok, s.g);
            // Direct accept: a converged sample whose regularity root is already at
            // the floor IS the eigenvalue.
            if (s.ok && std::isfinite(s.g) && std::abs(s.g) < kGtol) {
                direct_accept = true; ell_accept = s.ell; U_accept = s.U;
                if (kDiag)
                    std::printf("[OUTER]   DIRECT-ACCEPT ell_in=%.5f g=%.4e (<%.1e) — 𝒩₁(r_s) at floor\n",
                                s.ell, s.g, kGtol);
                return false;   // stop scanning; caller handles direct_accept
            }
            // Warm-start the next trial from a converged neighbour to stay in-basin.
            if (s.ok) Ubase = s.U;
            S.push_back(std::move(s));
        }
        // Find an adjacent pair of CONVERGED samples that straddle g=0.
        for (size_t k = 1; k < S.size(); ++k) {
            if (S[k-1].ok && S[k].ok && S[k-1].g * S[k].g <= 0.0
                && std::isfinite(S[k-1].g) && std::isfinite(S[k].g)) {
                ell_a = S[k-1].ell; g_a = S[k-1].g; Ua = S[k-1].U;
                ell_b = S[k].ell;   g_b = S[k].g;   Ub = S[k].U;
                return true;
            }
        }
        return false;
    };

    double ell_a=0, g_a=0, ell_b=0, g_b=0;
    std::vector<double> Ua, Ub;
    bool bracketed = false;
    // Window ladder: tight first, then progressively wider. (lo_frac, hi_frac, nsamp)
    const double windows[][3] = {
        {0.80, 1.00, 7},
        {0.60, 1.05, 10},
        {0.40, 1.10, 13},
    };
    for (const auto& w : windows) {
        if (kDiag)
            std::printf("[OUTER] scan window [%.2f, %.2f]·ellK_isco (ellK_isco=%.5f), %d samples\n",
                        w[0], w[1], ellK_isco, (int)w[2]);
        if (scan(w[0], w[1], (int)w[2], ell_a, g_a, Ua, ell_b, g_b, Ub)) { bracketed = true; break; }
        if (direct_accept) break;     // a sample already met |g|<kGtol
    }
    if (direct_accept) {
        U.swap(U_accept);
        U[4*N+0] = ell_accept;
        if (kDiag) std::printf("[OUTER] CONVERGED (direct) ell_in=%.6f — 𝒩₁(r_s) already at floor\n",
                               ell_accept);
        return true;
    }
    if (!bracketed) {
        if (kDiag) std::printf("[OUTER] NO sign change in g(ell_in) across all windows -> fallback\n");
        return false;
    }
    if (kDiag)
        std::printf("[OUTER] bracketed: g(%.5f)=%.4e , g(%.5f)=%.4e -> bisect\n",
                    ell_a, g_a, ell_b, g_b);

    // --- Bisect g(ℓ_in) to a tolerance on g (scaled) or on the ℓ_in interval. ---
    // Warm-start each inner solve from the bracket endpoint whose U is nearest.
    std::vector<double> Umid;
    int nbis = 0;
    for (; nbis < kMaxBisect; ++nbis) {
        const double ell_m = 0.5 * (ell_a + ell_b);
        // Warm-start from whichever endpoint is closer (keeps the inner in-basin).
        Ubase = (std::abs(ell_m - ell_a) < std::abs(ell_m - ell_b)) ? Ua : Ub;
        double g_m = 0.0;
        const bool ok = eval_g(ell_m, Umid, g_m);
        if (kDiag)
            std::printf("[OUTER] bisect %d: ell_m=%.6f inner_ok=%d g=%.4e  [%.6f,%.6f]\n",
                        nbis, ell_m, (int)ok, g_m, ell_a, ell_b);
        if (!ok) {
            // Inner failed at the midpoint: that side has no regular sonic point.
            // Collapse the bracket toward the converged endpoint with smaller |g|.
            if (std::abs(g_a) <= std::abs(g_b)) { ell_b = ell_m; }
            else                                { ell_a = ell_m; }
            continue;
        }
        if (std::abs(g_m) < kGtol) {
            U.swap(Umid);
            U[4*N+0] = ell_m;
            if (kDiag) std::printf("[OUTER] CONVERGED ell_in=%.6f g=%.4e (%d bisections)\n",
                                   ell_m, g_m, nbis + 1);
            return true;
        }
        if (g_a * g_m <= 0.0) { ell_b = ell_m; g_b = g_m; Ub = Umid; }
        else                  { ell_a = ell_m; g_a = g_m; Ua = Umid; }
        if (std::abs(ell_b - ell_a) < 1e-9 * std::max(std::abs(ellK_isco), 1e-30)) {
            // Interval collapsed: accept the better endpoint as the eigenvalue.
            const double ell_acc = (std::abs(g_a) <= std::abs(g_b)) ? ell_a : ell_b;
            Ubase = (std::abs(g_a) <= std::abs(g_b)) ? Ua : Ub;
            if (eval_g(ell_acc, Umid, g_m)) {
                U.swap(Umid); U[4*N+0] = ell_acc;
                if (kDiag) std::printf("[OUTER] interval collapsed -> ell_in=%.6f g=%.4e (%d bisections)\n",
                                       ell_acc, g_m, nbis + 1);
                return true;
            }
            break;
        }
    }
    if (kDiag) std::printf("[OUTER] bisection did not reach g-tol in %d steps -> fallback\n", nbis);
    return false;
}
} // anonymous namespace

// ---------------------------------------------------------------------------
// Transonic radial solver: Ṁ-continuation driver wrapping the outer ℓ_in bracket
// (spec §7 REVISED 2026-06-09 — two-level hybrid)
// ---------------------------------------------------------------------------
// The Ṁ ladder wraps the OUTER ℓ_in bracket (which wraps the INNER fixed-ℓ_in
// relaxation). At low Ṁ the disk is thin (sonic≈ISCO, ℓ_in≈ℓ_K(ISCO)) so the first
// bracket is easy; each successive (geometric) rung warm-starts the inner U and the
// ℓ_in search window from the last converged result. Honest empty fallback if any
// rung's bracket fails (never fabricates a profile).
SlimDiskRadial solve_slim_disk_radial(const SlimDiskInputs& in, const OpacityLUTs& opacity) {
    using namespace constants;
    const bool kDiag = std::getenv("SLIM_DIAG") != nullptr;

    // Eddington accretion rate (textbook η=0.1 convention, trap #12):
    //   L_Edd = 4πG M m_p c / σ_T   (≡ 4πGM c / κ_es with κ_es = σ_T/m_p),
    //   Ṁ_Edd = 10 L_Edd / c² = 10·(4πGM m_p)/(σ_T c).
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;   // M [g] (r_g=GM/c²)
    const double kappa_es = 0.34;                                    // cm²/g (Thomson, X≈0.7)
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es; // erg/s
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);          // g/s
    const double Mdot_lo  = 0.02 * Mdot_Edd;

    // Build a geometric Ṁ ladder from Mdot_lo up to in.mdot (≤2× per rung, ≤12).
    // If already thin (≤ the low rung), the single rung IS in.mdot.
    std::vector<double> rungs;
    if (!(in.mdot > Mdot_lo)) {
        rungs.push_back(in.mdot);
    } else {
        rungs.push_back(Mdot_lo);
        while (rungs.back() * 2.0 < in.mdot && (int)rungs.size() < 11)
            rungs.push_back(rungs.back() * 2.0);
        rungs.push_back(in.mdot);   // final rung = the requested target
    }

    SlimDiskRadial out;
    std::vector<double> U;          // warm state, carried across rungs
    bool have_warm = false;

    for (size_t k = 0; k < rungs.size(); ++k) {
        SlimDiskInputs in_rung = in;
        in_rung.mdot = rungs[k];
        if (kDiag)
            std::printf("[SLIM] === Mdot rung %zu/%zu: Mdot=%.3e (Mdot_Edd=%.3e, f_Edd=%.3f) ===\n",
                        k + 1, rungs.size(), rungs[k], Mdot_Edd, rungs[k] / Mdot_Edd);
        // Warm-start U from the previous rung, or build the clean thin-disk seed.
        if (!have_warm) U = build_thin_disk_seed(in_rung, opacity);

        const bool ok = solve_outer_bracket(in_rung, opacity, U);
        if (!ok) {
            if (kDiag)
                std::printf("[SLIM] rung %zu (Mdot=%.3e) bracket FAILED -> honest fallback\n",
                            k + 1, rungs[k]);
            return SlimDiskRadial{};   // honest fallback (empty, converged=false)
        }
        have_warm = true;
    }

    // Final rung converged: PHYSICAL-VALIDITY GATE on the accepted profile (Task 2).
    // The outer bracket has driven 𝒩₁(r_s)→0 (the eigenvalue), so here we apply the
    // FULL gate INCLUDING the 𝒩₁(r_s)≈0 sonic-regularity check: mass conserved,
    // V<0 & Σ>0, BOTH 𝒟₀(r_s)≈0 AND 𝒩₁(r_s)≈0, r_s<r_isco, profile smooth.  Only if
    // the gate passes do we accept; otherwise honest fallback (no fabricated profile).
    // This makes the returned converged=true mean "a physically valid slim disk at
    // the achievable FD precision", not merely "the bracket closed".
    {
        SlimDiskInputs in_final = in;   // in.mdot is the target
        const ValidityResult v = slim_validity_gate(in_final, opacity, U, /*require_N1=*/true);
        if (kDiag) {
            std::printf("[SLIM] FINAL validity gate: mass=%d(%.2e) sign=%d D0=%d(%.2e) N1=%d(%.2e) "
                        "rs=%d(%.4f<%.4f) smooth=%d(%.2fx) -> %s\n",
                        (int)v.mass_ok, v.mass_maxrel, (int)v.sign_ok,
                        (int)v.reg_D0_ok, v.D0_scaled, (int)v.reg_N1_ok, v.N1_scaled,
                        (int)v.rs_ok, v.r_s, v.r_isco, (int)v.smooth_ok, v.sigma_max_jump,
                        v.all(true) ? "VALID" : "INVALID");
        }
        if (!v.all(/*require_N1=*/true)) {
            if (kDiag) std::printf("[SLIM] FINAL gate FAILED -> honest fallback (no fabricated profile)\n");
            return SlimDiskRadial{};   // honest fallback (empty, converged=false)
        }

        unpack_profile(in_final, opacity, U, out);
        out.converged = true;
        out.iters = (int)rungs.size();
        std::vector<double> Rf;
        slim_radial_residual(U, in_final, opacity, Rf);
        out.final_residual = slim_scaled_residual_norm(U, Rf, in_final);
        if (kDiag) {
            const int N = std::max(in.n_nodes, 4);
            const GroupScales gs = slim_group_scales(U, in_final);
            std::printf("[SLIM] FINAL converged: ell_in=%.6f r_sonic=%.5f final_residual=%.3e | "
                        "D0(r_s)=%.3e (scaled %.3e)  N1(r_s)=%.3e (scaled %.3e)\n",
                        out.ell_in, out.r_sonic, out.final_residual,
                        Rf[4*N+0], Rf[4*N+0]/std::max(gs.reg_D0,1e-300),
                        Rf[4*N+1], Rf[4*N+1]/std::max(gs.reg_N1,1e-300));
        }
    }
    return out;
}
} // namespace grrt
