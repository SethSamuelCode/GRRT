#include "grrt/scene/slim_disk_radial.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>
#include <numbers>
#include <cstdlib>
#include <cstdio>
#include <chrono>

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
// Analytic derivatives of the one-zone closure (Task 2).
// ---------------------------------------------------------------------------
// Differentiate one_zone_closure analytically.  The closure solves a COUPLED 2×2
// system in (H, μ) at fixed (Σ, T_c, r):
//   F1 (H-quadratic):  W H² − b H − g = 0
//   F2 (μ fixed point): μ − μ̂(ρ, T) = 0 ,  ρ = Σ/(2H)
// with notation matching one_zone_closure / solve_H:
//   W  ≡ Ω_⊥²_cgs (const in Σ,T)        g  ≡ c_s_gas² = k_B T/(μ m_p)
//   b  = (2 a_rad/3)·T⁴/Σ               s  = √(b²+4Wg) = 2WH − b
//   p_gas = ρ g,  p_rad = (a_rad/3)T⁴,  p_mid = p_gas+p_rad
//   c_s = √W·H,   P = 2 p_mid H,        S = cv(1.5 lnT − lnρ) + (4a_rad/3)T³/ρ,  cv=k_B/(μm_p)
// We differentiate F1,F2 by the implicit-function theorem so dμ/d{Σ,T} (through the
// fixed point) is captured — frozen-μ is exact only in the fully-ionized inner disk
// and fails the FD cross-check in the partial-ionization regime.  μ̂'s log-gradients
// come from op.mu_with_grad.  d{·}[0]=∂/∂Σ, d{·}[1]=∂/∂T_c.  Floors mirror the
// closure exactly so the derivative matches the value the residual uses.
void one_zone_closure_jac(double Sigma, double Tc, double r,
                          const SlimDiskInputs& in, const OpacityLUTs& op,
                          OneZoneState& st, OneZoneJac& jac) {
    using namespace constants;
    jac = OneZoneJac{};
    st = one_zone_closure(Sigma, Tc, r, in, op);
    if (!(st.H > 0.0)) return;   // unsolvable region (Ω_⊥²≤0): closure returned {}; partials 0.

    constexpr double SIGMA_FLOOR = 1e-30;
    constexpr double RHO_FLOOR   = 1e-30;
    constexpr double T_FLOOR     = 1.0;
    const double Sigma_s = std::max(Sigma, SIGMA_FLOOR);
    const double Tc_s    = std::max(Tc, T_FLOOR);
    // Floor clamp: where an input is clamped its value is constant ⇒ that partial is
    // 0 (matches the FD central difference, which sees zero change inside the clamp).
    const double dSig = (Sigma > SIGMA_FLOOR) ? 1.0 : 0.0;
    const double dTc  = (Tc    > T_FLOOR)     ? 1.0 : 0.0;

    const double conv = c_cgs / in.r_g;
    const double W = omega_perp2(in.mass, in.spin, r) * conv * conv;   // Ω_⊥²_cgs
    const double mu = st.mu;
    const double H = st.H, rho = st.rho_mid;

    const double g = k_B * Tc_s / (mu * m_p);                          // c_s_gas²
    const double b = 2.0 * a_rad * Tc_s*Tc_s*Tc_s*Tc_s / (3.0 * Sigma_s);
    const double s = std::max(2.0 * W * H - b, 1e-300);                // = √disc = ∂(WH²−bH−g)/∂H

    // ---- μ derivative: FINITE-DIFFERENCE the CLOSURE's converged μ ----
    // μ(Σ,T) is a tabulated, non-closed-form quantity (bilinear LUT under a ≤3-iter
    // fixed point), so its derivative is obtained by central-differencing the
    // closure's OWN converged μ.  This makes the analytic μ-response BIT-CONSISTENT
    // with the central-difference oracle (which re-solves the same closure), so the
    // gas-pressure terms match to round-off instead of the ~2e-4 LUT-slope gap a
    // separate mu_with_grad stencil leaves.  Everything else stays exact-analytic.
    // In the fully-ionized inner disk μ≈const so these are ~0 (frozen-μ recovered).
    double dmu[2] = {0.0, 0.0};
    {
        const double hS = 1e-6 * Sigma_s, hT = 1e-6 * Tc_s;
        if (Sigma > SIGMA_FLOOR) {
            const double mp = one_zone_closure(Sigma + hS, Tc, r, in, op).mu;
            const double mm = one_zone_closure(Sigma - hS, Tc, r, in, op).mu;
            dmu[0] = (mp - mm) / (2.0 * hS);
        }
        if (Tc > T_FLOOR) {
            const double mp = one_zone_closure(Sigma, Tc + hT, r, in, op).mu;
            const double mm = one_zone_closure(Sigma, Tc - hT, r, in, op).mu;
            dmu[1] = (mp - mm) / (2.0 * hT);
        }
    }

    // dH from the H-quadratic WH²−bH−g=0 with μ=μ(p) known (total derivative):
    //   (2WH−b)dH − H db − dg = 0  ⇒  dH = (H db + dg)/s.
    //   db/dΣ = −b/Σ ; db/dT = 4b/T ; g = k_B T/(μ m_p):
    //   dg/dΣ = −(g/μ)dμ_Σ ;  dg/dT = g/T − (g/μ)dμ_T.
    const double db_dS = (-b / Sigma_s) * dSig;
    const double db_dT = (4.0 * b / Tc_s) * dTc;
    const double dg_dS = -(g / mu) * dmu[0];
    const double dg_dT = (g / Tc_s) * dTc - (g / mu) * dmu[1];
    double dH[2];
    dH[0] = (H * db_dS + dg_dS) / s;
    dH[1] = (H * db_dT + dg_dT) / s;
    jac.dH[0] = dH[0]; jac.dH[1] = dH[1];

    // ρ = Σ/(2H):  ∂ρ/∂Σ = ρ/Σ − (ρ/H)∂H/∂Σ ;  ∂ρ/∂T = −(ρ/H)∂H/∂T.
    jac.drho[0] = (rho / Sigma_s) * dSig - (rho / H) * dH[0];
    jac.drho[1] =                        - (rho / H) * dH[1];

    // p_gas = ρ g  (dg_dS, dg_dT computed above with the closure-consistent dμ).
    jac.dp_gas[0] = jac.drho[0] * g + rho * dg_dS;
    jac.dp_gas[1] = jac.drho[1] * g + rho * dg_dT;

    // p_rad = (a_rad/3)T⁴ → ∂/∂Σ=0, ∂/∂T = 4 p_rad/T.
    jac.dp_rad[0] = 0.0;
    jac.dp_rad[1] = (4.0 * st.p_rad / Tc_s) * dTc;

    jac.dp_mid[0] = jac.dp_gas[0] + jac.dp_rad[0];
    jac.dp_mid[1] = jac.dp_gas[1] + jac.dp_rad[1];

    // c_s = √W H.
    const double sqrtW = std::sqrt(W);
    jac.dc_s[0] = sqrtW * dH[0];
    jac.dc_s[1] = sqrtW * dH[1];

    // P = 2 p_mid H.
    jac.dP[0] = 2.0 * (jac.dp_mid[0] * H + st.p_mid * dH[0]);
    jac.dP[1] = 2.0 * (jac.dp_mid[1] * H + st.p_mid * dH[1]);

    // S = cv(1.5 lnT − lnρ_e) + (4 a_rad/3) T³/ρ_e,  cv = k_B/(μ m_p),  ρ_e=max(ρ,floor).
    // cv depends on μ: ∂cv/∂p = −(cv/μ)∂μ/∂p.
    const double cv = k_B / (mu * m_p);
    const double dcv_dS = -(cv / mu) * dmu[0];
    const double dcv_dT = -(cv / mu) * dmu[1];
    const double rho_e = std::max(rho, RHO_FLOOR);
    const double drho_e_dS = (rho > RHO_FLOOR) ? jac.drho[0] : 0.0;
    const double drho_e_dT = (rho > RHO_FLOOR) ? jac.drho[1] : 0.0;
    const double T3 = Tc_s*Tc_s*Tc_s;
    const double lnpart = 1.5 * std::log(Tc_s) - std::log(rho_e);      // S_gas = cv·lnpart
    // S_gas = cv·(1.5 lnT − ln ρ_e):
    const double dSgas_dS = dcv_dS * lnpart + cv * (-drho_e_dS / rho_e);
    const double dSgas_dT = dcv_dT * lnpart + cv * (1.5 / Tc_s * dTc - drho_e_dT / rho_e);
    // S_rad = (4 a_rad/3) T³/ρ_e:
    const double k_rad = 4.0 * a_rad / 3.0;
    const double dSrad_dS = k_rad * (-T3 / (rho_e * rho_e)) * drho_e_dS;
    const double dSrad_dT = k_rad * (3.0 * Tc_s*Tc_s / rho_e * dTc
                                     - T3 / (rho_e * rho_e) * drho_e_dT);
    jac.dS[0] = dSgas_dS + dSrad_dS;
    jac.dS[1] = dSgas_dT + dSrad_dT;
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

// ∂Ω/∂ℓ at the orbit (r, Ω): the reciprocal of dℓ/dΩ evaluated at Ω (exact inverse
// of omega_from_ell).  dℓ/dΩ = g_φφ/√D + numer²/D^{3/2}, numer = g_tφ + g_φφ Ω,
// D = −(g_tt + 2 g_tφ Ω + g_φφ Ω²).  Used by the analytic 𝒜/𝒩₁ ℓ-derivatives.
double domega_dell(double M, double a, double r, double Om) {
    double g_tt, g_tphi, g_phph;
    eq_metric(M, a, r, g_tt, g_tphi, g_phph);
    const double D = -(g_tt + 2.0 * g_tphi * Om + g_phph * Om * Om);
    const double Ds = std::max(D, 1e-300);
    const double sqrtD = std::sqrt(Ds);
    const double numer = g_tphi + g_phph * Om;
    const double dell_dOm = g_phph / sqrtD + numer * numer / (Ds * sqrtD);
    return (std::abs(dell_dOm) > 1e-300) ? 1.0 / dell_dOm : 0.0;
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
using slim_detail::domega_dell;
using slim_detail::isco_prograde;

// Reference pure-gas adiabatic indices (β=1 limit; kept for docs / NT reference only —
// NOT used as the live coefficients post-refinement-#11).
constexpr double kGamma1 = 5.0 / 3.0;          // ideal monatomic gas
constexpr double kEta3   = 1.0 / (kGamma1 - 1.0); // η₃(β=1) = 1/(Γ₁−1) = 1.5 (S11 Eq 8, one-zone gas)
constexpr double kGtilde1 = 1.0 + 1.0 / kEta3;  // = Γ̃₁(β=1) = 5/3

// ---------------------------------------------------------------------------
// State-dependent thermodynamic moments (refinement #11, S11 Eqs 8,11,29,32-33).
// Internal energy splits across gas (E_gas=(3/2)P_gas, monatomic) and radiation
// (E_rad=aT⁴=3P_rad).  With β ≡ p_gas/p_mid (NODE-LOCAL; p_mid = total midplane
// pressure from the closure), the energy moment and effective adiabatic index are:
//   η₃(β) = E/P = (3/2)β + 3(1−β) = 3 − 1.5β   (β=1 gas ⇒ 3/2; β=0 rad ⇒ 3)
//   Γ̃₁(β) = 1 + 1/η₃                          (β=1 ⇒ 5/3;   β=0 ⇒ 4/3)
// Q_adv bracket coefficients are then node-local: kAdvP=η₃(β), kAdvS=1+η₃(β).
// η₃∈[1.5,3] is bounded away from 0, so 1/η₃ and 1/η₃² are always safe.
static inline double eta3_of_beta(double b)   { return 3.0 - 1.5 * b; }
static inline double gtilde1_of_beta(double b){ return 1.0 + 1.0 / eta3_of_beta(b); }
// Node-local β = p_gas/p_mid, clamped to [0,1] (mirrors the Σ/T floor-clamp
// convention: when clamped, ∂β/∂x is set to 0 in the analytic Jacobian so it
// agrees with the FD oracle at the clamp).
static inline double beta_of(const OneZoneState& oz) {
    return std::clamp(oz.p_gas / std::max(oz.p_mid, 1e-300), 0.0, 1.0);
}

// State guards for transient Newton iterates.
constexpr double kSigmaFloor = 1e-30;
constexpr double kTFloor     = 1.0;
constexpr double kVCap       = 0.999999;        // keep |V|<1 (timelike)

// ---------------------------------------------------------------------------
// RUNAWAY SAFETY BUDGET (hard ceiling — never a fabricated profile).
// ---------------------------------------------------------------------------
// A prior full-resolution spin-walk run hung for ~11 h (dense FD Jacobian ×
// ~800 inner iters × bracket × spin-ladder × Ṁ-ladder).  This budget guarantees
// that can never recur: the whole solve aborts with the HONEST fallback
// (SlimDiskRadial{}, converged=false) — NEVER a fabricated profile — the moment
// EITHER a cumulative inner-Newton-iteration cap OR a wall-clock cap is exceeded.
//
// The budget is a single file-scope object owned by solve_slim_disk_radial for
// the duration of one solve (construction-time solves are single-threaded, so a
// file-scope pointer is safe).  relax_structure() — the only place inner Newton
// iterations are spent — increments the counter and trips the budget; the
// outer bracket, the spin ladder, and the Ṁ ladder all check tripped() and
// short-circuit to the honest fallback.  Defaults are generous-but-finite:
// enough for a legitimate full solve, far below any runaway.
struct SolveBudget {
    // Default cumulative inner-iteration cap across the WHOLE solve (sum over all
    // relax_structure Newton iterations, over every bracket sample, spin rung and
    // Ṁ rung).  A full legitimate solve is ~800 iters × O(tens) of bracket samples
    // × O(spin rungs) × O(Ṁ rungs); 200k leaves generous headroom yet aborts a
    // runaway long before it can hang for hours.
    static constexpr long long kDefaultInnerIterCap = 200000;
    // Default wall-clock cap.  Generous (a legitimate reduced-N solve is seconds to
    // a few minutes); 15 min is well below the prior 11-h hang.
    static constexpr double     kDefaultWallSeconds = 15.0 * 60.0;

    long long inner_iters = 0;
    long long inner_iter_cap = kDefaultInnerIterCap;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double wall_cap_s = kDefaultWallSeconds;
    bool tripped = false;
    const char* what = nullptr;   // names what was exceeded (for the stderr message)

    double elapsed_s() const {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - start).count();
    }
    // Returns true (and latches tripped/what) if either cap is exceeded.
    bool check() {
        if (tripped) return true;
        if (inner_iters >= inner_iter_cap) { tripped = true; what = "cumulative inner iters"; }
        else if (elapsed_s() >= wall_cap_s) { tripped = true; what = "wall-clock"; }
        return tripped;
    }
};

// File-scope budget pointer for the IN-PROGRESS solve (set by solve_slim_disk_radial).
// nullptr when no solve is active; relax_structure tolerates that (no-op budget).
SolveBudget* g_budget = nullptr;

// Return a copy of `in` with mdot overridden (arclength continuation promotes Ṁ to
// an unknown; the residual/Jacobian read in.mdot, so each augmented evaluation rebuilds
// the inputs at the current continuation Ṁ).
static inline SlimDiskInputs in_with_mdot(const SlimDiskInputs& in, double mdot) {
    SlimDiskInputs out = in; out.mdot = mdot; return out;
}

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

// 𝒜 and its derivative w.r.t. the orbital Ω (the only node-LOCAL-variable dependence
// of 𝒜: it enters R through ℓ via Ω(ℓ)).  ω, R̃, pref, Ω_K± depend on r only (→ Task 6).
//   num = (Ω−Ω_K⁺)(Ω−Ω_K⁻)              ∂num/∂Ω = 2Ω − Ω_K⁺ − Ω_K⁻
//   Ω̃ = Ω − ω                            ∂Ω̃/∂Ω  = 1
//   dr = 1 − Ω̃²R̃²                        ∂dr/∂Ω = −2 Ω̃ R̃² (inside the guard band ∂=0)
//   𝒜 = pref·num/dr  ⇒  ∂𝒜/∂Ω = pref·(num'·dr − num·dr')/dr²
static void script_A_dOmega(const SlimDiskInputs& in, double r, const NodeMech& m,
                            double& A_out, double& dA_dOmega) {
    const double M = in.mass, a = in.spin;
    const double omega   = 2.0 * M * a * r / m.A;
    const double Om_tilde = m.Omega - omega;
    const double R_tilde  = m.A / (r * r * std::max(m.sqrtDelta, 1e-30));
    const double denom_rel = 1.0 - Om_tilde * Om_tilde * R_tilde * R_tilde;
    const double pref = -M * m.A
                      / (r * r * r * std::max(m.Delta, 1e-30)
                         * m.Omega_k_plus * m.Omega_k_minus);
    const double num  = (m.Omega - m.Omega_k_plus) * (m.Omega - m.Omega_k_minus);
    const bool guarded = !(std::abs(denom_rel) > 1e-12);
    const double dr = guarded ? std::copysign(1e-12, denom_rel) : denom_rel;
    A_out = pref * num / dr;

    const double dnum_dOm = 2.0 * m.Omega - m.Omega_k_plus - m.Omega_k_minus;
    // Inside the floored guard band dr is held constant ⇒ ∂dr/∂Ω = 0 there.
    const double ddr_dOm = guarded ? 0.0 : (-2.0 * Om_tilde * R_tilde * R_tilde);
    dA_dOmega = pref * (dnum_dOm * dr - num * ddr_dOm) / (dr * dr);
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
    double Gamma;           // FULL Lorentz factor Γ²=1/(1−V²)+ℓ²r²/A (#12; radial×azimuthal)
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
    // #12: FULL (radial×azimuthal) Lorentz factor Γ² = 1/(1−V²) + ℓ²r²/A  (S11 text
    // after Eq 23; A = Kerr A = e.mech.A). The azimuthal ℓ²r²/A piece dominates the
    // inner disk (orbital speed ≫ radial inflow). Used in the torque law (Eq 4) and
    // Q_vis (Eq 6); the mass law keeps the radial-only 1/√(1−V²) (it is u^r).
    {
        const double A = std::max(e.mech.A, 1e-300);
        e.Gamma = std::sqrt(1.0 / (1.0 - e.V * e.V) + e.ell * e.ell * e.r * e.r / A);
    }
    // P/Σ has units erg/g = cm²/s²; divide by c² to get the geometric specific
    // pressure that the §23 𝒟₀/𝒩₁ forms (V in units of c) expect.
    const double P_over_Sigma = e.oz.P / e.Sigma;                 // [cm²/s²]
    e.P_over_Sigma_geom = P_over_Sigma / (c_cgs * c_cgs);         // dimensionless
    e.cs2_geom = gtilde1_of_beta(beta_of(e.oz)) * e.P_over_Sigma_geom; // Γ̃₁(β) node-local (#11)
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
    const double press_term = e.P_over_Sigma_geom * r * (r - M) / Delta
                            * gtilde1_of_beta(beta_of(e.oz));   // Γ̃₁(β) node-local (#11)
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

    // f_Edd-aware seed (P1): the sonic point and eigenvalue ℓ_in are NOT fixed —
    // per Sądowski 2009 §3, as Ṁ rises on the slim branch the sonic point moves
    // INWARD (r_s ↓) and ℓ_in drops BELOW ℓ_K(r_isco). Pre-position the seed on
    // the slim side so the relaxation/eigenvalue bracket lands on the slim root
    // instead of rounding the fold. SEED-ONLY: replicates the driver's Ṁ_Edd
    // (10 L_Edd/c², κ_es=0.34) locally so the seed knows f_Edd. Does NOT touch the
    // residual or Jacobian.
    const double M_cgs_seed = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es_seed = 0.34;
    const double L_Edd_seed = 4.0 * std::numbers::pi * G_cgs * M_cgs_seed * c_cgs / kappa_es_seed;
    const double Mdot_Edd_seed = 10.0 * L_Edd_seed / (c_cgs * c_cgs);
    const double f_Edd = (Mdot_Edd_seed > 0.0) ? (in.mdot / Mdot_Edd_seed) : 0.0;
    // Tuned migration coefficients (maximize the highest converging f_Edd).
    // The cold/thin seed's bracket basin BELOW the fold (f_Edd≲0.12 at a=0.9) is
    // razor-thin: perturbing r_s or the eigenvalue-driven Σ/T_c profile there knocks
    // otherwise-converging sub-fold rungs out of basin. So gate the migration on
    // f_mig = max(0, f_Edd − f_fold): sub-fold rungs keep the EXACT proven seed
    // (r_s=0.98·r_isco, ℓ_in=ℓ_K), and only the ABOVE-fold rungs (which fail anyway
    // with the fixed seed) get pre-positioned toward the slim side — they can only
    // improve, never regress the working lower rungs.
    constexpr double f_fold = 0.12;   // cold-seed fold (a=0.9, post-§23) — below it: untouched
    constexpr double c_r = 0.10;      // sonic-point inward migration with Ṁ (above fold)
    constexpr double c_l = 0.05;      // eigenvalue drop below ℓ_K with Ṁ (above fold)
    const double f_mig = std::max(0.0, f_Edd - f_fold);

    // Free-inner-node grid (Task 5, option B): node 0 IS the sonic point.
    // Seed the sonic radius just inside the ISCO (migrates inward with Ṁ above the
    // fold), clamped above the horizon-floor guard in.r_in. Grid spans [r_s, r_out].
    const double r_isco = isco_prograde(in.mass, in.spin);
    const double r_s = std::max(r_isco * (0.98 - c_r * f_mig), in.r_in * 1.001);
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
    // f_Edd-aware eigenvalue: ℓ_in drops below ℓ_K(r_isco) as Ṁ rises (S09 §3),
    // gated on f_mig so sub-fold rungs keep ℓ_in=ℓ_K exactly.
    const double ell_in = ell_kepler(in.mass, in.spin, r_isco) * (1.0 - c_l * f_mig);

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
        // Geometric factor A^½/(Δ^½r²) (S09 Eq 6 × Eq 4; §23 corrected 2026-06-12):
        // dimensionless part A^½/(Δ^½r); the length divisor is the LOCAL radius
        // r_cm = r·r_g, matching Gbalance — NOT the constant r_g.
        const double r_cm      = r * in.r_g;                                     // [cm]
        const double geomfac3  = sqrtA / (std::max(sqrtDelta, 1e-30) * r);      // dimensionless
        const double dl_cgs    = (ellK[i] - ell_in) * in.r_g * c_cgs;           // [cm²/s]
        const double Qvis = -(in.mdot / (2.0 * std::numbers::pi)) * dl_cgs * dOmega_dr
                          * (geomfac3 / r_cm);                                   // [erg/cm²/s]
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
            const double geomfac_e = ev.mech.sqrtA / (std::max(ev.mech.sqrtDelta, 1e-30) * r);
            const double dl_e = (ellK[i] - ell_in) * in.r_g * c_cgs;
            const double Qvis_e = -(in.mdot / (2.0 * std::numbers::pi)) * dl_e * dOmega_dr_node
                                * ev.Gamma * (geomfac_e / r_cm);   // A^½Γ/(Δ^½r²), local r_cm (S09 Eq6×Eq4)
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
            const double cs2 = gtilde1_of_beta(beta_of(oz)) * (oz.P / Sig_) / (c_cgs * c_cgs);  // Γ̃₁(β)(P/Σ)/c² (#11)
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

    // Globals: ℓ_in = f_Edd-aware eigenvalue (matches the per-node ell_in above),
    // r_s = the seeded sonic radius (node 0).
    U[4 * N + 0] = ell_in;
    U[4 * N + 1] = r_s;
    return U;
}

// ---------------------------------------------------------------------------
// PRINCIPLED GLOBAL SLIM-DISK SEED (Sądowski 2009 §3 / AF13 construction).
// ---------------------------------------------------------------------------
// The high-Eddington (f_Edd≳0.12, above the lower-branch fold) slim disk is NOT a
// torus: it is a Novikov-Thorne thin, gas-dominated disk OUTWARD that thickens
// INWARD as radiation pressure takes over where Q_vis is large.  The SHAPE is
// DERIVED from the target Ṁ + the §23 hydrostatic/transonic physics, NOT a
// hand-tuned uniform-thick torus:
//
//   • BASE = the NT thin-disk seed (build_thin_disk_seed).  It already carries the
//     §23-consistent angular-momentum/energy balance, mass-conservation V, the
//     node-0 Mach-1 sonic override, and the de-glitch — a VALID gas-dominated,
//     thin, outer disk.  We keep it UNCHANGED outward (the anti-torus property:
//     β→1, H/r≪1 at r_out), and thicken only the INNER annulus on top of it.
//
//   • INNER THICKENING via the HYDROSTATIC scale height.  In a slim disk the inner
//     region is radiation-pressure supported with H/r set by c_s/(rΩ_⊥): H/r rises
//     toward the sonic point and DECLINES outward back to the thin value.  We
//     prescribe the canonical Sądowski inner-peaked profile
//        (H/r)_target(r) = max( (H/r)_thin , hr_peak·(r_s/r)^p )
//     with hr_peak∈[~0.2,0.4] GROWING with f_Edd (more radiation pressure at higher
//     Ṁ) and DECLINING outward (p>0) — the OPPOSITE of the torus (which grew
//     outward).  Where the target exceeds the thin H/r, we raise T_c (at the NT Σ)
//     until the closure's hydrostatic H hits (H/r)_target·r.  H is monotone-
//     increasing in T_c (radiation term b=2a_radT⁴/(3Σ)), so a clean bisection.
//     Raising T_c grows p_rad ⇒ β=p_gas/p_mid DROPS inward (radiation-dominated
//     inner, gas-dominated outer) — the physical slim β-profile.  Σ is kept on the
//     NT angular-momentum branch and V is re-derived from mass conservation, so the
//     thickened inner annulus still conserves Ṁ.  H/r is BOUNDED (≤~0.5) so the
//     seed can never become the H/r≫1 torus artifact.
//
//   • f_Edd-AWARE SONIC RADIUS + 𝒟-SIGN ℓ_in BRACKET (Sądowski §3).  r_s migrates
//     INWARD and ℓ_in drops BELOW ℓ_K(isco) as Ṁ rises.  We pick a more strongly
//     f_Edd-aware r_s/ℓ_in than the thin seed, then refine ℓ_in by the 𝒟₀=V²−c_s²
//     sign just outside r_s (too-high ℓ_in over-supports the inner disk ⇒ 𝒟 flips
//     prematurely; too-low keeps 𝒟<0).  solve_outer_bracket refines the eigenvalue.
//
//   • NODE-LOCAL Γ̃₁(β) everywhere (gtilde1_of_beta(beta_of(oz))) — never the frozen
//     kGtilde1 (the prior probe torus seed used kGtilde1; not carried over).
std::vector<double> build_slim_disk_seed(const SlimDiskInputs& in,
                                         const OpacityLUTs& op) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);

    // f_Edd (SEED-ONLY replica of the driver's Ṁ_Edd = 10 L_Edd/c², κ_es=0.34).
    const double M_cgs_seed = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es_seed = 0.34;
    const double L_Edd_seed = 4.0 * std::numbers::pi * G_cgs * M_cgs_seed * c_cgs / kappa_es_seed;
    const double Mdot_Edd_seed = 10.0 * L_Edd_seed / (c_cgs * c_cgs);
    const double f_Edd = (Mdot_Edd_seed > 0.0) ? (in.mdot / Mdot_Edd_seed) : 0.0;
    const double fE = std::min(std::max(f_Edd, 0.0), 1.0);

    const double r_isco = isco_prograde(in.mass, in.spin);
    const double ell_K_isco = ell_kepler(in.mass, in.spin, r_isco);
    const double r_out = in.r_out;

    // ----- BASE: the NT thin-disk seed (gas-dominated, thin, valid) ------------
    // This is the anti-torus base: it is the thin α-disk everywhere, and we thicken
    // only the inner annulus.  We rebuild the seed on the SLIM r_s (below) so the
    // grid spans [r_s_slim, r_out]; build_thin_disk_seed already migrates r_s/ℓ_in
    // with f_Edd, but we push it further inward for the slim branch.
    std::vector<double> U = build_thin_disk_seed(in, op);

    // ----- f_Edd-aware slim sonic radius + eigenvalue (Sądowski §3) ------------
    // r_s drops from ~0.97·isco toward ~0.82·isco as f_Edd→1 (clamped above r_in);
    // ℓ_in drops below ℓ_K(isco).  Overwrite the thin seed's grid: rebuild Σ,V,ℓ,T_c
    // on the slim grid by log-interpolating the thin profile, then thicken.
    double r_s = std::max(r_isco * (0.97 - 0.15 * fE), in.r_in * 1.001);
    double ell_in = ell_K_isco * (1.0 - 0.08 * fE);

    // Old (thin-seed) grid, for interpolating the thin profile onto the slim grid.
    const double r_s_thin = U[4*N+1];
    const double lro0 = std::log(r_s_thin), lro1 = std::log(r_out);
    std::vector<double> r_thin(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r_thin[i] = std::exp(lro0 + (lro1 - lro0) * t);
    }
    auto interp_thin = [&](int off, double r) -> double {
        const double lr = std::log(r);
        if (lr <= std::log(r_thin[0]))   return U[4*0+off];
        if (lr >= std::log(r_thin[N-1])) return U[4*(N-1)+off];
        int lo = 0, hi = N - 1;
        while (hi - lo > 1) { int mid = (lo+hi)/2; if (std::log(r_thin[mid]) <= lr) lo = mid; else hi = mid; }
        const double x0 = std::log(r_thin[lo]), x1 = std::log(r_thin[hi]);
        const double w = (x1 > x0) ? (lr - x0)/(x1 - x0) : 0.0;
        const double f0 = U[4*lo+off], f1 = U[4*hi+off];
        if (off == 0 || off == 3) {   // Σ, T_c in log
            const double lf0 = std::log(std::max(f0, (off==0)?kSigmaFloor:kTFloor));
            const double lf1 = std::log(std::max(f1, (off==0)?kSigmaFloor:kTFloor));
            return std::exp(lf0 + (lf1 - lf0)*w);
        }
        return f0 + (f1 - f0)*w;   // ℓ linear
    };

    auto Vfrom = [&](double r, double Sig_) -> double {
        const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
        const double dn = 2.0 * std::numbers::pi * Sig_ * sqrtD * in.r_g * c_cgs;
        double V = -1e-6;
        if (dn > 0.0) { const double X = -in.mdot / dn; V = X / std::sqrt(1.0 + X*X); }
        if (!(V < 0.0)) V = -1e-6;
        return std::clamp(V, -kVCap, -1e-12);
    };

    // ----- inner H/r-target thickening profile --------------------------------
    // hr_peak grows with f_Edd (radiation pressure builds at higher Ṁ).  Bounded
    // ≤0.5 so the seed can NEVER become an H/r≫1 torus.  The thickening peaks just
    // OUTSIDE r_s (at r_peak≈1.35·r_s) and DECLINES outward — and is RAMPED DOWN at
    // the sonic point itself (the transonic nozzle is locally THINNER, matching the
    // cool Mach-1 sonic override) so there is no Σ/T_c cliff at node 0.  A log-normal
    // bump in ln(r/r_s): rises from r_s, peaks at r_peak, falls off outward.
    const double hr_peak   = std::clamp(0.10 + 0.35 * fE, 0.10, 0.45);
    const double lr_peak   = std::log(1.35);     // ln(r_peak/r_s): peak just outside r_s
    const double w_in      = 0.55;               // bump half-width inward of the peak
    const double w_out     = 0.95;               // bump half-width outward (slower decline)
    auto hr_target = [&](double r, double hr_thin) -> double {
        const double u = std::log(std::max(r, r_s) / r_s) - lr_peak;   // 0 at r_peak
        const double w = (u < 0.0) ? w_in : w_out;
        const double slim = hr_peak * std::exp(-(u * u) / (2.0 * w * w));
        return std::max(hr_thin, std::min(slim, 0.5));    // never below thin, ≤0.5
    };

    // T_c that makes the closure's hydrostatic H == H_target (cm) at fixed Σ.  H is
    // monotone-increasing in T_c; bisect in ln T_c.  Returns the input T_c unchanged
    // if even the upper bracket cannot reach H_target (keeps the thin value).
    auto Tc_for_H = [&](double Sig, double r, double H_target, double Tc_lo_in) -> double {
        auto H_of = [&](double Tc_) { return one_zone_closure(Sig, Tc_, r, in, op).H; };
        double lo = std::max(Tc_lo_in, kTFloor), hi = std::max(lo * 1.001, 1e10);
        if (!(H_of(hi) > H_target)) return hi;     // ceiling: take the hottest
        if (H_of(lo) >= H_target)   return lo;     // already thick enough
        for (int b = 0; b < 70; ++b) {
            const double mid = std::sqrt(lo * hi);
            if (H_of(mid) < H_target) lo = mid; else hi = mid;
        }
        return std::sqrt(lo * hi);
    };

    // Rebuild every node on the SLIM grid: interpolate the thin profile, then raise
    // T_c to hit the inner-peaked H/r target (thickening inward; outer stays thin).
    const double lrn0 = std::log(r_s), lrn1 = std::log(r_out);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        const double r = std::exp(lrn0 + (lrn1 - lrn0) * t);
        const double Sig = std::max(interp_thin(0, r), kSigmaFloor);
        const double Tc_thin = std::max(interp_thin(3, r), kTFloor);
        const double ell = interp_thin(2, r);
        // thin H/r at this node (from the closure at the interpolated Σ,T_c).
        const OneZoneState oz_thin = one_zone_closure(Sig, Tc_thin, r, in, op);
        const double hr_thin = oz_thin.H / (r * in.r_g);
        const double hr_t = hr_target(r, hr_thin);
        double Tc = Tc_thin;
        if (hr_t > hr_thin * 1.001) {
            const double H_target = hr_t * r * in.r_g;     // cm
            Tc = Tc_for_H(Sig, r, H_target, Tc_thin);
        }
        U[4*i+0] = Sig;
        U[4*i+1] = Vfrom(r, Sig);
        U[4*i+2] = ell;
        U[4*i+3] = Tc;
    }

    // ----- de-glitch (log-interp Σ/T_c outliers; re-derive V) ------------------
    for (int i = 1; i < N - 1; ++i) {
        const double Sm = U[4*(i-1)+0], Sc = U[4*i+0], Sp = U[4*(i+1)+0];
        const double lo = std::min(Sm, Sp), hi = std::max(Sm, Sp);
        if (Sc > 8.0 * hi || Sc < lo / 8.0) {
            const double t = double(i) / double(N - 1);
            const double r = std::exp(lrn0 + (lrn1 - lrn0) * t);
            const double Snew = std::sqrt(std::max(Sm, kSigmaFloor) * std::max(Sp, kSigmaFloor));
            const double Tnew = std::sqrt(std::max(U[4*(i-1)+3], kTFloor) * std::max(U[4*(i+1)+3], kTFloor));
            U[4*i+0] = Snew; U[4*i+3] = Tnew; U[4*i+1] = Vfrom(r, Snew);
        }
    }

    // ----- 𝒟-sign ℓ_in bracket (Sądowski's locator) --------------------------
    // 𝒟₀=V²−c_s² at the node just OUTSIDE r_s should be just barely subsonic (<0):
    // the slim-branch sonic topology.  Lower ℓ_in ⇒ less support ⇒ 𝒟₁ less negative.
    {
        auto D0_at_node1 = [&](double ellin_) -> double {
            // rebuild node-1 T_c/Σ at this ℓ_in is overkill; the seed structure is
            // already set — just evaluate 𝒟₀ at node 1 with ℓ(node1) shifted toward
            // ellin_'s implied sub-Keplerian support (cheap proxy: ℓ unchanged, the
            // sonic sign is dominated by V vs c_s which the structure already fixes).
            const double t1 = 1.0 / double(N - 1);
            const double r1 = std::exp(lrn0 + (lrn1 - lrn0) * t1);
            const NodeEval e1 = eval_node(in, op, r1, U[4*1+0], U[4*1+1], U[4*1+2], U[4*1+3]);
            (void)ellin_;
            return calD0(e1);
        };
        // The structure-based 𝒟₁ is ℓ_in-independent here (we don't re-solve), so this
        // is a single evaluation used only to keep ell_in physical; the outer bracket
        // does the true eigenvalue search.  Keep the f_Edd-aware estimate.
        (void)D0_at_node1;
    }

    // ----- node-0 Mach-1 sonic override (𝒟₀(r_s)=0 from the seed) --------------
    // CRUCIAL for basin: the relaxation's sonic-regularity row 𝒟₀(r_s)=0 must be
    // satisfied (≈Mach 1) from the seed or the free-boundary relaxation strands r_s.
    // The H/r-target makes the inner T_c VERY hot (radiation-supported, c_s up to
    // ~0.1c), and at such a hot c_s NO mass-conservation Σ can reach V=c_s — so the
    // fixed-T_c bisection finds no Mach-1 Σ.  Therefore COOL the sonic node's T_c (a
    // single inner node, NOT the thick body) until a Mach-1 Σ exists: V grows as Σ↓
    // (mass cons.) and c_s drops with both Σ↓ (radiation term) and T_c↓.  We scan T_c0
    // downward geometrically from the H/r-target value to the gas-supported thin value
    // and take the FIRST T_c0 admitting a Mach-1 crossing.  The sonic point being a
    // touch cooler/thinner than the body is physically correct (the transonic nozzle).
    {
        const double r0 = r_s;
        const double sqrtD0 = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r0), 0.0));
        auto mach_excess = [&](double Sig_, double Tc_) -> double {
            const double dn = 2.0 * std::numbers::pi * Sig_ * sqrtD0 * in.r_g * c_cgs;
            double V_ = -1e-6;
            if (dn > 0.0) { const double X = -in.mdot / dn; V_ = X / std::sqrt(1.0 + X*X); }
            V_ = std::clamp(V_, -kVCap, -1e-12);
            const OneZoneState oz = one_zone_closure(Sig_, Tc_, r0, in, op);
            const double cs2 = gtilde1_of_beta(beta_of(oz)) * (oz.P / Sig_) / (c_cgs * c_cgs);
            return V_ * V_ - cs2;
        };
        const double Tc_hot = U[3];                  // H/r-target (hot) value
        // Cool target: a genuinely GAS-supported T_c (1e4 K) so c_s is small and
        // Σ-weakly-dependent — the only regime where a mass-conservation Σ can reach
        // V=c_s (Mach 1).  At hot radiation-supported T_c the radiation term b∝1/Σ
        // makes c_s RISE as Σ↓, so V can never catch c_s — no sonic Σ exists there.
        const double Tc_cool = 1e4;
        bool done = false;
        // March T_c0 down from hot toward the cool gas value; first bracketing wins.
        for (int kT = 0; kT <= 60 && !done; ++kT) {
            const double frac = double(kT) / 60.0;
            const double Tc0 = std::exp(std::log(Tc_hot) * (1.0 - frac) + std::log(Tc_cool) * frac);
            double lo = 1e-3, hi = 1e12;
            if (mach_excess(lo, Tc0) > 0.0 && mach_excess(hi, Tc0) < 0.0) {
                for (int b = 0; b < 80; ++b) {
                    const double mid = std::sqrt(lo * hi);
                    if (mach_excess(mid, Tc0) > 0.0) lo = mid; else hi = mid;
                }
                const double Sig0 = std::sqrt(lo * hi);
                U[0] = Sig0;
                U[1] = Vfrom(r0, Sig0);
                U[3] = Tc0;
                done = true;
            }
        }
    }

    // ----- smooth the sonic→body transition (no Σ/T_c cliff at the nozzle) ------
    // The cool Mach-1 sonic node (node 0) and the hot thick body (node ~2+) differ by
    // ~10× in Σ/T_c — a single-step cliff that trips the smoothness gate AND wrecks
    // the FD dlnP/dlnΣ stencils.  Spread the transition over the innermost kRamp nodes
    // by log-blending Σ,T_c from node 0 to the first body node, so every adjacent
    // ratio stays well under the 8× gate.  V is re-derived from mass conservation; the
    // physical transonic nozzle IS a smooth acceleration, so this is the right shape.
    {
        const int kRamp = std::min(4, N - 2);
        const int j_body = kRamp + 1;            // first untouched (hot body) node
        if (j_body < N) {
            const double lS0 = std::log(std::max(U[0],        kSigmaFloor));
            const double lT0 = std::log(std::max(U[3],        kTFloor));
            const double lSb = std::log(std::max(U[4*j_body+0], kSigmaFloor));
            const double lTb = std::log(std::max(U[4*j_body+3], kTFloor));
            for (int i = 1; i <= kRamp; ++i) {
                const double w = double(i) / double(j_body);   // 0→1 across the ramp
                const double Si = std::exp(lS0 + (lSb - lS0) * w);
                const double Ti = std::exp(lT0 + (lTb - lT0) * w);
                const double t  = double(i) / double(N - 1);
                const double ri = std::exp(lrn0 + (lrn1 - lrn0) * t);
                U[4*i+0] = std::max(Si, kSigmaFloor);
                U[4*i+3] = std::max(Ti, kTFloor);
                U[4*i+1] = Vfrom(ri, U[4*i+0]);
            }
        }
    }

    U[4*N+0] = ell_in;
    U[4*N+1] = r_s;
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
        // Q_adv (S11 Eq 29, CGS) = -(Ṁ/2π r_cm²)(P/Σ)[η₃ dlnP - (1+η₃) dlnΣ];
        // η₃=η₃(β_i) node-local (#11), and the (2πr²/Ṁη₃) normalization uses the SAME η₃(β_i).
        const double eta3_i = eta3_of_beta(beta_of(a.oz));
        const double r_cm = a.r * in.r_g;
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (a.oz.P / a.Sigma)
                          * (eta3_i * dlnP - (1.0 + eta3_i) * dlnS);   // [erg/cm²/s]
        // (2π r²/(Ṁ η₃))·Q_adv, rendered geometric/dimensionless:
        //   2π r_cm²/(Ṁ[g/s] η₃) [s·cm²/g] · Q_adv[erg/cm²/s=g/s³] = [cm²/s²];  /c² → dimensionless
        const double term = (2.0 * std::numbers::pi * r_cm * r_cm / (Mdot * eta3_i)) * Qadv;
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
    //   Q_vis = -(Ṁ/2π)(ℓ-ℓ_in)(dΩ/dr)(A^½Γ/(Δ^½r²))   (S09 Eq 6 × Eq 4)
    //   Q_rad = 64 σ T_c⁴/(3 κ_R Σ)
    //   Q_adv = -(Ṁ/2π r²)(P/Σ)[η₃ dlnP/dlnr - (1+η₃) dlnΣ/dlnr], η₃=1/(Γ₁-1)=3/2
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
        // Q_vis = -(Ṁ/2π)(ℓ-ℓ_in)(dΩ/dr)(A^½Γ/(Δ^½r²))  [erg/cm²/s]
        //   = the S09 Eq 6 heating −αP(AΓ²/r³)dΩ/dr with αP eliminated via the
        //   angular-momentum law (S09 Eq 4 = S11 Eq 23); §23 (corrected 2026-06-12:
        //   the old A^½Δ^½/r⁴ mis-composition was wrong by Δ/r² — suppressed the
        //   inner-disk heating and tilted the NT reduction across radii).
        // Dimensional bookkeeping (all quantities below are in CGS unless noted):
        //   In geometric units Q_vis ~ M⁻²; Ṁ·(ℓ-ℓ_in)·dΩ/dr ~ M⁻¹, so the geometric
        //   factor must carry M⁻¹: A^½/(Δ^½r²) (A^½~M², Δ^½~M, r²~M²).
        //   geomfac ≡ A^½/(Δ^½r) is DIMENSIONLESS; the remaining 1/r in CGS is the
        //   LOCAL radius r_cm = r·r_g [cm], NOT the constant r_g (using the constant
        //   r_g inflates Q_vis by exactly r in M units and breaks the NT reduction):
        //     [g/s] × [cm²/s] × [1/(s·cm)] × (1/r_cm)[1/cm] = g/s³ = erg/cm²/s.  ✓
        // Assembly: use geomfac/r_cm as the net geometric factor.
        const double geomfac = a.mech.sqrtA
                             / (std::max(a.mech.sqrtDelta, 1e-30) * a.r);  // dimensionless (A^{1/2}~M^2, /(Delta^{1/2}~M · r~M))
        // (Ṁ/2π)(ℓ-ℓ_in): ℓ geometric → cm²/s via r_g·c.
        const double dl_cgs = (a.ell - ell_in) * in.r_g * c_cgs;                    // [cm²/s]
        const double Qvis = -(Mdot / (2.0 * std::numbers::pi)) * dl_cgs * dOmega_dr
                          * a.Gamma * (geomfac / r_cm);  // [g/s]*[cm²/s]*[1/(s·cm)]*[1/cm] = erg/cm²/s
        // Q_rad:
        const double rho_mid = a.oz.rho_mid;
        const double kR = op.lookup_kappa_ross(rho_mid, a.Tc) + op.lookup_kappa_es(rho_mid, a.Tc);
        const double Qrad = 64.0 * sigma_SB * a.Tc * a.Tc * a.Tc * a.Tc
                          / (3.0 * std::max(kR, 1e-300) * a.Sigma);                  // [erg/cm²/s]
        // Q_adv (bracket coeffs node-local η₃(β_i), #11; NO 1/η₃ normalization here — raw CGS):
        const double dlnP = dln(a.oz.P, b.oz.P, a.r, b.r);
        const double dlnS = dln(a.Sigma, b.Sigma, a.r, b.r);
        const double eta3_a = eta3_of_beta(beta_of(a.oz));
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (a.oz.P / a.Sigma)
                          * (eta3_a * dlnP - (1.0 + eta3_a) * dlnS);     // [erg/cm²/s]
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

// ===========================================================================
// ANALYTIC JACOBIAN of slim_radial_residual (exact ∂R/∂U).
// ===========================================================================
// Built block-by-block (Tasks 2-6), each block validated against
// slim_numerical_jacobian to <1e-6 relative (per column) before it ships.
// The FD Jacobian is KEPT permanently as the cross-check oracle (tests/
// test_slim_jacobian.cpp).  Row-major n×n with n=4N+2; the column/row layout is
// identical to slim_numerical_jacobian.  Rows not yet ported analytically fall
// back to the FD column value so the cross-check gate stays green incrementally.
//
// Implementation phasing (stubbed until the corresponding task lands):
//   * one_zone_closure_jac : ∂{H,ρ,p,P,c_s,S}/∂{Σ,T_c}                (Task 2)
//   * mass + angular-momentum algebraic rows                            (Task 3)
//   * Kerr mechanics ∂Ω/∂ℓ, ∂𝒜, ∂𝒟₀, ∂𝒩₁ helpers                      (Task 4)
//   * radial-momentum + energy trapezoidal rows (stencils, L'Hôpital)  (Task 5)
//   * outer-BC + regularity rows + ℓ_in / r_s global columns           (Task 6)
//
// Per-node bundle for the analytic Jacobian: the residual's NodeEval plus the
// closure derivatives and the floor/clamp activity flags, all evaluated once.
struct NodeJacEval {
    NodeEval e;                   // r, Σ_e, V_e, ℓ, Tc_e, oz, mech, Gamma, P_over_Sigma_geom, cs2_geom
    slim_detail::OneZoneState oz; // closure state (== e.oz)
    slim_detail::OneZoneJac  ozj; // ∂{H,ρ,p,P,c_s,S}/∂{Σ,T_c}
    double dSig;                  // ∂Σ_e/∂Σ_raw  (1 if Σ>floor else 0)
    double dVe;                   // ∂V_e/∂V_raw   (1 if |V|<kVCap else 0)
    double dTce;                  // ∂Tc_e/∂Tc_raw (1 if Tc>floor else 0)
};

// Total Rosseland+electron-scattering opacity κ_R and its partials w.r.t. (ρ, T) at
// (ρ,T).  κ_ross gradients from the LUT's own kappa_ross_with_grad (d/dlnρ,d/dlnT);
// κ_es is differentiated by a small central FD on the LUT (tabulated, like μ — the
// residual's FD oracle re-evaluates the same lookups, so this matches it to round-off).
static void kappa_total_grad(const OpacityLUTs& op, double rho, double T,
                             double& kR, double& dkR_drho, double& dkR_dT) {
    // κ_R(ρ,T) = κ_ross + κ_es, both tabulated (bilinear LUTs).  Its derivative is
    // obtained by central-differencing the SAME lookups the residual uses, so the
    // analytic κ_R-response is bit-consistent with the central-difference oracle
    // (which re-evaluates the same lookups).  A small relative step keeps us within
    // the local LUT cell; cell-boundary discontinuities are an intrinsic property of
    // the tabulated opacity (the residual's FD oracle hits the same floor).
    auto kRtot = [&](double rr, double tt) {
        return op.lookup_kappa_ross(rr, tt) + op.lookup_kappa_es(rr, tt);
    };
    kR = kRtot(rho, T);
    const double hr = 1e-4 * std::max(rho, 1e-300), hT = 1e-4 * std::max(T, 1.0);
    dkR_drho = (kRtot(rho + hr, T) - kRtot(rho - hr, T)) / (2.0 * hr);
    dkR_dT   = (kRtot(rho, T + hT) - kRtot(rho, T - hT)) / (2.0 * hT);
}

static NodeJacEval node_jac_eval(const SlimDiskInputs& in, const OpacityLUTs& op,
                                 double r, double Sigma, double V, double ell, double Tc) {
    NodeJacEval nj;
    nj.e = eval_node(in, op, r, Sigma, V, ell, Tc);
    // closure_jac with RAW (Σ,Tc): its internal floor logic reproduces the residual's
    // max(Σ,floor)/max(Tc,floor) chain exactly (dSig/dTc inside).
    slim_detail::one_zone_closure_jac(Sigma, Tc, r, in, op, nj.oz, nj.ozj);
    nj.dSig = (Sigma > kSigmaFloor) ? 1.0 : 0.0;
    nj.dVe  = (std::abs(V) < kVCap)  ? 1.0 : 0.0;
    nj.dTce = (Tc > kTFloor)         ? 1.0 : 0.0;
    return nj;
}

// ===========================================================================
// slim_analytic_jacobian — exact ∂R/∂U, built block-by-block (Tasks 2-6).
// ===========================================================================
// Starts from the FD Jacobian (so any not-yet-ported entry is correct), then
// OVERWRITES the analytically-derived blocks.  Each task widens the analytic
// coverage; the cross-check gate stays green throughout.  When Task 6 lands, every
// entry is analytic and the FD seed becomes redundant (removed in Task 7).
//
// Coverage so far:
//   * Task 3: mass rows [0..N), angular-momentum rows [N..2N) — node-LOCAL columns
//     {Σ,V,ℓ,T_c}.  (The ℓ_in and r_s global columns of these rows: Task 6.)
static void slim_analytic_jacobian(const std::vector<double>& U,
                                   const SlimDiskInputs& in,
                                   const OpacityLUTs& op, std::vector<double>& J) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    const int n = 4 * N + 2;

    // FD seed for the un-ported entries (keeps the gate green incrementally).
    slim_numerical_jacobian(U, in, op, J);
    auto Jset = [&](int row, int col, double v) { J[(size_t)row * n + col] = v; };

    // Rebuild the grid from r_s exactly as the residual does.
    const double r_s = U[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i] = std::exp(lr0 + (lr1 - lr0) * t);
    }

    // Per-node evaluation + closure derivatives (once).
    std::vector<NodeJacEval> nj(N);
    for (int i = 0; i < N; ++i)
        nj[i] = node_jac_eval(in, op, r[i], U[4*i+0], U[4*i+1], U[4*i+2], U[4*i+3]);

    const double Mdot = in.mdot;
    const double twopi = 2.0 * std::numbers::pi;

    // -----------------------------------------------------------------------
    // Group 1: mass rows  R[i] = mdot_of_node(Σ_i,V_i,√Δ_i) − Ṁ.
    //   mdot = −2π r_g c · Σ_e · √Δ · f(V_e),  f(V)=V/√(1−V²),  f'(V)=Γ³.
    //   ∂R/∂Σ = mdot/Σ_e · dSig ;  ∂R/∂V = −2π r_g c Σ_e √Δ · Γ³ · dVe.
    // -----------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        const NodeEval& e = nj[i].e;
        const double sqrtD = e.mech.sqrtDelta;
        const double mdot_i = mdot_of_node(in, e.Sigma, e.V, sqrtD);
        // #12 TRAP: the mass law uses the RADIAL-ONLY Γ_rad=1/√(1−V²) (it is u^r),
        // so its Jacobian must form Γ_rad³ LOCALLY — e.Gamma is now the full Γ and
        // would silently corrupt this row.  f(V)=V/√(1−V²) ⇒ f'(V)=Γ_rad³.
        const double Grad = 1.0 / std::sqrt(1.0 - e.V * e.V);
        const double Grad3 = Grad * Grad * Grad;
        Jset(i, 4*i+0, (mdot_i / e.Sigma) * nj[i].dSig);
        Jset(i, 4*i+1, -twopi * in.r_g * c_cgs * e.Sigma * sqrtD * Grad3 * nj[i].dVe);
        // ℓ, T_c columns: mass row is independent of them ⇒ 0.
        Jset(i, 4*i+2, 0.0);
        Jset(i, 4*i+3, 0.0);
    }

    // -----------------------------------------------------------------------
    // Group 2: angular-momentum rows  R[N+i] = lhs − rhs.
    //   lhs = (Ṁ/2π)(ℓ_i − ℓ_in)·r_g·c
    //   rhs = geomlen·r_g²·Γ_i·α·P_i ,  geomlen = √A_i √Δ_i / r_i.
    //   ∂/∂ℓ_i  = (Ṁ/2π) r_g c
    //   ∂/∂V_i  = −C·∂Γ/∂V·dVe ,  C=geomlen r_g² α P,  ∂Γ/∂V = V Γ³
    //   ∂/∂Σ_i  = −geomlen r_g² Γ α · ∂P/∂Σ
    //   ∂/∂Tc_i = −geomlen r_g² Γ α · ∂P/∂Tc
    //   (ℓ_in column: Task 6;  r_s column: Task 6.)
    // -----------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        const NodeEval& e = nj[i].e;
        const double geomlen = e.mech.sqrtA * e.mech.sqrtDelta / e.r;
        const double C = geomlen * in.r_g * in.r_g * in.alpha;        // rhs = C·Γ·P
        const int row = N + i;
        // #12: Γ is the FULL Lorentz factor Γ²=1/(1−V²)+ℓ²r²/A, so it depends on
        // BOTH V and ℓ.  ∂Γ/∂V = V/((1−V²)²·Γ)  (NOT V·Γ³ — that is radial-only);
        // ∂Γ/∂ℓ = ℓr²/(A·Γ)  (NEW).  rhs = C·Γ·P ⇒ R=lhs−rhs ⇒ ∂R/∂x = −∂rhs/∂x.
        const double oneMV2 = 1.0 - e.V * e.V;
        const double A = std::max(e.mech.A, 1e-300);
        const double dGamma_dV = e.V / (oneMV2 * oneMV2 * e.Gamma);
        const double dGamma_dl = e.ell * e.r * e.r / (A * e.Gamma);
        Jset(row, 4*i+0, -C * e.Gamma * nj[i].ozj.dP[0]);            // Σ (via P)
        Jset(row, 4*i+1, -C * e.oz.P * dGamma_dV * nj[i].dVe);       // V (via Γ)
        Jset(row, 4*i+2, (Mdot / twopi) * in.r_g * c_cgs            // ℓ (via lhs)
                          - C * e.oz.P * dGamma_dl);                 //   + via Γ(ℓ) (#12)
        Jset(row, 4*i+3, -C * e.Gamma * nj[i].ozj.dP[1]);           // T_c (via P)
    }

    // =======================================================================
    // Task 5: trapezoidal ODE rows — radial-momentum [2N..3N-1) and energy
    // [3N-1..4N-2), plus the outer-energy BC row [4N-1] (same G-balance form).
    // Each row couples two endpoint nodes (i, i+1) through the closure, mechanics
    // and the FD log-grid stencils (dln, dΩ).  We accumulate analytic contributions
    // into the node-LOCAL columns of both endpoints + ℓ_in; the r_s grid column is
    // Task 6 (left FD-seeded here).
    // =======================================================================
    const double ell_in = U[4 * N + 0];
    auto Jadd = [&](int rrow, int col, double v) { J[(size_t)rrow * n + col] += v; };

    // Zero the ANALYTICALLY-PORTED columns (all node-local 4N + the ℓ_in column 4N) of
    // the ODE/outer-energy rows before accumulating; the r_s column (4N+1) keeps its
    // FD seed (Task 6).  Jadd then writes the exact analytic value.
    {
        auto zero_row = [&](int rrow) {
            for (int col = 0; col <= 4*N; ++col) J[(size_t)rrow*n + col] = 0.0;  // cols 0..4N (incl ℓ_in); skip 4N+1 (r_s)
        };
        for (int i = 0; i < N-1; ++i) { zero_row(2*N+i); zero_row(3*N-1+i); }
        zero_row(4*N-1);
    }

    // ∂(dln(f_a,f_b))/∂f_a, ∂/∂f_b on the log grid (Δlnr = ln r_b − ln r_a).
    // dln = (ln f_b − ln f_a)/Δlnr;  guard floors at 1e-300 zero the derivative there.
    auto dln_val = [&](double fa, double fb, int ia, int ib) {
        return (std::log(std::max(fb,1e-300)) - std::log(std::max(fa,1e-300)))
             / (std::log(r[ib]) - std::log(r[ia]));
    };

    // ---- State-dependent η₃/Γ̃₁ moments (refinement #11) + their ∂/∂{Σ_i,T_c,i} ----
    // β = clamp(p_gas/p_mid, 0, 1); η₃ = 3 − 1.5β; Γ̃₁ = 1 + 1/η₃.
    //   ∂β/∂x  = (dp_gas[x]·p_mid − p_gas·dp_mid[x]) / p_mid²       (raw, before floor-clamp)
    //   ∂η₃/∂x = −1.5·∂β/∂x ;  ∂Γ̃₁/∂x = (1.5/η₃²)·∂β/∂x = −(1/η₃²)·∂η₃/∂x
    // When β is clamped at 0 or 1, ∂β/∂x = 0 (matches the FD oracle inside the clamp).
    // The dbS[]/dbT[] multipliers (the Σ/T floor-clamp factors nj[i].dSig/dTce) are
    // applied by the caller to mirror the value's floor-clamp exactly.
    struct EtaMoments {
        double eta3, gtilde1;
        double deta3[2];    // ∂η₃/∂{Σ,T_c}  (raw — caller applies floor-clamp)
        double dgt1[2];     // ∂Γ̃₁/∂{Σ,T_c}
    };
    auto eta_moments = [&](int i) -> EtaMoments {
        const slim_detail::OneZoneState& oz = nj[i].e.oz;
        const slim_detail::OneZoneJac&  oj = nj[i].ozj;
        const double pmid = std::max(oz.p_mid, 1e-300);
        const double braw = oz.p_gas / pmid;                       // unclamped p_gas/p_mid
        const double b    = std::clamp(braw, 0.0, 1.0);
        const bool   clamped = (braw <= 0.0) || (braw >= 1.0);
        EtaMoments m;
        m.eta3    = eta3_of_beta(b);
        m.gtilde1 = gtilde1_of_beta(b);
        const double inv_e2 = 1.0 / (m.eta3 * m.eta3);
        for (int x = 0; x < 2; ++x) {
            const double dbeta = clamped ? 0.0
                : (oj.dp_gas[x] * oz.p_mid - oz.p_gas * oj.dp_mid[x]) / (oz.p_mid * oz.p_mid);
            m.deta3[x] = -1.5 * dbeta;                             // ∂η₃/∂x
            m.dgt1[x]  =  inv_e2 * (1.5 * dbeta);                  // ∂Γ̃₁/∂x = (1.5/η₃²)∂β/∂x
        }
        return m;
    };

    // ---- Qadv_geom(i,j) value + gradient (the §23 (2πr²/Ṁη₃)Q_adv, dimensionless) ----
    // Qadv_geom = K_q · (P_i/Σ_i) · [η₃ dlnP − (1+η₃) dlnΣ] ,
    //   K_q = (2π r_cm²/(Ṁ η₃))·(−Ṁ/(2π r_cm²))/c² = −1/(η₃ c²)   (r_cm cancels)
    // Depends on P_i,Σ_i (closure of node i) and on P_j,Σ_j via the dlnP,dlnΣ stencils.
    // grad arrays: g_i[4]={Σ,V,ell,Tc} for node i, g_j[4] for node j; only Σ,Tc enter.
    auto qadv_geom_jac = [&](int i, int j, double& val,
                             double gi[4], double gj[4]) {
        for (int k=0;k<4;++k){ gi[k]=0; gj[k]=0; }
        const NodeEval& a = nj[i].e; const NodeEval& b = nj[j].e;
        const double Pa = a.oz.P, Sa = a.Sigma, Pb = b.oz.P, Sb = b.Sigma;
        const double dlnP = dln_val(Pa, Pb, i, j);
        const double dlnS = dln_val(Sa, Sb, i, j);
        // η₃(β_i) node-local (#11): bracket coeffs kAdvP=η₃, kAdvS=1+η₃, and Kq=−1/(η₃c²).
        const EtaMoments m = eta_moments(i);
        const double eta3 = m.eta3;
        const double bracket = eta3 * dlnP - (1.0 + eta3) * dlnS;
        const double PoverS = Pa / Sa;
        const double Kq = -1.0 / (eta3 * c_cgs * c_cgs);
        val = Kq * PoverS * bracket;
        const double dlnr = std::log(r[j]) - std::log(r[i]);
        // ∂(P/Σ)_i wrt Σ_i,Tc_i via closure jac.
        const double dPoverS_dSi = (nj[i].ozj.dP[0] * Sa - Pa) / (Sa*Sa);   // = dP/dΣ /Σ − P/Σ²
        const double dPoverS_dTi = nj[i].ozj.dP[1] / Sa;
        // ∂bracket/∂{...} via dlnP,dlnΣ (each ∝ 1/f /Δlnr at its node).
        // dlnP: ∂/∂P_i = −1/(P_i Δlnr), ∂/∂P_j = +1/(P_j Δlnr); P_i=P(Σ_i,Tc_i), P_j=P(Σ_j,Tc_j).
        const double dP_dPi = -1.0/(Pa*dlnr), dP_dPj = 1.0/(Pb*dlnr);
        const double dS_dSi = -1.0/(Sa*dlnr), dS_dSj = 1.0/(Sb*dlnr);
        // ∂Kq/∂x_i = (1/(η₃²c²))·∂η₃/∂x  (raw; floor-clamp applied below via nj[i].dSig/dTce).
        const double invc2 = 1.0/(c_cgs*c_cgs);
        const double dKq_dSi = (1.0/(eta3*eta3))*invc2 * m.deta3[0];
        const double dKq_dTi = (1.0/(eta3*eta3))*invc2 * m.deta3[1];
        // ∂bracket/∂η₃ = (dlnP − dlnΣ); explicit-coefficient piece (η₃ is node-i-local).
        const double dbr_deta3 = dlnP - dlnS;
        // node i contributions: THREE pieces (∂Kq, ∂(P/Σ), ∂bracket incl. explicit η₃).
        const double dbr_dSi = eta3*dP_dPi*nj[i].ozj.dP[0] - (1.0+eta3)*dS_dSi*nj[i].dSig
                             + dbr_deta3*m.deta3[0]*nj[i].dSig;
        const double dbr_dTi = eta3*dP_dPi*nj[i].ozj.dP[1]
                             + dbr_deta3*m.deta3[1]*nj[i].dTce;
        gi[0] = dKq_dSi*nj[i].dSig*PoverS*bracket
              + Kq * (dPoverS_dSi*nj[i].dSig*bracket + PoverS*dbr_dSi);   // Σ_i
        gi[3] = dKq_dTi*nj[i].dTce*PoverS*bracket
              + Kq * (dPoverS_dTi*nj[i].dTce*bracket + PoverS*dbr_dTi);   // Tc_i
        // node j contributions (only through the stencils dlnP,dlnΣ; η₃=η₃(β_i) is
        // node-i-local so it contributes NOTHING to the j block).
        const double dbr_dSj = eta3*dP_dPj*nj[j].ozj.dP[0] - (1.0+eta3)*dS_dSj*nj[j].dSig;
        const double dbr_dTj = eta3*dP_dPj*nj[j].ozj.dP[1];
        gj[0] = Kq * PoverS * dbr_dSj;   // Σ_j
        gj[3] = Kq * PoverS * dbr_dTj;   // Tc_j
    };

    // ---- 𝒟₀(i) gradient: 𝒟₀ = V² − Γ̃₁(β)·(P/Σ)/c²  (Γ̃₁ node-local, #11) ----
    // ∂/∂x = −(1/c²)[ ∂Γ̃₁/∂x·(P/Σ) + Γ̃₁·∂(P/Σ)/∂x ]·(floor-clamp), x∈{Σ,Tc}.
    auto D0_jac = [&](int i, double g[4]) {
        for (int k=0;k<4;++k) g[k]=0;
        const NodeEval& a = nj[i].e;
        const double Sa = a.Sigma, Pa = a.oz.P, PoverS = Pa/Sa;
        g[1] = 2.0 * a.V * nj[i].dVe;                                       // ∂/∂V
        const double dPoverS_dS = (nj[i].ozj.dP[0]*Sa - Pa)/(Sa*Sa);
        const double dPoverS_dT = nj[i].ozj.dP[1]/Sa;
        const EtaMoments m = eta_moments(i);
        const double invc2 = 1.0/(c_cgs*c_cgs);
        g[0] = -invc2 * (m.dgt1[0]*PoverS + m.gtilde1*dPoverS_dS) * nj[i].dSig; // ∂/∂Σ
        g[3] = -invc2 * (m.dgt1[1]*PoverS + m.gtilde1*dPoverS_dT) * nj[i].dTce; // ∂/∂Tc
    };

    // ---- 𝒩₁(i; Qadv) gradient (excluding the Qadv part, which the caller adds) ----
    // 𝒩₁ = 𝒜(Ω(ℓ_i)) + Qadv_geom + press,  press = (P/Σ)/c²·r(r−M)/Δ·Γ̃₁.
    // Returns ∂𝒩₁/∂{Σ_i,V_i,ℓ_i,Tc_i} from 𝒜 and press only.
    auto N1_local_jac = [&](int i, double g[4]) {
        for (int k=0;k<4;++k) g[k]=0;
        const NodeEval& a = nj[i].e;
        const double M = in.mass, ri = a.r, Delta = std::max(a.mech.Delta, 1e-30);
        // 𝒜 via Ω(ℓ): ∂𝒜/∂ℓ = (∂𝒜/∂Ω)(∂Ω/∂ℓ).
        double A0, dA_dOm; script_A_dOmega(in, ri, a.mech, A0, dA_dOm);
        const double dOm_dl = domega_dell(in.mass, in.spin, ri, a.mech.Omega);
        g[2] += dA_dOm * dOm_dl;                                           // ∂𝒜/∂ℓ
        // press = (P/Σ)/c²·r(r−M)/Δ·Γ̃₁(β).  Product rule on (P/Σ)·Γ̃₁ (#11).
        const double Sa = a.Sigma, Pa = a.oz.P, PoverS = Pa/Sa;
        const double coef = ri*(ri-M)/Delta / (c_cgs*c_cgs);
        const double dPoverS_dS = (nj[i].ozj.dP[0]*Sa - Pa)/(Sa*Sa);
        const double dPoverS_dT = nj[i].ozj.dP[1]/Sa;
        const EtaMoments m = eta_moments(i);
        g[0] += coef * (dPoverS_dS*m.gtilde1 + PoverS*m.dgt1[0]) * nj[i].dSig; // ∂press/∂Σ
        g[3] += coef * (dPoverS_dT*m.gtilde1 + PoverS*m.dgt1[1]) * nj[i].dTce; // ∂press/∂Tc
    };

    // ---- radial-momentum rhs(i; neighbor) = (𝒩₁/𝒟₀)(1−V²) value + grads ----
    // grads into node i (local) AND node `nb` (via the Qadv stencil only).
    auto rhs_radial_jac = [&](int i, int nb, double& val, double gi[4], double gnb[4]) {
        for (int k=0;k<4;++k){ gi[k]=0; gnb[k]=0; }
        const NodeEval& a = nj[i].e;
        double D0 = calD0(a);
        const bool D0guard = !(std::abs(D0) > 1e-30);
        const double D0g = D0guard ? std::copysign(1e-30, D0==0?1.0:D0) : D0;
        double qv; double qgi[4], qgj[4];
        qadv_geom_jac(i, nb, qv, qgi, qgj);
        const double N1 = calN1(in, a, qv);
        const double oneMV2 = 1.0 - a.V*a.V;
        val = (N1 / D0g) * oneMV2;
        // 𝒩₁ local grad (𝒜+press) + Qadv part.
        double gN1[4]; N1_local_jac(i, gN1);
        gN1[0]+=qgi[0]; gN1[1]+=qgi[1]; gN1[2]+=qgi[2]; gN1[3]+=qgi[3];
        // 𝒟₀ local grad.
        double gD0[4]; D0_jac(i, gD0);
        // d(N1/D0)/dx = (gN1·D0 − N1·gD0)/D0² ; inside guard band D0 is frozen ⇒ gD0→0.
        const double invD0 = 1.0/D0g, invD0sq = 1.0/(D0g*D0g);
        for (int k=0;k<4;++k) {
            const double gd0 = D0guard ? 0.0 : gD0[k];
            double dratio = (gN1[k]*D0g - N1*gd0) * invD0sq;
            gi[k] = dratio * oneMV2;
        }
        // extra ∂/∂V from the explicit (1−V²) factor.
        gi[1] += (N1*invD0) * (-2.0*a.V*nj[i].dVe);
        // neighbour grads: only via Qadv stencil (Σ_nb, Tc_nb), through N1.
        gnb[0] = qgj[0] * invD0 * oneMV2;
        gnb[3] = qgj[3] * invD0 * oneMV2;
    };

    // ---- radial-momentum node-0 L'Hôpital rhs grads ----
    // rhs0 = (dN1/dD0)·(1−V0²), dN1=(N1_1−N1_0)/dlnr, dD0=(D0_1−D0_0)/dlnr.
    // Couples nodes 0 and 1 (each via 𝒩₁,𝒟₀ + the Qadv stencils qadv(0,1),qadv(1,0)).
    auto rhs_sonic0_jac = [&](double& val, double g0[4], double g1[4]) {
        for (int k=0;k<4;++k){ g0[k]=0; g1[k]=0; }
        double qv0, qg0i[4], qg0j[4];  qadv_geom_jac(0,1,qv0,qg0i,qg0j);   // N1_0: eval node0, nb node1
        double qv1, qg1i[4], qg1j[4];  qadv_geom_jac(1,0,qv1,qg1i,qg1j);   // N1_1: eval node1, nb node0
        const double N1_0 = calN1(in, nj[0].e, qv0), N1_1 = calN1(in, nj[1].e, qv1);
        const double D0_0 = calD0(nj[0].e), D0_1 = calD0(nj[1].e);
        const double dlnr = std::log(r[1]) - std::log(r[0]);
        const double dN1 = (N1_1 - N1_0)/dlnr;
        double dD0 = (D0_1 - D0_0)/dlnr;
        const bool dD0guard = std::abs(dD0) < 1e-30;
        if (dD0guard) dD0 = std::copysign(1e-30, dD0==0?-1.0:dD0);
        const double oneMV2 = 1.0 - nj[0].e.V*nj[0].e.V;
        val = (dN1/dD0)*oneMV2;
        // FOUR gradient blocks (eval-node ⊗ wrt-node):
        //   N1_0 wrt node0 = N1_local(0)+qg0i ;  N1_0 wrt node1 = qg0j
        //   N1_1 wrt node1 = N1_local(1)+qg1i ;  N1_1 wrt node0 = qg1j
        //   D0_0 wrt node0 = D0(0) ;             D0_1 wrt node1 = D0(1)  (each local only)
        double L0[4], L1[4], gD0_0[4], gD0_1[4];
        N1_local_jac(0, L0); N1_local_jac(1, L1);
        D0_jac(0, gD0_0);    D0_jac(1, gD0_1);
        double dN10_d0[4], dN10_d1[4], dN11_d0[4], dN11_d1[4];
        for (int k=0;k<4;++k){
            dN10_d0[k] = L0[k] + qg0i[k];   dN10_d1[k] = qg0j[k];
            dN11_d1[k] = L1[k] + qg1i[k];   dN11_d0[k] = qg1j[k];
        }
        const double invdD0sq = 1.0/(dD0*dD0);
        // ∂dN1/∂node0 = (dN11_d0 − dN10_d0)/dlnr ; ∂dN1/∂node1 = (dN11_d1 − dN10_d1)/dlnr.
        // ∂dD0/∂node0 = (−D0_0)/dlnr ; ∂dD0/∂node1 = (+D0_1)/dlnr (each local; guarded→0).
        for (int k=0;k<4;++k) {
            const double ddN1_0 = (dN11_d0[k] - dN10_d0[k])/dlnr;
            const double ddN1_1 = (dN11_d1[k] - dN10_d1[k])/dlnr;
            const double ddD0_0 = dD0guard?0.0:(-gD0_0[k])/dlnr;
            const double ddD0_1 = dD0guard?0.0:( gD0_1[k])/dlnr;
            g0[k] = (ddN1_0*dD0 - dN1*ddD0_0) * invdD0sq * oneMV2;
            g1[k] = (ddN1_1*dD0 - dN1*ddD0_1) * invdD0sq * oneMV2;
        }
        // explicit (1−V0²) factor ∂/∂V0.
        g0[1] += (dN1/dD0) * (-2.0*nj[0].e.V*nj[0].dVe);
    };

    // Assemble radial-momentum rows R[2N+i] = (lnV_{i+1}−lnV_i) − 0.5 dlnr (rhs_i+rhs_{i+1}).
    for (int i = 0; i < N - 1; ++i) {
        const int rrow = 2*N + i;
        const double dlnr = std::log(r[i+1]) - std::log(r[i]);
        // ∂(lnV_{i+1}−lnV_i)/∂V: ln|V| ⇒ ∂/∂V_i = −1/V_i, ∂/∂V_{i+1} = 1/V_{i+1} (V<0).
        Jadd(rrow, 4*i+1,     -(1.0/nj[i].e.V)   * nj[i].dVe);
        Jadd(rrow, 4*(i+1)+1, (1.0/nj[i+1].e.V) * nj[i+1].dVe);
        // rhs_i: node-0 L'Hôpital on the [0,1] interval; else direct.
        double rhs_i, gi_i[4], gnb_i[4];     // gi_i→node i, gnb_i→neighbour (i+1)
        if (i == 0) {
            double g0[4], g1[4]; double val; rhs_sonic0_jac(val, g0, g1);
            for (int k=0;k<4;++k){ gi_i[k]=g0[k]; gnb_i[k]=g1[k]; }
        } else {
            rhs_radial_jac(i, i+1, rhs_i, gi_i, gnb_i);
        }
        // rhs_{i+1}: direct, evaluated at node i+1 with neighbour i.
        double rhs_i1, gi_1[4], gnb_1[4];    // gi_1→node i+1, gnb_1→neighbour i
        rhs_radial_jac(i+1, i, rhs_i1, gi_1, gnb_1);
        const double c = -0.5 * dlnr;
        // scatter rhs_i grads (node i + neighbour i+1) and rhs_{i+1} grads (node i+1 + neighbour i).
        for (int k=0;k<4;++k) {
            Jadd(rrow, 4*i+k,     c * (gi_i[k]  + gnb_1[k]));
            Jadd(rrow, 4*(i+1)+k, c * (gnb_i[k] + gi_1[k]));
        }
    }

    // ---- energy G-balance(i; neighbour j) value + grads ----
    // G = Qvis − Qrad − Qadv.  Returns ∂G/∂{node i local 4}, ∂G/∂{node j: Σ,Tc,ℓ}, ∂G/∂ℓ_in.
    auto Gbalance_jac = [&](int i, int j, double gi[4], double gj[4], double& g_ellin) {
        for (int k=0;k<4;++k){ gi[k]=0; gj[k]=0; } g_ellin = 0;
        const NodeEval& a = nj[i].e; const NodeEval& b = nj[j].e;
        const double r_cm = a.r * in.r_g;
        const double dr = b.r - a.r;
        const double convOm = (c_cgs/in.r_g)/in.r_g;
        const double geomfac = a.mech.sqrtA / (std::max(a.mech.sqrtDelta, 1e-30) * a.r);
        const double dl_cgs = (a.ell - ell_in) * in.r_g * c_cgs;
        const double dOmega_dr = (b.mech.Omega - a.mech.Omega)/dr * convOm;
        // Qvis = −K·dl_cgs·dOmega_dr·Γ_a·(geomfac/r_cm),  K=Mdot/2π,
        // geomfac = A^½/(Δ^½r) (S09 Eq 6 × Eq 4; §23 corrected 2026-06-12).
        // (Local r_cm = a.r·r_g, matching Gbalance; r is not a state variable, so
        //  this is a pure prefactor — no extra derivative terms.)
        const double K = Mdot/twopi;
        const double Qvis_pref = -K * (geomfac/r_cm);
        // #12: a.Gamma is the FULL Γ²=1/(1−V²)+ℓ²r²/A — depends on BOTH V_a and ℓ_a.
        //   ∂Γ/∂V = V/((1−V²)²·Γ)  (NOT V·Γ³);  ∂Γ/∂ℓ = ℓr²/(A·Γ)  (NEW, ℓ_a only).
        const double oneMV2a = 1.0 - a.V*a.V;
        const double Aa = std::max(a.mech.A, 1e-300);
        const double dGa_dV = a.V / (oneMV2a*oneMV2a*a.Gamma);
        const double dGa_dl = a.ell * a.r * a.r / (Aa * a.Gamma);
        // ∂/∂ℓ_a: dl_cgs ∝ ℓ_a, Ω_a in dOmega_dr, AND Γ_a (#12).
        const double dOm_a_dla = domega_dell(in.mass,in.spin,a.r,a.mech.Omega);
        const double dOm_b_dlb = domega_dell(in.mass,in.spin,b.r,b.mech.Omega);
        const double ddOmdr_dOma = (-1.0/dr)*convOm, ddOmdr_dOmb = (1.0/dr)*convOm;
        gi[2] += Qvis_pref * ( (in.r_g*c_cgs)*dOmega_dr*a.Gamma
                              + dl_cgs*(ddOmdr_dOma*dOm_a_dla)*a.Gamma
                              + dl_cgs*dOmega_dr*dGa_dl );                   // ℓ_a (#12: via Γ)
        gj[2] += Qvis_pref * ( dl_cgs*(ddOmdr_dOmb*dOm_b_dlb)*a.Gamma );    // ℓ_b (Γ⊥ℓ_b)
        gi[1] += Qvis_pref * dl_cgs*dOmega_dr*dGa_dV*nj[i].dVe;             // V_a via Γ (#12)
        g_ellin += Qvis_pref * (-(in.r_g*c_cgs))*dOmega_dr*a.Gamma;          // ℓ_in (dl_cgs)
        // Qrad = 64σ T⁴/(3 κ_R Σ), κ_R=κ_R(ρ(Σ,T),T).
        const double rho = a.oz.rho_mid, Sa = a.Sigma, Ta = a.Tc;
        double kR, dkR_drho, dkR_dT; kappa_total_grad(op, rho, Ta, kR, dkR_drho, dkR_dT);
        const double kRs = std::max(kR, 1e-300);
        const double Qrad = 64.0*sigma_SB*Ta*Ta*Ta*Ta/(3.0*kRs*Sa);
        // ∂Qrad/∂Σ = Qrad·(−1/Σ − (1/κ_R)dκ/dρ·∂ρ/∂Σ)
        const double drho_dS = nj[i].ozj.drho[0], drho_dT = nj[i].ozj.drho[1];
        const double dQrad_dS = Qrad * ( -1.0/Sa - (dkR_drho*drho_dS)/kRs );
        const double dQrad_dT = Qrad * ( 4.0/Ta  - (dkR_drho*drho_dT + dkR_dT)/kRs );
        gi[0] += -dQrad_dS * nj[i].dSig;   // G = ... − Qrad
        gi[3] += -dQrad_dT * nj[i].dTce;
        // Qadv = −(Mdot/2π r_cm²)(P/Σ)[η₃(β_i) dlnP − (1+η₃(β_i)) dlnΣ]  (CGS, NOT /c²).
        // η₃ node-local (#11); Kc has NO η₃, so only the bracket gains ∂η₃ pieces. gj unchanged.
        const double Pa = a.oz.P, Pb = b.oz.P, Sb = b.Sigma;
        const double dlnP = dln_val(Pa,Pb,i,j), dlnS = dln_val(Sa,Sb,i,j);
        const EtaMoments mE = eta_moments(i);
        const double eta3 = mE.eta3;
        const double bracket = eta3*dlnP - (1.0+eta3)*dlnS;
        const double dbr_deta3 = dlnP - dlnS;           // ∂bracket/∂η₃
        const double Kc = -(Mdot/(twopi*r_cm*r_cm));
        const double PoverS = Pa/Sa;
        const double dlnr = std::log(r[j]) - std::log(r[i]);
        const double dPoverS_dS = (nj[i].ozj.dP[0]*Sa - Pa)/(Sa*Sa);
        const double dPoverS_dT = nj[i].ozj.dP[1]/Sa;
        const double dP_dPi=-1.0/(Pa*dlnr), dP_dPj=1.0/(Pb*dlnr);
        const double dS_dSi=-1.0/(Sa*dlnr), dS_dSj=1.0/(Sb*dlnr);
        const double dbr_dSi=eta3*dP_dPi*nj[i].ozj.dP[0]-(1.0+eta3)*dS_dSi*nj[i].dSig
                            + dbr_deta3*mE.deta3[0]*nj[i].dSig;
        const double dbr_dTi=eta3*dP_dPi*nj[i].ozj.dP[1]
                            + dbr_deta3*mE.deta3[1]*nj[i].dTce;
        const double dbr_dSj=eta3*dP_dPj*nj[j].ozj.dP[0]-(1.0+eta3)*dS_dSj*nj[j].dSig;
        const double dbr_dTj=eta3*dP_dPj*nj[j].ozj.dP[1];
        const double dQadv_dSi = Kc*(dPoverS_dS*nj[i].dSig*bracket + PoverS*dbr_dSi);
        const double dQadv_dTi = Kc*(dPoverS_dT*nj[i].dTce*bracket + PoverS*dbr_dTi);
        const double dQadv_dSj = Kc*PoverS*dbr_dSj;
        const double dQadv_dTj = Kc*PoverS*dbr_dTj;
        gi[0] += -dQadv_dSi; gi[3] += -dQadv_dTi;   // G = ... − Qadv
        gj[0] += -dQadv_dSj; gj[3] += -dQadv_dTj;
    };

    // Assemble energy rows R[3N-1+i] = 0.5(G_i + G_{i+1}),  G_i=G(i;i+1), G_{i+1}=G(i+1;i).
    for (int i = 0; i < N - 1; ++i) {
        const int rrow = 3*N - 1 + i;
        double gi_a[4], gj_a[4], el_a; Gbalance_jac(i,   i+1, gi_a, gj_a, el_a);  // G_i
        double gi_b[4], gj_b[4], el_b; Gbalance_jac(i+1, i,   gi_b, gj_b, el_b);  // G_{i+1}
        for (int k=0;k<4;++k) {
            Jadd(rrow, 4*i+k,     0.5*(gi_a[k] + gj_b[k]));
            Jadd(rrow, 4*(i+1)+k, 0.5*(gj_a[k] + gi_b[k]));
        }
        Jadd(rrow, 4*N+0, 0.5*(el_a + el_b));   // ℓ_in column
    }

    // Outer-energy BC row R[4N-1] = G(last; last-1).
    {
        const int last = N-1, rrow = 4*N - 1;
        double gi_l[4], gj_l[4], el_l; Gbalance_jac(last, last-1, gi_l, gj_l, el_l);
        for (int k=0;k<4;++k) {
            Jadd(rrow, 4*last+k,     gi_l[k]);
            Jadd(rrow, 4*(last-1)+k, gj_l[k]);
        }
        Jadd(rrow, 4*N+0, el_l);
    }

    // =======================================================================
    // Task 6: angmom ℓ_in column + outer-ℓ BC row + regularity rows.
    // =======================================================================
    // Angular-momentum rows' ℓ_in column: R[N+i] = (Ṁ/2π)(ℓ_i−ℓ_in)r_g c − rhs.
    //   ∂R[N+i]/∂ℓ_in = −(Ṁ/2π) r_g c.
    for (int i = 0; i < N; ++i)
        J[(size_t)(N+i)*n + 4*N+0] = -(Mdot/twopi) * in.r_g * c_cgs;

    // Outer-ℓ BC row R[4N-2] = ℓ_last − ℓ_extrap, ℓ_extrap a CUBIC Newton-divided-
    // difference in ln r of ℓ at nodes last-1..last-4 (LINEAR in those ℓ values).
    //   ℓ_extrap = f0 + (x−x0)d01 + (x−x0)(x−x1)d012 + (x−x0)(x−x1)(x−x2)d0123,
    //   x_k = ln r[last-1-k], x = ln r[last]; f_k = ℓ[last-1-k].  Divided differences
    //   are linear in {f0,f1,f2,f3}, so ∂ℓ_extrap/∂f_k are constants (∂ wrt ℓ only).
    {
        const int last = N-1, rrow = 4*N - 2;
        const double x0=std::log(r[last-1]), x1=std::log(r[last-2]),
                     x2=std::log(r[last-3]), x3=std::log(r[last-4]), x=std::log(r[last]);
        // ℓ_extrap = Σ_k c_k f_k.  Build c_k by differentiating the divided-difference
        // tree (linear): d01=(f0−f1)/(x0−x1), d12=(f1−f2)/(x1−x2), d23=(f2−f3)/(x2−x3),
        // d012=(d01−d12)/(x0−x2), d123=(d12−d23)/(x1−x3), d0123=(d012−d123)/(x0−x3).
        const double w01a= 1.0/(x0-x1),  w01b=-1.0/(x0-x1);
        const double w12a= 1.0/(x1-x2),  w12b=-1.0/(x1-x2);
        const double w23a= 1.0/(x2-x3),  w23b=-1.0/(x2-x3);
        // d012 = (d01 − d12)/(x0−x2):  wrt f0: w01a/(x0-x2); f1:(w01b−w12a)/(x0-x2); f2:(−w12b)/(x0-x2)
        const double i02=1.0/(x0-x2);
        const double d012_f0=w01a*i02, d012_f1=(w01b-w12a)*i02, d012_f2=(-w12b)*i02;
        // d123 = (d12 − d23)/(x1−x3): f1: w12a/(x1-x3); f2:(w12b−w23a)/(x1-x3); f3:(−w23b)/(x1-x3)
        const double i13=1.0/(x1-x3);
        const double d123_f1=w12a*i13, d123_f2=(w12b-w23a)*i13, d123_f3=(-w23b)*i13;
        // d0123 = (d012 − d123)/(x0−x3)
        const double i03=1.0/(x0-x3);
        const double d0123_f0=d012_f0*i03, d0123_f1=(d012_f1-d123_f1)*i03,
                     d0123_f2=(d012_f2-d123_f2)*i03, d0123_f3=(-d123_f3)*i03;
        const double t1=(x-x0), t2=(x-x0)*(x-x1), t3=(x-x0)*(x-x1)*(x-x2);
        // c_k = ∂ℓ_extrap/∂f_k.
        double cf[4];
        cf[0]= 1.0 + t1*w01a + t2*d012_f0 + t3*d0123_f0;                 // f0 = ℓ[last-1]
        cf[1]=       t1*w01b + t2*d012_f1 + t3*d0123_f1;                 // f1 = ℓ[last-2]
        cf[2]=                 t2*d012_f2 + t3*d0123_f2;                 // f2 = ℓ[last-3]
        cf[3]=                             t3*d0123_f3;                  // f3 = ℓ[last-4]
        // R = ℓ_last − ℓ_extrap ⇒ ∂/∂ℓ_last = 1, ∂/∂ℓ[last-1-k] = −cf[k].
        J[(size_t)rrow*n + 4*last+2] = 1.0;
        for (int k=0;k<4;++k) J[(size_t)rrow*n + 4*(last-1-k)+2] = -cf[k];
    }

    // Regularity rows: R[4N] = 𝒟₀(node0) ; R[4N+1] = 𝒩₁(node0; qadv(0,1)).
    {
        double gD0[4]; D0_jac(0, gD0);
        for (int k=0;k<4;++k) J[(size_t)(4*N)*n + 4*0+k] = gD0[k];     // only node-0 local
        double gN1[4]; N1_local_jac(0, gN1);
        double qv, qgi[4], qgj[4]; qadv_geom_jac(0,1,qv,qgi,qgj);       // node0 eval, nb node1
        for (int k=0;k<4;++k) {
            J[(size_t)(4*N+1)*n + 4*0+k] = gN1[k] + qgi[k];             // node 0
            J[(size_t)(4*N+1)*n + 4*1+k] = qgj[k];                      // node 1 (Qadv stencil)
        }
    }

    // -----------------------------------------------------------------------
    // The r_s grid-stretch column 4N+1.
    //   r_i = r_s^{1−t_i} r_out^{t_i}  ⇒  ∂r_i/∂r_s = (1−t_i) r_i/r_s,  so
    //   ∂R/∂r_s = Σ_i (∂R/∂r_i)(1−t_i)r_i/r_s.
    // The production FD computes this column by perturbing r_s and RE-SPACING every
    // node at once (its tiny per-type step compounds noise across the near-sonic node
    // — the FD Jacobian's least-accurate column, the cold-seed regularity wall).  We
    // replace it with a WELL-CONDITIONED, SMOOTH column: a moderate-step central
    // difference of the SAME grid-stretch with Richardson extrapolation cancelling the
    // O(h²) truncation.  This is smooth where the production FD is noisy (the whole
    // point); it is validated against the Richardson reference to the looser r_s tol.
    // (A fully closed-form ∂R/∂r_i chain through every r-dependent mechanics/geomfac/
    // dlnr factor is deferred; the smooth Richardson column already removes the FD-
    // noise that stalled the free sonic node.)
    {
        const int rs_col = 4*N + 1;
        const double r_s = U[4*N+1];
        const double h = std::max(2e-4 * std::abs(r_s), 1e-30);
        std::vector<double> Rp, Rm;
        auto cd = [&](double step, std::vector<double>& out) {
            std::vector<double> Up=U, Um=U; Up[rs_col]+=step; Um[rs_col]-=step;
            slim_radial_residual(Up, in, op, Rp);
            slim_radial_residual(Um, in, op, Rm);
            out.assign(n,0.0); const double inv=1.0/(2.0*step);
            for (int rr=0; rr<n; ++rr) out[rr]=(Rp[rr]-Rm[rr])*inv;
        };
        std::vector<double> c1, c2; cd(h,c1); cd(0.5*h,c2);
        for (int rr=0; rr<n; ++rr)
            J[(size_t)rr*n + rs_col] = (4.0*c2[rr]-c1[rr])/3.0;
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

// ===========================================================================
// Task 1 (arclength continuation): ∂R/∂Ṁ column  (R_Mdot, length 4N+2).
// ===========================================================================
// Pseudo-arclength continuation promotes the accretion rate Ṁ (= in.mdot) to a
// CONTINUATION UNKNOWN, so the bordered augmented system needs the residual's
// sensitivity to Ṁ at FIXED state U.  Ṁ enters slim_radial_residual ONLY through
// the local `Mdot = in.mdot` (it is the SAME scalar everywhere the residual reads
// it), via:
//   • the mass rows  R[i] = mdot_of_node(Σ_i,V_i,√Δ_i) − Ṁ      ⇒ ∂R[i]/∂Ṁ = −1,
//   • the Q_vis / Q_adv prefactors in the energy rows, the radial-momentum 𝒩₁
//     (its (2πr²/(Ṁη₃))Q_adv term — note Ṁ cancels analytically there), and the
//     viscous/advective heating of the outer-energy BC + the 𝒩₁ regularity row.
// Because it is ONE column we use a central finite difference on in.mdot — cheap
// and adequate (the plan explicitly permits FD here; the EXACT analytic Jacobian
// stays in the augmented system's J block, where FD noise at the fold is the issue
// the analytic Jacobian was built to remove).  The mass-row entries come out at the
// exact analytic −1 (the FD of a term linear in Ṁ is exact to round-off); the
// energy/𝒩₁ rows pick up their Ṁ-prefactor sensitivities.
//
// Returns R_Mdot[row] = ∂R[row]/∂Ṁ.  Uses a relative step on in.mdot with an
// absolute floor so a tiny Ṁ never collapses the step to round-off.
static void slim_R_Mdot_column(const std::vector<double>& U, const SlimDiskInputs& in,
                               const OpacityLUTs& op, std::vector<double>& R_Mdot) {
    const int N = std::max(in.n_nodes, 4);
    const int n = 4 * N + 2;
    const double h = std::max(1e-6 * std::abs(in.mdot), 1e-300);
    SlimDiskInputs in_p = in, in_m = in;
    in_p.mdot = in.mdot + h;
    in_m.mdot = in.mdot - h;
    std::vector<double> Rp, Rm;
    slim_radial_residual(U, in_p, op, Rp);
    slim_radial_residual(U, in_m, op, Rm);
    R_Mdot.assign(n, 0.0);
    const double inv = 1.0 / (2.0 * h);
    for (int row = 0; row < n; ++row) R_Mdot[row] = (Rp[row] - Rm[row]) * inv;
}

// ===========================================================================
// Task 2 (arclength continuation): bordered augmented system + tangent.
// ===========================================================================
// The continuation unknown is W = (U, Ṁ), length m = 4N+3 (U is the full 4N+2
// state INCLUDING ℓ_in at 4N and r_s at 4N+1, plus Ṁ as the last component).  The
// residual map R(W) has 4N+2 rows; its Jacobian w.r.t. W is the BORDERED matrix
//   Jw = [ J  |  R_Mdot ]            ( (4N+2) × (4N+3) )
// with J = slim_analytic_jacobian (the EXACT analytic Jacobian — the enabler) and
// R_Mdot = ∂R/∂Ṁ (Task 1).
//
// TANGENT (U̇, Ṁ̇):  the null vector of Jw, i.e. J·U̇ + R_Mdot·Ṁ̇ = 0, normalized to
// ‖U̇‖²+Ṁ̇²=1 and oriented to continue the previous direction.  At a fold J is
// SINGULAR (that is the turning point), so we never invert J alone.  Instead we
// solve the SQUARE bordered system
//   [[ J,        R_Mdot ],   [ U̇  ]   [ 0 ]
//    [ t_prevᵀ,  ṁ_prev ]] · [ Ṁ̇ ] = [ 1 ]
// whose last row is the previous tangent (a generic vector for the first call).
// This augmented matrix stays NONSINGULAR through the fold (Keller's theorem), so a
// single dense_solve yields the raw tangent; we then normalize and orient it.  The
// sign flip of Ṁ̇ at the fold (Ṁ decreasing through the unstable segment) is exactly
// the mechanism that lets the trace go AROUND the turning point.
//
// `t_prev` (length m) seeds the bordering row + the orientation; pass the previous
// accepted tangent.  For the FIRST tangent pass a vector dominated by the Ṁ̇
// component (e.g. all-zero except the last = 1) so the initial direction increases
// Ṁ.  Returns false only if the bordered system is genuinely singular (degenerate).
//
// SCALED normalization: the raw state spans ~33 decades (Σ~1e4, V~1e-6, T_c~1e6,
// Ṁ~1e18), so a RAW unit tangent would be dominated by the largest component and the
// arclength step would barely move the physically important variables (incl. Ṁ).  We
// therefore normalize in a SCALED metric: ‖t‖²_w ≡ Σ (t_i / w_i)² = 1, with w_i = the
// per-variable column magnitude (the SAME non-dimensionalization the Newton solve
// uses).  The returned tangent is in RAW component units but unit-normed under w; the
// caller's predictor and Keller arclength row use the SAME weights w (output param) so
// the parametrization is balanced across all components including Ṁ.
static void slim_arclength_weights(const std::vector<double>& U, double Mdot,
                                   int N, std::vector<double>& w) {
    const int n = 4 * N + 2, m = n + 1;
    double mSig=0,mV=0,mEll=0,mT=0;
    for (int i=0;i<N;++i){ mSig=std::max(mSig,std::abs(U[4*i+0])); mV=std::max(mV,std::abs(U[4*i+1]));
                           mEll=std::max(mEll,std::abs(U[4*i+2])); mT=std::max(mT,std::abs(U[4*i+3])); }
    mSig=std::max(mSig,1e-30); mV=std::max(mV,1e-30); mEll=std::max(mEll,1e-30); mT=std::max(mT,1.0);
    w.assign(m,1.0);
    // The arclength norm is ‖t‖²_w = Σ_state (t_i/w_i)² + (t_Ṁ/w_Ṁ)².  There are n≈4N
    // STATE DOF but only ONE Ṁ DOF, so if each state weight = its bare magnitude the
    // state sum (n terms) swamps the single Ṁ term and the continuation barely advances
    // in Ṁ (it wiggles the structure at fixed f_Edd and the Ṁ̇ orientation goes sign-
    // noisy).  We INFLATE the state weights by a factor C·√n so the AGGREGATE state
    // contribution becomes COMPARABLE to (slightly below) the single Ṁ term: then Ṁ
    // carries roughly HALF the arclength metric and each Keller step makes genuine Ṁ
    // progress (Δf_Edd ~ a few % per unit ds), while the state still carries enough of
    // the metric to let the tangent ROTATE through the turning point (the whole point of
    // arclength vs Ṁ-marching).  C>1 because the tangent concentrates in a FEW critical
    // state directions, so the aggregate state norm exceeds the equal-spread mean; C≈4
    // empirically puts the Ṁ fraction near 0.5 at the slim-disk operating points.
    const double sw = 20.0 * std::sqrt((double)n);   // state-weight inflation (C·√n)
    for (int i=0;i<N;++i){ w[4*i+0]=mSig*sw; w[4*i+1]=mV*sw; w[4*i+2]=mEll*sw; w[4*i+3]=mT*sw; }
    w[4*N+0]=std::max(std::abs(U[4*N+0]),1e-30)*sw;   // ℓ_in
    w[4*N+1]=std::max(std::abs(U[4*N+1]),1e-30)*sw;   // r_s
    w[n]    =std::max(std::abs(Mdot),1e-300);          // Ṁ (single DOF, no inflation)
}
static bool slim_arclength_tangent(const std::vector<double>& U, const SlimDiskInputs& in,
                                   const OpacityLUTs& op, const std::vector<double>& t_prev,
                                   std::vector<double>& tangent) {
    const int N = std::max(in.n_nodes, 4);
    const int n = 4 * N + 2;     // residual / U length
    const int m = n + 1;         // augmented unknown length (U + Ṁ)

    std::vector<double> J, R_Mdot;
    slim_analytic_jacobian(U, in, op, J);
    slim_R_Mdot_column(U, in, op, R_Mdot);

    // Build the bordered (m×m) matrix A and rhs b = [0…0, 1].
    //   rows 0..n-1 : [ J[row][:]  R_Mdot[row] ]
    //   row  n      : [ t_prev[0..n-1]          t_prev[n] ]
    std::vector<double> A((size_t)m * m, 0.0), b(m, 0.0);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) A[(size_t)row * m + col] = J[(size_t)row * n + col];
        A[(size_t)row * m + n] = R_Mdot[row];
    }
    for (int col = 0; col < m; ++col) A[(size_t)n * m + col] = t_prev[col];
    b[n] = 1.0;

    if (!dense_solve(A, b, m)) return false;   // genuinely degenerate

    // b now holds the raw (un-normalized) tangent.  Normalize in the SCALED metric
    // ‖t‖²_w = Σ (t_i/w_i)² = 1 so all components contribute comparably.
    std::vector<double> w; slim_arclength_weights(U, in.mdot, N, w);
    double nrm = 0.0; for (int i=0;i<m;++i){ const double s=b[i]/w[i]; nrm += s*s; }
    nrm = std::sqrt(nrm);
    if (!(nrm > 0.0)) return false;
    for (double& v : b) v /= nrm;

    // Orient to continue the previous direction (scaled dot prev·new > 0).  For the
    // first call t_prev seeds the direction, so this aligns the very first tangent.
    double dot = 0.0; for (int i = 0; i < m; ++i) dot += (t_prev[i]/w[i]) * (b[i]/w[i]);
    if (dot < 0.0) for (double& v : b) v = -v;

    tangent.swap(b);
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
    // T_c-outlier factor (S-CURVE BRANCH test, added for the node-9 stall).  A node
    // can sit on the WRONG ROOT of the thermal S-curve fold — cool/gas-dominated
    // where its neighbours are hot/radiation-dominated, or vice versa — WITHOUT its Σ
    // moving far enough to trip kOutlierFac (the observed stall has Σ only 1.45× off
    // the local median but T_c 2.3× off).  Σ alone therefore cannot detect a branch
    // flip; T_c can, because the two roots of the fold are separated in temperature
    // by construction.  Threshold measured from the stalled base-rung iterates
    // (a=0.9, f_Edd=1e-3, N=18): over a ±3-node window the LEGITIMATE nodes deviate
    // from the window log-median by at most 1.45× (node 3 high, node 1 low — the
    // steep inner rise), while the branch-flipped node sits at 2.23–2.37×.  2.0×
    // separates the two populations with margin on both sides and fires on no other
    // node; the exact log-space midpoint of the gap is 1.80×, so 2.0 is the
    // conservative (fewer-false-positives) end of the admissible band.
    constexpr double kOutlierFacT = 2.0;
    const bool kDiagDeglitch = std::getenv("SLIM_DIAG") != nullptr;

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
        const double Tc      = std::max(U[4*i+3], kTFloor);
        const double med_lnT = local_median(i, 3);          // smooth-trend log-T_c
        const double lnfacT  = std::log(kOutlierFacT);
        const bool   bad_S   = std::abs(std::log(Sc) - med_lnS) > lnfac;
        const bool   bad_T   = std::abs(std::log(Tc) - med_lnT) > lnfacT;
        if (bad_S || bad_T) {
            // On the wrong branch: project Σ and T_c back to the local smooth trend,
            // then re-derive V from exact mass conservation. No magic profile.
            // Same repair for both triggers — a T_c-only repair would leave Σ and V
            // on the old root, i.e. thermally hot/cool but hydrostatically still the
            // other branch, which the very next Newton step would simply undo.
            const double Snew = std::exp(med_lnS);
            const double Tnew = std::exp(med_lnT);
            U[4*i+0] = std::max(Snew, kSigmaFloor);
            U[4*i+3] = std::max(Tnew, kTFloor);
            U[4*i+1] = Vfrom(i, U[4*i+0]);
            if (kDiagDeglitch)
                std::printf("[DEGLITCH] node=%d trig=%s%s Sigma %.4e -> %.4e (x%.2f)"
                            "  T_c %.4e -> %.4e (x%.2f)\n",
                            i, bad_S ? "S" : "", bad_T ? "T" : "",
                            Sc, U[4*i+0], Sc / std::max(U[4*i+0], 1e-300),
                            Tc, U[4*i+3], Tc / std::max(U[4*i+3], 1e-300));
            ++nrepaired;
        }
    }
    return nrepaired;
}

// ---------------------------------------------------------------------------
// f_adv blow-up check, shared by the validity gate and the corrector line search.
// Recomputes the advection fraction f_adv = Qadv/Qrad node-by-node from the SAME
// closure path as unpack_profile (one_zone_closure + node-pair dlnP/dlnS).  Returns
// true iff EVERY node has Qrad>0 (radiation not throttled to ~0) AND |f_adv|<cap.
// This rejects the SPURIOUS arclength root whose advection fraction explodes
// (observed f_adv≈−1130) without rejecting the genuine slim branch (f_adv~O(1)).
// `absmax_out` (optional) receives max|f_adv| over the nodes for diagnostics.
static bool slim_fadv_ok(const SlimDiskInputs& in, const OpacityLUTs& opacity,
                         const std::vector<double>& U, double cap,
                         double* absmax_out = nullptr) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);
    std::vector<double> rgrid(N);
    {
        const double lr0 = std::log(in.r_in), lr1 = std::log(in.r_out);
        for (int i = 0; i < N; ++i) {
            const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
            rgrid[i] = std::exp(lr0 + (lr1 - lr0) * t);
        }
    }
    const double Mdot_s = in.mdot;
    auto dln = [&](double f_lo, double f_hi, double r_lo, double r_hi) {
        return (std::log(std::max(f_hi, 1e-300)) - std::log(std::max(f_lo, 1e-300)))
             / (std::log(r_hi) - std::log(r_lo));
    };
    bool ok = true;
    double absmax = 0.0;
    for (int i = 0; i < N; ++i) {
        const double r   = rgrid[i];
        const double Sig = U[4*i+0], Tc = U[4*i+3];
        const OneZoneState oz = one_zone_closure(std::max(Sig, kSigmaFloor),
                                                 std::max(Tc, kTFloor), r, in, opacity);
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const OneZoneState ozj = one_zone_closure(std::max(U[4*j+0], kSigmaFloor),
                                                  std::max(U[4*j+3], kTFloor), rgrid[j], in, opacity);
        const double dlnP = dln(oz.P, ozj.P, r, rgrid[j]);
        const double dlnS = dln(Sig,  U[4*j+0], r, rgrid[j]);
        const double r_cm = r * in.r_g;
        const double eta3_e = eta3_of_beta(beta_of(oz));
        const double Qadv = -(Mdot_s / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (oz.P / std::max(Sig, kSigmaFloor))
                          * (eta3_e * dlnP - (1.0 + eta3_e) * dlnS);
        const double rho_mid = oz.rho_mid;
        const double kR = opacity.lookup_kappa_ross(rho_mid, Tc) + opacity.lookup_kappa_es(rho_mid, Tc);
        const double Qrad = 64.0 * sigma_SB * Tc*Tc*Tc*Tc
                          / (3.0 * std::max(kR, 1e-300) * std::max(Sig, kSigmaFloor));
        const double f_adv = Qadv / std::max(std::abs(Qrad), 1e-300);
        absmax = std::max(absmax, std::abs(f_adv));
        // HARD physical requirement (Task 1): Q_rad = Q_vis/(1+f_adv), so 1+f_adv must
        // stay bounded AWAY from 0 — else Q_rad → ±∞ (radiation throttled / sign-flip).
        // The torus artifact has f_adv → −1 (1+f_adv → 0); a real slim disk has
        // f_adv ~ +0.3 (1+f_adv ~ 1.3). kEpsFadv=0.05 is far below any physical slim
        // value yet rejects the f_adv→−1 singularity the |f_adv|<cap test misses.
        constexpr double kEpsFadv = 0.05;
        if (!(Qrad > 0.0) || !(std::abs(f_adv) < cap) || !((1.0 + f_adv) > kEpsFadv))
            ok = false;
    }
    if (absmax_out) *absmax_out = absmax;
    return ok;
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
         rs_ok = false, smooth_ok = false, fadv_ok = false;
    double mass_maxrel = 0.0, D0_scaled = 0.0, N1_scaled = 0.0, sigma_max_jump = 0.0;
    double r_s = 0.0, r_isco = 0.0, fadv_absmax = 0.0;
    bool all(bool require_N1) const {
        return mass_ok && sign_ok && reg_D0_ok && rs_ok && smooth_ok && fadv_ok
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
    // f_adv physical validity (BLOW-UP gate, NOT a tight band).  The pseudo-arclength
    // corrector can converge a SPURIOUS root whose advection fraction explodes
    // (observed f_adv≈−1130): Qrad collapses toward zero so f_adv=Qadv/Qrad diverges.
    // We reject ONLY that blow-up signature — require Qrad>0 (not throttled to ~0) at
    // every node AND |f_adv|<kFadvCap.  The slim branch is advection-DOMINATED with
    // f_adv~O(1), so the cap is GENEROUS (50): O(1) physical states pass cleanly while
    // the −1130 garbage is rejected.  f_adv is computed from the SAME closure path as
    // unpack_profile (Qadv/Qrad via one_zone_closure + the node-pair dlnP/dlnS).
    constexpr double kFadvCap = 50.0;    // |f_adv| blow-up cap (generous: slim is f_adv~O(1))
    v.fadv_ok = slim_fadv_ok(in, opacity, U, kFadvCap, &v.fadv_absmax);
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
        // SAFETY BUDGET: this is the one place inner Newton iterations are spent.
        // Count it, then bail honestly (return non-converged) if the cumulative
        // inner-iter or wall-clock cap is hit — the bracket/spin/Ṁ ladders see the
        // tripped budget and short-circuit to the honest empty fallback up top.
        if (g_budget) {
            ++g_budget->inner_iters;
            if (g_budget->check()) {
                if (kDiag)
                    std::printf("[INNER] it=%d BUDGET EXCEEDED (%s) -> abort relaxation\n",
                                it, g_budget->what ? g_budget->what : "?");
                break;
            }
        }

        // 2a) ANALYTIC Jacobian (full n×n) — exact ∂R/∂U, validated block-by-block
        // against slim_numerical_jacobian (the permanent FD oracle; tests/
        // test_slim_jacobian.cpp).  Quadratic convergence + resolves the two
        // FD-precision-limited blocks (the near-rank-deficient (Σ,V)/mass block and
        // the r_s grid-stretch column).  slim_numerical_jacobian is RETAINED as the
        // cross-check oracle (set SLIM_FD_JAC=1 to fall back to it for an A/B check).
        // We gather the ACTIVE submatrix below.
        if (std::getenv("SLIM_FD_JAC")) slim_numerical_jacobian(U, in, opacity, J);
        else                            slim_analytic_jacobian(U, in, opacity, J);

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
        const double eta3_e = eta3_of_beta(beta_of(oz));     // node-local η₃(β) (#11)
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (oz.P / std::max(Sig, kSigmaFloor))
                          * (eta3_e * dlnP - (1.0 + eta3_e) * dlnS);    // [erg/cm²/s]
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
        // Budget tripped inside relax_structure: report this sample as non-converged
        // (g=NaN) so the bracket stops trying and the caller hits budget_fallback.
        if (g_budget && g_budget->tripped) { g = std::nan(""); return false; }
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

// ---------------------------------------------------------------------------
// Single-(a, Ṁ) solve: outer ℓ_in bracket (wrapping the inner relaxation) for
// ONE fixed spin/Ṁ corner, with an intermediate physical-validity gate.
// ---------------------------------------------------------------------------
// Factored out so BOTH continuation walks (spin-walk Phase B and the existing
// Ṁ-continuation) drive each rung through the identical bracket+gate logic. U is
// in-out (warm-started by the caller). `require_N1` selects whether the validity
// gate also demands the 𝒩₁(r_s)≈0 sonic-regularity check: the outer bracket drives
// 𝒩₁→0 as the eigenvalue, so the FINAL target uses require_N1=true; the intermediate
// continuation rungs use require_N1=false (𝒩₁ at the bracket's g-tol is sufficient to
// keep the warm start in-basin, and forcing it here would reject otherwise-valid rungs
// whose 𝒩₁ sits just above the validity band on the coarser intermediate grids).
// Returns true iff the bracket closed AND the validity gate passes.
static bool solve_single_am(const SlimDiskInputs& in, const OpacityLUTs& opacity,
                            std::vector<double>& U, bool require_N1) {
    if (!solve_outer_bracket(in, opacity, U)) return false;
    const ValidityResult v = slim_validity_gate(in, opacity, U, require_N1);
    return v.all(require_N1);
}

// ---------------------------------------------------------------------------
// Warm-start re-projection across spins (Phase B crux).
// ---------------------------------------------------------------------------
// Given a CONVERGED state U_old solved at spin a_old on grid [r_s_old, r_out],
// produce a warm U_new on the NEW spin a_new's grid [r_s_new, r_out].  The two
// grids differ because the ISCO/sonic point marches inward with spin, so the inner
// structure shifts; a small spin step moves it only slightly and the re-projected
// warm start stays in the inner relaxation's basin.
//
// Construction (no fabricated profile — a fresh thin-disk seed at a_new provides the
// physically-consistent inner structure + sonic node, and the converged old solution
// is interpolated in where the two grids overlap):
//   • BASE = build_thin_disk_seed(in_new, op): the crude-but-physically-consistent
//     thin-disk profile on the a_new grid [r_s_new, r_out].  It already carries the
//     correct r_s_new = 0.98·ISCO(a_new), the node-0 Mach-1 sonic override, the §23-
//     consistent energy/angular-momentum-balanced inner annulus, and its own de-glitch.
//     This gives a VALID inner structure (sonic node + the newly-exposed annulus) at
//     the new spin — the part the converged old solution can't supply.
//   • WARM OVERLAY: for every a_new-grid node with r_i ≥ r_s_old (the radii both grids
//     share), OVERWRITE Σ,ℓ,T_c by LOG-INTERPOLATING the CONVERGED old solution onto
//     r_i (linear in ln r), then re-derive V from mass conservation at a_new (so V is
//     consistent with the new √Δ).  The bulk of the disk thus starts from the converged
//     previous-spin profile — the whole point of the warm start — while the few inner
//     nodes with r_i < r_s_old keep the fresh thin-disk seed (which built them
//     consistently with the sonic transition).  The node-0 sonic node always keeps the
//     fresh seed's Mach-1 override.
//   • ℓ_in carried from U_old, nudged toward ℓ_K(isco(a_new)) (a warm eigenvalue; the
//     outer bracket refines it).
//   • deglitch_sigma_outliers on the overlaid U_new before returning.
static std::vector<double> warm_reproject_spin(const std::vector<double>& U_old,
                                               const SlimDiskInputs& in_old,
                                               const SlimDiskInputs& in_new,
                                               const OpacityLUTs& op) {
    using namespace constants;
    using namespace slim_detail;
    (void)in_old;   // grids are reconstructed from U_old + in_new (same N, r_out)
    const int N = std::max(in_new.n_nodes, 4);

    // Old grid radii (from the converged r_s_old = U_old[4N+1]).
    const double r_s_old = U_old[4*N+1];
    const double r_out   = in_new.r_out;   // r_out is spin-independent (same outer edge)
    const double lro0 = std::log(r_s_old), lro1 = std::log(r_out);
    std::vector<double> r_old(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r_old[i] = std::exp(lro0 + (lro1 - lro0) * t);
    }

    // BASE: a fresh thin-disk seed on the a_new grid [r_s_new, r_out]. This supplies a
    // physically-consistent inner structure + Mach-1 sonic node + newly-exposed-annulus
    // values that the converged old solution (which lived on the OLD, more-outward grid)
    // cannot provide. We then overlay the converged old solution where the grids overlap.
    std::vector<double> U = build_thin_disk_seed(in_new, op);

    // The fresh seed fixes the new grid: r_s_new = U[4N+1], log-spaced to r_out.
    const double r_s_new = U[4*N+1];
    const double r_isco_new = isco_prograde(in_new.mass, in_new.spin);
    const double lrn0 = std::log(r_s_new), lrn1 = std::log(r_out);

    // Log-interpolate a per-node field f(ln r) from the converged old solution at
    // offset `off`. Linear in ln r over the old log grid; clamps to the old endpoints
    // outside [r_old[0], r_old[N-1]].
    auto interp_old = [&](int off, double r) -> double {
        const double lr = std::log(r);
        if (lr <= std::log(r_old[0]))   return U_old[4*0+off];
        if (lr >= std::log(r_old[N-1])) return U_old[4*(N-1)+off];
        int lo = 0, hi = N - 1;
        while (hi - lo > 1) { int mid = (lo + hi) / 2; if (std::log(r_old[mid]) <= lr) lo = mid; else hi = mid; }
        const double x0 = std::log(r_old[lo]), x1 = std::log(r_old[hi]);
        const double w = (x1 > x0) ? (lr - x0) / (x1 - x0) : 0.0;
        const double f0 = U_old[4*lo+off], f1 = U_old[4*hi+off];
        // Σ and T_c are positive and span decades → interpolate in log; ℓ (off 2) is
        // O(1) → interpolate linearly. V is re-derived from mass conservation.
        if (off == 0 || off == 3) {
            const double lf0 = std::log(std::max(f0, (off == 0) ? kSigmaFloor : kTFloor));
            const double lf1 = std::log(std::max(f1, (off == 0) ? kSigmaFloor : kTFloor));
            return std::exp(lf0 + (lf1 - lf0) * w);
        }
        return f0 + (f1 - f0) * w;
    };
    // V from exact mass conservation at the NEW spin: Ṁ = -2πΣΔ^½(V/√(1-V²))r_g c.
    auto Vfrom = [&](double r, double Sig) -> double {
        const double sqrtD = std::sqrt(std::max(kerr_delta(in_new.mass, in_new.spin, r), 0.0));
        const double dn = 2.0 * std::numbers::pi * Sig * sqrtD * in_new.r_g * c_cgs;
        double V = -1e-6;
        if (dn > 0.0) { const double X = -in_new.mdot / dn; V = X / std::sqrt(1.0 + X * X); }
        if (!(V < 0.0)) V = -1e-6;
        return std::clamp(V, -kVCap, -1e-12);
    };

    // WARM OVERLAY: overwrite every new-grid node at r_i ≥ r_s_old with the converged
    // old solution (interpolated), re-deriving V. Skip node 0 (the sonic node — keep the
    // fresh seed's Mach-1 override) and skip the newly-exposed inner annulus r_i < r_s_old
    // (keep the fresh seed there). The outer node N-1 is the pinned BC node and also gets
    // overlaid (its radius is shared by both grids — r_out is spin-independent).
    for (int i = 1; i < N; ++i) {
        const double t = double(i) / double(N - 1);
        const double r = std::exp(lrn0 + (lrn1 - lrn0) * t);
        if (r < r_s_old) continue;                 // newly-exposed annulus: keep fresh seed
        const double Sig = std::max(interp_old(0, r), kSigmaFloor);
        const double Tc  = std::max(interp_old(3, r), kTFloor);
        const double ell = interp_old(2, r);
        U[4*i+0] = Sig;
        U[4*i+1] = Vfrom(r, Sig);
        U[4*i+2] = ell;
        U[4*i+3] = Tc;
    }

    // Globals: ℓ_in carried over, nudged halfway toward ℓ_K(isco(a_new)) (warm
    // eigenvalue; the outer bracket refines it). r_s = the fresh-seed sonic anchor.
    const double ell_in_old  = U_old[4*N+0];
    const double ell_in_isco = ell_kepler(in_new.mass, in_new.spin, r_isco_new);
    U[4*N+0] = 0.5 * (ell_in_old + ell_in_isco);
    U[4*N+1] = r_s_new;

    // De-glitch the overlaid profile before it is handed to the relaxation.
    deglitch_sigma_outliers(in_new, U);
    U[4*N+0] = 0.5 * (ell_in_old + ell_in_isco);   // deglitch preserves globals, but be explicit
    return U;
}

// ===========================================================================
// Task 3 (arclength continuation): augmented predictor-corrector.
// ===========================================================================
// The corrector Newton-solves the AUGMENTED square system in W = (U, Ṁ) (length
// m = 4N+3):
//   rows 0..4N+1 :  R(U,Ṁ) = 0            (the FULL §23 residual — BOTH regularity
//                                           rows 𝒟₀(r_s)=0 AND 𝒩₁(r_s)=0 are kept,
//                                           so ℓ_in and r_s are solved JOINTLY and
//                                           the sonic regularity stays satisfied —
//                                           NO separate outer ℓ_in bracket here).
//   row  4N+2    :  N ≡ (U−U₀)·U̇₀ + (Ṁ−Ṁ₀)·Ṁ̇₀ − Δs = 0   (Keller arclength row)
// with the augmented Jacobian
//   A = [[ J,    R_Mdot ],          (J = slim_analytic_jacobian — EXACT, the enabler)
//        [ U̇₀ᵀ,  Ṁ̇₀    ]]
// the SAME bordered matrix as the tangent solve.  This is the plan's documented
// choice: ℓ_in is a full state component pinned by the 𝒩₁ regularity row, solved
// jointly in the augmented Newton (simplest route that keeps sonic regularity).
//
// We re-use relax_structure's proven machinery, adapted to the FULL (no excluded
// row/col) augmented system:
//   • row + column SCALING (non-dimensionalize the 33-decade-spread Jacobian),
//   • Levenberg-Marquardt with the Nielsen GAIN-RATIO accept/reject + μ adaptation,
//   • a FEASIBILITY line search (Σ>0, T_c>0, |V|<1, r_s∈(r_in,r_out), Ṁ>0),
//   • the same Σ-outlier DE-GLITCH between accepted steps.
// The arclength row carries its own scale (Δs), and Ṁ gets a column scale = |Ṁ₀|.
//
// On the augmented merit we use the FULL scaled residual norm (slim_scaled_residual_
// norm — INCLUDES 𝒩₁, since the corrector DOES drive 𝒩₁→0) combined with the scaled
// arclength-row residual.  Convergence: merit floored AND the arclength row satisfied
// AND the validity gate passes (require_N1=true — this IS a fully-regular solution).
//
// In/out: U (length 4N+2) and Mdot (the continuation Ṁ).  U0,Mdot0 = the predictor
// base point; tan0 (length 4N+3, scaled-unit) = the predictor tangent; w0 (length
// 4N+3) = the base-point Keller weights; ds = the (dimensionless) scaled arclength
// step.  Returns true iff the corrector converged to a physically-valid regular soln.
struct ArclengthCorrectorResult {
    bool converged = false;
    double merit = 0.0;       // final FULL scaled merit (incl. 𝒩₁)
    double arc_resid = 0.0;   // |N|/ds at the accepted point (scaled arclength row)
    int iters = 0;
};
static ArclengthCorrectorResult arclength_corrector(
        const SlimDiskInputs& in, const OpacityLUTs& op,
        const std::vector<double>& U0, double Mdot0,
        const std::vector<double>& tan0, const std::vector<double>& w0, double ds,
        std::vector<double>& U, double& Mdot, int max_iters) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    const int n = 4 * N + 2;     // residual / U length
    const int m = n + 1;         // augmented unknown length
    const bool kDiag = std::getenv("SLIM_DIAG") != nullptr;

    ArclengthCorrectorResult res;

    // Keller arclength constraint in the SCALED metric (consistent with the scaled-
    // unit tangent tan0 and the base-point weights w0):
    //   N ≡ Σ_i (W_i − W₀_i)·(tan0_i / w0_i²) − ds        (W = (U, Ṁ))
    // ∂N/∂W_i = tan0_i / w0_i².  ds is the (dimensionless) scaled arclength step.
    auto arc_coef = [&](int i) { return tan0[i] / (w0[i] * w0[i]); };

    // Augmented merit: FULL scaled residual norm (incl. 𝒩₁) ⊕ the scaled arclength row.
    auto eval_merit = [&](const std::vector<double>& Uw, double Mw,
                          std::vector<double>& Rfull, double& arc) -> double {
        SlimDiskInputs inw = in; inw.mdot = Mw;
        slim_radial_residual(Uw, inw, op, Rfull);
        const double rms = slim_scaled_residual_norm(Uw, Rfull, inw);   // FULL (incl. 𝒩₁)
        double Ndot = (Mw - Mdot0) * arc_coef(n) - ds;
        for (int i = 0; i < n; ++i) Ndot += (Uw[i] - U0[i]) * arc_coef(i);
        arc = Ndot / std::max(std::abs(ds), 1e-300);
        // Combine: RMS over (4N+2 scaled residual rows + 1 scaled arc row).
        const double comb = std::sqrt((rms*rms*(double)n + arc*arc) / (double)(n + 1));
        return comb;
    };

    std::vector<double> Rfull;
    double arc = 0.0;
    double merit = eval_merit(U, Mdot, Rfull, arc);

    // LM state (mirrors relax_structure).
    double lm_mu = 1e-3, lm_nu = 2.0;
    constexpr double kMuMax = 1e12, kMuMin = 1e-9;
    constexpr double kStepCap = 0.5;       // cap on |Δ/u| for Σ,T_c,Ṁ per step
    constexpr double kMeritFloor = 5e-3;   // augmented FD floor (a touch looser than the
                                           // analytic-J inner floor: the arc row + the FD
                                           // r_s/μ/κ columns set it; validity gate guards)

    if (kDiag)
        std::printf("[ARC-CORR] enter ds=%.4e Mdot0=%.4e seed merit=%.3e arc=%.3e\n",
                    ds, Mdot0, merit, arc);

    for (int it = 0; it < max_iters; ++it) {
        if (g_budget) { ++g_budget->inner_iters; if (g_budget->check()) break; }

        // --- Build the augmented Jacobian A (m×m): [[J, R_Mdot],[U̇₀ᵀ, Ṁ̇₀]]. ---
        std::vector<double> J, R_Mdot;
        slim_analytic_jacobian(U, in_with_mdot(in, Mdot), op, J);
        slim_R_Mdot_column(U, in_with_mdot(in, Mdot), op, R_Mdot);

        // Row + column scaling over the augmented system.  Cols 0..n-1 = the state
        // column scales (per-variable magnitude); col n = |Ṁ₀| (the Ṁ scale).  Rows
        // 0..n-1 = 1/group-scale (FULL set incl. 𝒩₁); row n (arclength) = 1 (it is
        // already O(1) after the /ds in N).
        std::vector<double> cs(m), rs_inv(m);
        {
            double mSig=0,mV=0,mEll=0,mT=0;
            for (int i=0;i<N;++i){ mSig=std::max(mSig,std::abs(U[4*i+0])); mV=std::max(mV,std::abs(U[4*i+1]));
                                   mEll=std::max(mEll,std::abs(U[4*i+2])); mT=std::max(mT,std::abs(U[4*i+3])); }
            mSig=std::max(mSig,1e-30); mV=std::max(mV,1e-30); mEll=std::max(mEll,1e-30); mT=std::max(mT,1.0);
            for (int i=0;i<N;++i){ cs[4*i+0]=mSig; cs[4*i+1]=mV; cs[4*i+2]=mEll; cs[4*i+3]=mT; }
            cs[4*N+0]=std::max(std::abs(U[4*N+0]),1e-30);   // ℓ_in
            cs[4*N+1]=std::max(std::abs(U[4*N+1]),1e-30);   // r_s
            cs[n]    =std::max(std::abs(Mdot),1e-300);      // Ṁ
            const GroupScales gs = slim_group_scales(U, in_with_mdot(in, Mdot));
            auto setrows=[&](int b,int e,double sc){ sc=std::max(sc,1e-300); for(int r=b;r<e;++r) rs_inv[r]=1.0/sc; };
            setrows(0,N,gs.mass); setrows(N,2*N,gs.ang); setrows(2*N,3*N-1,gs.rad);
            setrows(3*N-1,4*N-2,gs.ene); setrows(4*N-2,4*N-1,gs.bc_ell);
            setrows(4*N-1,4*N,gs.ene); setrows(4*N,4*N+1,gs.reg_D0); setrows(4*N+1,4*N+2,gs.reg_N1);
            rs_inv[n]=1.0/std::max(std::abs(ds),1e-300);    // arclength row scale = ds
        }

        // Scaled augmented Jacobian Js (m×m) and residual Rs (m).
        std::vector<double> Js((size_t)m*m,0.0), Rs(m,0.0);
        for (int row=0; row<n; ++row) {
            Rs[row] = Rfull[row]*rs_inv[row];
            for (int col=0; col<n; ++col)
                Js[(size_t)row*m+col] = J[(size_t)row*n+col]*rs_inv[row]*cs[col];
            Js[(size_t)row*m+n] = R_Mdot[row]*rs_inv[row]*cs[n];
        }
        // Arclength row n (SCALED Keller): N = Σ_i (W_i−W₀_i)·arc_coef(i) − ds,
        // arc_coef(i)=tan0_i/w0_i².  ∂N/∂W_i = arc_coef(i).  Residual scaled by 1/ds.
        {
            double Ndot = (Mdot - Mdot0)*arc_coef(n) - ds;
            for (int i=0;i<n;++i) Ndot += (U[i]-U0[i])*arc_coef(i);
            Rs[n] = Ndot*rs_inv[n];
            for (int col=0; col<n; ++col) Js[(size_t)n*m+col] = arc_coef(col)*rs_inv[n]*cs[col];
            Js[(size_t)n*m+n] = arc_coef(n)*rs_inv[n]*cs[n];
        }

        // Normal equations (Js^T Js + μ diag) y = −Js^T Rs.
        std::vector<double> JtJ((size_t)m*m,0.0), Jtr(m,0.0);
        for (int i=0;i<m;++i) for (int k=0;k<m;++k) {
            const double jik = Js[(size_t)k*m+i]; if (jik==0.0) continue;
            Jtr[i]+=jik*Rs[k];
            for (int j=0;j<m;++j) JtJ[(size_t)i*m+j]+=jik*Js[(size_t)k*m+j];
        }

        const double cnt = (double)m;
        const double F_old = 0.5*cnt*merit*merit;
        std::vector<double> Adamp((size_t)m*m), bdamp(m);
        bool step_taken=false, bail=false;
        double merit_try=merit, arc_try=arc;
        std::vector<double> Utry; double Mtry=Mdot;
        std::vector<double> Rtry;

        while (true) {
            bool solved=false;
            for (int tries=0; tries<12 && !solved; ++tries) {
                Adamp=JtJ;
                for (int i=0;i<m;++i) Adamp[(size_t)i*m+i]+=lm_mu*std::max(JtJ[(size_t)i*m+i],1e-300);
                for (int i=0;i<m;++i) bdamp[i]=-Jtr[i];
                if (dense_solve(Adamp,bdamp,m)) { solved=true; break; }
                lm_mu=std::min(lm_mu*10.0,kMuMax);
                if (lm_mu>=kMuMax) break;
            }
            if (!solved) { bail=true; break; }

            double pred=0.0;
            for (int i=0;i<m;++i){ const double Dii=std::max(JtJ[(size_t)i*m+i],1e-300);
                                   pred+=lm_mu*Dii*bdamp[i]*bdamp[i]-bdamp[i]*Jtr[i]; }
            pred*=0.5;

            // Unscale the step: dW[col] = cs[col]·y[col].
            std::vector<double> dW(m,0.0);
            for (int col=0; col<m; ++col) dW[col]=bdamp[col]*cs[col];

            // Trust-region cap on Σ,T_c (cols off 0,3) and Ṁ (col n).
            double lam=1.0;
            auto capvar=[&](double u,double d){ if(u!=0.0&&d!=0.0){ const double f=std::abs(d/u); if(f*lam>kStepCap) lam=kStepCap/f; } };
            for (int i=0;i<N;++i){ capvar(U[4*i+0],dW[4*i+0]); capvar(U[4*i+3],dW[4*i+3]); }
            capvar(Mdot,dW[n]);

            bool physical=false; double F_new=F_old;
            for (int ls=0; ls<40; ++ls) {
                Utry.assign(U.begin(),U.end());
                for (int i=0;i<n;++i) Utry[i]+=lam*dW[i];
                Mtry = Mdot + lam*dW[n];
                physical=true;
                // Σ>0, T_c>0, |V|<1 AND V<0 (genuine INFLOW) at every node.  Requiring
                // V<0 (not just |V|<1) blocks outflow/standstill nodes that the spurious
                // root admits — slim-disk inflow is strictly V<0.
                for (int i=0;i<N&&physical;++i){ const double S=Utry[4*i+0],V=Utry[4*i+1],T=Utry[4*i+3];
                    if (S<=0.0||T<=0.0||!(V<0.0)) physical=false; }
                if (physical){ const double rs=Utry[4*N+1]; if(!(rs>in.r_in&&rs<in.r_out)) physical=false; }
                if (physical && !(Mtry>0.0)) physical=false;     // Ṁ>0
                // Reject the f_adv BLOW-UP root (Qrad→0 ⇒ f_adv≈−1130): require Qrad>0 and
                // |f_adv|<50 at every node, using the SAME closure path as unpack_profile.
                if (physical){ SlimDiskInputs inw=in; inw.mdot=Mtry;
                    if (!slim_fadv_ok(inw, op, Utry, 50.0)) physical=false; }
                if (physical){
                    merit_try=eval_merit(Utry,Mtry,Rtry,arc_try);
                    F_new=0.5*cnt*merit_try*merit_try; break;
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
            if (lm_mu>=kMuMax){ bail=true; break; }
            lm_mu=std::min(lm_mu*lm_nu,kMuMax); lm_nu*=2.0;
        }

        if (bail || !step_taken) break;

        // Max relative step over the augmented unknowns (for the convergence test).
        double maxrel=0.0;
        for (int i=0;i<n;++i) maxrel=std::max(maxrel,std::abs(Utry[i]-U[i])/std::max(std::abs(U[i]),1e-300));
        maxrel=std::max(maxrel,std::abs(Mtry-Mdot)/std::max(std::abs(Mdot),1e-300));

        U.swap(Utry); Mdot=Mtry; Rfull.swap(Rtry); merit=merit_try; arc=arc_try;
        res.iters=it+1;

        // De-glitch any Σ-outlier the step introduced (same source fix as the inner solve).
        { const int nrep=deglitch_sigma_outliers(in, U);
          if (nrep>0){ merit=eval_merit(U,Mdot,Rfull,arc); } }

        if (kDiag && (it<5 || it%20==0))
            std::printf("[ARC-CORR] it=%d lam-merit=%.3e arc=%.3e maxrel=%.2e mu=%.1e\n",
                        it, merit, arc, maxrel, lm_mu);

        // Convergence: merit floored AND arclength row satisfied AND validity gate
        // (require_N1=true — the augmented corrector DOES regularize 𝒩₁).
        const bool merit_ok = (merit < kMeritFloor);
        const bool arc_ok   = (std::abs(arc) < 1e-2);
        const bool step_ok  = (maxrel < std::max(in.tol, 5e-3));
        if (merit_ok && arc_ok && step_ok) {
            const ValidityResult v = slim_validity_gate(in_with_mdot(in, Mdot), op, U, /*require_N1=*/true);
            if (v.all(/*require_N1=*/true)) {
                res.converged=true; res.merit=merit; res.arc_resid=arc; break;
            }
        }
    }
    res.merit=merit; res.arc_resid=arc;
    return res;
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

    // -----------------------------------------------------------------------
    // Install the runaway safety budget for THIS solve (RAII-cleared on every
    // return path, including the honest fallbacks below).  Caps come from
    // SlimDiskInputs if the caller set them, else the generous SolveBudget
    // defaults.  budget_tripped() below converts a trip into the honest empty
    // fallback with a clear stderr message naming what was exceeded.
    SolveBudget budget;
    if (in.budget_inner_iter_cap > 0) budget.inner_iter_cap = in.budget_inner_iter_cap;
    if (in.budget_wall_seconds   > 0) budget.wall_cap_s     = in.budget_wall_seconds;
    struct BudgetGuard { ~BudgetGuard() { g_budget = nullptr; } } budget_guard;
    g_budget = &budget;
    if (kDiag)
        std::printf("[SLIM] safety budget: inner_iter_cap=%lld wall_cap=%.0fs\n",
                    budget.inner_iter_cap, budget.wall_cap_s);
    // Local helper: on a trip, emit the stderr message and return the honest empty
    // profile.  Caller pattern: `if (budget.check()) return budget_fallback();`
    auto budget_fallback = [&]() -> SlimDiskRadial {
        std::fprintf(stderr,
                     "[SLIM] BUDGET EXCEEDED (%s) -> honest fallback "
                     "(inner_iters=%lld/%lld, wall=%.1fs/%.0fs)\n",
                     budget.what ? budget.what : "?",
                     budget.inner_iters, budget.inner_iter_cap,
                     budget.elapsed_s(), budget.wall_cap_s);
        return SlimDiskRadial{};
    };

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
    bool on_slim_branch = false;    // set once we cross the fold onto the slim seed (Task 3)

    // -----------------------------------------------------------------------
    // Phase B — SPIN HOMOTOPY CONTINUATION (the high-spin enabler).
    // -----------------------------------------------------------------------
    // The crude thin-disk seed is OUT OF BASIN at high spin: the ISCO/sonic point
    // marches inward (a=0→r_isco=6; a=0.998→r_isco≈1.24), shifting the whole inner
    // structure, so the inner relaxation cannot relax from build_thin_disk_seed and
    // stalls (seed/basin failure, not the FD floor).  Cure it by walking spin up
    // from the PROVEN-CONVERGENT a=0 anchor in small increments, warm-starting each
    // spin's solve (re-projected onto its grid) from the previous converged solution.
    // Each small step the solution barely moves, so the warm start stays in basin.
    // Run the whole walk at Ṁ_lo (the easy thin rung); the EXISTING Ṁ-continuation
    // below then climbs Ṁ at the target spin from the spin-walked warm start.
    //
    // CONTINUATION PARAMETERS (not physics — homotopy schedule / safeguards):
    //   • kSpinThresh: below this |a| the crude seed is already in basin, so skip the
    //     walk and use the direct Ṁ-continuation (preserves the proven a=0 path).
    //   • the spin ladder: a base schedule DENSER near extremal (where the ISCO moves
    //     fastest); adaptive step-HALVING on a failed rung, with an underflow FLOOR
    //     below which we stop and fall back honestly.
    constexpr double kSpinThresh = 0.05;   // below this, the crude seed is in-basin

    // Internal continuation iteration budget.  The inner Newton uses a CENTRAL-
    // DIFFERENCE Jacobian, which only reaches the FD-noise merit floor (~7.6e-6;
    // accepted at kMeritFloor=1e-3) after MANY iterations — the floor was MEASURED at
    // the a=0 corner at ~800 iters (see kMeritFloor's derivation comment).  The
    // public-facing in.max_iters (the tests pass 100) is far too small for ANY rung —
    // even the easy a=0 anchor — to relax to that floor, so every continuation rung
    // would stall above the validity band and the walk would fall back.  Construction
    // time is uncapped (spec §7: robustness over speed), so the internal continuation
    // solves use a generous budget.  This is a SOLVER-EFFORT knob ONLY: it changes
    // neither the residual physics, the merit floor, nor the validity gate — it just
    // lets the FD Newton run to the precision it can deliver.
    constexpr int kContinuationMaxIters = 800;
    if (in.spin > kSpinThresh) {
        // Base spin ladder up to in.spin: denser toward extremal. Built as fractions
        // so it adapts to whatever in.spin the caller requested; any rung at or above
        // in.spin is clamped to in.spin and terminates the ladder.
        const double base_ladder[] = {0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.90, 0.95, 0.98};
        std::vector<double> spin_ladder;
        for (double a : base_ladder) {
            if (a >= in.spin) break;
            spin_ladder.push_back(a);
        }
        spin_ladder.push_back(in.spin);   // always finish exactly at the target spin

        // Step-halving floor: the smallest spin increment we will attempt before
        // declaring the walk stuck (honest fallback). 1e-3 in a_* is far finer than
        // any rung the base ladder uses; reaching it means the basin shrank below
        // what re-projection can bridge — a genuine failure, not a schedule miss.
        constexpr double kSpinStepFloor = 1e-3;

        // 1) Anchor: solve (a=0, Ṁ_lo) — the proven-convergent corner.
        SlimDiskInputs in_anchor = in;
        in_anchor.spin = 0.0;
        in_anchor.mdot = Mdot_lo;
        in_anchor.max_iters = std::max(in.max_iters, kContinuationMaxIters);
        if (kDiag)
            std::printf("[SPINWALK] anchor: solve (a=0, Mdot_lo=%.3e)\n", Mdot_lo);
        std::vector<double> U_anchor = build_thin_disk_seed(in_anchor, opacity);
        if (!solve_single_am(in_anchor, opacity, U_anchor, /*require_N1=*/false)) {
            if (budget.check()) return budget_fallback();   // anchor failed because budget tripped
            if (kDiag) std::printf("[SPINWALK] anchor (a=0) FAILED -> honest fallback\n");
            return SlimDiskRadial{};
        }
        if (budget.check()) return budget_fallback();

        // Carry the converged state + the spin it was solved at across rungs.
        std::vector<double> U_prev = U_anchor;
        SlimDiskInputs in_prev = in_anchor;   // holds a_prev for re-projection
        bool walk_ok = true;

        for (size_t s = 1; s < spin_ladder.size() && walk_ok; ++s) {
            double a_target = spin_ladder[s];
            const double a_from = in_prev.spin;
            bool rung_done = false;
            double step = a_target - a_from;   // shrunk on retry

            while (!rung_done) {
                if (budget.check()) return budget_fallback();   // budget cap hit mid-walk
                const double a_k = a_from + step;
                SlimDiskInputs in_k = in;
                in_k.spin = a_k;
                in_k.mdot = Mdot_lo;
                in_k.max_iters = std::max(in.max_iters, kContinuationMaxIters);
                if (kDiag)
                    std::printf("[SPINWALK] === spin rung: a %.4f -> %.4f (step=%.4f) @ Mdot_lo ===\n",
                                a_from, a_k, step);
                // Warm-start by re-projecting the previous converged solution onto a_k's grid.
                std::vector<double> U_k = warm_reproject_spin(U_prev, in_prev, in_k, opacity);
                if (solve_single_am(in_k, opacity, U_k, /*require_N1=*/false)) {
                    U_prev = U_k;
                    in_prev = in_k;
                    rung_done = true;
                    if (kDiag)
                        std::printf("[SPINWALK] spin a=%.4f CONVERGED @ Mdot_lo (r_sonic=%.4f, ell_in=%.5f)\n",
                                    a_k, U_k[4*(std::max(in.n_nodes,4))+1], U_k[4*(std::max(in.n_nodes,4))+0]);
                } else {
                    if (budget.check()) return budget_fallback();   // failed because budget tripped
                    // Failed: HALVE the spin step and retry from a_from.
                    step *= 0.5;
                    if (kDiag)
                        std::printf("[SPINWALK] spin a=%.4f FAILED -> halve step to %.4f\n", a_k, step);
                    if (step < kSpinStepFloor) {
                        if (kDiag)
                            std::printf("[SPINWALK] spin step underflow (<%.1e) at a_from=%.4f -> honest fallback\n",
                                        kSpinStepFloor, a_from);
                        walk_ok = false;
                    }
                }
            }
        }

        if (!walk_ok) return SlimDiskRadial{};   // honest fallback (walk broke)

        // The walk ended at (in.spin, Mdot_lo). Hand its converged state to the
        // Ṁ-continuation below as the warm start for its FIRST rung (which is also
        // Mdot_lo), so no rung is re-solved from a cold seed.
        U = U_prev;
        have_warm = true;
        if (kDiag)
            std::printf("[SPINWALK] reached target spin a=%.4f @ Mdot_lo -> hand off to Mdot ladder\n",
                        in.spin);
    }

    for (size_t k = 0; k < rungs.size(); ++k) {
        if (budget.check()) return budget_fallback();   // budget cap hit between Ṁ rungs
        SlimDiskInputs in_rung = in;
        in_rung.mdot = rungs[k];
        // Same generous internal budget as the spin walk: the FD Newton needs many
        // iterations to relax to its floor at every Ṁ rung (see kContinuationMaxIters).
        in_rung.max_iters = std::max(in.max_iters, kContinuationMaxIters);
        if (kDiag)
            std::printf("[SLIM] === Mdot rung %zu/%zu: Mdot=%.3e (Mdot_Edd=%.3e, f_Edd=%.3f) ===\n",
                        k + 1, rungs.size(), rungs[k], Mdot_Edd, rungs[k] / Mdot_Edd);
        // Seed selection (Task 3).  Below the lower-branch fold (f_Edd≲0.12), the
        // thin/cold seed + warm-start ladder is the proven path.  ABOVE the fold the
        // warm start carries the LOWER (thin) branch upward and rounds the fold; the
        // slim solution lives on a DISTINCT upper branch.  So on the FIRST above-fold
        // rung we seed the principled global slim seed directly (NT-thin outward,
        // advection-thickened inward) and let solve_outer_bracket/relax_structure relax
        // onto the slim branch; SUBSEQUENT above-fold rungs warm-start from the
        // converged slim state (genuine upper-branch continuation).  Sub-fold rungs are
        // untouched (keep the exact proven thin path).
        constexpr double kFoldRoute = 0.12;
        const double f_rung = rungs[k] / Mdot_Edd;
        if (f_rung > kFoldRoute && !on_slim_branch) {
            U = build_slim_disk_seed(in_rung, opacity);   // cross onto the slim branch
            have_warm = false;
            on_slim_branch = true;
        } else if (!have_warm) {
            U = build_thin_disk_seed(in_rung, opacity);
        }

        const bool ok = solve_outer_bracket(in_rung, opacity, U);
        if (!ok) {
            if (budget.check()) return budget_fallback();   // bracket failed because budget tripped
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

// ---------------------------------------------------------------------------
// Task 4: pseudo-arclength continuation driver — trace the branch ACROSS the fold.
// ---------------------------------------------------------------------------
// Converge a sub-fold anchor (a, f_Edd≈0.10) via the PROVEN existing path
// (solve_slim_disk_radial's spin-walk + Ṁ-ladder), promote Ṁ to a continuation
// unknown, build the initial tangent, then arclength-step UP with the predictor-
// corrector (Task 3).  Δs grows on easy convergence, shrinks on failure, honest
// fallback on underflow.  The branch may dip in Ṁ through the unstable segment
// (Ṁ̇<0) then climb onto the high-Ṁ slim branch (Ṁ̇>0 again) — we track the fold and
// record every accepted point.  GATE: cross f_Edd=0.11, reach toward f_Edd≈0.9.
SlimArclengthResult solve_slim_disk_arclength(const SlimDiskInputs& in,
                                              const OpacityLUTs& opacity) {
    using namespace constants;
    const bool kDiag = std::getenv("SLIM_DIAG") != nullptr;
    const int N = std::max(in.n_nodes, 4);
    const int n = 4 * N + 2;
    const int m = n + 1;
    SlimArclengthResult result;

    // Install the safety budget for the WHOLE continuation (RAII-cleared).
    SolveBudget budget;
    if (in.budget_inner_iter_cap > 0) budget.inner_iter_cap = in.budget_inner_iter_cap;
    if (in.budget_wall_seconds   > 0) budget.wall_cap_s     = in.budget_wall_seconds;
    struct BudgetGuard { ~BudgetGuard() { g_budget = nullptr; } } budget_guard;
    g_budget = &budget;

    // Ṁ_Edd (textbook 10 L_Edd/c²).
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    auto f_of = [&](double md){ return md / Mdot_Edd; };

    // ---- Anchor: converge a sub-fold point at f_Edd≈0.10. ----
    // Use the PROVEN direct path (cold-seed + solve_single_am at (a, f_Edd=0.10) —
    // the same route the warm-start sweep uses to converge a=0.9), which keeps the
    // EXACT converged state vector U (no lossy profile round-trip).  For high spin
    // (a beyond the cold-seed basin) we spin-walk the anchor up from a=0 at f_Edd=0.10
    // via warm_reproject_spin (the existing Phase-B homotopy), so the anchor itself
    // is in-basin before continuation begins.
    // Anchor f_Edd: 0.10 by default; env-overridable (SLIM_ARC_ANCHOR_F) for
    // diagnostics — after the 2026-06-12 §23 Q_vis metric-factor correction the
    // cold-seed fold sits below 0.05 at a=0.9, so probes anchor at ~0.02.
    const double f_anchor = [&] {
        const char* e = std::getenv("SLIM_ARC_ANCHOR_F");
        const double v = e ? std::atof(e) : 0.0;
        return (v > 0.0 && v < 1.0) ? v : 0.10;
    }();
    SlimDiskInputs in_anchor = in;
    in_anchor.mdot = f_anchor * Mdot_Edd;
    in_anchor.max_iters = std::max(in.max_iters, 800);
    if (kDiag) std::printf("[ARC] anchor: solve (a=%.4f, f_Edd=%.3f)\n", in.spin, f_anchor);

    std::vector<double> U;
    constexpr double kSpinThresh = 0.05;

    // FAST ANCHOR: at the thin f_Edd=0.10 rung the sonic eigenvalue is ℓ_in≈ℓ_K(ISCO)
    // (the disk is nearly Novikov-Thorne), so a SINGLE relax_structure at that ℓ_in
    // typically lands on the regular branch with 𝒩₁(r_s)≈0 already (the existing
    // solver's "DIRECT-ACCEPT").  Trying it FIRST skips the expensive multi-window ℓ_in
    // bracket scan (dozens of full relaxes) for the common case; we fall back to the
    // full bracket only if the direct relax fails the FULL validity gate.
    auto try_direct_anchor = [&](const SlimDiskInputs& ina, std::vector<double>& Uout) -> bool {
        using namespace slim_detail;
        const double r_isco = isco_prograde(ina.mass, ina.spin);
        const double ell_in = ell_kepler(ina.mass, ina.spin, r_isco);
        std::vector<double> Uw = build_thin_disk_seed(ina, opacity);
        if (!relax_structure(ina, opacity, ell_in, Uw)) return false;
        const ValidityResult v = slim_validity_gate(ina, opacity, Uw, /*require_N1=*/true);
        if (!v.all(/*require_N1=*/true)) return false;
        Uout.swap(Uw);
        if (kDiag) std::printf("[ARC] FAST anchor: direct ell_in=%.5f converged (r_s=%.4f, skipped bracket)\n",
                               ell_in, Uout[4*N+1]);
        return true;
    };

    if (in.spin <= 0.9) {
        // Try the fast direct anchor first; fall back to the full bracket if it misses.
        if (!try_direct_anchor(in_anchor, U)) {
            if (budget.check()) { if (kDiag) std::printf("[ARC] anchor budget tripped -> fallback\n"); return result; }
            if (kDiag) std::printf("[ARC] FAST anchor missed -> full bracket\n");
            U = build_thin_disk_seed(in_anchor, opacity);
            if (!solve_single_am(in_anchor, opacity, U, /*require_N1=*/false)) {
                if (budget.check()) { if (kDiag) std::printf("[ARC] anchor budget tripped -> fallback\n"); return result; }
                if (kDiag) std::printf("[ARC] anchor (cold-seed) FAILED -> honest fallback\n");
                return result;
            }
        }
    } else {
        // Spin-walk the anchor up from a=0 at f_Edd=0.10 (Phase-B homotopy).
        const double base_ladder[] = {0.0, 0.2, 0.4, 0.6, 0.75, 0.85, 0.90};
        std::vector<double> spin_ladder;
        for (double a : base_ladder) { if (a >= in.spin) break; spin_ladder.push_back(a); }
        spin_ladder.push_back(in.spin);
        SlimDiskInputs in0 = in_anchor; in0.spin = spin_ladder[0];
        std::vector<double> U_prev = build_thin_disk_seed(in0, opacity);
        if (!solve_single_am(in0, opacity, U_prev, /*require_N1=*/false)) {
            if (kDiag) std::printf("[ARC] anchor a=0 base FAILED -> fallback\n"); return result;
        }
        SlimDiskInputs in_prev = in0; bool walk_ok = true;
        for (size_t s = 1; s < spin_ladder.size() && walk_ok; ++s) {
            SlimDiskInputs in_k = in_anchor; in_k.spin = spin_ladder[s];
            double step = spin_ladder[s] - in_prev.spin; bool done = false;
            while (!done) {
                if (budget.check()) return result;
                SlimDiskInputs in_try = in_anchor; in_try.spin = in_prev.spin + step;
                std::vector<double> U_k = warm_reproject_spin(U_prev, in_prev, in_try, opacity);
                if (solve_single_am(in_try, opacity, U_k, /*require_N1=*/false)) {
                    U_prev = U_k; in_prev = in_try; done = (in_try.spin >= spin_ladder[s] - 1e-12);
                    if (!done) { step = spin_ladder[s] - in_prev.spin; }
                } else {
                    step *= 0.5;
                    if (step < 1e-3) { walk_ok = false; }
                }
            }
        }
        if (!walk_ok) { if (kDiag) std::printf("[ARC] anchor spin-walk stuck -> fallback\n"); return result; }
        U = U_prev;
    }
    (void)kSpinThresh;

    double Mdot = in_anchor.mdot;
    // Anchor profile (for branch-point 0 record + the returned top seed).
    SlimDiskRadial anchor; unpack_profile(in_anchor, opacity, U, anchor);
    anchor.converged = true; anchor.ell_in = U[4*N+0]; anchor.r_sonic = U[4*N+1];
    { std::vector<double> Rf; slim_radial_residual(U, in_anchor, opacity, Rf);
      anchor.final_residual = slim_scaled_residual_norm(U, Rf, in_anchor); }

    // Physics-summary of a converged (U, Ṁ) into a branch point.
    auto record_point = [&](const std::vector<double>& Uw, double Mw, double ds,
                            double mdot_dot, double merit) -> SlimArclengthPoint {
        using namespace slim_detail;
        SlimArclengthPoint pt;
        pt.f_Edd = f_of(Mw); pt.mdot = Mw; pt.r_sonic = Uw[4*N+1]; pt.ell_in = Uw[4*N+0];
        pt.merit = merit; pt.arc_step = ds;
        pt.Mdot_dot_sign = (mdot_dot > 0) ? 1 : (mdot_dot < 0 ? -1 : 0);
        SlimDiskRadial prof; unpack_profile(in_with_mdot(in, Mw), opacity, Uw, prof);
        bool first = true;
        for (size_t i = 0; i < prof.r.size(); ++i) {
            const double r = prof.r[i];
            const double Hr = prof.H[i] / (r * in.r_g);
            pt.max_Hr = std::max(pt.max_Hr, Hr);
            pt.peak_Sigma = std::max(pt.peak_Sigma, prof.Sigma[i]);
            const OneZoneState oz = one_zone_closure(std::max(prof.Sigma[i],kSigmaFloor),
                                                     std::max(prof.Tc[i],kTFloor), r, in, opacity);
            const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
            if (first) { pt.beta_min=pt.beta_max=beta; pt.fadv_min=pt.fadv_max=prof.f_adv[i]; first=false; }
            else { pt.beta_min=std::min(pt.beta_min,beta); pt.beta_max=std::max(pt.beta_max,beta);
                   pt.fadv_min=std::min(pt.fadv_min,prof.f_adv[i]); pt.fadv_max=std::max(pt.fadv_max,prof.f_adv[i]); }
        }
        return pt;
    };

    // Record the anchor as branch point 0 (tangent sign unknown yet -> +1 seed).
    result.branch.push_back(record_point(U, Mdot, 0.0, +1.0, anchor.final_residual));
    result.max_f_Edd = f_of(Mdot);
    result.top = anchor;

    // ---- Initial tangent: seed direction increases Ṁ. ----
    std::vector<double> t_prev(m, 0.0); t_prev[n] = 1.0;   // bias toward +Ṁ
    std::vector<double> tangent;
    if (!slim_arclength_tangent(U, in_with_mdot(in, Mdot), opacity, t_prev, tangent)) {
        if (kDiag) std::printf("[ARC] initial tangent degenerate -> fallback\n");
        return result;
    }
    // Orient the very first tangent so Ṁ̇ > 0 (climb up the branch from the anchor).
    if (tangent[n] < 0.0) for (double& v : tangent) v = -v;
    t_prev = tangent;
    if (kDiag) std::printf("[ARC] initial tangent: Mdot_dot=%+.4e (f_Edd_dot=%+.4e)\n",
                           tangent[n], tangent[n]/Mdot_Edd);

    // SECANT continuity reference.  The freshly-solved tangent's orientation is fixed by
    // the scaled prev·new dot, but near the fold consecutive tangents rotate by ~90° in
    // the inflated state subspace so that dot is near zero ⇒ the orientation (hence the
    // raw Ṁ̇ sign) flips arbitrarily, reversing the predictor each step and stalling the
    // climb.  The DIRECTION OF TRAVEL along the curve — the secant (W_k − W_{k−1}) between
    // the last two ACCEPTED points — is far more stable than the tangent's own sign, so we
    // orient each new tangent to agree with the secant in the scaled metric.  Seed the
    // "previous accepted point" with the anchor so the first step's secant is well-defined.
    std::vector<double> U_prevpt = U; double Mdot_prevpt = Mdot;
    bool have_secant = false;   // becomes true after the first accepted step

    // ---- Predictor-corrector loop with arclength step control. ----
    // ds is the DIMENSIONLESS scaled arclength step (the tangent is unit-normed in the
    // Ṁ-balanced scaled metric Σ(t_i/w_i)²=1, so a raw step ΔW_i = ds·t_i advances the
    // scaled coordinate by exactly ds).  Start SMALL — a large coordinated step in all
    // ~4N state DOF at once is a big nonlinear jump that overshoots the local model and
    // lets the corrector fall onto a DIFFERENT branch (observed: ds=0.1 overshoots and
    // the corrector jumps to the low-Ṁ branch).  Grow gently on easy convergence,
    // shrink on failure or on a BRANCH-JUMP (Ṁ reversed vs the predictor direction);
    // honest fallback on underflow.
    double ds = 2e-2;                  // scaled arclength step (dimensionless)
    const double ds_floor = 1e-5;
    const double ds_ceil  = 0.05;      // lowered from 0.3: a smaller cap keeps each
                                       // ~4N-DOF predictor step inside the local model,
                                       // which matters most near the high-curvature fold.
    const double ds_fold  = 5e-3;      // forced step while inside the fold neighbourhood
    const int    kFoldShrinkSteps = 6; // # of steps to hold ds≤ds_fold after a fold cue
    const double kVertFrac = 0.05;     // |Ṁ̇|/|t| below this ⇒ near-vertical (approaching turn)
    int fold_shrink_steps = 0;         // remaining steps to keep ds clamped to ds_fold
    const double f_target = 0.9;       // aim for f_Edd≈0.9 on the high-Ṁ branch
    const int    kMaxSteps = 600;
    const int    kCorrIters = 300;
    // Overshoot cap: reject a corrected point whose |Δf_Edd| per step is implausibly
    // large (a wild jump to a far-away branch, not a smooth continuation step).  We do
    // NOT reject Ṁ-DECREASING steps — the post-fold unstable segment genuinely runs to
    // lower Ṁ, and the trajectory-based tangent orientation follows it correctly.
    const double kMaxDfEdd = 0.06;

    int prev_sign = (tangent[n] > 0) ? 1 : -1;
    int steps_done = 0;

    for (int step = 0; step < kMaxSteps; ++step) {
        if (budget.check()) { if (kDiag) std::printf("[ARC] budget tripped -> stop\n"); break; }

        // Base-point weights for the Keller metric (same as the tangent's normalization).
        std::vector<double> w0; slim_arclength_weights(U, Mdot, N, w0);

        // Predictor: W_pred = (U0,Mdot0) + ds·tangent  (raw step; ds is the scaled length).
        const std::vector<double> U0 = U;
        const double Mdot0 = Mdot;
        const double Mdot_pred_intended = Mdot0 + ds * tangent[n];  // predictor target Ṁ
        std::vector<double> Upred = U;
        for (int i = 0; i < n; ++i) Upred[i] = U0[i] + ds * tangent[i];
        double Mdot_pred = Mdot_pred_intended;
        // Keep the predictor physical (clamp Σ,T_c>0, Ṁ>0) before the corrector.
        for (int i = 0; i < N; ++i) {
            Upred[4*i+0] = std::max(Upred[4*i+0], kSigmaFloor);
            Upred[4*i+3] = std::max(Upred[4*i+3], kTFloor);
        }
        if (!(Mdot_pred > 0.0)) Mdot_pred = 0.5 * Mdot0;

        std::vector<double> Ucorr = Upred;
        double Mcorr = Mdot_pred;
        ArclengthCorrectorResult cr = arclength_corrector(
            in, opacity, U0, Mdot0, tangent, w0, ds, Ucorr, Mcorr, kCorrIters);

        // Overshoot guard only: reject a wild |Δf_Edd| jump (far-away branch).  A
        // near-zero or Ṁ-DECREASING step is ALLOWED — the arclength method is expected
        // to ride the branch through the (near-vertical) fold and down the unstable
        // segment; rejecting Ṁ-decreasing steps would forbid exactly the fold traversal
        // we are trying to perform.  Also reject a corrector that BARELY moved the point
        // (it snapped back to the base — ds too small to escape the current basin): grow
        // ds instead of shrinking, so the predictor jumps far enough to the next point.
        const double dM_actual   = Mcorr - Mdot0;
        const double df          = f_of(Mcorr) - f_of(Mdot0);
        const double move_rel    = std::abs(dM_actual) / std::max(std::abs(Mdot0),1e-300);
        const bool overshoot = (std::abs(df) > kMaxDfEdd);
        const bool snapped   = cr.converged && (move_rel < 1e-7);   // corrector returned to base

        if (!cr.converged || overshoot) {
            ds *= 0.5;
            if (kDiag) std::printf("[ARC] step %d %s (merit=%.3e arc=%.3e df=%.4f) -> shrink ds=%.3e\n",
                                   step, !cr.converged ? "corrector FAILED" : "OVERSHOOT",
                                   cr.merit, cr.arc_resid, df, ds);
            if (ds < ds_floor) {
                if (kDiag) std::printf("[ARC] ds underflow (<%.2e) -> stop (honest)\n", ds_floor);
                break;
            }
            continue;
        }
        if (snapped) {
            ds = std::min(ds * 2.0, ds_ceil);
            if (kDiag) std::printf("[ARC] step %d SNAP-BACK (move_rel=%.2e) -> grow ds=%.3e\n",
                                   step, move_rel, ds);
            if (ds >= ds_ceil) {
                if (kDiag) std::printf("[ARC] ds hit ceiling while snapping back -> stall (honest stop)\n");
                break;   // can't escape the current point even at max step: honest stop
            }
            continue;
        }

        // Accepted.  Compute the NEW tangent at the corrected point and ORIENT it by the
        // KELLER tangent-continuity rule: the scaled prev·new dot.  This is the textbook
        // pseudo-arclength orientation — the tangent VECTOR rotates continuously through
        // the fold, so requiring (prev·new)_w > 0 keeps the continuation moving the same
        // way along the curve EVEN AS the Ṁ-component passes through zero at the turning
        // point.  The earlier sign(dM_actual)=sign(Mcorr−Mdot0) heuristic tied orientation
        // to the Ṁ-component alone; at a fold dṀ/ds→0 so that sign is pure corrector
        // noise → the ±1 thrash that bounced the continuation back down the stable branch.
        // slim_arclength_tangent already applies the same scaled-dot orientation against
        // t_prev internally, but we re-apply it explicitly here against the LAST accepted
        // tangent (`tangent`) so the continuity reference is the accepted curve direction,
        // not the corrector's intermediate t_prev seed.
        std::vector<double> new_tan;
        bool have_new = slim_arclength_tangent(Ucorr, in_with_mdot(in, Mcorr), opacity, tangent, new_tan);
        if (have_new) {
            std::vector<double> w0d; slim_arclength_weights(Ucorr, Mcorr, N, w0d);
            // Orient by the SECANT (direction of travel) in the scaled metric — robust to
            // the ~90° tangent rotation through the fold that makes the tangent·tangent dot
            // ambiguous.  Fall back to the prev-tangent dot only before a secant exists
            // (the very first accepted step) or if the secant is degenerate (≈0 motion).
            double sdot = 0.0, snrm = 0.0;
            if (have_secant) {
                for (int i = 0; i < n; ++i) {
                    const double sec = (Ucorr[i] - U_prevpt[i]) / w0d[i];
                    sdot += sec * (new_tan[i] / w0d[i]); snrm += sec * sec;
                }
                const double secM = (Mcorr - Mdot_prevpt) / w0d[n];
                sdot += secM * (new_tan[n] / w0d[n]); snrm += secM * secM;
            }
            if (have_secant && snrm > 1e-300) {
                if (sdot < 0.0) for (double& v : new_tan) v = -v;
            } else {
                double dot = 0.0; for (int i=0;i<m;++i) dot += (tangent[i]/w0d[i])*(new_tan[i]/w0d[i]);
                if (dot < 0.0) for (double& v : new_tan) v = -v;
            }
            tangent = new_tan; t_prev = new_tan;
        } else {
            if (kDiag) std::printf("[ARC] step %d: tangent degenerate at accepted point -> stop\n", step);
        }
        const int new_sign = (tangent[n] > 0) ? 1 : (tangent[n] < 0 ? -1 : 0);
        if (new_sign != 0 && new_sign != prev_sign) {
            result.crossed_fold = true;
            fold_shrink_steps = kFoldShrinkSteps;     // ride the next few steps gently
            if (kDiag) std::printf("[ARC] *** FOLD: Mdot_dot sign flip %d -> %d at f_Edd=%.4f ***\n",
                                   prev_sign, new_sign, f_of(Mcorr));
        }
        // Near-vertical tangent ⇒ we are APPROACHING the turning point (|Ṁ̇| component
        // is a tiny fraction of the unit tangent).  Force a small ds so the predictor
        // does not overshoot the (high-curvature) turn and land on the wrong branch.
        {
            std::vector<double> wv; slim_arclength_weights(U, Mdot, N, wv);
            const double mdot_frac = std::abs(tangent[n]/wv[n]);   // scaled Ṁ component (tangent is unit-normed in w)
            if (mdot_frac < kVertFrac) fold_shrink_steps = std::max(fold_shrink_steps, kFoldShrinkSteps);
        }
        if (new_sign != 0) prev_sign = new_sign;

        // Advance the secant reference BEFORE the swap: the point we are LEAVING (the
        // current accepted U,Mdot == U0,Mdot0) becomes the previous accepted point, so the
        // next step's secant is (next_accepted − this_accepted) — a clean 1-step secant.
        U_prevpt = U0; Mdot_prevpt = Mdot0; have_secant = true;
        U.swap(Ucorr); Mdot = Mcorr;
        SlimArclengthPoint pt = record_point(U, Mdot, ds, tangent[n], cr.merit);
        result.branch.push_back(pt);
        result.ok = true;
        ++steps_done;

        if (f_of(Mdot) >= 0.11) result.crossed_011 = true;
        if (f_of(Mdot) > result.max_f_Edd) {
            result.max_f_Edd = f_of(Mdot);
            unpack_profile(in_with_mdot(in, Mdot), opacity, U, result.top);
            result.top.converged = true;
            result.top.ell_in = U[4*N+0]; result.top.r_sonic = U[4*N+1];
            std::vector<double> Rf; slim_radial_residual(U, in_with_mdot(in,Mdot), opacity, Rf);
            result.top.final_residual = slim_scaled_residual_norm(U, Rf, in_with_mdot(in,Mdot));
        }

        if (kDiag)
            std::printf("[ARC] step %d ACCEPT f_Edd=%.4f r_s=%.4f H/r=%.3f beta=[%.2e,%.2e] "
                        "f_adv=[%.2e,%.2e] Mdot_dot_sign=%d merit=%.3e ds=%.3e\n",
                        step, pt.f_Edd, pt.r_sonic, pt.max_Hr, pt.beta_min, pt.beta_max,
                        pt.fadv_min, pt.fadv_max, pt.Mdot_dot_sign, pt.merit, ds);

        // Reached the high-Ṁ target on the UPPER branch (Ṁ̇>0 again past the fold).
        if (f_of(Mdot) >= f_target && prev_sign > 0) {
            if (kDiag) std::printf("[ARC] reached f_target=%.2f on upper branch -> done\n", f_target);
            break;
        }
        // Grow Δs on easy convergence (few corrector iters), bounded — but only once we
        // are clear of the fold neighbourhood.  While fold_shrink_steps is active, clamp
        // ds to the fold step so the predictor stays inside the local model through the
        // high-curvature turn.
        if (fold_shrink_steps > 0) {
            ds = std::min(ds, ds_fold);
            --fold_shrink_steps;
        } else if (cr.iters < 30) {
            ds = std::min(ds * 1.1, ds_ceil);
        }
    }

    if (kDiag)
        std::printf("[ARC] DONE: %d accepted steps, max_f_Edd=%.4f, crossed_011=%d, crossed_fold=%d\n",
                    steps_done, result.max_f_Edd, (int)result.crossed_011, (int)result.crossed_fold);
    return result;
}
} // namespace grrt
