#include "grrt/scene/slim_disk_radial.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>

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

} // namespace slim_detail

// ---------------------------------------------------------------------------
// Radial solver stub (implemented in Tasks 4-5)
// ---------------------------------------------------------------------------
SlimDiskRadial solve_slim_disk_radial(const SlimDiskInputs& in, const OpacityLUTs& opacity) {
    (void)in; (void)opacity;
    SlimDiskRadial out;
    return out;  // not yet implemented (Task 5 wires the Newton relaxation)
}
} // namespace grrt
