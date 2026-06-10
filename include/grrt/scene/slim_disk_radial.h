#ifndef GRRT_SLIM_DISK_RADIAL_H
#define GRRT_SLIM_DISK_RADIAL_H
#include "grrt/color/opacity.h"
#include "grrt_export.h"
#include <cmath>
#include <vector>
namespace grrt {

/// Inputs for the relativistic transonic slim-disk radial solve.
/// Geometric mechanics (G=c=1, M sets the scale); CGS thermodynamics via r_g.
struct SlimDiskInputs {
    double mass = 1.0;      ///< M (geometric)
    double spin = 0.0;      ///< a, |a|<M
    double mdot = 0.0;      ///< accretion rate Mdot [g/s]
    double alpha = 0.1;     ///< Shakura-Sunyaev viscosity
    double r_g = 0.0;       ///< gravitational radius [cm] (geometric->cm)
    double r_in = 0.0;      ///< inner edge of the grid [M] (>= horizon)
    double r_out = 50.0;    ///< outer edge [M]
    int    n_nodes = 400;
    int    max_iters = 100;
    double tol = 1e-8;
};

/// Converged transonic radial structure. Index 0 = inner edge, back = outer.
struct SlimDiskRadial {
    std::vector<double> r;       ///< radius [M]
    std::vector<double> Sigma;   ///< surface density [g/cm^2]
    std::vector<double> V;       ///< radial velocity (corotating frame), <0 = inflow
    std::vector<double> Omega;   ///< orbital angular velocity [1/s]
    std::vector<double> Tc;      ///< midplane temperature [K]
    std::vector<double> H;       ///< scale height [cm]
    std::vector<double> f_adv;   ///< advected fraction Q_adv/Q_vis
    double ell_in = 0.0;         ///< inner specific angular momentum (eigenvalue)
    double r_sonic = 0.0;        ///< sonic radius [M]
    bool   converged = false;
    int    iters = 0;
    double final_residual = 0.0;
};

/// Kerr relativistic factor functions (geometric units, G=c=1; prograde, equatorial).
/// Verified against formula reference §22. r in units of M.
namespace slim_detail {
inline double omega_k(double M, double a, double r) {          // prograde Kerr Keplerian Ω_K
    return std::sqrt(M) / (r * std::sqrt(r) + a * std::sqrt(M));
}
inline double calC(double M, double a, double r) {             // 𝒞 = 1 − 3M/r + 2a√M/r^{3/2}
    return 1.0 - 3.0 * M / r + 2.0 * a * std::sqrt(M) / (r * std::sqrt(r));
}
inline double calD(double M, double a, double r) {             // 𝒟 = 1 − 2M/r + a²/r² = Δ/r²
    return 1.0 - 2.0 * M / r + a * a / (r * r);
}
inline double calH(double M, double a, double r) {             // ℋ = 1 − 4a√M/r^{3/2} + 3a²/r²
    return 1.0 - 4.0 * a * std::sqrt(M) / (r * std::sqrt(r)) + 3.0 * a * a / (r * r);
}
inline double omega_perp2(double M, double a, double r) {      // vertical epicyclic Ω_⊥² = Ω_K²·ℋ  (= omega_z_sq)
    const double ok = omega_k(M, a, r);
    return ok * ok * calH(M, a, r);
}
inline double kerr_delta(double M, double a, double r) {       // Δ = r² − 2Mr + a²
    return r * r - 2.0 * M * r + a * a;
}
inline double kerr_A(double M, double a, double r) {           // A = r⁴ + r²a² + 2Mra²
    return r*r*r*r + r*r*a*a + 2.0*M*r*a*a;
}

/// Thermodynamic state returned by the one-zone vertical closure.
/// All quantities are in CGS.
struct OneZoneState {
    double H     = 0.0;  ///< scale height [cm]
    double rho_mid = 0.0;///< midplane density [g/cm^3]
    double c_s   = 0.0;  ///< total sound speed [cm/s] (= H · Ω_⊥ by construction)
    double p_mid = 0.0;  ///< total midplane pressure p_gas + p_rad [erg/cm^3]
    double p_gas = 0.0;  ///< gas pressure at midplane [erg/cm^3]
    double p_rad = 0.0;  ///< radiation pressure at midplane [erg/cm^3]
    double P     = 0.0;  ///< vertically-integrated pressure 2·p_mid·H [erg/cm^2] (α-stress; trap #9)
    double S     = 0.0;  ///< specific entropy (gas + radiation) [erg/(g·K)]
    double mu    = 0.0;  ///< mean molecular weight used
};

/// One-zone height-integrated vertical closure.
///
/// Given the surface density Sigma [g/cm²], midplane temperature Tc [K], and
/// radius r [M], returns the self-consistent scale height, midplane
/// density/pressure, integrated pressure, sound speed, and specific entropy.
///
/// The scale height follows from hydrostatic balance H = c_s / Ω_⊥ with the
/// TOTAL sound speed c_s² = p_mid/ρ_mid.  Substituting p_mid = p_gas + p_rad
/// and ρ_mid = Σ/(2H) leads to the quadratic
///   Ω_⊥² H² − b H − c_s_gas² = 0 ,  b = 2 a_rad T_c⁴ / (3 Σ)
/// whose positive root is H = (b + sqrt(b² + 4 Ω_⊥² c_s_gas²)) / (2 Ω_⊥²).
/// In the gas-dominated limit b→0 this reduces to H = c_s_gas / Ω_⊥.
///
/// A single fixed-point pass is done for μ: first solve with μ = mu_fully_ionized,
/// then look up μ(ρ_mid, T_c) and recompute.
///
/// Specific entropy (per unit mass, additive constant irrelevant — only dS/dr enters Q_adv):
///   S = (k_B / (μ m_p)) ln(T_c^{3/2} / ρ_mid) + (4 a_rad T_c³) / (3 ρ_mid)
///
/// @param Sigma  surface density [g/cm²]
/// @param Tc     midplane temperature [K]
/// @param r      radius [M]
/// @param in     slim-disk inputs (M, a, r_g, α)
/// @param op     opacity LUTs (for μ look-up)
GRRT_EXPORT OneZoneState one_zone_closure(double Sigma, double Tc, double r,
                                         const SlimDiskInputs& in, const OpacityLUTs& op);

/// Prograde Keplerian specific angular momentum ℓ_K = u_φ on a circular
/// equatorial Kerr orbit at radius r [M] (Bardeen-Press-Teukolsky 1972).
/// Used by the seed and the outer/regularity boundary conditions.
GRRT_EXPORT double ell_kepler(double M, double a, double r);

/// Prograde Kerr ISCO radius (Bardeen-Press-Teukolsky 1972), in units of M.
/// Inputs are dimensional M and a; the BPT72 formula scales linearly with M,
/// so we evaluate the dimensionless r_isco(a/M) and multiply by M.  For M=1 it
/// matches the bare BPT72 expression used by the tests.  The seed uses r_isco as
/// the inner-node anchor and the energy outer BC uses it as the NT zero-torque radius.
GRRT_EXPORT double isco_prograde(double M, double a);

/// Invert the equatorial Kerr Ω↔ℓ relation: given the covariant specific
/// angular momentum ℓ = u_φ at radius r, return the orbital angular velocity
/// Ω = u^φ/u^t [geometric, 1/M]. A few Newton iterations seeded from Ω_K.
/// See slim_disk_radial.cpp for the derivation; documented as a robust local solve.
GRRT_EXPORT double omega_from_ell(double M, double a, double r, double ell);

} // namespace slim_detail

/// Build a crude thin-disk seed state vector U (length 4N+2) for the radial
/// residual: power-law Σ, mass-conservation V<0, Keplerian ℓ, NT-ish T_c, plus
/// the two globals (ℓ_in, r_s). Refined by the relaxation (Task 5). See .cpp.
GRRT_EXPORT std::vector<double> build_thin_disk_seed(const SlimDiskInputs& in,
                                                     const OpacityLUTs& opacity);

/// Evaluate the transonic radial residual R (length 4N+2) for state U.
/// Row layout (see disk-physics-formulas.md §22/§23 and the .cpp header comment):
///   [0 .. N-1]      mass conservation (algebraic, per node)
///   [N .. 2N-1]     angular momentum (algebraic, per node)
///   [2N .. 3N-2]    radial-momentum transonic ODE (trapezoidal, N-1 intervals)
///   [3N-1 .. 4N-3]  energy Q_vis=Q_rad+Q_adv ODE (trapezoidal, N-1 intervals)
///   [4N-2, 4N-1]    outer boundary conditions (ℓ_out matched-slope radial-equilibrium
///                   extrapolation, outer-node §23 energy balance for T_c,out)
///   [4N, 4N+1]      sonic-point regularity (𝒟₀(r_s)=0, 𝒩₁(r_s)=0)
GRRT_EXPORT void slim_radial_residual(const std::vector<double>& U,
                                      const SlimDiskInputs& in,
                                      const OpacityLUTs& opacity,
                                      std::vector<double>& R);

/// Solve the relativistic transonic slim-disk radial structure
/// (see docs/superpowers/references/disk-physics-formulas.md §22).
GRRT_EXPORT SlimDiskRadial solve_slim_disk_radial(const SlimDiskInputs& in,
                                                  const OpacityLUTs& opacity);
} // namespace grrt
#endif
