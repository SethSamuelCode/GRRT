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
} // namespace slim_detail

/// Solve the relativistic transonic slim-disk radial structure
/// (see docs/superpowers/references/disk-physics-formulas.md §22).
GRRT_EXPORT SlimDiskRadial solve_slim_disk_radial(const SlimDiskInputs& in,
                                                  const OpacityLUTs& opacity);
} // namespace grrt
#endif
