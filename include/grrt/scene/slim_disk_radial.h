#ifndef GRRT_SLIM_DISK_RADIAL_H
#define GRRT_SLIM_DISK_RADIAL_H
#include "grrt/color/opacity.h"
#include "grrt_export.h"
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

/// Solve the relativistic transonic slim-disk radial structure
/// (see docs/superpowers/references/disk-physics-formulas.md §22).
GRRT_EXPORT SlimDiskRadial solve_slim_disk_radial(const SlimDiskInputs& in,
                                                  const OpacityLUTs& opacity);
} // namespace grrt
#endif
