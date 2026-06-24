#ifndef GRRT_DISK_COLUMN_COUPLED_H
#define GRRT_DISK_COLUMN_COUPLED_H
#include "grrt/scene/disk_column_bvp.h"
#include "grrt/color/opacity.h"
#include "grrt_export.h"
#include <vector>
namespace grrt {
struct ColumnCoupledInputs {
    double Sigma_target;  ///< Σ [g/cm²] (radial unknown)
    double Tc;            ///< midplane T_c [K] (radial unknown)
    double f_adv;         ///< advected fraction (radial input)
    double shear, omega_z, alpha, rho_mid_guess;
    int n_nodes = 96; int max_iters = 300; double tol = 1e-8;
    double Teff_guess = 0.0;  ///< warm-start for T_eff (0 ⇒ derive from Tc)
    bool naive_seed = false;  ///< true ⇒ skip the secant bring-up; seed from a SINGLE
                              ///< base solve at a grey-relation T_eff guess (no Σ0
                              ///< root-find) — sits off the coupled root, exercising
                              ///< the equilibrated Newton from a non-trivial seed.
};
struct ColumnClosure {
    double F = 0.0;     ///< emergent flux σT_eff⁴ [erg/cm²/s]
    double z0 = 0.0;    ///< photosphere half-thickness [cm] (→ H)
    double eta3 = 0.0;  ///< filled by C2 (Task 4); leave 0 here
    double eta4 = 0.0;  ///< filled by C2 (Task 5); leave 0 here
    double T_eff = 0.0; ///< converged surface temperature (warm-start carrier)
    bool converged = false;
    ColumnBVPSolution sol;  ///< converged profile (for C2/C3)
};
/// C1: solve the column with (Σ, T_c, f_adv) fixed and T_eff/F free (BC row-swap).
GRRT_EXPORT ColumnClosure solve_column_coupled(const ColumnCoupledInputs& in,
                                               const OpacityLUTs& op,
                                               const std::vector<double>* warm_start);
} // namespace grrt
#endif
