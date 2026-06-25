#ifndef GRRT_DISK_COLUMN_COUPLED_H
#define GRRT_DISK_COLUMN_COUPLED_H
#include "grrt/scene/disk_column_bvp.h"
#include "grrt/color/opacity.h"
#include "grrt_export.h"
#include <vector>
namespace grrt {
struct ColumnCoupledInputs {
    double Sigma_target;  ///< Σ [g/cm²] (radial unknown) — pinned via Σ0
    double Tc;            ///< midplane T_c [K] (radial unknown) — pinned via T(0)
    double shear, omega_z, alpha, rho_mid_guess;
    int n_nodes = 96; int max_iters = 300; double tol = 1e-8;
    double Teff_guess = 0.0;  ///< warm-start for T_eff (0 ⇒ derive from Tc)
    bool naive_seed = false;  ///< true ⇒ skip the secant bring-up; seed from a SINGLE
                              ///< base solve at a grey-relation T_eff guess (no Σ0
                              ///< root-find) — sits off the coupled root, exercising
                              ///< the equilibrated Newton from a non-trivial seed.
    bool allow_continuation = true;  ///< if the affine-invariant Newton fails from the
                                     ///< seed AND a consistent reference is available,
                                     ///< fall back to a (Σ,T_c) homotopy from that
                                     ///< reference to the target. false ⇒ primary only.
};
struct ColumnClosure {
    double F = 0.0;     ///< emergent flux σT_eff⁴ [erg/cm²/s]
    double z0 = 0.0;    ///< photosphere half-thickness [cm] (→ H)
    double eta3 = 0.0;  ///< filled by C2 (Task 4); leave 0 here
    double eta4 = 0.0;  ///< filled by C2 (Task 5); leave 0 here
    double T_eff = 0.0; ///< converged surface temperature (warm-start carrier)
    double f_adv = 0.0; ///< back-solved advected fraction (OUTPUT, not input)
    bool converged = false;
    ColumnBVPSolution sol;  ///< converged profile (for C2/C3)
};
/// C1: solve the column with (Σ, T_c) pinned and (T_eff, f_adv) freed as global
/// unknowns (augmented 4N+4 system). f_adv is a back-solved OUTPUT (S11 §3.1-3.2).
GRRT_EXPORT ColumnClosure solve_column_coupled(const ColumnCoupledInputs& in,
                                               const OpacityLUTs& op,
                                               const std::vector<double>* warm_start);

/// C2: vertical energy moment from a converged profile. η₃ = ∫E dz / ∫P dz with
/// E = (3/2)P_gas + 3·P_rad (P_rad=(a_rad/3)T⁴), P = total. η₄ is Task 5 (stub=0). Pure.
GRRT_EXPORT void column_moments(const ColumnBVPSolution& s, double& eta3, double& eta4);

/// C3: column-output sensitivities dC/d{Σ,T_c} for C={F,z0,η3,η4,f_adv}, via the IFT
/// through the augmented column Jacobian. Index [0]=∂/∂Σ_target, [1]=∂/∂T_c.
struct ColumnSensitivity { double dF[2], dz0[2], deta3[2], deta4[2], dfadv[2]; };
GRRT_EXPORT ColumnSensitivity column_sensitivity(const ColumnClosure& c,
                                                 const ColumnCoupledInputs& in,
                                                 const OpacityLUTs& op);
} // namespace grrt
#endif
