#ifndef GRRT_DISK_COLUMN_BVP_H
#define GRRT_DISK_COLUMN_BVP_H

#include "grrt/color/opacity.h"
#include "grrt_export.h"
#include <vector>

namespace grrt {

/// Inputs for one disc column's vertical-structure BVP (all CGS).
struct ColumnInputs {
    double T_eff;        ///< effective temperature [K]
    double shear;        ///< Kerr shear rate |r dΩ/dr| [1/s] (drives viscous heating; exact, not (3/2)Ω)
    double omega_z;      ///< vertical epicyclic frequency Ω_z [1/s] (gravity)
    double alpha;        ///< Shakura-Sunyaev viscosity
    double f_adv = 0.0;  ///< radial advected fraction Q_adv/Q_rad; reduces flux generation by 1/(1+f_adv) (S11 Eq 13). Default 0 = thin/no reduction.
    double rho_mid_guess;///< midplane density estimate [g/cm^3] (seed; e.g. rho_est)
    int    n_nodes = 150;///< grid points on q ∈ [0,1]
    int    max_iters = 60;
    double tol = 1e-8;   ///< Newton convergence: max |ΔU/U|
};

/// Converged vertical structure on the column-mass-fraction grid q ∈ [0,1]
/// (index 0 = midplane, n_nodes-1 = surface). All CGS.
struct ColumnBVPSolution {
    std::vector<double> q;     ///< grid coordinate [0,1]
    std::vector<double> z;     ///< height [cm]
    std::vector<double> P;     ///< TOTAL pressure (gas + radiation) [erg/cm^3]
    std::vector<double> P_gas; ///< GAS pressure [erg/cm^3] — the cancellation-free Newton state variable; pack THIS (not P) for warm starts
    std::vector<double> Q;     ///< vertical flux [erg/cm^2/s]
    std::vector<double> T;     ///< temperature [K]
    std::vector<double> rho;   ///< density [g/cm^3]
    double z0 = 0.0;           ///< disc half-thickness [cm]   (= z_max)
    double Sigma0 = 0.0;       ///< full surface density Σ = 2∫₀^{z₀} ρ dz [g/cm^2]
    double tau_mid = 0.0;      ///< vertical optical depth midplane↔surface
    bool   converged = false;  ///< true if Newton met tol
    int    iters = 0;
    double final_residual = 0.0;
};

/// EOS: density from total pressure and temperature.
/// ρ = (P − a T⁴/3) · μ m_p / (k_B T). Returns <= 0 if radiation pressure
/// exceeds total pressure (non-physical input) — caller must guard.
GRRT_EXPORT double eos_rho(double P, double T);

/// Solve the grey vertical-structure BVP for one column (Newton relaxation).
/// @param warm_start optional initial state U (length 4·n_nodes+2) from a
///        converged neighbouring column (numerical continuation). When null or
///        wrong-sized, the solver builds its own flux-balanced analytic seed.
///        On a size mismatch the solution's `converged`/`iters` fields therefore
///        reflect the cold-start run, not a warm one.
/// On non-convergence the returned solution has `converged=false`, all profile
/// vectors (q,z,P,Q,T,rho) empty, and z0/Sigma0/tau_mid = 0; callers MUST check
/// `converged` before indexing the profile.
GRRT_EXPORT ColumnBVPSolution solve_column_bvp(const ColumnInputs& in,
                                               const OpacityLUTs& opacity,
                                               const std::vector<double>* warm_start = nullptr);

/// Test hook: build a crude EOS-valid state and evaluate the BVP residual on it.
/// Fills U (length 4N+2) and R (length 4N+2). For unit tests only.
GRRT_EXPORT void column_residual_test(const ColumnInputs& in, const OpacityLUTs& op,
                                      std::vector<double>& U, std::vector<double>& R);

/// Test hook: build the analytic seed and fill a DENSE (4N+2)×(4N+2) numerical
/// (finite-difference) Jacobian of the residual, row-major in Jdense; n = 4N+2.
GRRT_EXPORT void column_numerical_jacobian_test(const ColumnInputs& in, const OpacityLUTs& op,
                                                std::vector<double>& Jdense, int& n);

/// Test hook: at the seed, fill BOTH the analytic (Ja) and numerical (Jn) dense
/// (4N+2)^2 Jacobians, row-major; n = 4N+2. For the analytic-vs-numerical cross-check.
GRRT_EXPORT void column_jacobians_test(const ColumnInputs& in, const OpacityLUTs& op,
                                       std::vector<double>& Ja, std::vector<double>& Jn, int& n);

} // namespace grrt

#endif
