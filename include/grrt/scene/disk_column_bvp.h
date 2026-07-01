#ifndef GRRT_DISK_COLUMN_BVP_H
#define GRRT_DISK_COLUMN_BVP_H

#include "grrt/color/opacity.h"
#include "grrt/math/constants.h"
#include "grrt_export.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace grrt {

/// Closed-form gas+radiation thermodynamic helpers for mixing-length convection
/// (refinement #13; see docs/superpowers/references/disk-physics-formulas.md §24).
/// Both are opus+Wolfram-derived from the gas+radiation specific entropy
/// (s = R_g ln(T^{3/2}/ρ) + 4aT³/3ρ, monatomic γ=5/3), no partial ionization.
/// β = P_gas / P_total ∈ [0,1]. Header-inline so both src/disk_column_bvp.cpp and
/// the linked tests can reach them as grrt::detail_bvp::<fn>.
namespace detail_bvp {

/// Adiabatic temperature gradient ∇_ad = dlnT/dlnP at constant entropy, gas+radiation.
///   ∇_ad = (4 − 3β) / (16 − 12β − (3/2)β²)
/// Limits: β=1 (pure gas) → 0.40 = (γ−1)/γ for γ=5/3; β=0 (pure radiation) → 0.25.
inline double nabla_ad(double beta) {
    const double b = std::clamp(beta, 0.0, 1.0);
    return (4.0 - 3.0 * b) / (16.0 - 12.0 * b - 1.5 * b * b);
}

/// Specific heat at constant pressure C_p for the gas+radiation mixture [erg/(g K)].
///   C_p = R_g · (16/β² − 12/β − 3/2),   R_g = k_B / (μ m_p)
/// Limits: β=1 → (5/2)R_g (monatomic ideal gas); β→0 → ∞ (radiation-dominated).
inline double c_p_gas_rad(double beta) {
    using namespace grrt::constants;
    const double b = std::max(beta, 1e-12);   // diverges as β→0; floor to stay finite
    const double R_g = k_B / (mu_fully_ionized * m_p);
    return R_g * (16.0 / (b * b) - 12.0 / b - 1.5);
}

/// Solve the MLT efficiency cubic  A y^3 + w y^2 + w^2 y - w = 0  for the unique y>0
/// (§24 Eq 20). F(0)=-w<0, monotone increasing for y>0 (A,w>0) -> guarded Newton.
inline double mlt_solve_y(double A, double w) {
    double y = (w > 1.0) ? 1.0/w : 1.0;
    for (int it = 0; it < 40; ++it) {
        const double F  = A*y*y*y + w*y*y + w*w*y - w;
        const double dF = 3.0*A*y*y + 2.0*w*y + w*w;
        const double step = F/dF;
        y -= step;
        if (y <= 0.0) y = 1e-12;
        if (std::abs(step) <= 1e-12*(std::abs(y)+1e-12)) break;
    }
    return std::max(y, 0.0);
}

/// Returns dT/dz (the quantity node_deriv multiplies by dz_dq). Stable (∇_rad≤∇_ad):
/// bare radiative gradient (BIT-IDENTICAL). Unstable: ∇_conv·(T/Ptot)·dP/dz. §24 Eqs 16-21.
inline double convective_gradient(double rho, double T, double Ptot, double Q, double kR,
                                  double z, double omega_z,
                                  double& nabla_out, bool& convective) {
    using namespace grrt::constants;
    const double dTdz_rad = -3.0*kR*rho*Q/(16.0*sigma_SB*T*T*T);
    const double dPdz     = -rho*omega_z*omega_z*z;
    convective = false;
    if (!(z > 0.0) || !(dPdz < 0.0) || !(Q > 0.0)) { nabla_out = 0.0; return dTdz_rad; }
    const double Pg   = Ptot - (a_rad/3.0)*T*T*T*T;
    const double beta = (Ptot > 0.0) ? std::clamp(Pg/Ptot, 0.0, 1.0) : 1.0;
    const double nab_rad = (Ptot/T) * (dTdz_rad/dPdz);
    const double nab_ad  = nabla_ad(beta);
    nabla_out = nab_rad;
    if (nab_rad <= nab_ad) return dTdz_rad;                 // STABLE -> bit-identical radiative
    const double Hp  = Ptot / (rho*omega_z*omega_z*z + std::sqrt(Ptot*rho)*omega_z);
    const double Hml = Hp;                                    // α_MLT = 1 (Sądowski 2011)
    const double tau = rho*kR*Hml;
    // Optically-thin mixing length (τ_ml→0, e.g. κ_R→0): convection carries no flux —
    // the MLT cubic's own continuous limit is ∇_conv→∇_rad (Wolfram-verified). Guard the
    // τ=0 singularity in `pref` (else pref→∞ → w=0,A=0 → mlt_solve_y(0,0)=NaN cascade).
    if (!(tau > 0.0)) { nabla_out = nab_rad; return dTdz_rad; }
    const double Cp  = c_p_gas_rad(beta);
    const double delta = (4.0 - 3.0*beta)/std::max(beta,1e-12);   // SIGN-RESOLVED (>0)
    const double T6 = T*T*T*T*T*T;
    const double pref = (3.0+tau*tau)/(3.0*tau);
    const double inv_w2 = pref*pref
        * (omega_z*omega_z * z * Hml*Hml * rho*rho * Cp*Cp) / (512.0*sigma_SB*sigma_SB*T6*Hp)
        * delta * (nab_rad - nab_ad);
    const double w = 1.0/std::sqrt(std::max(inv_w2, 1e-300));
    const double A = (9.0/4.0) * (tau*tau)/(3.0 + tau*tau);
    const double y = mlt_solve_y(A, w);
    double frac = std::clamp(y*(y + w), 0.0, 1.0);
    const double nab_conv = nab_ad + (nab_rad - nab_ad)*frac;
    nabla_out = nab_conv;
    convective = true;
    return nab_conv * (T/Ptot) * dPdz;
}

} // namespace detail_bvp

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

/// Dense LU with partial pivoting, split into reusable factor + solve so one
/// factorization (O(n³)) serves many RHS back-substitutions (O(n²) each) — the
/// IFT column-sensitivity needs ∂R_c/∂p solved against the same ∂R_c/∂U_c factor.
GRRT_EXPORT bool column_lu_factor(std::vector<double>& A, std::vector<int>& piv, int n);
GRRT_EXPORT void column_lu_solve(const std::vector<double>& LU, const std::vector<int>& piv,
                                 std::vector<double>& b, int n);

} // namespace grrt

#endif
