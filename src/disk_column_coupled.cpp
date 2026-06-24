// C1: the causality re-pose of the grey vertical-structure column.
//
// The base solver (disk_column_bvp.cpp, solve_column_bvp) is T_eff-DRIVEN: you
// hand it T_eff and it solves for the column, returning Sigma0 as a free output.
// The radial slim-disk coupling needs the INVERSE map: hand it (Σ_target, T_c)
// and let T_eff (hence the emergent flux F = σT_eff⁴) FLOAT.
//
// We implement that inverse as a BC ROW-SWAP (the 2026-06-20 robustness decision),
// NOT a secant root-find on T_eff. The row-swap keeps ONE differentiable Newton
// solve, so the later IFT analytic sensitivity (Tasks 6/7) is exact.
//
// REUSE: this TU is #included into the test TU AFTER disk_column_bvp.cpp, so the
// anonymous-namespace node_deriv and the static column residual/Jacobian machinery
// are in scope. We reuse node_deriv for the interior ODE rows and mirror the exact
// row layout of column_residual / analytic_jacobian.
//
// THE RE-POSE (state length unchanged at 4N+2):
//   * Reinterpret the global slot U[4N+1]: it now carries T_eff (was Sigma0).
//   * Sigma0 is FIXED to the constant Σ_target everywhere it entered the interior
//     z-integration (dz/dq = Sigma0/(2ρ) inside node_deriv).
//   * Interior ODE rows, and the BC rows Q(0)=0, z(0)=0, Q(N-1)=σT_eff⁴,
//     z(N-1)=z0, P_tot(N-1)=(2/3)Ω_z²z0/κ_s are UNCHANGED IN FORM, except that the
//     two rows that read in.T_eff (the surface-Q and surface-T pins) now read the
//     free unknown U[4N+1].
//   * The surface-temperature pin  T(N-1) − T_eff = 0  is REPLACED by the
//     midplane pin  T(0) − T_c = 0.
// Net: Sigma0 is no longer a free unknown; T_eff is, determined by the surface-flux
// row Q(N-1) − σ·U[4N+1]⁴ = 0.

#include "grrt/scene/disk_column_coupled.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <numbers>
#include <vector>
#include <cstdio>

namespace grrt {

// -----------------------------------------------------------------------------
// Coupled residual (row-swapped). Σ0 is the CONSTANT in.Sigma_target; the global
// slot U[4N+1] carries the free T_eff.
// -----------------------------------------------------------------------------
static void coupled_column_residual(const std::vector<double>& U,
                                    const ColumnCoupledInputs& in,
                                    const OpacityLUTs& op, std::vector<double>& R) {
    using namespace constants;
    const int N = in.n_nodes;
    const double z0      = U[4*N];
    const double T_eff   = U[4*N+1];          // FREED global unknown (was Sigma0)
    const double Sigma0  = in.Sigma_target;   // FIXED constant (was U[4N+1])
    const double dq = 1.0 / (N - 1);
    auto P = [&](int i){ return U[4*i+0]; };   // GAS pressure Pg (the state variable)
    auto Q = [&](int i){ return U[4*i+1]; };
    auto T = [&](int i){ return U[4*i+2]; };
    auto z = [&](int i){ return U[4*i+3]; };

    R.assign(4*N + 2, 0.0);
    int row = 0;
    for (int i = 0; i < N - 1; ++i) {
        Deriv di = node_deriv(P(i),   Q(i),   T(i),   z(i),   Sigma0, in.alpha, in.shear, in.omega_z, in.f_adv, op);
        Deriv dj = node_deriv(P(i+1), Q(i+1), T(i+1), z(i+1), Sigma0, in.alpha, in.shear, in.omega_z, in.f_adv, op);
        R[row++] = (p_total(P(i+1),T(i+1)) - p_total(P(i),T(i))) - 0.5*dq*(di.dP + dj.dP);
        R[row++] = Q(i+1) - Q(i) - 0.5*dq*(di.dQ + dj.dQ);
        R[row++] = T(i+1) - T(i) - 0.5*dq*(di.dT + dj.dT);
        R[row++] = z(i+1) - z(i) - 0.5*dq*(di.dz + dj.dz);
    }
    const double Q_surf  = surface_flux(T_eff);                 // = σ·U[4N+1]⁴ (free)
    const double rho_srf = std::max(rho_from_gas(P(N-1), T(N-1)), RHO_GHOST_FLOOR);
    const double kR_srf  = kappa_total(op, rho_srf, T(N-1));
    R[row++] = Q(0);                                            // midplane Q=0
    R[row++] = z(0);                                            // midplane z=0
    R[row++] = Q(N-1) - Q_surf;                                // surface Q DETERMINES T_eff
    R[row++] = T(0) - in.Tc;                                    // ROW-SWAP: midplane T pin (was T(N-1)-T_eff)
    R[row++] = z(N-1) - z0;                                     // surface z=z0
    R[row++] = p_total(P(N-1),T(N-1)) - (2.0/3.0)*in.omega_z*in.omega_z*z0/kR_srf;
    assert(row == 4*N + 2);
}

// -----------------------------------------------------------------------------
// Coupled analytic Jacobian. Mirrors analytic_jacobian, then adjusts for the
// re-pose:
//   * The Sigma0 column (cS = 4N+1) is GONE — Sigma0 is now the constant
//     in.Sigma_target, so every ∂R/∂Sigma0 entry vanishes. That column instead
//     becomes ∂R/∂T_eff, which is nonzero ONLY for the surface-Q row
//     (Q(N-1) − σT_eff⁴ → ∂/∂T_eff = −4σT_eff³).
//   * The swapped temperature pins: drop ∂(T(N-1)−T_eff)/∂T(N-1)=1, add
//     ∂(T(0)−T_c)/∂T(0)=1.
//   * Interior rows that used Sigma0 now use the constant, so their ∂/∂U[4N+1]
//     entries are zero (achieved simply by not writing the cS column there).
// -----------------------------------------------------------------------------
static void coupled_column_jacobian(const std::vector<double>& U,
                                    const ColumnCoupledInputs& in,
                                    const OpacityLUTs& op, std::vector<double>& J) {
    using namespace constants;
    const int N = in.n_nodes;
    const int n = 4*N + 2;
    const double z0     = U[4*N];
    const double T_eff  = U[4*N+1];           // free unknown
    const double Sigma0 = in.Sigma_target;    // constant
    const double dq = 1.0 / (N - 1);
    const double oz2 = in.omega_z * in.omega_z;
    const double as  = in.alpha * in.shear;

    J.assign((size_t)n * n, 0.0);
    auto at = [&](int row, int col) -> double& { return J[(size_t)row * n + col]; };

    // Per-node partials, identical to analytic_jacobian's node_jac. The Sigma0
    // partials (*_dS) are STILL computed (the interior derivatives genuinely depend
    // on Σ0 through dz/dq) but are NOT written into the cS column, because Σ0 is a
    // constant here, not a state variable.
    struct NodeJac {
        double dP_dP, dP_dz, dP_dS;
        double dQ_dP, dQ_dT, dQ_dS;
        double dT_dP, dT_dQ, dT_dT, dT_dS;
        double dz_dP, dz_dT, dz_dS;
    };
    auto node_jac = [&](int i) -> NodeJac {
        const double Pg = U[4*i+0], Q = U[4*i+1], T = U[4*i+2], z = U[4*i+3];
        const double rho  = std::max(rho_from_gas(Pg, T), RHO_GHOST_FLOOR);
        const double Ptot = p_total(Pg, T);
        const double drho_dP = mu_fully_ionized * m_p / (k_B * T);
        const double drho_dT = -rho / T;
        const double dPtot_dT = (4.0 * a_rad / 3.0) * T*T*T;
        double kappa, dk_dlnrho, dk_dlnT;
        kappa_total_with_grad(op, rho, T, kappa, dk_dlnrho, dk_dlnT);
        const double dk_dP = (dk_dlnrho / rho) * drho_dP;
        const double dk_dT = (dk_dlnrho / rho) * drho_dT + dk_dlnT / T;

        NodeJac Jn{};
        Jn.dP_dP = 0.0;
        Jn.dP_dz = -oz2 * Sigma0 / 2.0;
        Jn.dP_dS = -oz2 * z / 2.0;
        const double fadv_inv = 1.0 / (1.0 + in.f_adv);
        Jn.dQ_dP = fadv_inv * as * Sigma0 / 2.0 * (1.0 / rho) * (1.0 - (Ptot / rho) * drho_dP);
        Jn.dQ_dT = fadv_inv * as * Sigma0 / 2.0 * (1.0 / rho) * (dPtot_dT - (Ptot / rho) * drho_dT);
        Jn.dQ_dS = fadv_inv * as * Ptot / (2.0 * rho);
        const double T3 = T*T*T;
        Jn.dT_dQ = -3.0 * kappa * Sigma0 / (32.0 * sigma_SB * T3);
        Jn.dT_dS = -3.0 * kappa * Q / (32.0 * sigma_SB * T3);
        Jn.dT_dP = -3.0 * Q * Sigma0 / (32.0 * sigma_SB * T3) * dk_dP;
        Jn.dT_dT = -3.0 * Q * Sigma0 / (32.0 * sigma_SB) * (dk_dT / T3 - 3.0 * kappa / (T3 * T));
        Jn.dz_dP = -Sigma0 / (2.0 * rho*rho) * drho_dP;
        Jn.dz_dT = -Sigma0 / (2.0 * rho*rho) * drho_dT;
        Jn.dz_dS = 1.0 / (2.0 * rho);
        return Jn;
    };

    // --- Trapezoidal ODE rows --- (identical to analytic_jacobian, sans cS column)
    int row = 0;
    for (int i = 0; i < N - 1; ++i) {
        const NodeJac ji = node_jac(i);
        const NodeJac jj = node_jac(i+1);
        const int ci = 4*i;
        const int cj = 4*(i+1);
        const double half_dq = 0.5 * dq;

        // --- R_P row ---
        {
            const int r = row++;
            const double Ti = U[ci+2], Tj = U[cj+2];
            const double dPtot_dTi = (4.0 * a_rad / 3.0) * Ti*Ti*Ti;
            const double dPtot_dTj = (4.0 * a_rad / 3.0) * Tj*Tj*Tj;
            at(r, ci+0) += -1.0;
            at(r, cj+0) +=  1.0;
            at(r, ci+2) += -dPtot_dTi;
            at(r, cj+2) +=  dPtot_dTj;
            at(r, ci+3) += -half_dq * ji.dP_dz;
            at(r, cj+3) += -half_dq * jj.dP_dz;
            // (Sigma0 is constant: no cS column.)
        }
        // --- R_Q row ---
        {
            const int r = row++;
            at(r, ci+1) += -1.0;
            at(r, cj+1) +=  1.0;
            at(r, ci+0) += -half_dq * ji.dQ_dP;
            at(r, ci+2) += -half_dq * ji.dQ_dT;
            at(r, cj+0) += -half_dq * jj.dQ_dP;
            at(r, cj+2) += -half_dq * jj.dQ_dT;
        }
        // --- R_T row ---
        {
            const int r = row++;
            at(r, ci+2) += -1.0;
            at(r, cj+2) +=  1.0;
            at(r, ci+0) += -half_dq * ji.dT_dP;
            at(r, ci+1) += -half_dq * ji.dT_dQ;
            at(r, ci+2) += -half_dq * ji.dT_dT;
            at(r, cj+0) += -half_dq * jj.dT_dP;
            at(r, cj+1) += -half_dq * jj.dT_dQ;
            at(r, cj+2) += -half_dq * jj.dT_dT;
        }
        // --- R_z row ---
        {
            const int r = row++;
            at(r, ci+3) += -1.0;
            at(r, cj+3) +=  1.0;
            at(r, ci+0) += -half_dq * ji.dz_dP;
            at(r, ci+2) += -half_dq * ji.dz_dT;
            at(r, cj+0) += -half_dq * jj.dz_dP;
            at(r, cj+2) += -half_dq * jj.dz_dT;
        }
    }

    // --- Boundary-condition rows ---
    const int cz0   = 4*N;        // z0 column
    const int cTeff = 4*N + 1;    // T_eff column (re-purposed from Sigma0)
    const double P_s = U[4*(N-1)+0], T_s = U[4*(N-1)+2];
    const double rho_s = std::max(rho_from_gas(P_s, T_s), RHO_GHOST_FLOOR);
    const double drho_s_dP = mu_fully_ionized * m_p / (k_B * T_s);
    const double drho_s_dT = -rho_s / T_s;
    const double dPtot_s_dT = (4.0 * a_rad / 3.0) * T_s*T_s*T_s;
    double kappa_s, dk_s_dlnrho, dk_s_dlnT;
    kappa_total_with_grad(op, rho_s, T_s, kappa_s, dk_s_dlnrho, dk_s_dlnT);
    const double dks_dP = (dk_s_dlnrho / rho_s) * drho_s_dP;
    const double dks_dT = (dk_s_dlnrho / rho_s) * drho_s_dT + dk_s_dlnT / T_s;

    // R = Q(0):              ∂/∂Q(0) = 1
    at(row++, 4*0 + 1) = 1.0;
    // R = z(0):              ∂/∂z(0) = 1
    at(row++, 4*0 + 3) = 1.0;
    // R = Q(N-1) - σT_eff⁴:  ∂/∂Q(N-1) = 1 ; ∂/∂T_eff = -4σT_eff³  (determines T_eff)
    { const int r = row++; at(r, 4*(N-1) + 1) = 1.0; at(r, cTeff) = -4.0 * sigma_SB * T_eff*T_eff*T_eff; }
    // ROW-SWAP: R = T(0) - T_c:  ∂/∂T(0) = 1   (was T(N-1)-T_eff)
    at(row++, 4*0 + 2) = 1.0;
    // R = z(N-1) - z0:       ∂/∂z(N-1) = 1, ∂/∂z0 = -1
    { const int r = row++; at(r, 4*(N-1) + 3) = 1.0; at(r, cz0) = -1.0; }
    // R = p_total(N-1) - (2/3) oz2 z0 / kappa_s:
    {
        const int r = row++;
        const double f = (2.0/3.0) * oz2 * z0 / (kappa_s * kappa_s);
        at(r, 4*(N-1) + 0) = 1.0        + f * dks_dP;
        at(r, 4*(N-1) + 2) = dPtot_s_dT + f * dks_dT;
        at(r, cz0)         = -(2.0/3.0) * oz2 / kappa_s;
    }
    assert(row == n);
}

// -----------------------------------------------------------------------------
// Central-difference FD of coupled_column_residual — the analytic-Jacobian safety
// net (cross-check helper). Returns the dense (4N+2)² FD Jacobian.
// -----------------------------------------------------------------------------
static void coupled_numerical_jacobian(const std::vector<double>& U,
                                       const ColumnCoupledInputs& in,
                                       const OpacityLUTs& op, std::vector<double>& J) {
    const int n = (int)U.size();
    const int N = in.n_nodes;
    J.assign((size_t)n * n, 0.0);

    double sP=0, sQ=0, sT=0, sZ=0;
    for (int i = 0; i < N; ++i) {
        sP = std::max(sP, std::abs(U[4*i+0])); sQ = std::max(sQ, std::abs(U[4*i+1]));
        sT = std::max(sT, std::abs(U[4*i+2])); sZ = std::max(sZ, std::abs(U[4*i+3]));
    }
    const double floorP = 1e-7 * std::max(sP, 1e-30), floorQ = 1e-7 * std::max(sQ, 1e-30);
    const double floorT = 1e-7 * std::max(sT, 1e-30), floorZ = 1e-7 * std::max(sZ, 1e-30);
    // z0 (slot 4N) and T_eff (slot 4N+1) are both large positive; one shared floor.
    const double floorG = 1e-7 * std::max(std::max(std::abs(U[4*N]), std::abs(U[4*N+1])), 1e-30);

    std::vector<double> Up, Um, Rp, Rm;
    for (int j = 0; j < n; ++j) {
        double absfloor;
        if (j < 4*N) { switch (j & 3) { case 0: absfloor=floorP; break; case 1: absfloor=floorQ; break;
                                        case 2: absfloor=floorT; break; default: absfloor=floorZ; } }
        else absfloor = floorG;
        const double delta = std::max(1e-7 * std::abs(U[j]), absfloor);
        Up = U; Um = U;
        Up[j] += delta; Um[j] -= delta;
        coupled_column_residual(Up, in, op, Rp);
        coupled_column_residual(Um, in, op, Rm);
        for (int row = 0; row < n; ++row)
            J[(size_t)row * n + j] = (Rp[row] - Rm[row]) / (2.0 * delta);
    }
}

// Max relative analytic-vs-FD Jacobian mismatch (row-scale-guarded, matching the
// base solver's column_jacobians cross-check metric). Returns the worst rel error.
static double coupled_jacobian_fd_mismatch(const std::vector<double>& U,
                                           const ColumnCoupledInputs& in,
                                           const OpacityLUTs& op) {
    std::vector<double> Ja, Jn;
    const int n = (int)U.size();
    coupled_column_jacobian(U, in, op, Ja);
    coupled_numerical_jacobian(U, in, op, Jn);
    std::vector<double> rowmax((size_t)n, 0.0);
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c)
        rowmax[r] = std::max(rowmax[r], std::abs(Jn[(size_t)r*n+c]));
    double max_rel = 0.0;
    for (int r = 0; r < n; ++r) for (int c = 0; c < n; ++c) {
        const double a = Ja[(size_t)r*n+c], num = Jn[(size_t)r*n+c];
        const double scale = std::max(std::abs(num), 1e-6 * rowmax[r]);
        const double rel = std::abs(a - num) / scale;
        if (rel > max_rel && std::abs(a - num) > 1e-6 * rowmax[r]) max_rel = rel;
    }
    return max_rel;
}

// -----------------------------------------------------------------------------
// Scale-balanced residual merit for the coupled residual. Mirrors the base
// solver's scaled_residual_norm, but the BC-row scales reflect the SWAPPED rows:
//   Q(0)=0, z(0)=0, Q(surf), T(0)-T_c, z(surf)=z0, P(surf) -> mQ,mZ,mQ,mT,mZ,mP
// (Order matches coupled_column_residual: the 4th BC row is now a T-row, T(0)-T_c.)
// -----------------------------------------------------------------------------
static double coupled_residual_norm(const std::vector<double>& U,
                                    const std::vector<double>& R,
                                    const ColumnCoupledInputs& in) {
    const int N = in.n_nodes;
    double mP=0, mQ=0, mT=0, mZ=0;
    for (int i = 0; i < N; ++i) {
        mP += std::abs(U[4*i+0]); mQ += std::abs(U[4*i+1]);
        mT += std::abs(U[4*i+2]); mZ += std::abs(U[4*i+3]);
    }
    mP=std::max(mP/N,1e-300); mQ=std::max(mQ/N,1e-300);
    mT=std::max(mT/N,1e-300); mZ=std::max(mZ/N,1e-300);
    const double bc_scale[6] = { mQ, mZ, mQ, mT, mZ, mP };

    double sum = 0.0; int row = 0;
    for (int i = 0; i < N-1; ++i) {
        double sP=R[row++]/mP; double sQ=R[row++]/mQ; double sT=R[row++]/mT; double sZ=R[row++]/mZ;
        sum += sP*sP + sQ*sQ + sT*sT + sZ*sZ;
    }
    for (int b = 0; b < 6; ++b) { double s = R[row++] / std::max(bc_scale[b],1e-300); sum += s*s; }
    return std::sqrt(sum / (double)R.size());
}

// -----------------------------------------------------------------------------
// Map a ColumnCoupledInputs onto a ColumnInputs for the base (T_eff-driven) solver.
static ColumnInputs base_inputs_from(const ColumnCoupledInputs& in, double T_eff) {
    ColumnInputs b{};
    b.T_eff = T_eff; b.shear = in.shear; b.omega_z = in.omega_z; b.alpha = in.alpha;
    b.f_adv = in.f_adv; b.rho_mid_guess = in.rho_mid_guess;
    b.n_nodes = in.n_nodes; b.max_iters = in.max_iters; b.tol = in.tol;
    return b;
}

// Estimate a starting T_eff from (Σ_target, T_c) via the grey-diffusion relation.
// In an optically-thick grey column the midplane and surface temperatures obey
//   T_c⁴ ≈ (3/4) τ_mid T_eff⁴   (interior-dominated; τ_mid = κ Σ/2 the half-column
// optical depth). With electron-scattering opacity κ ≈ κ_es this gives a closed
// first guess T_eff = T_c / (3 τ_mid / 4)^{1/4}. The base solver then refines from
// a real Newton solve; this only needs to land within its (wide) convergence basin.
static double estimate_Teff_guess(const ColumnCoupledInputs& in, const OpacityLUTs& op) {
    using namespace constants;
    // Representative midplane density from the Gaussian Σ0 = sqrt(2π) ρ_mid H closure.
    const double cs2_gas = k_B * in.Tc / (mu_fully_ionized * m_p);
    const double P_rad   = (a_rad / 3.0) * in.Tc * in.Tc * in.Tc * in.Tc;
    double rho_mid = std::max(in.rho_mid_guess, 1e-20);
    for (int it = 0; it < 30; ++it) {
        const double H = std::sqrt(cs2_gas + P_rad / rho_mid) / in.omega_z;
        const double rho_new = in.Sigma_target / (std::sqrt(2.0 * std::numbers::pi) * H);
        if (std::abs(rho_new - rho_mid) <= 1e-12 * rho_mid) { rho_mid = rho_new; break; }
        rho_mid = std::max(rho_new, 1e-30);
    }
    const double kappa = kappa_total(op, std::max(rho_mid, RHO_GHOST_FLOOR), in.Tc);
    const double tau_mid = std::max(0.5 * kappa * in.Sigma_target, 1.0);   // κ Σ/2, floored
    const double Teff = in.Tc / std::pow(0.75 * tau_mid, 0.25);
    return std::max(Teff, 1.0);
}

// Build the coupled initial state by REUSING the base T_eff-driven solver as a
// robust BRING-UP that lands the differentiable row-swapped Newton essentially AT
// its solution.
//
// The coupled solution (Σ0 fixed = Σ_target, T_eff free) is, by definition, the
// T_eff at which the base T_eff-driven column produces exactly Σ0(T_eff)=Σ_target.
// Because the base solver is monotone and well-conditioned in T_eff (hotter columns
// are thinner ⇒ smaller Σ0), a few SECANT iterations on Σ0(T_eff)−Σ_target=0 nail
// that T_eff with the full base profile in hand. We pack that converged column into
// the coupled state with U[4N+1]=T_eff. The row-swapped differentiable Newton then
// only POLISHES it (≈1 step) — but it is the differentiable residual/Jacobian that
// drives the final solve and that Tasks 6/7's IFT sensitivity attach to. The secant
// is bring-up ONLY (it produces a seed; it is NOT the differentiable driver).
//
// Note the base solve at this T_eff already satisfies T(0)≈T_c whenever (Σ_target,
// T_c) come from a self-consistent column (the physical use case), because both are
// outputs of the same T_eff-driven structure. Returns false (empty U) if the base
// bring-up cannot converge.
static bool build_coupled_seed(const ColumnCoupledInputs& in, const OpacityLUTs& op,
                               std::vector<double>& U) {
    const int N = in.n_nodes;

    // Σ0(T_eff) from the base solver; returns <0 if that base solve fails.
    auto sigma_of = [&](double Te, ColumnBVPSolution& sout) -> double {
        ColumnInputs b = base_inputs_from(in, Te);
        sout = solve_column_bvp(b, op);
        return sout.converged ? sout.Sigma0 : -1.0;
    };

    const double Te0_guess = (in.Teff_guess > 0.0) ? in.Teff_guess
                                                   : estimate_Teff_guess(in, op);
    ColumnBVPSolution s0, s1, sbest;
    double T0 = Te0_guess;
    double f0 = sigma_of(T0, s0) - in.Sigma_target;
    if (s0.Sigma0 <= 0.0 && !s0.converged) {
        // First guess failed to converge — try a small grid of fallback T_eff.
        bool ok = false;
        for (double m : {0.5, 2.0, 0.25, 4.0, 0.1, 10.0}) {
            T0 = Te0_guess * m;
            f0 = sigma_of(T0, s0) - in.Sigma_target;
            if (s0.converged) { ok = true; break; }
        }
        if (!ok) return false;
    }
    double T1 = T0 * 1.2;
    double f1 = sigma_of(T1, s1) - in.Sigma_target;
    if (!s1.converged) { T1 = T0 * 0.8; f1 = sigma_of(T1, s1) - in.Sigma_target; }
    if (!s1.converged) return false;

    sbest = (std::abs(f1) < std::abs(f0)) ? s1 : s0;
    double Te_best = (std::abs(f1) < std::abs(f0)) ? T1 : T0;
    const double sig_tol = 1e-10 * in.Sigma_target;

    for (int k = 0; k < 40; ++k) {
        if (std::abs(f1) < sig_tol) break;
        double denom = (f1 - f0);
        double T2 = (std::abs(denom) > 0.0) ? T1 - f1 * (T1 - T0) / denom : T1;
        if (!(T2 > 0.0)) T2 = 0.5 * (T0 + T1);
        ColumnBVPSolution s2;
        double f2 = sigma_of(T2, s2) - in.Sigma_target;
        if (!s2.converged) {  // step landed outside the basin: damp toward the bracket
            T2 = 0.5 * (T1 + T2);
            f2 = sigma_of(T2, s2) - in.Sigma_target;
            if (!s2.converged) break;
        }
        T0 = T1; f0 = f1; T1 = T2; f1 = f2;
        if (std::abs(f1) < std::abs(sbest.Sigma0 - in.Sigma_target) || !sbest.converged) {
            sbest = s2; Te_best = T2;
        }
    }
    if (!sbest.converged) return false;

    U.assign(4*N + 2, 0.0);
    for (int i = 0; i < N; ++i) {
        U[4*i+0] = sbest.P_gas[i]; U[4*i+1] = sbest.Q[i];
        U[4*i+2] = sbest.T[i];     U[4*i+3] = sbest.z[i];
    }
    U[4*N]   = sbest.z0;   // z0
    U[4*N+1] = Te_best;    // FREED global: T_eff (NOT Sigma0; Sigma0 is the fixed input)
    return true;
}

// -----------------------------------------------------------------------------
// C1 driver: row-swapped coupled Newton. Mirrors solve_column_bvp's loop:
//   residual -> coupled_column_jacobian -> column_lu_factor/solve -> step cap +
//   damped line search on the scaled merit -> converge at in.tol.
// On success fills F=Q(N-1), z0=U[4N], T_eff=U[4N+1], copies the profile into sol.
// On failure returns {converged=false} (no fabricated profile).
// -----------------------------------------------------------------------------
GRRT_EXPORT ColumnClosure solve_column_coupled(const ColumnCoupledInputs& in,
                                               const OpacityLUTs& op,
                                               const std::vector<double>* warm_start) {
    using namespace constants;
    const int N = in.n_nodes;
    const int n = 4*N + 2;
    ColumnClosure out;

    std::vector<double> U;
    assert(warm_start == nullptr || (int)warm_start->size() == n);
    if (warm_start && (int)warm_start->size() == n) {
        U = *warm_start;
    } else if (!build_coupled_seed(in, op, U)) {
        return ColumnClosure{};   // base seed solve failed -> no coupled solution
    }

    std::vector<double> R, J, Jcopy, rhs, Utry, Rtry;
    std::vector<int> piv;

    // One-shot analytic-vs-FD Jacobian cross-check at the seed (the safety net the
    // spec requires). Printed; must be < 1e-6 for a robust Newton.
    {
        const double mism = coupled_jacobian_fd_mismatch(U, in, op);
        std::printf("  [coupled] analytic-vs-FD Jacobian max rel mismatch (seed) = %.3e\n", mism);
    }

    coupled_column_residual(U, in, op, R);
    double merit = coupled_residual_norm(U, R, in);

    for (int it = 0; it < in.max_iters; ++it) {
        coupled_column_jacobian(U, in, op, J);
        Jcopy = J;
        if (!column_lu_factor(Jcopy, piv, n)) break;     // singular -> bail
        rhs.assign(R.begin(), R.end());
        for (double& r : rhs) r = -r;
        column_lu_solve(Jcopy, piv, rhs, n);
        const std::vector<double>& dU = rhs;

        // Trust-region cap: limit fractional change of any positive variable (P, T)
        // AND the free global T_eff to STEP_CAP, then run the merit line search.
        constexpr double STEP_CAP = 0.5;
        double lambda = 1.0;
        auto cap = [&](double u, double d){
            if (u != 0.0 && d != 0.0) {
                const double frac = std::abs(d / u);
                if (frac * lambda > STEP_CAP) lambda = STEP_CAP / frac;
            }
        };
        for (int i = 0; i < N; ++i) { cap(U[4*i+0], dU[4*i+0]); cap(U[4*i+2], dU[4*i+2]); }
        cap(U[4*N+1], dU[4*N+1]);   // T_eff (the freed global)

        bool accepted = false;
        double merit_try = merit;
        for (int ls = 0; ls < 40; ++ls) {
            Utry.assign(U.begin(), U.end());
            for (int i = 0; i < n; ++i) Utry[i] += lambda * dU[i];
            bool physical = true;
            for (int i = 0; i < N && physical; ++i) {
                const double Pgi = Utry[4*i+0], Ti = Utry[4*i+2];
                if (Ti <= 0.0 || rho_from_gas(Pgi, Ti) <= 0.0) physical = false;
            }
            if (physical && Utry[4*N+1] <= 0.0) physical = false;   // T_eff must stay > 0
            if (physical) {
                coupled_column_residual(Utry, in, op, Rtry);
                merit_try = coupled_residual_norm(Utry, Rtry, in);
                if (merit_try < merit) { accepted = true; break; }
            }
            lambda *= 0.5;
        }
        if (!accepted) break;

        double maxrel = 0.0;
        for (int i = 0; i < n; ++i) {
            const double rel = std::abs(lambda * dU[i]) / std::max(std::abs(U[i]), 1e-300);
            maxrel = std::max(maxrel, rel);
        }

        U.swap(Utry);
        R.swap(Rtry);
        merit = merit_try;

        if (maxrel < in.tol && merit < 1e-6) { out.converged = true; break; }
    }

    if (!out.converged) {
        return ColumnClosure{};   // {converged=false}, no fabricated profile
    }

    // Final analytic-vs-FD cross-check at the CONVERGED state (the meaningful one
    // for Tasks 6/7 — the sensitivity is evaluated at the solution). Printed.
    {
        const double mism = coupled_jacobian_fd_mismatch(U, in, op);
        std::printf("  [coupled] analytic-vs-FD Jacobian max rel mismatch (converged) = %.3e\n", mism);
    }

    // Unpack: F = emergent flux = Q(N-1); z0 = U[4N]; T_eff = U[4N+1].
    out.F     = U[4*(N-1) + 1];
    out.z0    = U[4*N];
    out.T_eff = U[4*N+1];
    out.converged = true;

    // Build the converged ColumnBVPSolution (mirror solve_column_bvp's unpack).
    ColumnBVPSolution& s = out.sol;
    s.q.resize(N); s.z.resize(N); s.P.resize(N); s.P_gas.resize(N);
    s.Q.resize(N); s.T.resize(N); s.rho.resize(N);
    for (int i = 0; i < N; ++i) {
        const double Pgi = U[4*i+0], Qi = U[4*i+1], Ti = U[4*i+2], zi = U[4*i+3];
        s.q[i] = (double)i / (N - 1);
        s.P_gas[i] = Pgi;
        s.P[i] = p_total(Pgi, Ti); s.Q[i] = Qi; s.T[i] = Ti; s.z[i] = zi;
        s.rho[i] = std::max(rho_from_gas(Pgi, Ti), 0.0);
    }
    s.z0     = U[4*N];
    s.Sigma0 = in.Sigma_target;   // fixed input (the re-pose's defining constant)
    s.converged = true;

    double tau = 0.0;
    for (int i = 0; i + 1 < N; ++i) {
        const double kRi = kappa_total(op, std::max(s.rho[i],   RHO_GHOST_FLOOR), s.T[i]);
        const double kRj = kappa_total(op, std::max(s.rho[i+1], RHO_GHOST_FLOOR), s.T[i+1]);
        const double dz = std::abs(s.z[i+1] - s.z[i]);
        tau += 0.5 * (kRi*s.rho[i] + kRj*s.rho[i+1]) * dz;
    }
    s.tau_mid = tau;

    return out;
}

} // namespace grrt
