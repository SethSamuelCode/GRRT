// C1: the causality re-pose of the grey vertical-structure column.
//
// The base solver (disk_column_bvp.cpp, solve_column_bvp) is T_eff-DRIVEN: you
// hand it T_eff (and f_adv) and it solves for the column, returning Sigma0 as a
// free output. The radial slim-disk coupling needs the INVERSE map: hand it
// (Σ_target, T_c) and let the emergent flux F = σT_eff⁴ AND the advected fraction
// f_adv float as determined OUTPUTS.
//
// WHY f_adv FLOATS (S11 §3.1-3.2): the vertical structure is a TWO-parameter
// family (T_c, f_adv) ≡ (T_c, Σ). Given (Σ, T_c) the advected fraction f_adv is
// DETERMINED — it is an output, not an input. The earlier row-swap pinned
// (Σ, T_c) AND fixed f_adv (3 constraints on a 2-parameter family) → over-
// determined → spurious "folds". The numerical probe (slim_fadv_freedom_probe)
// confirmed: freeing f_adv makes the previously-"folded" targets (T_c, 1.3Σ) and
// (T_c, 0.7Σ) converge to physical solutions (f_adv = +1.13 and −0.63).
//
// THE RE-POSE — clean AUGMENTATION (4N+4):
//   State  U_c = [Pg,Q,T,z]×N + (z0, Σ0, T_eff, f_adv)   (length 4N+4)
//     * z0, Σ0 are the SAME two globals the base solver carries (Σ0 stays a genuine
//       unknown), PLUS T_eff and f_adv appended.
//   Residual (length 4N+4):
//     * Rows 0 .. 4N+1: the BASE column_residual EXACTLY, but reading
//       T_eff = U[4N+2] and f_adv = U[4N+3] from the state. This keeps ALL 6 base
//       BCs — including the surface blackbody closure T(N-1)=T_eff that the old
//       row-swap wrongly dropped, and Q(N-1)=σT_eff⁴.
//     * Row 4N+2 (NEW pin): T(0) − T_c = 0   (midplane temperature pinned).
//     * Row 4N+3 (NEW pin): Σ0 − Σ_target = 0 (column mass pinned).
//   Balanced: 4N+4 unknowns, 4N+4 equations. NO f_adv input parameter.
//
// REUSE: this TU is #included into the test TU AFTER disk_column_bvp.cpp, so the
// anonymous-namespace node_deriv and the static column residual/Jacobian machinery
// are in scope. We reuse node_deriv for the interior ODE rows and mirror the exact
// row layout of column_residual / analytic_jacobian.

#include "grrt/scene/disk_column_coupled.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <numbers>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace grrt {

// -----------------------------------------------------------------------------
// Coupled residual (augmented 4N+4). Σ0 = U[4N+1] (genuine unknown, pinned by the
// last row), T_eff = U[4N+2] and f_adv = U[4N+3] are freed globals.
// -----------------------------------------------------------------------------
static void coupled_column_residual(const std::vector<double>& U,
                                    const ColumnCoupledInputs& in,
                                    const OpacityLUTs& op, std::vector<double>& R) {
    using namespace constants;
    const int N = in.n_nodes;
    const double z0     = U[4*N];
    const double Sigma0 = U[4*N+1];   // genuine unknown (pinned by row 4N+3)
    const double T_eff  = U[4*N+2];   // freed global
    const double f_adv  = U[4*N+3];   // freed global (back-solved OUTPUT)
    const double dq = 1.0 / (N - 1);
    auto P = [&](int i){ return U[4*i+0]; };   // GAS pressure Pg (the state variable)
    auto Q = [&](int i){ return U[4*i+1]; };
    auto T = [&](int i){ return U[4*i+2]; };
    auto z = [&](int i){ return U[4*i+3]; };

    R.assign(4*N + 4, 0.0);
    int row = 0;
    for (int i = 0; i < N - 1; ++i) {
        Deriv di = node_deriv(P(i),   Q(i),   T(i),   z(i),   Sigma0, in.alpha, in.shear, in.omega_z, f_adv, op);
        Deriv dj = node_deriv(P(i+1), Q(i+1), T(i+1), z(i+1), Sigma0, in.alpha, in.shear, in.omega_z, f_adv, op);
        R[row++] = (p_total(P(i+1),T(i+1)) - p_total(P(i),T(i))) - 0.5*dq*(di.dP + dj.dP);
        R[row++] = Q(i+1) - Q(i) - 0.5*dq*(di.dQ + dj.dQ);
        R[row++] = T(i+1) - T(i) - 0.5*dq*(di.dT + dj.dT);
        R[row++] = z(i+1) - z(i) - 0.5*dq*(di.dz + dj.dz);
    }
    const double Q_surf  = surface_flux(T_eff);                 // = σ·U[4N+2]⁴
    const double rho_srf = std::max(rho_from_gas(P(N-1), T(N-1)), RHO_GHOST_FLOOR);
    const double kR_srf  = kappa_total(op, rho_srf, T(N-1));
    // --- 6 base BCs, IDENTICAL in form to column_residual ---
    R[row++] = Q(0);                                            // midplane Q=0
    R[row++] = z(0);                                            // midplane z=0
    R[row++] = Q(N-1) - Q_surf;                                // surface Q = σT_eff⁴
    R[row++] = T(N-1) - T_eff;                                 // surface BB closure (RESTORED)
    R[row++] = z(N-1) - z0;                                    // surface z=z0
    R[row++] = p_total(P(N-1),T(N-1)) - (2.0/3.0)*in.omega_z*in.omega_z*z0/kR_srf;
    // --- 2 NEW augmentation pins ---
    R[row++] = T(0)    - in.Tc;                                 // midplane T pin
    R[row++] = Sigma0  - in.Sigma_target;                      // column-mass pin
    assert(row == 4*N + 4);
}

// -----------------------------------------------------------------------------
// Coupled analytic Jacobian (4N+4)². Mirrors analytic_jacobian for rows 0..4N+1
// (KEEPING the Σ0 column — Σ0 is a genuine unknown again), then ADDS:
//   * col 4N+2 (T_eff): nonzero only on the surface rows — Q(N-1)−σT_eff⁴ → −4σT_eff³,
//     T(N-1)−T_eff → −1.
//   * col 4N+3 (f_adv): ∂(dQ/dq)/∂f_adv = −(α·shear·P_tot)/(1+f_adv)²·dz_dq per node,
//     entering the trapezoidal dQ rows.
//   * row 4N+2 (T(0)−T_c): ∂/∂T(0) = 1.
//   * row 4N+3 (Σ0−Σ_target): ∂/∂Σ0 = 1.
// -----------------------------------------------------------------------------
static void coupled_column_jacobian(const std::vector<double>& U,
                                    const ColumnCoupledInputs& in,
                                    const OpacityLUTs& op, std::vector<double>& J) {
    using namespace constants;
    const int N = in.n_nodes;
    const int n = 4*N + 4;
    const double z0     = U[4*N];
    const double Sigma0 = U[4*N+1];           // genuine unknown
    const double T_eff  = U[4*N+2];           // free unknown
    const double f_adv  = U[4*N+3];           // free unknown
    const double dq = 1.0 / (N - 1);
    const double oz2 = in.omega_z * in.omega_z;
    const double as  = in.alpha * in.shear;

    J.assign((size_t)n * n, 0.0);
    auto at = [&](int row, int col) -> double& { return J[(size_t)row * n + col]; };

    // Per-node partials, identical to analytic_jacobian's node_jac. The Σ0 partials
    // (*_dS) ARE written into the Σ0 column (cS) since Σ0 is a state variable. We also
    // need the per-node ∂(dQ/dq)/∂f_adv for the new f_adv column.
    struct NodeJac {
        double dP_dP, dP_dz, dP_dS;
        double dQ_dP, dQ_dT, dQ_dS, dQ_dfadv;
        double dT_dP, dT_dQ, dT_dT, dT_dS;
        double dz_dP, dz_dT, dz_dS;
    };
    const double fadv_inv = 1.0 / (1.0 + f_adv);
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
        Jn.dQ_dP = fadv_inv * as * Sigma0 / 2.0 * (1.0 / rho) * (1.0 - (Ptot / rho) * drho_dP);
        Jn.dQ_dT = fadv_inv * as * Sigma0 / 2.0 * (1.0 / rho) * (dPtot_dT - (Ptot / rho) * drho_dT);
        Jn.dQ_dS = fadv_inv * as * Ptot / (2.0 * rho);
        // dQ/dq = (as * P_tot /(1+f_adv)) * Sigma0/(2 rho).  ∂/∂f_adv = −1/(1+f_adv)·dQ/dq.
        //   = −fadv_inv² · as · P_tot · Sigma0/(2 rho).
        Jn.dQ_dfadv = -fadv_inv * fadv_inv * as * Ptot * Sigma0 / (2.0 * rho);
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

    const int cS    = 4*N + 1;    // Sigma0 column (genuine unknown)
    const int cTeff = 4*N + 2;    // T_eff column
    const int cfadv = 4*N + 3;    // f_adv column

    // --- Trapezoidal ODE rows --- (identical to analytic_jacobian, plus the f_adv
    //     column on the dQ rows).
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
            at(r, cS)   += -half_dq * (ji.dP_dS + jj.dP_dS);
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
            at(r, cS)   += -half_dq * (ji.dQ_dS + jj.dQ_dS);
            at(r, cfadv)+= -half_dq * (ji.dQ_dfadv + jj.dQ_dfadv);
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
            at(r, cS)   += -half_dq * (ji.dT_dS + jj.dT_dS);
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
            at(r, cS)   += -half_dq * (ji.dz_dS + jj.dz_dS);
        }
    }

    // --- Boundary-condition rows ---
    const int cz0 = 4*N;          // z0 column
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
    // R = Q(N-1) - σT_eff⁴:  ∂/∂Q(N-1) = 1 ; ∂/∂T_eff = -4σT_eff³
    { const int r = row++; at(r, 4*(N-1) + 1) = 1.0; at(r, cTeff) = -4.0 * sigma_SB * T_eff*T_eff*T_eff; }
    // R = T(N-1) - T_eff:    ∂/∂T(N-1) = 1 ; ∂/∂T_eff = -1   (surface BB closure RESTORED)
    { const int r = row++; at(r, 4*(N-1) + 2) = 1.0; at(r, cTeff) = -1.0; }
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
    // --- NEW augmentation pin rows ---
    // R = T(0) - T_c:        ∂/∂T(0) = 1
    at(row++, 4*0 + 2) = 1.0;
    // R = Sigma0 - Sigma_target: ∂/∂Sigma0 = 1
    at(row++, cS) = 1.0;
    assert(row == n);
}

// -----------------------------------------------------------------------------
// Central-difference FD of coupled_column_residual over the FULL augmented state —
// the analytic-Jacobian safety net (cross-check helper). Returns the dense (4N+4)² FD.
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
    // z0 (4N), Σ0 (4N+1), T_eff (4N+2) are large positive; f_adv (4N+3) is O(1) and may
    // be ≈0 — give it its own floor so its FD column stays resolvable.
    const double floorBig = 1e-7 * std::max(std::max(std::abs(U[4*N]), std::abs(U[4*N+1])),
                                            std::max(std::abs(U[4*N+2]), 1e-30));
    const double floorFadv = 1e-7 * std::max(std::abs(1.0 + U[4*N+3]), 1e-3);

    std::vector<double> Up, Um, Rp, Rm;
    for (int j = 0; j < n; ++j) {
        double absfloor;
        if (j < 4*N) { switch (j & 3) { case 0: absfloor=floorP; break; case 1: absfloor=floorQ; break;
                                        case 2: absfloor=floorT; break; default: absfloor=floorZ; } }
        else if (j == 4*N+3) absfloor = floorFadv;
        else absfloor = floorBig;
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
// Scale-balanced residual merit for the coupled residual. The first 6 BC rows use
// the base solver's scales; the 2 NEW pins use mT (T(0)−T_c) and mS (Σ0−Σ_target).
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
    const double mS = std::max(std::abs(U[4*N+1]), 1e-300);
    // Order matches coupled_column_residual's 8 BC/pin rows:
    //   Q(0), z(0), Q(surf), T(surf), z(surf)=z0, P(surf), T(0)-Tc, Σ0-Σ
    const double bc_scale[8] = { mQ, mZ, mQ, mT, mZ, mP, mT, mS };

    double sum = 0.0; int row = 0;
    for (int i = 0; i < N-1; ++i) {
        double sP=R[row++]/mP; double sQ=R[row++]/mQ; double sT=R[row++]/mT; double sZ=R[row++]/mZ;
        sum += sP*sP + sQ*sQ + sT*sT + sZ*sZ;
    }
    for (int b = 0; b < 8; ++b) { double s = R[row++] / std::max(bc_scale[b],1e-300); sum += s*s; }
    return std::sqrt(sum / (double)R.size());
}

// -----------------------------------------------------------------------------
// Map a ColumnCoupledInputs onto a ColumnInputs for the base (T_eff-driven) solver,
// at an explicit (T_eff, f_adv).
static ColumnInputs base_inputs_from(const ColumnCoupledInputs& in, double T_eff, double f_adv) {
    ColumnInputs b{};
    b.T_eff = T_eff; b.shear = in.shear; b.omega_z = in.omega_z; b.alpha = in.alpha;
    b.f_adv = f_adv; b.rho_mid_guess = in.rho_mid_guess;
    b.n_nodes = in.n_nodes; b.max_iters = in.max_iters; b.tol = in.tol;
    return b;
}

// Estimate a starting T_eff from (Σ_target, T_c) via the grey-diffusion relation.
// In an optically-thick grey column the midplane and surface temperatures obey
//   T_c⁴ ≈ (3/4) τ_mid T_eff⁴   (τ_mid = κ Σ/2 the half-column optical depth). With
// electron-scattering opacity κ ≈ κ_es this gives T_eff = T_c / (3 τ_mid / 4)^{1/4}.
// The augmented Newton then refines from a real solve; this only needs to land within
// its (wide) convergence basin.
static double estimate_Teff_guess(const ColumnCoupledInputs& in, const OpacityLUTs& op) {
    using namespace constants;
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
    const double tau_mid = std::max(0.5 * kappa * in.Sigma_target, 1.0);
    const double Teff = in.Tc / std::pow(0.75 * tau_mid, 0.25);
    return std::max(Teff, 1.0);
}

// Build the augmented coupled seed by a base (T_eff-driven) BRING-UP at f_adv=0 that
// lands the Σ0-pin already satisfied: a few SECANT iterations on Σ0(T_eff)−Σ_target=0
// (the base solver is monotone in T_eff — hotter columns are thinner ⇒ smaller Σ0).
// We pack that converged column into the 4N+4 state with U[4N+1]=Σ_target,
// U[4N+2]=T_eff, U[4N+3]=0.
//
// This is BRING-UP only (it produces a seed; the differentiable augmented Newton is the
// driver). It leaves REAL work for the augmented Newton: the seed is consistent at
// f_adv=0, so its midplane T(0) does NOT in general equal the pinned in.Tc (unless the
// pair is self-consistent, the round-trip case). The augmented Newton then moves T_eff
// AND f_adv to satisfy the T(0)=T_c pin — that is where f_adv is back-solved. A bare
// rough-T_eff single solve leaves the Σ0-pin residual huge and the stiff augmented step
// over-shoots under natural-monotonicity damping; the Σ0 bring-up removes that.
// Returns false if the base bring-up cannot converge (caller bails — no fabricated col).
static bool build_coupled_seed(const ColumnCoupledInputs& in, const OpacityLUTs& op,
                               std::vector<double>& U) {
    const int N = in.n_nodes;

    auto sigma_of = [&](double Te, ColumnBVPSolution& sout) -> double {
        ColumnInputs b = base_inputs_from(in, Te, 0.0);
        sout = solve_column_bvp(b, op);
        return sout.converged ? sout.Sigma0 : -1.0;
    };

    const double Te0_guess = (in.Teff_guess > 0.0) ? in.Teff_guess
                                                   : estimate_Teff_guess(in, op);
    ColumnBVPSolution s0, s1, sbest;
    double T0 = Te0_guess;
    double f0 = sigma_of(T0, s0) - in.Sigma_target;
    if (!s0.converged) {
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
        if (!s2.converged) {
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

    U.assign(4*N + 4, 0.0);
    for (int i = 0; i < N; ++i) {
        U[4*i+0] = sbest.P_gas[i]; U[4*i+1] = sbest.Q[i];
        U[4*i+2] = sbest.T[i];     U[4*i+3] = sbest.z[i];
    }
    U[4*N]   = sbest.z0;          // z0
    U[4*N+1] = in.Sigma_target;   // Σ0 pinned at target (Σ0-pin residual ≈ 0 at seed)
    U[4*N+2] = Te_best;           // T_eff (the Σ0-matched value)
    U[4*N+3] = 0.0;               // f_adv (back-solved OUTPUT; start at 0)
    return true;
}

// Build the augmented seed by a 2-D (T_eff, f_adv) BRING-UP that mirrors the independent
// slim_fadv_freedom_probe: outer Newton on the 2×2 map (T_eff, f_adv) → (T_c, Σ) with the
// base solve_column_bvp as the inner column solver, so the inner 4N+2 block residual is
// EXACTLY zero at every outer iterate (only the two outer constraints T(0)=T_c, Σ0=Σ are
// driven). This is far better conditioned than the monolithic 4N+4 solve far from the
// root, so it reaches the back-solved (T_eff, f_adv) for inconsistent (Σ,T_c) pairs where
// the monolithic Newton's basin is too small.
//
// This is BRING-UP ONLY (it produces a seed); the differentiable augmented Newton then
// POLISHES it (≈1 step) and IS the driver the IFT sensitivity attaches to. f_adv is the
// back-solved column OUTPUT. Returns false (empty U) if the bring-up cannot converge.
static bool build_coupled_seed_2d(const ColumnCoupledInputs& in, const OpacityLUTs& op,
                                  std::vector<double>& U) {
    const int N = in.n_nodes;
    const double Tc_t  = in.Tc;
    const double Sig_t = in.Sigma_target;

    // Inner column evaluation at (T_eff, f_adv); returns the achieved (T_c, Σ0) + profile.
    // (eval_col is a plain C++ lambda — the inner base-column solve — NOT a code-eval.)
    auto eval_col = [&](double Te, double fa, ColumnBVPSolution& s, double& Tc, double& Sig) -> bool {
        ColumnInputs b = base_inputs_from(in, Te, fa);
        s = solve_column_bvp(b, op);
        if (!s.converged || s.T.empty()) return false;
        Tc = s.T.front(); Sig = s.Sigma0; return true;
    };
    // Scaled residual r = (T_c/T_c_target − 1, Σ/Σ_target − 1).
    auto resid = [&](double Te, double fa, double& r0, double& r1, ColumnBVPSolution& s) -> bool {
        double Tc, Sig; if (!eval_col(Te, fa, s, Tc, Sig)) return false;
        r0 = Tc / Tc_t - 1.0; r1 = Sig / Sig_t - 1.0; return true;
    };

    double Te = (in.Teff_guess > 0.0) ? in.Teff_guess : estimate_Teff_guess(in, op);
    double fa = 0.0;
    ColumnBVPSolution s;
    double r0, r1;
    if (!resid(Te, fa, r0, r1, s)) {
        bool ok = false;
        for (double m : {0.5, 2.0, 0.25, 4.0, 0.1, 10.0}) {
            Te = ((in.Teff_guess > 0.0) ? in.Teff_guess : estimate_Teff_guess(in, op)) * m;
            if (resid(Te, fa, r0, r1, s)) { ok = true; break; }
        }
        if (!ok) return false;
    }

    bool solved = (std::max(std::abs(r0), std::abs(r1)) < 1e-6);
    for (int it = 0; it < 60 && !solved; ++it) {
        const double res_inf = std::max(std::abs(r0), std::abs(r1));
        // 2×2 forward-difference Jacobian of (r0,r1) w.r.t. (T_eff, f_adv).
        const double dTe = 1e-3 * Te;
        const double dfa = 1e-3;
        double rT0, rT1, rF0, rF1; ColumnBVPSolution st, sf;
        if (!resid(Te + dTe, fa, rT0, rT1, st)) break;
        if (!resid(Te, fa + dfa, rF0, rF1, sf)) break;
        const double J00 = (rT0 - r0) / dTe, J10 = (rT1 - r1) / dTe;
        const double J01 = (rF0 - r0) / dfa, J11 = (rF1 - r1) / dfa;
        const double det = J00 * J11 - J01 * J10;
        if (std::abs(det) < 1e-300) break;
        const double sTe = -( J11 * r0 - J01 * r1) / det;
        const double sFa = -(-J10 * r0 + J00 * r1) / det;

        // Line search: halve until ‖r‖_inf decreases AND the inner solve converges AND
        // 1+f_adv > 0 (the heating-reduction physicality gate).
        double lam = 1.0; bool acc = false;
        double nTe = Te, nFa = fa, nr0 = r0, nr1 = r1; ColumnBVPSolution ns;
        for (int bt = 0; bt < 30; ++bt) {
            const double tTe = Te + lam * sTe, tFa = fa + lam * sFa;
            if (tFa <= -0.999 || tTe <= 0.0) { lam *= 0.5; continue; }
            double tr0, tr1; ColumnBVPSolution ts;
            if (resid(tTe, tFa, tr0, tr1, ts) &&
                std::max(std::abs(tr0), std::abs(tr1)) < res_inf) {
                nTe = tTe; nFa = tFa; nr0 = tr0; nr1 = tr1; ns = ts; acc = true; break;
            }
            lam *= 0.5;
        }
        if (!acc) break;
        Te = nTe; fa = nFa; r0 = nr0; r1 = nr1; s = ns;
        solved = (std::max(std::abs(r0), std::abs(r1)) < 1e-6);
    }
    if (!solved) return false;

    U.assign(4*N + 4, 0.0);
    for (int i = 0; i < N; ++i) {
        U[4*i+0] = s.P_gas[i]; U[4*i+1] = s.Q[i];
        U[4*i+2] = s.T[i];     U[4*i+3] = s.z[i];
    }
    U[4*N]   = s.z0;
    U[4*N+1] = in.Sigma_target;   // pin satisfied to ~1e-6 by the 2-D bring-up
    U[4*N+2] = Te;
    U[4*N+3] = fa;                // back-solved advected fraction
    return true;
}

// LU pivot ratio max|U_kk| / min|U_kk| of A (a cheap conditioning proxy). Factors a
// COPY of A with the same partial-pivoting LU used by the Newton solve; returns -1 if
// singular. Used only for the seed diagnostic print (before/after equilibration).
static double lu_pivot_ratio(std::vector<double> A, int n) {
    std::vector<int> piv;
    if (!column_lu_factor(A, piv, n)) return -1.0;
    double pmax = 0.0, pmin = 1e300;
    for (int k = 0; k < n; ++k) {
        const double d = std::abs(A[(size_t)k*n+k]);
        pmax = std::max(pmax, d);
        pmin = std::min(pmin, d);
    }
    return (pmin > 0.0) ? pmax / pmin : -1.0;
}

// Build the Ruiz-equilibrated copy of J (for the pivot-ratio diagnostic only).
static void ruiz_scaled_copy(const std::vector<double>& J, int n, std::vector<double>& A) {
    A = J;
    constexpr int RUIZ_ITERS = 5;
    for (int sweep = 0; sweep < RUIZ_ITERS; ++sweep) {
        for (int i = 0; i < n; ++i) {
            double rmax = 0.0;
            for (int j = 0; j < n; ++j) rmax = std::max(rmax, std::abs(A[(size_t)i*n+j]));
            if (rmax > 0.0) { const double s = 1.0/std::sqrt(rmax); for (int j=0;j<n;++j) A[(size_t)i*n+j]*=s; }
        }
        for (int j = 0; j < n; ++j) {
            double cmax = 0.0;
            for (int i = 0; i < n; ++i) cmax = std::max(cmax, std::abs(A[(size_t)i*n+j]));
            if (cmax > 0.0) { const double s = 1.0/std::sqrt(cmax); for (int i=0;i<n;++i) A[(size_t)i*n+j]*=s; }
        }
    }
}

// -----------------------------------------------------------------------------
// Ruiz two-sided (inf-norm) equilibration + LU FACTOR of J. RETAINS Dr, Dc and the
// factored scaled matrix Ã = Dr·J·Dc so the affine-invariant Newton can (a) reuse the
// factor for cheap SIMPLIFIED corrections at trial points and (b) measure correction
// norms in SCALED solution space (the affine-invariant Deuflhard measure).
//
// The augmented coupled Jacobian is STIFF: its rows span many orders of magnitude
// (the surface-flux row ~1e17, the z rows ~1e4) and the −4σT_eff³ T_eff-column entry
// ~1e14. Ruiz equilibration computes diagonal scalings Dr (rows) and Dc (cols) so that
// Ã has all row and column inf-norms ≈ 1, collapsing the conditioning. We work with the
// EQUIVALENT scaled system Ã·ỹ = Dr·rhs and recover δ_j = Dc[j]·ỹ_j — a numerically-
// equivalent reconditioning: the root is unchanged.
struct EquilFactor {
    std::vector<double> LU;     // factored Ã (column_lu_factor in place)
    std::vector<int>    piv;
    std::vector<double> Dr, Dc; // row/column scalings
    int n = 0;
    bool ok = false;
};

static EquilFactor equilibrate_and_factor(const std::vector<double>& J, int n) {
    EquilFactor f;
    f.n = n;
    f.Dr.assign((size_t)n, 1.0);
    f.Dc.assign((size_t)n, 1.0);
    std::vector<double> A(J);  // working scaled copy Ã = Dr·J·Dc (built incrementally)

    constexpr int RUIZ_ITERS = 5;
    for (int sweep = 0; sweep < RUIZ_ITERS; ++sweep) {
        for (int i = 0; i < n; ++i) {
            double rmax = 0.0;
            for (int j = 0; j < n; ++j) rmax = std::max(rmax, std::abs(A[(size_t)i*n+j]));
            if (rmax > 0.0) {
                const double s = 1.0 / std::sqrt(rmax);
                f.Dr[i] *= s;
                for (int j = 0; j < n; ++j) A[(size_t)i*n+j] *= s;
            }
        }
        for (int j = 0; j < n; ++j) {
            double cmax = 0.0;
            for (int i = 0; i < n; ++i) cmax = std::max(cmax, std::abs(A[(size_t)i*n+j]));
            if (cmax > 0.0) {
                const double s = 1.0 / std::sqrt(cmax);
                f.Dc[j] *= s;
                for (int i = 0; i < n; ++i) A[(size_t)i*n+j] *= s;
            }
        }
    }
    f.ok = column_lu_factor(A, f.piv, n);
    f.LU.swap(A);
    return f;
}

// Solve Ã·ỹ = −Dr·R using the retained factor. Returns the SCALED correction ỹ
// (in ytilde) and the TRUE correction δ = Dc·ỹ (in delta).
static void simplified_correction(const EquilFactor& f, const std::vector<double>& R,
                                  std::vector<double>& ytilde, std::vector<double>& delta) {
    const int n = f.n;
    std::vector<double> b((size_t)n);
    for (int i = 0; i < n; ++i) b[i] = -f.Dr[i] * R[i];
    column_lu_solve(f.LU, f.piv, b, n);   // b now holds ỹ
    ytilde = b;
    delta.assign((size_t)n, 0.0);
    for (int j = 0; j < n; ++j) delta[j] = f.Dc[j] * b[j];
}

// Affine-invariant solution-space norm of a TRUE Newton correction δ: a BLOCK-SCALED
// RMS norm  ‖δ‖_w = sqrt( (1/n) Σ (δ_j / W_block(j))² ), where W_block is the
// characteristic magnitude (max |U|) of δ_j's VARIABLE TYPE (P, Q, T, z, or the globals
// z0/Σ0/T_eff/f_adv) over the whole column. See the original derivation: a Ruiz-scaled
// ‖ỹ‖ is a conditioning preconditioner, not a solution-space metric, so we measure the
// TRUE correction δ in this fixed weighting — affine-invariant w.r.t. residual rescaling
// (a function of J⁻¹R only) while staying a usable stopping/monotonicity measure.
static double correction_weighted_norm(const std::vector<double>& delta,
                                       const std::vector<double>& U, int N) {
    const int n = 4*N + 4;
    double sP=0, sQ=0, sT=0, sZ=0;
    for (int i = 0; i < N; ++i) {
        sP = std::max(sP, std::abs(U[4*i+0])); sQ = std::max(sQ, std::abs(U[4*i+1]));
        sT = std::max(sT, std::abs(U[4*i+2])); sZ = std::max(sZ, std::abs(U[4*i+3]));
    }
    const double wP=std::max(sP,1e-300), wQ=std::max(sQ,1e-300);
    const double wT=std::max(sT,1e-300), wZ=std::max(sZ,1e-300);
    const double wz0   = std::max(std::abs(U[4*N]),   1e-300);
    const double wS    = std::max(std::abs(U[4*N+1]), 1e-300);
    const double wTeff = std::max(std::abs(U[4*N+2]), 1e-300);
    // f_adv is O(1) and may be near zero; weight against (1+f_adv) so its correction is
    // measured against the physical scale of the heating-reduction factor.
    const double wfadv = std::max(std::abs(1.0 + U[4*N+3]), 1e-3);
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        double w;
        if (j < 4*N) { switch (j & 3) { case 0: w=wP; break; case 1: w=wQ; break;
                                        case 2: w=wT; break; default: w=wZ; } }
        else if (j == 4*N)   w = wz0;
        else if (j == 4*N+1) w = wS;
        else if (j == 4*N+2) w = wTeff;
        else                 w = wfadv;
        const double r = delta[j] / w;
        sum += r*r;
    }
    return std::sqrt(sum / (double)n);
}

// Physicality test for a coupled trial state (T>0, ρ>0 at every node, z0>0, Σ0>0,
// T_eff>0, 1+f_adv>0).
static bool coupled_state_physical(const std::vector<double>& U, int N) {
    for (int i = 0; i < N; ++i) {
        const double Pgi = U[4*i+0], Ti = U[4*i+2];
        if (!(Ti > 0.0) || !(rho_from_gas(Pgi, Ti) > 0.0)) return false;
    }
    if (!(U[4*N]   > 0.0)) return false;       // z0 > 0
    if (!(U[4*N+1] > 0.0)) return false;       // Σ0 > 0
    if (!(U[4*N+2] > 0.0)) return false;       // T_eff > 0
    if (!(1.0 + U[4*N+3] > 0.0)) return false; // 1 + f_adv > 0 (heating-reduction gate)
    return true;
}

// -----------------------------------------------------------------------------
// Affine-invariant (Deuflhard natural-monotonicity / NLEQ-ERR) Newton on the coupled
// residual, from an explicit seed U. Returns true on convergence (U is left at the
// root). Judges progress by the SCALED ordinary/simplified correction norm in solution
// space — invariant to residual scaling. No fabricated profile on failure.
static bool affine_invariant_newton(std::vector<double>& U, const ColumnCoupledInputs& in,
                                    const OpacityLUTs& op, int* iters_out = nullptr) {
    const int N = in.n_nodes;
    const int n = 4*N + 4;
    constexpr double LAMBDA_MIN = 1e-8;
    const bool dbg = std::getenv("AIN_DBG") != nullptr;

    std::vector<double> R, J, ytilde, delta, Utry, Rtry;
    double lambda_prev = 1.0;
    int used = 0;

    for (int it = 0; it < in.max_iters; ++it) {
        used = it + 1;
        coupled_column_residual(U, in, op, R);
        coupled_column_jacobian(U, in, op, J);

        EquilFactor f = equilibrate_and_factor(J, n);
        if (!f.ok) break;                                  // singular -> bail

        // Ordinary (equilibrated) Newton correction δ_k = Dc·ỹ_k and its affine-invariant
        // solution-space norm (the convergence measure).
        simplified_correction(f, R, ytilde, delta);
        const double dk_norm = correction_weighted_norm(delta, U, N);
        const double merit_k = coupled_residual_norm(U, R, in);
        if (dbg)
            std::printf("    [AIN] it=%d dk_norm=%.4e resid_merit=%.4e\n", it, dk_norm, merit_k);

        // Convergence: (a) the affine-invariant scaled correction is small AND the residual
        // is small; OR (b) the scaled residual merit is at its physical floor. Criterion (b)
        // is necessary because the achievable dk_norm floor is set by the INHERITED ~3e-4
        // opacity-Jacobian inexactness (the same bilinear-LUT slope limit the base solver
        // lives with): near the root the Newton direction is dominated by that inexactness,
        // so dk_norm bottoms out at ~few×1e-8 (just above a 1e-8 tol) while the residual is
        // already ~1e-10 — physically converged. The base solver likewise stops on a
        // merit floor (its 1e-6), not on an arbitrarily tight step norm. MERIT_FLOOR=1e-7
        // sits comfortably below any unconverged iterate's merit (~1e-3) and above the
        // ~1e-10 root-floor, so it never declares a false positive.
        constexpr double MERIT_FLOOR = 1e-7;
        // The merit-floor branch (b) additionally requires a moderately small step
        // (dk_norm < 100·tol): a residual at its physical floor only counts as converged if
        // the Newton step is not still large. This closes the one masking path — a large-step
        // iterate can no longer be declared converged on residual alone. The passing cases
        // already bottom at dk_norm ~1e-8 ≪ 100·tol, so this is a no-op for them; it only
        // blocks a false positive on a stalled-but-large-step iterate (the C4 warm-start risk).
        if ((dk_norm < in.tol && merit_k < std::max(1e-6, 1e4 * in.tol)) ||
            (merit_k < MERIT_FLOOR && dk_norm < 100.0 * in.tol)) {
            if (iters_out) *iters_out = used;
            return true;
        }

        // Globalization: residual-merit Armijo backtracking on the FULL Newton direction,
        // started at λ=1 each iteration (try the full quadratically-convergent step first).
        // The earlier Deuflhard natural-monotonicity damping (simplified correction with the
        // frozen x_k factor) over-rejects when the viscous heating is strongly nonlinear in
        // the freed f_adv — it stalls λ→0 on the inconsistent pairs even while the residual
        // drops. A monotone scaled-residual descent is the robust globalization (it is what
        // the independent slim_fadv_freedom_probe used); the affine-invariant correction
        // norm above stays the authoritative convergence criterion.
        constexpr double ARMIJO_C = 1e-4;
        // Trust-region cap on the initial step (mirrors solve_column_bvp's STEP_CAP). The
        // full Newton step in the stiff outer layers can change P/T by orders of magnitude
        // and overshoot the steep opacity/heating nonlinearity, so the residual line search
        // never recovers. Cap λ so no positive P or T node changes by more than STEP_CAP
        // fractionally, AND so the freed f_adv (whose heating enters as 1/(1+f_adv)) moves
        // by at most FADV_CAP per step — its nonlinearity is what stalled the inconsistent
        // pairs. The merit Armijo line search then refines from this capped start.
        constexpr double STEP_CAP = 0.5;
        constexpr double FADV_CAP = 0.25;
        double lambda = 1.0;
        for (int i = 0; i < N; ++i) {
            for (int c : {0, 2}) {                 // P (offset 0) and T (offset 2)
                const double u = U[4*i+c], d = delta[4*i+c];
                if (u != 0.0 && d != 0.0) {
                    const double frac = std::abs(d / u);
                    if (frac * lambda > STEP_CAP) lambda = STEP_CAP / frac;
                }
            }
        }
        {
            const double df = std::abs(delta[4*N+3]);   // |Δf_adv|
            if (df * lambda > FADV_CAP) lambda = FADV_CAP / df;
        }
        bool accepted = false;
        double merit_try = merit_k;
        for (int ls = 0; ls < 60; ++ls) {
            Utry.assign(U.begin(), U.end());
            for (int i = 0; i < n; ++i) Utry[i] += lambda * delta[i];

            if (!coupled_state_physical(Utry, N)) {
                lambda *= 0.5;                              // unphysical -> shrink
                if (lambda < LAMBDA_MIN) break;
                continue;
            }
            coupled_column_residual(Utry, in, op, Rtry);
            merit_try = coupled_residual_norm(Utry, Rtry, in);

            if (dbg)
                std::printf("      [AIN] ls=%d lambda=%.4e merit=%.4e/%.4e\n",
                            ls, lambda, merit_try, merit_k);
            if (merit_try <= (1.0 - ARMIJO_C * lambda) * merit_k) {
                accepted = true;
                break;
            }
            lambda *= 0.5;
            if (lambda < LAMBDA_MIN) break;
        }
        if (!accepted) break;                              // line search failed -> bail

        for (int i = 0; i < n; ++i) U[i] += lambda * delta[i];
        lambda_prev = lambda;  // (unused by the λ=1 restart; kept for diagnostics)
    }
    if (iters_out) *iters_out = used;
    return false;
}

// Recover the column surface density Σ a packed coupled state is consistent with:
// Σ = 2∫₀^{z0} ρ dz, trapezoid on the z-grid. (The augmented state stores Σ0 directly
// in U[4N+1], but the continuation anchor uses the integrated value for a perturbed-Σ
// warm start whose stored Σ0 already equals its own consistent Σ.)
static double sigma_from_state(const std::vector<double>& U, int N) {
    double half = 0.0;
    for (int i = 0; i + 1 < N; ++i) {
        const double rho_i = std::max(rho_from_gas(U[4*i+0],   U[4*i+2]),   0.0);
        const double rho_j = std::max(rho_from_gas(U[4*(i+1)+0], U[4*(i+1)+2]), 0.0);
        const double dz = std::abs(U[4*(i+1)+3] - U[4*i+3]);
        half += 0.5 * (rho_i + rho_j) * dz;
    }
    return 2.0 * half;
}

// (Σ,T_c)-continuation fallback: homotopy from a known-consistent state to the target.
// At each interpolation step re-converge the affine-invariant Newton, warm-started from
// the previous root. Adaptively halves the remaining step on a failed sub-solve.
static bool sigma_continuation(std::vector<double>& U, const ColumnCoupledInputs& target,
                               const OpacityLUTs& op,
                               double Sigma_start, double Tc_start,
                               int* substeps_out = nullptr) {
    constexpr double MIN_STEP = 1.0 / 256.0;
    double t = 0.0;
    double step = 1.0 / 8.0;
    std::vector<double> Uprev = U;
    int substeps = 0;

    while (t < 1.0 - 1e-12) {
        double t_try = std::min(1.0, t + step);
        ColumnCoupledInputs sub = target;
        sub.Sigma_target = Sigma_start + (target.Sigma_target - Sigma_start) * t_try;
        sub.Tc           = Tc_start    + (target.Tc           - Tc_start)    * t_try;

        std::vector<double> Utry = Uprev;
        ++substeps;
        if (affine_invariant_newton(Utry, sub, op, nullptr)) {
            Uprev.swap(Utry);
            t = t_try;
            step = std::min(1.0 / 8.0, step * 1.5);
        } else {
            step *= 0.5;
            if (step < MIN_STEP) {
                if (substeps_out) *substeps_out = substeps;
                return false;
            }
        }
    }
    U.swap(Uprev);
    if (substeps_out) *substeps_out = substeps;
    return true;
}

// -----------------------------------------------------------------------------
// C1 driver: augmented (Σ,T_c)-driven column via an AFFINE-INVARIANT Newton (Deuflhard
// natural monotonicity, NLEQ-ERR core). Seed -> primary affine-invariant Newton; on
// failure, if allowed and a consistent reference is available, fall back to a (Σ,T_c)
// homotopy. On success fills F=Q(N-1), z0=U[4N], T_eff=U[4N+2], f_adv=U[4N+3] and copies
// the profile into sol. On failure returns {converged=false} (no fabricated profile).
// -----------------------------------------------------------------------------
GRRT_EXPORT ColumnClosure solve_column_coupled(const ColumnCoupledInputs& in,
                                               const OpacityLUTs& op,
                                               const std::vector<double>* warm_start) {
    using namespace constants;
    const int N = in.n_nodes;
    const int n = 4*N + 4;
    ColumnClosure out;

    std::vector<double> U;
    bool have_consistent_ref = false;      // is U a converged (consistent) column?
    double ref_Sigma = 0.0, ref_Tc = 0.0;  // the (Σ,T_c) that U is consistent with

    assert(warm_start == nullptr || (int)warm_start->size() == n);
    if (warm_start && (int)warm_start->size() == n) {
        U = *warm_start;
        // A warm_start is a converged column ⇒ consistent at its OWN (Σ,T_c). Its Σ is
        // recovered from the profile (= its stored Σ0); its T_c is its midplane T(0).
        // These are the continuation start if a homotopy is needed.
        have_consistent_ref = true;
        ref_Sigma = sigma_from_state(U, N);
        ref_Tc    = U[2];                  // T(0) of the warm start
    } else {
        // Seed by the 2-D (T_eff, f_adv) bring-up (generalizes the 1-D f_adv=0 secant; it
        // lands BOTH the Σ0-pin and the T_c-pin to ~1e-6 with f_adv back-solved, so the
        // augmented Newton only polishes). Fall back to the f_adv=0 secant seed if the 2-D
        // bring-up cannot converge (the augmented Newton + continuation then do the work).
        if (!build_coupled_seed_2d(in, op, U) && !build_coupled_seed(in, op, U))
            return ColumnClosure{};   // no seed -> no coupled solution
    }

    // One-shot analytic-vs-FD Jacobian cross-check at the seed (the safety net the
    // spec requires). Printed.
    {
        std::vector<double> J;
        const double mism = coupled_jacobian_fd_mismatch(U, in, op);
        std::printf("  [coupled] analytic-vs-FD Jacobian max rel mismatch (seed) = %.3e\n", mism);
        coupled_column_jacobian(U, in, op, J);
        const double raw_ratio = lu_pivot_ratio(J, n);
        std::vector<double> Jeq; ruiz_scaled_copy(J, n, Jeq);
        const double eq_ratio = lu_pivot_ratio(Jeq, n);
        std::printf("  [coupled] LU pivot ratio: raw = %.3e  Ruiz-equilibrated = %.3e\n",
                    raw_ratio, eq_ratio);
    }

    // --- Primary: affine-invariant Newton from the seed. ---
    int iters = 0;
    bool converged = affine_invariant_newton(U, in, op, &iters);
    std::printf("  [coupled] affine-invariant Newton: converged=%d in %d iters\n",
                converged, iters);

    // --- Fallback: (Σ,T_c) continuation from a CONSISTENT anchor (rarely needed; the
    //     augmented system is balanced and should converge directly). ---
    auto try_continuation = [&](const std::vector<double>& Uref,
                                double s_Sigma, double s_Tc, const char* tag) -> bool {
        int substeps = 0;
        std::vector<double> Uc = Uref;
        const bool ok = sigma_continuation(Uc, in, op, s_Sigma, s_Tc, &substeps);
        std::printf("  [coupled] Σ-continuation (%s): converged=%d in %d sub-solves"
                    " (anchor Σ=%.4e,Tc=%.4e -> target Σ=%.4e,Tc=%.4e)\n",
                    tag, ok, substeps, s_Sigma, s_Tc, in.Sigma_target, in.Tc);
        if (ok) { U.swap(Uc); converged = true; }
        return ok;
    };
    if (!converged && in.allow_continuation) {
        // (A) fresh consistent anchor at the target Σ. build_coupled_seed lands a base
        //     column converged at (Σ_target, f_adv=0); that IS an augmented root at
        //     (Σ_target, Tc_anchor=U[2]) with f_adv=0 (all 6 base BCs hold, T(0)=Tc_anchor
        //     by definition, Σ0=Σ_target by the pin). So homotope T_c from Tc_anchor to
        //     in.Tc directly from this seed — NO re-solve precondition (the over-determined
        //     direct solve is exactly what we are routing around). f_adv tracks along the
        //     homotopy, reaching its back-solved value at the target.
        std::vector<double> Uanchor;
        if (build_coupled_seed(in, op, Uanchor)) {
            try_continuation(Uanchor, Uanchor[4*N+1], Uanchor[2], "fresh-anchor");
        }
        // (B) caller warm start as a (possibly off-target) consistent anchor.
        if (!converged && have_consistent_ref) {
            try_continuation(*warm_start, ref_Sigma, ref_Tc, "warm-start-anchor");
        }
    }

    if (!converged) {
        return ColumnClosure{};   // {converged=false}, no fabricated profile (honest no-root/fold)
    }

    // Final analytic-vs-FD cross-check at the CONVERGED state. Printed.
    {
        const double mism = coupled_jacobian_fd_mismatch(U, in, op);
        std::printf("  [coupled] analytic-vs-FD Jacobian max rel mismatch (converged) = %.3e\n", mism);
    }

    // Unpack: F = emergent flux = Q(N-1); z0 = U[4N]; T_eff = U[4N+2]; f_adv = U[4N+3].
    out.F     = U[4*(N-1) + 1];
    out.z0    = U[4*N];
    out.T_eff = U[4*N+2];
    out.f_adv = U[4*N+3];
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
    s.Sigma0 = U[4*N+1];   // converged Σ0 (= Σ_target by the pin)
    s.converged = true;

    double tau = 0.0;
    for (int i = 0; i + 1 < N; ++i) {
        const double kRi = kappa_total(op, std::max(s.rho[i],   RHO_GHOST_FLOOR), s.T[i]);
        const double kRj = kappa_total(op, std::max(s.rho[i+1], RHO_GHOST_FLOOR), s.T[i+1]);
        const double dz = std::abs(s.z[i+1] - s.z[i]);
        tau += 0.5 * (kRi*s.rho[i] + kRj*s.rho[i+1]) * dz;
    }
    s.tau_mid = tau;

    // C2: fill the vertical energy moment η₃ from the converged profile (η₄ stub=0).
    column_moments(out.sol, out.eta3, out.eta4);

    return out;
}

// -----------------------------------------------------------------------------
// C2: vertical energy moment of a converged column profile.
//   η₃ = ∫E dz / ∫P dz,  E = (3/2)P_gas + 3·P_rad,  P_rad = (a_rad/3)T⁴,  P = total.
// Trapezoidal in z, using the SAME radiation constant (constants::a_rad) the base
// column's p_total uses. One-zone reduction (constant β = P_gas/P_total): E =
// (3/2)βP + 3(1−β)P = (3−1.5β)P ⇒ η₃ → 3 − 1.5β. η₄ (S11 (1/Σ)∫ρz²dz) is Task 5.
// -----------------------------------------------------------------------------
GRRT_EXPORT void column_moments(const ColumnBVPSolution& s, double& eta3, double& eta4) {
    using namespace constants;
    const int N = (int)s.z.size();
    double intE = 0.0, intP = 0.0;
    auto Eden = [&](int k){
        const double Prad = (a_rad / 3.0) * std::pow(s.T[k], 4);
        return 1.5 * s.P_gas[k] + 3.0 * Prad;
    };
    for (int i = 0; i + 1 < N; ++i) {
        const double dz = s.z[i+1] - s.z[i];
        intE += 0.5 * (Eden(i) + Eden(i+1)) * dz;
        intP += 0.5 * (s.P[i]  + s.P[i+1])  * dz;
    }
    eta3 = (intP > 0.0) ? intE / intP : 0.0;

    // η₄ = (∫ρz²dz)/(∫ρdz) = density-weighted <z²> (S11 density 2nd moment; reference §23).
    // Convention-free over the stored half-profile (the both-faces/Σ factors cancel).
    double m2 = 0.0, m0 = 0.0;
    for (size_t i = 0; i + 1 < s.z.size(); ++i) {
        const double dz = s.z[i+1] - s.z[i];
        m2 += 0.5*(s.rho[i]*s.z[i]*s.z[i] + s.rho[i+1]*s.z[i+1]*s.z[i+1])*dz;
        m0 += 0.5*(s.rho[i] + s.rho[i+1])*dz;
    }
    eta4 = (m0 > 0.0) ? (m2 / m0) : 0.0;
}

} // namespace grrt
