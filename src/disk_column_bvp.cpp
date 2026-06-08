#include "grrt/scene/disk_column_bvp.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <numbers>
#include <vector>

namespace grrt {

double eos_rho(double P, double T) {
    using namespace constants;
    const double P_gas = P - (a_rad / 3.0) * T * T * T * T;   // P - P_rad
    if (P_gas <= 0.0 || T <= 0.0) return 0.0;                 // non-physical
    return P_gas * mu_fully_ionized * m_p / (k_B * T);
}

} // namespace grrt

namespace {
using namespace grrt::constants;

// Numerical guards (not physics):
constexpr double RHO_GHOST_FLOOR = 1e-30;  // g/cm^3 — guards 1/rho on transient non-physical Newton iterates (~12 dex below any real disk density)
constexpr double T_LUT_MIN       = 3000.0; // K — opacity-LUT lower temperature edge; clamp lookups into the table

// Surface flux Q at the photosphere = sigma_SB * T_eff^4.
inline double surface_flux(double T_eff) {
    return sigma_SB * T_eff * T_eff * T_eff * T_eff;
}

// Total Rosseland opacity for diffusion / optical depth: absorption + electron scattering.
inline double kappa_total(const grrt::OpacityLUTs& op, double rho, double T) {
    const double Tk = std::max(T, T_LUT_MIN);
    return op.lookup_kappa_ross(rho, Tk) + op.lookup_kappa_es(rho, Tk);
}

// Total Rosseland opacity (absorption + electron scattering) and its LOGARITHMIC
// gradients d/dlnrho, d/dlnT at (rho, T).
//
// The gradients are central finite differences of the SAME kappa_total() the
// residual evaluates (ross + es), in log rho and log T, mirroring its T_LUT_MIN
// clamp. The step is deliberately SMALL (h = 1e-3 in natural-log): the LUT lookups
// are bilinear interpolants whose slope is piecewise constant within a cell, so the
// residual's local derivative is the slope of the current cell. A wide stencil (e.g.
// the h = 0.01 used by kappa_ross_with_grad) can straddle a cell boundary and return
// the average of two adjacent cell slopes, disagreeing with the residual's local
// slope by ~5% near an edge — which would corrupt the opacity-coupled Jacobian rows.
// h = 1e-3 stays inside a single cell here (cell width ~4e-3 in log10) while staying
// well clear of roundoff, so the analytic opacity gradient matches the residual's
// local interpolation slope (validated by the Task-7 numerical cross-check).
inline void kappa_total_with_grad(const grrt::OpacityLUTs& op, double rho, double T,
                                  double& k, double& dk_dlnrho, double& dk_dlnT) {
    const double Tk = std::max(T, T_LUT_MIN);
    auto kt = [&](double rr, double tt) {
        return op.lookup_kappa_ross(rr, tt) + op.lookup_kappa_es(rr, tt);
    };
    constexpr double h = 1e-3;  // natural-log step; stays within one LUT cell
    // NOTE: below T_LUT_MIN the minus-side T probe (Tk*exp(-h)) clamps to the LUT
    // edge, so the T-gradient there is effectively one-sided (asymmetric). This is
    // the shallow-opacity surface layer; the analytic-vs-numerical cross-check
    // (2.5e-9) confirms the resulting Jacobian is well within tolerance.
    k         = kt(rho, Tk);
    dk_dlnrho = (kt(rho * std::exp(h), Tk) - kt(rho * std::exp(-h), Tk)) / (2.0 * h);
    dk_dlnT   = (kt(rho, Tk * std::exp(h)) - kt(rho, Tk * std::exp(-h))) / (2.0 * h);
}

struct Deriv { double dP, dQ, dT, dz; };
// dX/dq at a node; dz/dq = Sigma0/(2 rho).
Deriv node_deriv(double P, double Q, double T, double z,
                 double Sigma0, double alpha, double shear, double omega_z,
                 const grrt::OpacityLUTs& op) {
    const double rho = std::max(grrt::eos_rho(P, T), RHO_GHOST_FLOOR);
    const double kR  = kappa_total(op, rho, T);
    const double dz_dq = Sigma0 / (2.0 * rho);
    Deriv d;
    // Each dX/dq = (dX/dz) * dz/dq, with dz/dq = Sigma0/(2 rho):
    //   dP/dz = -rho Omega_z^2 z            vertical hydrostatic (gravity uses Omega_z)
    //   dQ/dz = alpha * shear * P           Shakura-Sunyaev viscous heating (shear = exact Kerr |r dΩ/dr|)
    //   dT/dz = -3 kR rho Q /(16 sigma T^3) grey Rosseland radiative diffusion (16 = 4ac/sigma)
    //   dz/dq = Sigma0/(2 rho)              column-mass coordinate (2 = both disc faces)
    d.dz = dz_dq;
    d.dP = (-rho * omega_z * omega_z * z) * dz_dq;
    d.dQ = ( alpha * shear * P) * dz_dq;     // viscous heating: q+ = alpha P |r dΩ/dr|
    d.dT = (-3.0 * kR * rho * Q / (16.0 * sigma_SB * T * T * T)) * dz_dq;
    return d;
}
} // namespace

namespace grrt {

static void column_residual(const std::vector<double>& U, const ColumnInputs& in,
                            const OpacityLUTs& op, std::vector<double>& R) {
    using namespace constants;
    const int N = in.n_nodes;
    const double z0 = U[4*N], Sigma0 = U[4*N+1];
    const double dq = 1.0 / (N - 1);
    auto P = [&](int i){ return U[4*i+0]; };
    auto Q = [&](int i){ return U[4*i+1]; };
    auto T = [&](int i){ return U[4*i+2]; };
    auto z = [&](int i){ return U[4*i+3]; };

    R.assign(4*N + 2, 0.0);
    int row = 0;
    for (int i = 0; i < N - 1; ++i) {
        Deriv di = node_deriv(P(i),   Q(i),   T(i),   z(i),   Sigma0, in.alpha, in.shear, in.omega_z, op);
        Deriv dj = node_deriv(P(i+1), Q(i+1), T(i+1), z(i+1), Sigma0, in.alpha, in.shear, in.omega_z, op);
        R[row++] = P(i+1) - P(i) - 0.5*dq*(di.dP + dj.dP);
        R[row++] = Q(i+1) - Q(i) - 0.5*dq*(di.dQ + dj.dQ);
        R[row++] = T(i+1) - T(i) - 0.5*dq*(di.dT + dj.dT);
        R[row++] = z(i+1) - z(i) - 0.5*dq*(di.dz + dj.dz);
    }
    const double Q_surf  = surface_flux(in.T_eff);
    const double rho_srf = std::max(eos_rho(P(N-1), T(N-1)), RHO_GHOST_FLOOR);
    const double kR_srf  = kappa_total(op, rho_srf, T(N-1));
    R[row++] = Q(0);                                                   // midplane Q=0
    R[row++] = z(0);                                                   // midplane z=0
    R[row++] = Q(N-1) - Q_surf;                                        // surface Q
    R[row++] = T(N-1) - in.T_eff;                                      // surface T
    R[row++] = z(N-1) - z0;                                            // surface z=z0
    R[row++] = P(N-1) - (2.0/3.0)*in.omega_z*in.omega_z*z0/kR_srf;     // surface pressure
    assert(row == 4*N + 2);
}

/// Build a radiation-aware Gaussian column seed state (length 4N+2).
/// Isothermal (T = T_eff), linear z grid up to 4H, Gaussian rho.
/// H uses the TOTAL (gas + radiation) sound speed so the seed width is correct
/// in the radiation-dominated (hot) regime; Newton converges from a seed that
/// matches the true column width rather than one that is far too thin.
/// Used by both column_residual_test and the numerical Jacobian hook.
static std::vector<double> build_seed(const ColumnInputs& in) {
    using namespace constants;
    const int N = in.n_nodes;
    const double cs2_gas = k_B * in.T_eff / (mu_fully_ionized * m_p);  // gas-only sound speed^2
    const double rho_mid = in.rho_mid_guess;
    const double P_rad   = (a_rad / 3.0) * in.T_eff * in.T_eff * in.T_eff * in.T_eff;
    // Total sound speed^2: (P_gas_mid + P_rad) / rho_mid = cs2_gas + P_rad/rho_mid
    const double cs2_total = cs2_gas + P_rad / rho_mid;
    const double H   = std::sqrt(cs2_total) / in.omega_z;
    const double z0  = 4.0 * H;                                   // ~99.97% of a Gaussian column
    const double Sigma0  = std::sqrt(2.0 * std::numbers::pi) * rho_mid * H;
    std::vector<double> U(4*N + 2, 0.0);
    for (int i = 0; i < N; ++i) {
        const double q  = (double)i / (N - 1);                    // 0 midplane → 1 surface
        const double zi = z0 * q;
        // 1e-20 floor: keep P>0 at the surface node (q=1, exp→0) so the first
        // residual eval is finite. Distinct from RHO_GHOST_FLOOR (1e-30), which
        // guards 1/rho on transient Newton iterates.
        const double rho = std::max(rho_mid * std::exp(-zi*zi/(2.0*H*H)), 1e-20);
        const double Ti = in.T_eff;                              // isothermal seed (Newton warms the midplane)
        const double Pi = rho * cs2_gas + (a_rad/3.0)*Ti*Ti*Ti*Ti;  // P_gas(local rho) + P_rad; cs2_gas not double-counted
        const double Qi = surface_flux(in.T_eff) * q;            // 0 midplane → σT_eff^4 surface
        U[4*i+0]=Pi; U[4*i+1]=Qi; U[4*i+2]=Ti; U[4*i+3]=zi;
    }
    U[4*N]=z0; U[4*N+1]=Sigma0;
    return U;
}

/// Dense central-difference Jacobian J[row*n + col] = ∂R_row/∂U_col.
static void numerical_jacobian(const std::vector<double>& U, const ColumnInputs& in,
                               const OpacityLUTs& op, std::vector<double>& J) {
    const int n = (int)U.size();
    const int N = in.n_nodes;
    J.assign((size_t)n * n, 0.0);

    // Per-variable absolute step floors. Some variables are EXACTLY zero at the
    // seed (Q and z at the midplane, q=0), where a purely relative step collapses
    // to ~1e-37 and the central difference returns garbage/0 — corrupting those
    // columns of the reference Jacobian (e.g. the structural -1 in the z-difference
    // row). Floor each component's step at a small fraction of the column's typical
    // magnitude for that variable type so the difference stays resolvable.
    double sP=0, sQ=0, sT=0, sZ=0;
    for (int i = 0; i < N; ++i) {
        sP = std::max(sP, std::abs(U[4*i+0])); sQ = std::max(sQ, std::abs(U[4*i+1]));
        sT = std::max(sT, std::abs(U[4*i+2])); sZ = std::max(sZ, std::abs(U[4*i+3]));
    }
    const double floorP = 1e-7 * std::max(sP, 1e-30), floorQ = 1e-7 * std::max(sQ, 1e-30);
    const double floorT = 1e-7 * std::max(sT, 1e-30), floorZ = 1e-7 * std::max(sZ, 1e-30);
    // z0 and Sigma0 differ in scale but are both large at the seed, so one
    // shared (conservative) floor is fine for the two-global layout.
    const double floorG = 1e-7 * std::max(std::max(std::abs(U[4*N]), std::abs(U[4*N+1])), 1e-30); // z0, Sigma0

    std::vector<double> Up, Um, Rp, Rm;
    for (int j = 0; j < n; ++j) {
        // Per-component relative step with a per-variable absolute floor; central
        // differences are insensitive to the exact value over a wide range. 1e-7
        // gives ~1e-9 Jacobian accuracy here, far inside the <1e-3 tolerance of the
        // Task-7 analytic cross-check.
        double absfloor;
        if (j < 4*N) { switch (j & 3) { case 0: absfloor=floorP; break; case 1: absfloor=floorQ; break;
                                        case 2: absfloor=floorT; break; default: absfloor=floorZ; } }
        else absfloor = floorG;
        const double delta = std::max(1e-7 * std::abs(U[j]), absfloor);
        Up = U; Um = U;
        Up[j] += delta; Um[j] -= delta;
        column_residual(Up, in, op, Rp);
        column_residual(Um, in, op, Rm);
        for (int row = 0; row < n; ++row)
            J[(size_t)row * n + j] = (Rp[row] - Rm[row]) / (2.0 * delta);
    }
}

/// Analytic dense Jacobian J[row*n + col] = ∂R_row/∂U_col, built block by block
/// from the hand-derived per-node partials of (dP,dQ,dT,dz)/dq. O(n) to assemble
/// (vs the numerical O(n²)); validated against numerical_jacobian by the Task-7
/// cross-check (column_jacobians_test). See node_deriv for the dX/dq definitions.
static void analytic_jacobian(const std::vector<double>& U, const ColumnInputs& in,
                              const OpacityLUTs& op, std::vector<double>& J) {
    using namespace constants;
    const int N = in.n_nodes;
    const int n = 4*N + 2;
    const double Sigma0 = U[4*N+1];
    const double z0     = U[4*N];
    const double dq = 1.0 / (N - 1);
    const double oz2 = in.omega_z * in.omega_z;
    const double as  = in.alpha * in.shear;

    J.assign((size_t)n * n, 0.0);
    auto at = [&](int row, int col) -> double& { return J[(size_t)row * n + col]; };

    // Per-node partials of each dX/dq w.r.t. that node's (P,Q,T) and the global Sigma0.
    // Layout matches the variable offsets: P=0, Q=1, T=2, z=3.
    struct NodeJac {
        // ∂(dP,dQ,dT,dz)/dq w.r.t. [P,Q,T,z] of this node, plus w.r.t. Sigma0.
        double dP_dP, dP_dz, dP_dS;                 // dP/dq partials (rho cancels)
        double dQ_dP, dQ_dT, dQ_dS;                 // dQ/dq partials
        double dT_dP, dT_dQ, dT_dT, dT_dS;          // dT/dq partials
        double dz_dP, dz_dT, dz_dS;                 // dz/dq partials
    };
    auto node_jac = [&](int i) -> NodeJac {
        const double P = U[4*i+0], Q = U[4*i+1], T = U[4*i+2], z = U[4*i+3];
        const double rho = std::max(eos_rho(P, T), RHO_GHOST_FLOOR);
        // EOS derivatives: rho = (P - aT^4/3) mu m_p/(k_B T)
        const double drho_dP = mu_fully_ionized * m_p / (k_B * T);
        const double drho_dT = -mu_fully_ionized * m_p * (P + a_rad * T*T*T*T) / (k_B * T * T);
        // Total opacity + log gradients, then convert to linear partials.
        double kappa, dk_dlnrho, dk_dlnT;
        kappa_total_with_grad(op, rho, T, kappa, dk_dlnrho, dk_dlnT);
        const double dk_dP = (dk_dlnrho / rho) * drho_dP;
        const double dk_dT = (dk_dlnrho / rho) * drho_dT + dk_dlnT / T;

        NodeJac J{};
        // dP/dq = -oz2 * z * Sigma0/2   (rho cancels)
        J.dP_dP = 0.0;
        J.dP_dz = -oz2 * Sigma0 / 2.0;
        J.dP_dS = -oz2 * z / 2.0;
        // dQ/dq = as * P * Sigma0/(2 rho)
        J.dQ_dP = as * Sigma0 / (2.0 * rho) * (1.0 - (P / rho) * drho_dP);
        J.dQ_dT = as * P * Sigma0 / 2.0 * (-1.0 / (rho*rho)) * drho_dT;
        J.dQ_dS = as * P / (2.0 * rho);
        // dT/dq = -3 kappa Q Sigma0 / (32 sigma T^3)
        const double T3 = T*T*T;
        J.dT_dQ = -3.0 * kappa * Sigma0 / (32.0 * sigma_SB * T3);
        J.dT_dS = -3.0 * kappa * Q / (32.0 * sigma_SB * T3);
        J.dT_dP = -3.0 * Q * Sigma0 / (32.0 * sigma_SB * T3) * dk_dP;
        J.dT_dT = -3.0 * Q * Sigma0 / (32.0 * sigma_SB) * (dk_dT / T3 - 3.0 * kappa / (T3 * T));
        // dz/dq = Sigma0/(2 rho)
        J.dz_dP = -Sigma0 / (2.0 * rho*rho) * drho_dP;
        J.dz_dT = -Sigma0 / (2.0 * rho*rho) * drho_dT;
        J.dz_dS = 1.0 / (2.0 * rho);
        return J;
    };

    // --- Trapezoidal ODE rows ---
    // R_X(i) = X(i+1) - X(i) - 0.5*dq*(dX_i + dX_{i+1}); X in {P(0),Q(1),T(2),z(3)}.
    int row = 0;
    for (int i = 0; i < N - 1; ++i) {
        const NodeJac ji = node_jac(i);
        const NodeJac jj = node_jac(i+1);
        const int ci = 4*i;        // base col of node i variables (P,Q,T,z)
        const int cj = 4*(i+1);    // base col of node i+1 variables
        const int cS = 4*N + 1;    // Sigma0 column
        const double half_dq = 0.5 * dq;

        // Helper to write one ODE row for variable Xoff (0=P,1=Q,2=T,3=z).
        // ∂R/∂var_i = -[X is var] - half_dq*∂dX_i/∂var_i
        // ∂R/∂var_{i+1} = +[X is var] - half_dq*∂dX_{i+1}/∂var_{i+1}
        // ∂R/∂Sigma0 = -half_dq*(∂dX_i/∂Sigma0 + ∂dX_{i+1}/∂Sigma0)

        // --- R_P row ---
        {
            const int r = row++;
            at(r, ci+0) += -1.0;                        // -[P is P]
            at(r, cj+0) +=  1.0;                        // +[P is P]
            at(r, ci+3) += -half_dq * ji.dP_dz;         // -half_dq ∂dP_i/∂z_i
            at(r, cj+3) += -half_dq * jj.dP_dz;         // -half_dq ∂dP_{i+1}/∂z_{i+1}
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
    const int cz0 = 4*N;       // z0 column
    // Surface-node opacity (for the photosphere pressure BC).
    const double P_s = U[4*(N-1)+0], T_s = U[4*(N-1)+2];
    const double rho_s = std::max(eos_rho(P_s, T_s), RHO_GHOST_FLOOR);
    const double drho_s_dP = mu_fully_ionized * m_p / (k_B * T_s);
    const double drho_s_dT = -mu_fully_ionized * m_p * (P_s + a_rad * T_s*T_s*T_s*T_s) / (k_B * T_s * T_s);
    double kappa_s, dk_s_dlnrho, dk_s_dlnT;
    kappa_total_with_grad(op, rho_s, T_s, kappa_s, dk_s_dlnrho, dk_s_dlnT);
    const double dks_dP = (dk_s_dlnrho / rho_s) * drho_s_dP;
    const double dks_dT = (dk_s_dlnrho / rho_s) * drho_s_dT + dk_s_dlnT / T_s;

    // R = Q(0):           ∂/∂Q(0) = 1
    at(row++, 4*0 + 1) = 1.0;
    // R = z(0):           ∂/∂z(0) = 1
    at(row++, 4*0 + 3) = 1.0;
    // R = Q(N-1) - Q_surf:∂/∂Q(N-1) = 1
    at(row++, 4*(N-1) + 1) = 1.0;
    // R = T(N-1) - T_eff: ∂/∂T(N-1) = 1
    at(row++, 4*(N-1) + 2) = 1.0;
    // R = z(N-1) - z0:    ∂/∂z(N-1) = 1, ∂/∂z0 = -1
    { const int r = row++; at(r, 4*(N-1) + 3) = 1.0; at(r, cz0) = -1.0; }
    // R = P(N-1) - (2/3) oz2 z0 / kappa_s:
    //   ∂/∂P(N-1) = 1 + (2/3) oz2 z0 / kappa^2 * dkappa/dP
    //   ∂/∂T(N-1) =     (2/3) oz2 z0 / kappa^2 * dkappa/dT
    //   ∂/∂z0     = -(2/3) oz2 / kappa
    {
        const int r = row++;
        const double f = (2.0/3.0) * oz2 * z0 / (kappa_s * kappa_s);
        at(r, 4*(N-1) + 0) = 1.0 + f * dks_dP;
        at(r, 4*(N-1) + 2) =       f * dks_dT;
        at(r, cz0)         = -(2.0/3.0) * oz2 / kappa_s;
    }
    assert(row == n);
}

void column_residual_test(const ColumnInputs& in, const OpacityLUTs& op,
                          std::vector<double>& U, std::vector<double>& R) {
    U = build_seed(in);
    column_residual(U, in, op, R);
}

void column_numerical_jacobian_test(const ColumnInputs& in, const OpacityLUTs& op,
                                    std::vector<double>& Jdense, int& n) {
    std::vector<double> U = build_seed(in);
    n = (int)U.size();
    numerical_jacobian(U, in, op, Jdense);
}

void column_jacobians_test(const ColumnInputs& in, const OpacityLUTs& op,
                           std::vector<double>& Ja, std::vector<double>& Jn, int& n) {
    std::vector<double> U = build_seed(in);
    n = (int)U.size();
    analytic_jacobian(U, in, op, Ja);
    numerical_jacobian(U, in, op, Jn);
}

/// Dense Gaussian elimination with partial pivoting. Solves A x = b; A is
/// row-major (n×n) and modified in place; the solution is returned in b.
/// Returns false if the matrix is (numerically) singular.
static bool dense_solve(std::vector<double>& A, std::vector<double>& b, int n) {
    for (int k = 0; k < n; ++k) {
        int piv = k; double maxv = std::abs(A[(size_t)k*n+k]);
        for (int i = k+1; i < n; ++i) { double v = std::abs(A[(size_t)i*n+k]); if (v>maxv){maxv=v;piv=i;} }
        if (maxv < 1e-300) return false;

        if (piv != k) { for (int j=0;j<n;++j) std::swap(A[(size_t)k*n+j],A[(size_t)piv*n+j]); std::swap(b[k],b[piv]); }

        const double akk = A[(size_t)k*n+k];
        for (int i = k+1; i < n; ++i) {
            const double f = A[(size_t)i*n+k]/akk;
            if (f != 0.0) { for (int j=k;j<n;++j) A[(size_t)i*n+j]-=f*A[(size_t)k*n+j]; b[i]-=f*b[k]; }
        }
    }
        // Back-substitution. Elimination above started at j=k and did NOT normalize
        // the diagonal, so A[i*n+i] still holds the original pivot — divide by it.
    for (int i = n-1; i >= 0; --i) { double sgi=b[i]; for (int j=i+1;j<n;++j) sgi-=A[(size_t)i*n+j]*b[j]; b[i]=sgi/A[(size_t)i*n+i]; }
    return true;
}

/// Scale-balanced residual merit (RMS of per-row-normalized residuals).
///
/// The residual rows span wildly different magnitudes — the P-equation rows are
/// ~1e15 while T-equation rows are ~1e5 — so a plain L2 norm is dominated by the
/// pressure equations and the line search makes no progress on temperature.
/// We normalize each ODE-difference row by the mean magnitude of the variable it
/// advances (P, Q, T, or z across all nodes), and each boundary-condition row by
/// its own representative magnitude, then take the RMS. This makes every equation
/// contribute on a comparable, dimensionless footing.
static double scaled_residual_norm(const std::vector<double>& U,
                                   const std::vector<double>& R,
                                   const ColumnInputs& in) {
    const int N = in.n_nodes;
    // Mean magnitude of each variable across nodes (the row scales for the ODEs).
    double mP=0, mQ=0, mT=0, mZ=0;
    for (int i = 0; i < N; ++i) {
        mP += std::abs(U[4*i+0]); mQ += std::abs(U[4*i+1]);
        mT += std::abs(U[4*i+2]); mZ += std::abs(U[4*i+3]);
    }
    mP=std::max(mP/N,1e-300); mQ=std::max(mQ/N,1e-300);
    mT=std::max(mT/N,1e-300); mZ=std::max(mZ/N,1e-300);
        // Order MUST match the 6 BC rows in column_residual, in sequence:
        //   Q(0)=0, z(0)=0, Q(surf), T(surf), z(surf)=z0, P(surf)  ->  mQ,mZ,mQ,mT,mZ,mP
    const double bc_scale[6] = { mQ, mZ, mQ, mT, mZ, mP };

    double sum = 0.0; int row = 0;
    for (int i = 0; i < N-1; ++i) {
        double sP=R[row++]/mP; double sQ=R[row++]/mQ; double sT=R[row++]/mT; double sZ=R[row++]/mZ;
        sum += sP*sP + sQ*sQ + sT*sT + sZ*sZ;
    }
    for (int b = 0; b < 6; ++b) { double s = R[row++] / std::max(bc_scale[b],1e-300); sum += s*s; }
    return std::sqrt(sum / (double)R.size());
}

ColumnBVPSolution solve_column_bvp(const ColumnInputs& in, const OpacityLUTs& op,
                                   const std::vector<double>* warm_start) {
    const int N = in.n_nodes;
    const int n = 4*N + 2;
    ColumnBVPSolution s;

    std::vector<double> U;
    std::vector<double> R, J, Jcopy, rhs, Utry, Rtry;

    // A non-null warm_start of the wrong size is a caller bug (mismatched n_nodes):
    // catch it in debug, but fall back to a cold start in release rather than
    // corrupt state by consuming a misaligned vector.
    assert(warm_start == nullptr || (int)warm_start->size() == n);
    if (warm_start && (int)warm_start->size() == n) {
        // Numerical continuation: start Newton from the converged neighbour.
        // It is already flux-balanced, so skip the analytic seed + rescale.
        U = *warm_start;
    } else {
        U = build_seed(in);
        // Flux-balance seed rescale (cold start only).
        //
        // T_eff is a fixed surface boundary condition, so in steady state the
        // height-integrated viscous heating must equal the radiated surface flux:
        //   ∫ alpha*shear*P dz  ≈  sigma_SB T_eff^4   (per face).
        // The user-supplied rho_mid_guess is only a rough density estimate and can be
        // many orders of magnitude away from the value that satisfies this balance
        // (e.g. for the cool gas-limit it overshoots by ~1e5). Starting Newton from a
        // grossly over-dense column drives the solver toward a runaway-hot state and
        // stalls the line search. We therefore rescale the Gaussian seed's density by
        // the single factor that makes the analytic heating integral match the surface
        // flux. This lands the seed within the Newton basin of the true (heating-
        // balanced) column, from which the relaxation converges quadratically.
        {
            using namespace constants;
            // Mirror the radiation-aware H from build_seed (cs2_gas + P_rad/rho_mid)
            // so scale cancels the H factor in Sigma0 and the seed lands on the correct
            // flux-balanced density regardless of how much radiation contributes to H.
            const double cs2_gas_r = k_B * in.T_eff / (mu_fully_ionized * m_p);
            const double rho_mid_seed = in.rho_mid_guess;
            const double P_rad_seed   = (a_rad / 3.0) * in.T_eff * in.T_eff * in.T_eff * in.T_eff;
            const double H_r = std::sqrt(cs2_gas_r + P_rad_seed / rho_mid_seed) / in.omega_z;
            // Heating per face for the current seed (Gaussian rho, P_gas ≈ rho cs2_gas):
            //   ∫0^∞ alpha*shear*rho_mid*cs2_gas*exp(-z^2/2H_r^2) dz
            //   = alpha * shear * P_gas_mid * H_r * sqrt(pi/2)
            // Using H_r here (not the old gas-only H) keeps scale consistent with the
            // radiation-aware Sigma0 in U, so the final Sigma0 = 2*flux/(alpha*shear*cs2_gas)
            // matches the gas-dominated limit identically.
            const double P_gas_mid_seed = rho_mid_seed * cs2_gas_r;
            const double heat_seed    = in.alpha * in.shear * P_gas_mid_seed * H_r * std::sqrt(std::numbers::pi / 2.0);
            const double flux_target  = surface_flux(in.T_eff);
            // scale can be large if shear/alpha are near-zero (heat_seed -> 0); the
            // caller is expected to supply physically reasonable (nonzero) inputs.
            double scale = (heat_seed > 0.0) ? flux_target / heat_seed : 1.0;
            // Density scales linearly with the column, so P_gas and Sigma do too.
            for (int i = 0; i < N; ++i) {
                const double T_i = U[4*i+2];
                const double rho_old = std::max(eos_rho(U[4*i+0], T_i), 0.0);
                const double rho_new = rho_old * scale;
                U[4*i+0] = rho_new * cs2_gas_r + (a_rad/3.0)*T_i*T_i*T_i*T_i;  // refresh P (gas + rad)
            }
            U[4*N+1] *= scale;                                          // Sigma0
        }
    }

    column_residual(U, in, op, R);
    double merit = scaled_residual_norm(U, R, in);

    for (int it = 0; it < in.max_iters; ++it) {
        // 1) Jacobian and Newton step  J dU = -R
        // Analytic block Jacobian (O(n) assembly); validated against the numerical
        // finite-difference Jacobian by the Task-7 cross-check (column_jacobians_test).
        analytic_jacobian(U, in, op, J);
        Jcopy = J;
        rhs.assign(R.begin(), R.end());
        for (double& r : rhs) r = -r;
        if (!dense_solve(Jcopy, rhs, n)) break;       // singular -> bail (non-converged)
        const std::vector<double>& dU = rhs;

        // 2) Trust-region cap on the step length: in the stiff outer layers the
        //    full Newton step can change T (and P) by orders of magnitude in one
        //    shot, overshooting the steep opacity nonlinearity so that no damping
        //    of the full direction decreases the merit. Cap the initial step so no
        //    positive variable (P or T) changes by more than STEP_CAP in fractional
        //    terms, then run the merit line search from there.
        constexpr double STEP_CAP = 0.5;
        double lambda = 1.0;
        for (int i = 0; i < N; ++i) {
            for (int c : {0, 2}) {                 // P (offset 0) and T (offset 2)
                const double u = U[4*i+c], d = dU[4*i+c];
                if (u != 0.0 && d != 0.0) {
                    const double frac = std::abs(d / u);
                    if (frac * lambda > STEP_CAP) lambda = STEP_CAP / frac;
                }
            }
        }
        bool accepted = false;
        double merit_try = merit;
        for (int ls = 0; ls < 40; ++ls) {
            Utry.assign(U.begin(), U.end());
            for (int i = 0; i < n; ++i) Utry[i] += lambda * dU[i];
            bool physical = true;
            for (int i = 0; i < N && physical; ++i) {
                const double Pi = Utry[4*i+0], Ti = Utry[4*i+2];
                if (Ti <= 0.0 || eos_rho(Pi, Ti) <= 0.0) physical = false;
            }
            if (physical) {
                column_residual(Utry, in, op, Rtry);
                merit_try = scaled_residual_norm(Utry, Rtry, in);
                if (merit_try < merit) { accepted = true; break; }
            }
            lambda *= 0.5;
        }
        if (!accepted) break;                          // stuck -> bail (non-converged)

        // 3) Convergence on relative step size.
        double maxrel = 0.0;
        for (int i = 0; i < n; ++i) {
            const double rel = std::abs(lambda * dU[i]) / std::max(std::abs(U[i]), 1e-300);
            maxrel = std::max(maxrel, rel);
        }

        U.swap(Utry);
        R.swap(Rtry);
        merit = merit_try;
        s.iters = it + 1;
        s.final_residual = merit;

        // The merit<1e-6 guard prevents a tiny improving line-search step (small
        // |lambda*dU|) from being mistaken for convergence while the residual is
        // still large. Both must hold: relative step small AND residual small.
        // 1e-6 floor: the scaled residual cannot reliably reach much lower with a
        // finite-difference numerical Jacobian (Jacobian truncation ~1e-7 step).
        if (maxrel < in.tol && merit < 1e-6) { s.converged = true; break; }
    }

    // No fallback (Approach A: fail or succeed, never a fabricated profile).
    // On non-convergence return EMPTY profile vectors; the caller MUST check
    // `converged` before reading the solution.
    if (!s.converged) {
        s.q.clear(); s.z.clear(); s.P.clear(); s.Q.clear(); s.T.clear(); s.rho.clear();
        s.z0 = 0.0; s.Sigma0 = 0.0; s.tau_mid = 0.0;
        return s;
    }

    // Unpack the converged state into the solution.
    s.q.resize(N); s.z.resize(N); s.P.resize(N); s.Q.resize(N); s.T.resize(N); s.rho.resize(N);
    for (int i = 0; i < N; ++i) {
        const double Pi = U[4*i+0], Qi = U[4*i+1], Ti = U[4*i+2], zi = U[4*i+3];
        s.q[i] = (double)i / (N - 1);
        s.P[i] = Pi; s.Q[i] = Qi; s.T[i] = Ti; s.z[i] = zi;
        s.rho[i] = std::max(eos_rho(Pi, Ti), 0.0);
    }
    s.z0 = U[4*N];
    s.Sigma0 = U[4*N+1];

    double tau = 0.0;
    for (int i = 0; i + 1 < N; ++i) {
        const double kRi = kappa_total(op, std::max(s.rho[i],   RHO_GHOST_FLOOR), s.T[i]);
        const double kRj = kappa_total(op, std::max(s.rho[i+1], RHO_GHOST_FLOOR), s.T[i+1]);
        const double dz = std::abs(s.z[i+1] - s.z[i]);
        tau += 0.5 * (kRi*s.rho[i] + kRj*s.rho[i+1]) * dz;
    }
    s.tau_mid = tau;

    return s;
}

} // namespace grrt
