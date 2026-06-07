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

ColumnBVPSolution solve_column_bvp(const ColumnInputs& in, const OpacityLUTs&) {
    ColumnBVPSolution s;                          // implemented in later tasks
    s.q.assign(in.n_nodes, 0.0);
    return s;
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

struct Deriv { double dP, dQ, dT, dz; };
// dX/dq at a node; dz/dq = Sigma0/(2 rho).
Deriv node_deriv(double P, double Q, double T, double z,
                 double Sigma0, double alpha, double shear, double omega_z,
                 const grrt::OpacityLUTs& op) {
    const double rho = std::max(grrt::eos_rho(P, T), RHO_GHOST_FLOOR);
    const double Tk  = std::max(T, T_LUT_MIN);
    const double kR  = op.lookup_kappa_ross(rho, Tk);
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
    const double kR_srf  = op.lookup_kappa_ross(rho_srf, std::max(T(N-1), T_LUT_MIN));
    R[row++] = Q(0);                                                   // midplane Q=0
    R[row++] = z(0);                                                   // midplane z=0
    R[row++] = Q(N-1) - Q_surf;                                        // surface Q
    R[row++] = T(N-1) - in.T_eff;                                      // surface T
    R[row++] = z(N-1) - z0;                                            // surface z=z0
    R[row++] = P(N-1) - (2.0/3.0)*in.omega_z*in.omega_z*z0/kR_srf;     // surface pressure
    assert(row == 4*N + 2);
}

/// Build a gas-pressure Gaussian column seed state (length 4N+2).
/// Isothermal (T = T_eff), linear z grid up to 4H, Gaussian rho.
/// Used by both column_residual_test and the numerical Jacobian hook.
static std::vector<double> build_seed(const ColumnInputs& in) {
    using namespace constants;
    const int N = in.n_nodes;
    const double cs2 = k_B * in.T_eff / (mu_fully_ionized * m_p);
    const double H   = std::sqrt(cs2) / in.omega_z;
    const double z0  = 4.0 * H;                                   // ~99.97% of a Gaussian column
    const double rho_mid = in.rho_mid_guess;
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
        const double Pi = rho * cs2 + (a_rad/3.0)*Ti*Ti*Ti*Ti;
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
    J.assign((size_t)n * n, 0.0);
    std::vector<double> Up, Um, Rp, Rm;
    for (int j = 0; j < n; ++j) {
        // Per-component relative step; central differences are insensitive to the
        // exact value over a wide range. 1e-7 gives ~1e-9 Jacobian accuracy here,
        // far inside the <1e-3 tolerance of the Task-7 analytic cross-check.
        const double delta = 1e-7 * std::max(std::abs(U[j]), 1e-30);
        Up = U; Um = U;
        Up[j] += delta; Um[j] -= delta;
        column_residual(Up, in, op, Rp);
        column_residual(Um, in, op, Rm);
        for (int row = 0; row < n; ++row)
            J[(size_t)row * n + j] = (Rp[row] - Rm[row]) / (2.0 * delta);
    }
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

} // namespace grrt
