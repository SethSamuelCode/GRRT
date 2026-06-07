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

ColumnBVPSolution solve_column_bvp(const ColumnInputs& in, const OpacityLUTs& op) {
    const int N = in.n_nodes;
    const int n = 4*N + 2;
    ColumnBVPSolution s;

    std::vector<double> U = build_seed(in);
    std::vector<double> R, J, Jcopy, rhs, Utry, Rtry;

    // Flux-balance seed rescale.
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
        const double cs2 = k_B * in.T_eff / (mu_fully_ionized * m_p);
        const double H   = std::sqrt(cs2) / in.omega_z;
        // Heating per face for the current (Gaussian, P≈rho cs2) seed:
        //   ∫0^∞ alpha*shear*rho_mid*cs2*exp(-z^2/2H^2) dz = alpha*shear*P_mid*H*sqrt(pi/2)
        const double rho_mid_seed = in.rho_mid_guess;
        const double P_mid_seed   = rho_mid_seed * cs2;
        const double heat_seed    = in.alpha * in.shear * P_mid_seed * H * std::sqrt(std::numbers::pi / 2.0);
        const double flux_target  = surface_flux(in.T_eff);
        // scale can be large if shear/alpha are near-zero (heat_seed -> 0); the
        // caller is expected to supply physically reasonable (nonzero) inputs.
        double scale = (heat_seed > 0.0) ? flux_target / heat_seed : 1.0;
        // Density scales linearly with the column, so P_gas and Sigma do too.
        for (int i = 0; i < N; ++i) {
            const double T_i = U[4*i+2];
            const double rho_old = std::max(eos_rho(U[4*i+0], T_i), 0.0);
            const double rho_new = rho_old * scale;
            U[4*i+0] = rho_new * cs2 + (a_rad/3.0)*T_i*T_i*T_i*T_i;  // refresh P (gas + rad)
        }
        U[4*N+1] *= scale;                                          // Sigma0
    }

    column_residual(U, in, op, R);
    double merit = scaled_residual_norm(U, R, in);

    for (int it = 0; it < in.max_iters; ++it) {
        // 1) Jacobian and Newton step  J dU = -R
        numerical_jacobian(U, in, op, J);
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

    // Unpack the (converged or best) state into the solution.
    s.q.resize(N); s.z.resize(N); s.P.resize(N); s.Q.resize(N); s.T.resize(N); s.rho.resize(N);
    for (int i = 0; i < N; ++i) {
        const double Pi = U[4*i+0], Qi = U[4*i+1], Ti = U[4*i+2], zi = U[4*i+3];
        s.q[i] = (double)i / (N - 1);
        s.P[i] = Pi; s.Q[i] = Qi; s.T[i] = Ti; s.z[i] = zi;
        s.rho[i] = std::max(eos_rho(Pi, Ti), 0.0);
    }
    s.z0 = U[4*N];
    s.Sigma0 = U[4*N+1];

    // tau_mid: trapezoidal integral of kappa_total * rho over z, midplane->surface.
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
