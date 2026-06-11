#ifndef GRRT_SLIM_DISK_RADIAL_H
#define GRRT_SLIM_DISK_RADIAL_H
#include "grrt/color/opacity.h"
#include "grrt_export.h"
#include <cmath>
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

    /// Runaway safety budget (hard ceiling on the WHOLE solve — never a fabricated
    /// profile; on exceed the solver returns SlimDiskRadial{} with converged=false).
    /// A prior full-resolution run hung ~11 h; these caps guarantee that can't recur.
    ///   • budget_inner_iter_cap: cumulative inner-Newton-iteration cap summed across
    ///     every bracket sample, spin rung and Ṁ rung.  <=0 uses the solver default
    ///     (~200000), generous for a full solve yet far below a runaway.
    ///   • budget_wall_seconds: wall-clock cap (steady_clock).  <=0 uses the default
    ///     (~900 s = 15 min).
    long long budget_inner_iter_cap = 0;   ///< <=0 -> solver default (~200000)
    double    budget_wall_seconds   = 0.0; ///< <=0 -> solver default (~900 s)
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

/// Kerr relativistic factor functions (geometric units, G=c=1; prograde, equatorial).
/// Verified against formula reference §22. r in units of M.
namespace slim_detail {
inline double omega_k(double M, double a, double r) {          // prograde Kerr Keplerian Ω_K
    return std::sqrt(M) / (r * std::sqrt(r) + a * std::sqrt(M));
}
inline double calC(double M, double a, double r) {             // 𝒞 = 1 − 3M/r + 2a√M/r^{3/2}
    return 1.0 - 3.0 * M / r + 2.0 * a * std::sqrt(M) / (r * std::sqrt(r));
}
inline double calD(double M, double a, double r) {             // 𝒟 = 1 − 2M/r + a²/r² = Δ/r²
    return 1.0 - 2.0 * M / r + a * a / (r * r);
}
inline double calH(double M, double a, double r) {             // ℋ = 1 − 4a√M/r^{3/2} + 3a²/r²
    return 1.0 - 4.0 * a * std::sqrt(M) / (r * std::sqrt(r)) + 3.0 * a * a / (r * r);
}
inline double omega_perp2(double M, double a, double r) {      // vertical epicyclic Ω_⊥² = Ω_K²·ℋ  (= omega_z_sq)
    const double ok = omega_k(M, a, r);
    return ok * ok * calH(M, a, r);
}
inline double kerr_delta(double M, double a, double r) {       // Δ = r² − 2Mr + a²
    return r * r - 2.0 * M * r + a * a;
}
inline double kerr_A(double M, double a, double r) {           // A = r⁴ + r²a² + 2Mra²
    return r*r*r*r + r*r*a*a + 2.0*M*r*a*a;
}

/// Thermodynamic state returned by the one-zone vertical closure.
/// All quantities are in CGS.
struct OneZoneState {
    double H     = 0.0;  ///< scale height [cm]
    double rho_mid = 0.0;///< midplane density [g/cm^3]
    double c_s   = 0.0;  ///< total sound speed [cm/s] (= H · Ω_⊥ by construction)
    double p_mid = 0.0;  ///< total midplane pressure p_gas + p_rad [erg/cm^3]
    double p_gas = 0.0;  ///< gas pressure at midplane [erg/cm^3]
    double p_rad = 0.0;  ///< radiation pressure at midplane [erg/cm^3]
    double P     = 0.0;  ///< vertically-integrated pressure 2·p_mid·H [erg/cm^2] (α-stress; trap #9)
    double S     = 0.0;  ///< specific entropy (gas + radiation) [erg/(g·K)]
    double mu    = 0.0;  ///< mean molecular weight used
};

/// One-zone height-integrated vertical closure.
///
/// Given the surface density Sigma [g/cm²], midplane temperature Tc [K], and
/// radius r [M], returns the self-consistent scale height, midplane
/// density/pressure, integrated pressure, sound speed, and specific entropy.
///
/// The scale height follows from hydrostatic balance H = c_s / Ω_⊥ with the
/// TOTAL sound speed c_s² = p_mid/ρ_mid.  Substituting p_mid = p_gas + p_rad
/// and ρ_mid = Σ/(2H) leads to the quadratic
///   Ω_⊥² H² − b H − c_s_gas² = 0 ,  b = 2 a_rad T_c⁴ / (3 Σ)
/// whose positive root is H = (b + sqrt(b² + 4 Ω_⊥² c_s_gas²)) / (2 Ω_⊥²).
/// In the gas-dominated limit b→0 this reduces to H = c_s_gas / Ω_⊥.
///
/// A single fixed-point pass is done for μ: first solve with μ = mu_fully_ionized,
/// then look up μ(ρ_mid, T_c) and recompute.
///
/// Specific entropy (per unit mass, additive constant irrelevant — only dS/dr enters Q_adv):
///   S = (k_B / (μ m_p)) ln(T_c^{3/2} / ρ_mid) + (4 a_rad T_c³) / (3 ρ_mid)
///
/// @param Sigma  surface density [g/cm²]
/// @param Tc     midplane temperature [K]
/// @param r      radius [M]
/// @param in     slim-disk inputs (M, a, r_g, α)
/// @param op     opacity LUTs (for μ look-up)
GRRT_EXPORT OneZoneState one_zone_closure(double Sigma, double Tc, double r,
                                         const SlimDiskInputs& in, const OpacityLUTs& op);

/// Exact partial derivatives of the one-zone closure outputs w.r.t. the two state
/// variables (Σ, T_c) at fixed radius.  d{field}[0] = ∂field/∂Σ, d{field}[1] =
/// ∂field/∂T_c.  Foundation for the analytic radial Jacobian (Task 2).
///
/// μ is held FIXED at the converged fixed-point value (∂μ/∂{Σ,T_c}=0): in the hot,
/// fully-ionized inner disk μ ≈ mu_fully_ionized is constant, so this is exact
/// there; at low T (partial ionization) it neglects the weak ∂μ sensitivity — the
/// FD cross-check validates that this is below tolerance at the operating points.
struct OneZoneJac {
    double dH[2];       ///< ∂H/∂{Σ,T_c}
    double drho[2];     ///< ∂ρ_mid/∂{Σ,T_c}
    double dp_gas[2];   ///< ∂p_gas/∂{Σ,T_c}
    double dp_rad[2];   ///< ∂p_rad/∂{Σ,T_c}
    double dp_mid[2];   ///< ∂p_mid/∂{Σ,T_c}
    double dc_s[2];     ///< ∂c_s/∂{Σ,T_c}
    double dP[2];       ///< ∂P/∂{Σ,T_c}
    double dS[2];       ///< ∂S/∂{Σ,T_c}
};

/// Analytic Jacobian of the one-zone vertical closure (see OneZoneJac).  Returns
/// the closure state itself (st) and its partials (jac) so callers evaluate both
/// in one pass.  Differentiates the H-quadratic Ω_⊥²H²−bH−c_s_gas²=0 by the
/// implicit-function theorem; μ frozen (documented above).
GRRT_EXPORT void one_zone_closure_jac(double Sigma, double Tc, double r,
                                      const SlimDiskInputs& in, const OpacityLUTs& op,
                                      OneZoneState& st, OneZoneJac& jac);

/// Prograde Keplerian specific angular momentum ℓ_K = u_φ on a circular
/// equatorial Kerr orbit at radius r [M] (Bardeen-Press-Teukolsky 1972).
/// Used by the seed and the outer/regularity boundary conditions.
GRRT_EXPORT double ell_kepler(double M, double a, double r);

/// Prograde Kerr ISCO radius (Bardeen-Press-Teukolsky 1972), in units of M.
/// Inputs are dimensional M and a; the BPT72 formula scales linearly with M,
/// so we evaluate the dimensionless r_isco(a/M) and multiply by M.  For M=1 it
/// matches the bare BPT72 expression used by the tests.  The seed uses r_isco as
/// the inner-node anchor and the energy outer BC uses it as the NT zero-torque radius.
GRRT_EXPORT double isco_prograde(double M, double a);

/// Invert the equatorial Kerr Ω↔ℓ relation: given the covariant specific
/// angular momentum ℓ = u_φ at radius r, return the orbital angular velocity
/// Ω = u^φ/u^t [geometric, 1/M]. A few Newton iterations seeded from Ω_K.
/// See slim_disk_radial.cpp for the derivation; documented as a robust local solve.
GRRT_EXPORT double omega_from_ell(double M, double a, double r, double ell);

} // namespace slim_detail

/// Build a crude thin-disk seed state vector U (length 4N+2) for the radial
/// residual: power-law Σ, mass-conservation V<0, Keplerian ℓ, NT-ish T_c, plus
/// the two globals (ℓ_in, r_s). Refined by the relaxation (Task 5). See .cpp.
GRRT_EXPORT std::vector<double> build_thin_disk_seed(const SlimDiskInputs& in,
                                                     const OpacityLUTs& opacity);

/// Evaluate the transonic radial residual R (length 4N+2) for state U.
/// Row layout (see disk-physics-formulas.md §22/§23 and the .cpp header comment):
///   [0 .. N-1]      mass conservation (algebraic, per node)
///   [N .. 2N-1]     angular momentum (algebraic, per node)
///   [2N .. 3N-2]    radial-momentum transonic ODE (trapezoidal, N-1 intervals)
///   [3N-1 .. 4N-3]  energy Q_vis=Q_rad+Q_adv ODE (trapezoidal, N-1 intervals)
///   [4N-2, 4N-1]    outer boundary conditions (ℓ_out matched-slope radial-equilibrium
///                   extrapolation, outer-node §23 energy balance for T_c,out)
///   [4N, 4N+1]      sonic-point regularity (𝒟₀(r_s)=0, 𝒩₁(r_s)=0)
GRRT_EXPORT void slim_radial_residual(const std::vector<double>& U,
                                      const SlimDiskInputs& in,
                                      const OpacityLUTs& opacity,
                                      std::vector<double>& R);

/// Solve the relativistic transonic slim-disk radial structure
/// (see docs/superpowers/references/disk-physics-formulas.md §22).
GRRT_EXPORT SlimDiskRadial solve_slim_disk_radial(const SlimDiskInputs& in,
                                                  const OpacityLUTs& opacity);

/// One accepted point on the pseudo-arclength continuation branch.  Recorded by
/// solve_slim_disk_arclength as it traces the (Σ,V)/mass branch THROUGH the
/// f_Edd≈0.11 turning point.  f_Edd = Ṁ/Ṁ_Edd (textbook 10 L_Edd/c²); Mdot_dot_sign
/// is the sign of the continuation tangent's Ṁ̇ component — it FLIPS at the fold.
struct SlimArclengthPoint {
    double f_Edd = 0.0;       ///< Eddington fraction Ṁ/Ṁ_Edd at this point
    double mdot = 0.0;        ///< Ṁ [g/s]
    double r_sonic = 0.0;     ///< sonic radius [M]
    double ell_in = 0.0;      ///< inner specific angular momentum eigenvalue
    double max_Hr = 0.0;      ///< max H/r over the disk
    double beta_min = 0.0;    ///< min gas-pressure fraction p_gas/p_mid (β→0 = slim)
    double beta_max = 0.0;    ///< max β
    double fadv_min = 0.0;    ///< min advected fraction Q_adv/Q_rad
    double fadv_max = 0.0;    ///< max f_adv
    double peak_Sigma = 0.0;  ///< peak surface density [g/cm^2] (the Σ branch)
    double merit = 0.0;       ///< final augmented scaled merit
    double arc_step = 0.0;    ///< the arclength step Δs that produced this point
    int    Mdot_dot_sign = 0; ///< sign of the tangent Ṁ̇ (flips at the fold)
};

/// Pseudo-arclength continuation result: the full traced branch plus the
/// highest-f_Edd accepted profile.
struct SlimArclengthResult {
    std::vector<SlimArclengthPoint> branch;  ///< every accepted continuation point
    SlimDiskRadial top;        ///< the highest-f_Edd accepted profile (converged)
    double max_f_Edd = 0.0;    ///< highest f_Edd reached on the branch
    bool   crossed_fold = false; ///< did Ṁ̇ flip sign anywhere (fold detected)?
    bool   crossed_011 = false;  ///< did the trace cross f_Edd=0.11 (beat the ceiling)?
    bool   ok = false;         ///< at least one point past the anchor was accepted
};

/// Keller pseudo-arclength continuation of the slim-disk branch (Tasks 1-4).
///
/// Converges a sub-fold anchor (a from in.spin, the requested Ṁ used as a target;
/// the anchor is taken at f_Edd≈0.10 the existing way), computes the initial
/// continuation tangent, then arclength-steps UP, tracing the (Σ,V)/mass branch
/// AROUND the f_Edd≈0.11 turning point that simple Ṁ-marching cannot cross.  The
/// continuation unknown is Ṁ; each step is a predictor (tangent) + an augmented
/// Newton corrector on {R=0, Keller arclength row=0} using the EXACT analytic
/// Jacobian.  Records every accepted point; the branch shows the fold (where Ṁ̇
/// flips sign) and the deep-slim physics (H/r, β, f_adv) at the top.
GRRT_EXPORT SlimArclengthResult solve_slim_disk_arclength(const SlimDiskInputs& in,
                                                          const OpacityLUTs& opacity);
} // namespace grrt
#endif
