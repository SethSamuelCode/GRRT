#include "grrt/scene/slim_disk_radial.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>
#include <numbers>

namespace grrt {

// ---------------------------------------------------------------------------
// One-zone vertical closure
// ---------------------------------------------------------------------------
namespace slim_detail {

/// Solve the scale-height quadratic for a fixed μ.
///
/// Hydrostatic balance H = c_s/Ω_⊥ with c_s² = p_mid/ρ_mid and
/// ρ_mid = Σ/(2H) gives:
///   Ω_⊥² H² − b H − c_s_gas² = 0 ,  b = 2 a_rad T_c⁴ / (3 Σ)
/// Positive root: H = (b + sqrt(b² + 4 Ω_⊥² c_s_gas²)) / (2 Ω_⊥²)
static double solve_H(double Sigma, double Tc, double mu, double Omega_perp2_cgs) {
    using namespace constants;
    const double cs_gas2 = k_B * Tc / (mu * m_p);                    // [cm²/s²]
    const double b       = 2.0 * a_rad * Tc*Tc*Tc*Tc / (3.0 * Sigma); // radiation term [cm²/s² per cm]
    // H = (b + sqrt(b² + 4 Ω_⊥² c_s_gas²)) / (2 Ω_⊥²)
    const double disc = b * b + 4.0 * Omega_perp2_cgs * cs_gas2;
    return (b + std::sqrt(disc)) / (2.0 * Omega_perp2_cgs);          // [cm]
}

OneZoneState one_zone_closure(double Sigma, double Tc, double r,
                              const SlimDiskInputs& in, const OpacityLUTs& op) {
    using namespace constants;

    // Vertical epicyclic frequency in CGS.
    //   ω_⊥² (geometric, 1/M²) → CGS (1/s²) by multiplying by (c_cgs/r_g)²
    const double conv = c_cgs / in.r_g;                               // [1/s per 1/M]
    const double Omega_perp2_geom = omega_perp2(in.mass, in.spin, r); // [1/M²]
    const double Omega_perp2_cgs  = Omega_perp2_geom * conv * conv;   // [1/s²]
    const double Omega_perp       = std::sqrt(Omega_perp2_cgs);       // [1/s]

    // Numerical guards for transient non-physical Newton iterates (Tasks 4-5).
    constexpr double SIGMA_FLOOR = 1e-30;  // g/cm^2 — guards b=2aT^4/(3Sigma) and rho_mid sign
    constexpr double RHO_FLOOR   = 1e-30;  // g/cm^3 — guards log(rho), 1/rho in the entropy
    constexpr double T_FLOOR     = 1.0;    // K — guards c_s_gas^2 and T^4
    if (!(Omega_perp2_cgs > 0.0)) return OneZoneState{};   // unsolvable here; caller checks st.H==0
    const double Sigma_s = std::max(Sigma, SIGMA_FLOOR);
    const double Tc_s    = std::max(Tc, T_FLOOR);

    // -----------------------------------------------------------------------
    // ≤3-iteration μ fixed point; μ is ~constant in the fully-ionized inner
    // disk but varies at low T where partial ionization matters.
    // -----------------------------------------------------------------------
    double mu = mu_fully_ionized;
    for (int it = 0; it < 3; ++it) {
        const double H_it   = solve_H(Sigma_s, Tc_s, mu, Omega_perp2_cgs);
        const double rho_it = Sigma_s / (2.0 * H_it);
        double mu_new = op.lookup_mu(rho_it, Tc_s);
        if (!(mu_new > 0.0) || !std::isfinite(mu_new)) mu_new = mu_fully_ionized;
        const bool conv = std::abs(mu_new - mu) <= 1e-6 * mu;
        mu = mu_new;
        if (conv) break;
    }
    const double H       = solve_H(Sigma_s, Tc_s, mu, Omega_perp2_cgs);
    const double rho_mid = Sigma_s / (2.0 * H);

    // -----------------------------------------------------------------------
    // Thermodynamic quantities
    // -----------------------------------------------------------------------

    const double p_gas = rho_mid * k_B * Tc_s / (mu * m_p);          // [erg/cm³]
    const double p_rad = (a_rad / 3.0) * Tc_s*Tc_s*Tc_s*Tc_s;        // [erg/cm³]
    const double p_mid = p_gas + p_rad;                               // [erg/cm³]

    // Total sound speed: by construction equals H·Ω_⊥.
    const double c_s = Omega_perp * H;                                // [cm/s]

    // Vertically-integrated pressure (α-stress, trap #9 — use P, not p_mid).
    const double P = 2.0 * p_mid * H;                                 // [erg/cm²]

    // Specific entropy per unit mass (gas + radiation; additive constant irrelevant).
    // S = (k_B/(μ m_p)) ln(T_c^{3/2}/ρ_mid) + (4 a_rad T_c³)/(3 ρ_mid)
    // Only dS/dr enters Q_adv in the radial solver.
    const double rho_e  = std::max(rho_mid, RHO_FLOOR);
    const double S_gas = (k_B / (mu * m_p)) * std::log(std::pow(Tc_s, 1.5) / rho_e);
    const double S_rad = (4.0 * a_rad * Tc_s*Tc_s*Tc_s) / (3.0 * rho_e);
    const double S     = S_gas + S_rad;                               // [erg/(g·K)] (up to additive const)

    OneZoneState st;
    st.H       = H;
    st.rho_mid = rho_mid;
    st.c_s     = c_s;
    st.p_mid   = p_mid;
    st.p_gas   = p_gas;
    st.p_rad   = p_rad;
    st.P       = P;
    st.S       = S;
    st.mu      = mu;
    return st;
}

// ---------------------------------------------------------------------------
// Equatorial Kerr orbital mechanics helpers (geometric units, G=c=1)
// ---------------------------------------------------------------------------
//
// Equatorial Kerr metric components (match VolumetricDisk::circular_velocity):
//   g_tt   = -(1 - 2M/r)
//   g_tφ   = -2 M a / r
//   g_φφ   =  r² + a² + 2 M a²/r
//
// For a circular orbit u^φ = Ω u^t, normalization u^μu_μ=-1 gives
//   (u^t)² · D(Ω) = 1,  D(Ω) = -(g_tt + 2 g_tφ Ω + g_φφ Ω²)
// and the covariant specific angular momentum is
//   ℓ(Ω) = u_φ = (g_tφ + g_φφ Ω) · u^t = (g_tφ + g_φφ Ω) / √D(Ω).
//
// The seed sets ℓ = ℓ_K (Keplerian); the residual needs the inverse Ω(ℓ).
// ℓ(Ω) is smooth and monotone over the physically relevant prograde branch,
// so we invert with a short damped Newton iteration seeded from Ω_K. This is a
// well-defined local solve (not a guess); documented per the task latitude note.

static inline void eq_metric(double M, double a, double r,
                             double& g_tt, double& g_tphi, double& g_phph) {
    g_tt   = -(1.0 - 2.0 * M / r);
    g_tphi = -2.0 * M * a / r;
    g_phph = r * r + a * a + 2.0 * M * a * a / r;
}

double ell_kepler(double M, double a, double r) {
    double g_tt, g_tphi, g_phph;
    eq_metric(M, a, r, g_tt, g_tphi, g_phph);
    const double Om = omega_k(M, a, r);                       // prograde Ω_K
    const double D  = -(g_tt + 2.0 * g_tphi * Om + g_phph * Om * Om);
    const double ut = 1.0 / std::sqrt(std::max(D, 1e-30));
    return (g_tphi + g_phph * Om) * ut;                       // ℓ_K = u_φ
}

double omega_from_ell(double M, double a, double r, double ell) {
    double g_tt, g_tphi, g_phph;
    eq_metric(M, a, r, g_tt, g_tphi, g_phph);
    // ℓ(Ω) = (g_tφ + g_φφ Ω)/√D,  D = -(g_tt + 2 g_tφ Ω + g_φφ Ω²)
    // dℓ/dΩ = g_φφ/√D + (g_tφ + g_φφ Ω)·(g_tφ + g_φφ Ω)/D^{3/2}
    //       = g_φφ/√D + (numer)²/D^{3/2}   (numer ≡ g_tφ + g_φφ Ω)
    double Om = omega_k(M, a, r);                             // seed from Keplerian
    for (int it = 0; it < 40; ++it) {
        const double D = -(g_tt + 2.0 * g_tphi * Om + g_phph * Om * Om);
        if (!(D > 0.0)) { Om *= 0.5; continue; }              // stay timelike
        const double sqrtD = std::sqrt(D);
        const double numer = g_tphi + g_phph * Om;
        const double f  = numer / sqrtD - ell;
        const double df = g_phph / sqrtD + numer * numer / (D * sqrtD);
        if (!(std::abs(df) > 0.0)) break;
        double step = f / df;
        // damp to keep D>0
        double Om_new = Om - step;
        double D_new  = -(g_tt + 2.0 * g_tphi * Om_new + g_phph * Om_new * Om_new);
        int guard = 0;
        while (!(D_new > 0.0) && guard++ < 30) {
            step *= 0.5;
            Om_new = Om - step;
            D_new  = -(g_tt + 2.0 * g_tphi * Om_new + g_phph * Om_new * Om_new);
        }
        Om = Om_new;
        if (std::abs(step) <= 1e-14 * (std::abs(Om) + 1e-30)) break;
    }
    return Om;
}

} // namespace slim_detail

// ---------------------------------------------------------------------------
// Transonic radial residual (Phase 1, one-zone)
// ---------------------------------------------------------------------------
//
// State vector U (length 4N+2), index 0 = inner edge, N-1 = outer edge:
//   per node i:  U[4i+0]=Σ_i [g/cm²]   U[4i+1]=V_i (corotating radial vel, <0 inflow)
//                U[4i+2]=ℓ_i [u_φ,geom] U[4i+3]=T_{c,i} [K]
//   globals:     U[4N]=ℓ_in (eigenvalue)   U[4N+1]=r_s (sonic radius, [M])
//
// GRID: logarithmic in r over [in.r_in, in.r_out]. Log spacing concentrates
//       nodes near the hot inner disk / sonic point. r_i = r_in·(r_out/r_in)^{i/(N-1)}.
//
// CONVENTIONS / SIMPLIFICATIONS (Phase 1, one-zone):
//   * Geometric mechanics; CGS thermodynamics. The one-zone closure
//     one_zone_closure() returns the vertically-integrated P [erg/cm²] and the
//     midplane p_mid, rho_mid, c_s used below.
//   * Γ₁ fixed = 5/3 (ideal monatomic gas) ⇒ η₃ = 1/(Γ₁-1) = 1.5, Γ̃₁ = 1+1/η₃ = 5/3.
//     One-zone ⇒ η₃,η₄ are CONSTANTS, so dlnη₃/dlnr = dlnη₄/dlnr = 0; the η-gradient
//     term in 𝒩₁ and the Ω_⊥²(η₄/η₃)dlnη₄/dlnr term BOTH vanish (§22/§23).
//   * Ṁ cgs↔geometric: §23 mass law Ṁ = -2π Σ Δ^½ V/√(1-V²) is geometric in length.
//     Σ is CGS [g/cm²], Δ^½ is geometric [M]. The physical mass rate [g/s] is
//       Ṁ = -2π Σ Δ^½ (V/√(1-V²)) · r_g · c_cgs
//     ( Δ^½·r_g → cm gives g/cm; the 4-velocity component ·c_cgs → cm/s gives g/s ).
//     We compare against in.mdot [g/s] directly.  (Trap #5/§19: lengths × r_g.)
//   * P [erg/cm²] in the angular-momentum and pressure-gradient terms is the
//     vertically-integrated pressure; P/Σ has units erg/g = cm²/s², i.e. (c_s)².
//     Mechanics terms multiply P/Σ by 1/c² to make it the geometric (dimensionless)
//     specific-pressure that the §23 𝒟₀/𝒩₁ forms expect (where V is in units of c).
//
// ROW LAYOUT (square 4N+2; see header):
//   N (mass) + N (ang.mom.) + (N-1) (radial ODE) + (N-1) (energy ODE)
//     + 2 (outer BC) + 2 (regularity) = 4N+2.
// ---------------------------------------------------------------------------
namespace {

using slim_detail::OneZoneState;
using slim_detail::one_zone_closure;
using slim_detail::omega_k;
using slim_detail::kerr_delta;
using slim_detail::kerr_A;
using slim_detail::omega_perp2;
using slim_detail::ell_kepler;
using slim_detail::omega_from_ell;

// Fixed Phase-1 adiabatic indices (documented simplification).
constexpr double kGamma1 = 5.0 / 3.0;          // ideal monatomic gas
constexpr double kEta3   = 1.0 / (kGamma1 - 1.0); // = 1.5
constexpr double kGtilde1 = 1.0 + 1.0 / kEta3;  // = Γ̃₁ = 5/3

// State guards for transient Newton iterates.
constexpr double kSigmaFloor = 1e-30;
constexpr double kTFloor     = 1.0;
constexpr double kVCap       = 0.999999;        // keep |V|<1 (timelike)

struct NodeMech {
    double Delta, sqrtDelta, A, sqrtA;
    double Omega;          // orbital Ω [geometric 1/M] from ℓ
    double Omega_k_plus;   // prograde Keplerian Ω_K⁺
    double Omega_k_minus;  // retrograde Keplerian Ω_K⁻
};

static NodeMech node_mech(const SlimDiskInputs& in, double r, double ell) {
    NodeMech m;
    m.Delta     = kerr_delta(in.mass, in.spin, r);
    m.sqrtDelta = std::sqrt(std::max(m.Delta, 0.0));
    m.A         = kerr_A(in.mass, in.spin, r);
    m.sqrtA     = std::sqrt(std::max(m.A, 0.0));
    m.Omega     = omega_from_ell(in.mass, in.spin, r, ell);
    const double sqM = std::sqrt(in.mass);
    const double r32 = r * std::sqrt(r);
    m.Omega_k_plus  =  sqM / (r32 + in.spin * sqM);   // Ω_K⁺ = +√M/(r^{3/2}+a√M)
    // Ω_K⁻ denominator → 0 near r ≈ (a√M)^{2/3} just outside the horizon for near-extremal a.
    // Guard: floor |denom| to 1e-30 so we never divide by ~0 if r_in is pushed toward the horizon.
    const double denom_minus = r32 - in.spin * sqM;
    m.Omega_k_minus = -sqM / (std::abs(denom_minus) > 1e-30 ? denom_minus
                                                             : std::copysign(1e-30, denom_minus));
    return m;
}

// Gravitational/centrifugal term 𝒜 (S09 Eq 3, §23):
//   𝒜 = -M·A/(r³ Δ Ω_K⁺ Ω_K⁻) · (Ω-Ω_K⁺)(Ω-Ω_K⁻)/(1 - Ω̃²R̃²)
//   Ω̃ = Ω - ω,  ω = 2Mar/A,  R̃ = A/(r²Δ^½)
static double script_A(const SlimDiskInputs& in, double r, const NodeMech& m) {
    const double M = in.mass, a = in.spin;
    const double omega   = 2.0 * M * a * r / m.A;
    const double Om_tilde = m.Omega - omega;
    const double R_tilde  = m.A / (r * r * std::max(m.sqrtDelta, 1e-30));
    const double denom_rel = 1.0 - Om_tilde * Om_tilde * R_tilde * R_tilde;
    const double pref = -M * m.A
                      / (r * r * r * std::max(m.Delta, 1e-30)
                         * m.Omega_k_plus * m.Omega_k_minus);
    const double num  = (m.Omega - m.Omega_k_plus) * (m.Omega - m.Omega_k_minus);
    // guard the (1 - Ω̃²R̃²) denominator (corotation singularity); floor magnitude.
    const double dr = (std::abs(denom_rel) > 1e-12) ? denom_rel
                                                    : std::copysign(1e-12, denom_rel);
    return pref * num / dr;
}

// Mdot from a node (CGS, [g/s]) per the mass law; sign(V<0)->Mdot>0 for inflow.
static double mdot_of_node(const SlimDiskInputs& in, double Sigma, double V,
                           double sqrtDelta) {
    using namespace constants;
    const double Vc = std::clamp(V, -kVCap, kVCap);
    const double Gamma = 1.0 / std::sqrt(1.0 - Vc * Vc);
    return -2.0 * std::numbers::pi * Sigma * sqrtDelta * Vc * Gamma * in.r_g * c_cgs;
}

// Bundle of per-node derived quantities used across rows.
struct NodeEval {
    double r, Sigma, V, ell, Tc;
    OneZoneState oz;
    NodeMech mech;
    double Gamma;           // Lorentz factor 1/√(1-V²)
    double P_over_Sigma_geom; // (P/Σ)/c²  [dimensionless, = (c_s/c)²-ish]
    double cs2_geom;        // Γ̃₁·(P/Σ)/c²  (geometric specific c_s²)
};

static NodeEval eval_node(const SlimDiskInputs& in, const OpacityLUTs& op,
                          double r, double Sigma, double V, double ell, double Tc) {
    using namespace constants;
    NodeEval e;
    e.r = r;
    e.Sigma = std::max(Sigma, kSigmaFloor);
    e.V   = std::clamp(V, -kVCap, kVCap);
    e.ell = ell;
    e.Tc  = std::max(Tc, kTFloor);
    e.oz  = one_zone_closure(e.Sigma, e.Tc, r, in, op);
    e.mech = node_mech(in, r, ell);
    e.Gamma = 1.0 / std::sqrt(1.0 - e.V * e.V);
    // P/Σ has units erg/g = cm²/s²; divide by c² to get the geometric specific
    // pressure that the §23 𝒟₀/𝒩₁ forms (V in units of c) expect.
    const double P_over_Sigma = e.oz.P / e.Sigma;                 // [cm²/s²]
    e.P_over_Sigma_geom = P_over_Sigma / (c_cgs * c_cgs);         // dimensionless
    e.cs2_geom = kGtilde1 * e.P_over_Sigma_geom;
    return e;
}

// One-zone transonic denominator/numerator (§23). Both dimensionless (V in c).
static double calD0(const NodeEval& e) {
    return e.V * e.V - e.cs2_geom;       // 𝒟₀ = V² - Γ̃₁(P/Σ)
}

// 𝒩₁ = 𝒜 + (2πr²/(Ṁ η₃))·Q_adv + (P/Σ)·r(r-M)/Δ·Γ̃₁   (one-zone; η-grad & Ω_⊥² drop)
// Q_adv passed in geometric (dimensionless) form already; see caller.
static double calN1(const SlimDiskInputs& in, const NodeEval& e, double Qadv_geom) {
    const double A_term = script_A(in, e.r, e.mech);
    const double M = in.mass, r = e.r, Delta = std::max(e.mech.Delta, 1e-30);
    // Qadv_geom = (2π r²/(Ṁ η₃))·Q_adv already reduced to dimensionless by the
    // caller (CGS form / c²); the pressure term is the geometric (P/Σ)/c² · r(r-M)/Δ · Γ̃₁.
    const double press_term = e.P_over_Sigma_geom * r * (r - M) / Delta * kGtilde1;
    return A_term + Qadv_geom + press_term;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Crude thin-disk seed builder
// ---------------------------------------------------------------------------

std::vector<double> build_thin_disk_seed(const SlimDiskInputs& in,
                                         const OpacityLUTs& op) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);
    std::vector<double> U((size_t)4 * N + 2, 0.0);

    // Logarithmic radial grid r_in .. r_out (matches the residual's grid).
    const double r_in = in.r_in, r_out = in.r_out;
    const double lr0 = std::log(r_in), lr1 = std::log(r_out);

    // Characteristic Σ scale: a plausible α-disk surface density near r_in.
    // Σ ~ Ṁ/(3π ν), ν = α c_s²/Ω.  Use a rough c_s from a guessed T to set scale;
    // exact value is irrelevant — Task 5 relaxes it.  We just need finite/EOS-valid.
    // Pick Σ_ref so that the mass-conservation V comes out comfortably subsonic.
    const double r_ref = r_in;
    const double Omega_ref_cgs = omega_k(in.mass, in.spin, r_ref) * c_cgs / in.r_g; // 1/s
    const double T_ref = 1e7;                                   // K, hot inner disk guess
    const double cs2_ref = k_B * T_ref / (mu_fully_ionized * m_p);
    const double nu_ref = in.alpha * cs2_ref / std::max(Omega_ref_cgs, 1e-30);
    double Sigma_ref = in.mdot / (3.0 * std::numbers::pi * std::max(nu_ref, 1e-30));
    if (!(Sigma_ref > 0.0) || !std::isfinite(Sigma_ref)) Sigma_ref = 1e4;

    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        const double r = std::exp(lr0 + (lr1 - lr0) * t);

        // Σ(r) ∝ r^{-1} scaled to Σ_ref at r_ref.
        const double Sigma = Sigma_ref * (r_ref / r);

        // T_c(r): rough Novikov-Thorne-ish T ∝ r^{-3/4} scaled to T_ref at r_ref.
        const double Tc = T_ref * std::pow(r_ref / r, 0.75);

        // ℓ(r) = Keplerian ℓ_K.
        const double ell = ell_kepler(in.mass, in.spin, r);

        // V(r) from mass conservation:  Ṁ = -2π Σ Δ^½ (V/√(1-V²)) r_g c.
        // Let X ≡ V/√(1-V²) = -Ṁ / (2π Σ Δ^½ r_g c)  (X<0 inflow).
        const double sqrtDelta = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
        const double denom = 2.0 * std::numbers::pi * Sigma * sqrtDelta * in.r_g * c_cgs;
        double V = -1e-6;
        if (denom > 0.0) {
            const double X = -in.mdot / denom;                 // V/√(1-V²)
            V = X / std::sqrt(1.0 + X * X);                    // invert (|V|<1)
        }
        if (!(V < 0.0)) V = -1e-6;                             // enforce inflow
        V = std::clamp(V, -kVCap, -1e-12);

        U[4 * i + 0] = Sigma;
        U[4 * i + 1] = V;
        U[4 * i + 2] = ell;
        U[4 * i + 3] = Tc;
    }

    // Globals: ℓ_in = ℓ_K(r_isco-ish ≈ r_in), r_s = r_in (initial guesses).
    U[4 * N + 0] = ell_kepler(in.mass, in.spin, r_in);
    U[4 * N + 1] = r_in;
    return U;
}

// ---------------------------------------------------------------------------
// Radial residual evaluation
// ---------------------------------------------------------------------------
void slim_radial_residual(const std::vector<double>& U, const SlimDiskInputs& in,
                          const OpacityLUTs& op, std::vector<double>& R) {
    using namespace constants;
    using namespace slim_detail;
    const int N = std::max(in.n_nodes, 4);
    R.assign((size_t)4 * N + 2, 0.0);

    // Rebuild the (log) radial grid — same as the seed builder.
    const double lr0 = std::log(in.r_in), lr1 = std::log(in.r_out);
    std::vector<double> r(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i] = std::exp(lr0 + (lr1 - lr0) * t);
    }

    // Unpack + evaluate every node once.
    std::vector<NodeEval> e(N);
    for (int i = 0; i < N; ++i) {
        e[i] = eval_node(in, op, r[i],
                         U[4 * i + 0], U[4 * i + 1], U[4 * i + 2], U[4 * i + 3]);
    }
    const double ell_in = U[4 * N + 0];
    const double r_s    = U[4 * N + 1];
    const double Mdot   = in.mdot;                              // [g/s]

    // -----------------------------------------------------------------------
    // Group 1: mass conservation (N algebraic rows).  R = Ṁ_node - Ṁ_target.
    // -----------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        const double mdot_i = mdot_of_node(in, e[i].Sigma, e[i].V, e[i].mech.sqrtDelta);
        R[i] = mdot_i - Mdot;
    }

    // -----------------------------------------------------------------------
    // Group 2: angular momentum (N algebraic rows, §23):
    //   R = (Ṁ/2π)(ℓ_i - ℓ_in) - (A^½ Δ^½ Γ / r)·α·P
    //
    // ℓ convention: stored in GEOMETRIC units (u_φ, dimensionless ~ [M]).
    //   ℓ_geom × r_g × c_cgs → [cm²/s]  (specific angular momentum in CGS).
    // LHS: (Ṁ/2π)(ℓ-ℓ_in): Ṁ [g/s] × [cm²/s] = [g·cm²/s²] = erg.
    // RHS dimensional count: A^½~[M²], Δ^½~[M], /r~[1/M] → geomlen [M²].
    //   geomlen × r_g² → [cm²];  × Γ(1) × α(1) × P[erg/cm²] = erg.
    // Both sides in erg. ✓
    // -----------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        const NodeEval& ei = e[i];
        // LHS: (Ṁ/2π)(ℓ-ℓ_in), ℓ geometric [M] → CGS [cm²/s] via r_g·c_cgs.
        const double dl_cgs = (ei.ell - ell_in) * in.r_g * c_cgs; // [cm²/s]
        const double lhs = (Mdot / (2.0 * std::numbers::pi)) * dl_cgs;  // [g·cm²/s²] = erg
        // RHS: (A^½Δ^½/r) is [M^2] (A^{1/2}~M^2, Delta^{1/2}~M, /r~M); × r_g² → [cm²].
        const double geomlen = ei.mech.sqrtA * ei.mech.sqrtDelta / ei.r;  // [M^2]
        const double rhs = geomlen * in.r_g * in.r_g * ei.Gamma * in.alpha * ei.oz.P;  // [cm^2]*[erg/cm^2] = erg
        R[N + i] = lhs - rhs;
    }

    // -----------------------------------------------------------------------
    // Group 3: radial-momentum transonic ODE (N-1 trapezoidal rows, §23):
    //   dlnV/dlnr = (𝒩₁/𝒟₀)(1-V²)
    //   R = (lnV_{i+1}-lnV_i) - 0.5(lnr_{i+1}-lnr_i)(rhs_i + rhs_{i+1})
    // 𝒩₁ needs Q_adv (geometric).  Build Q_adv per node here (uses FD gradients
    // for the energy group too — but the radial-momentum term needs the geometric
    // scaled (2πr²/(Ṁ η₃))Q_adv, which we assemble inline).
    // -----------------------------------------------------------------------
    // Precompute geometric Q_adv-in-𝒩₁ contribution per node using FD gradients
    // of lnP and lnΣ.  Use one-sided/centred FD on the log grid.
    auto dln = [&](double f_lo, double f_hi, double r_lo, double r_hi) {
        return (std::log(std::max(f_hi, 1e-300)) - std::log(std::max(f_lo, 1e-300)))
             / (std::log(r_hi) - std::log(r_lo));
    };

    // Geometric Ṁ for the (2πr²/(Ṁη₃)) factor: invert the cgs conversion.
    //   Ṁ_geom (the bare -2πΣΔ^½V/√(1-V²) with Σ,length geometric) relates to the
    //   physical Ṁ [g/s] by Ṁ = Ṁ_geom · (Σ_unit · r_g · c).  But the §23 𝒩₁ uses
    //   the SAME geometric Ṁ that appears in its geometric Q_adv, so we keep BOTH
    //   in geometric form and let the (Σ_unit) factor cancel: i.e. evaluate
    //   (2πr²/(Ṁ_g η₃))·Q_adv,g with Q_adv,g = Q_adv,phys/(Σ_unit r_g c) ... this is
    //   fiddly.  Phase-1 expedient: assemble (2πr²/(Ṁ_phys η₃))·Q_adv_phys with BOTH
    //   in CGS, then divide by c² to render dimensionless (matching 𝒜 / 𝒟₀).
    //   Documented: only structural finiteness is required in Phase 1; Tasks 5-8
    //   validate the exact normalization.
    auto qadv_term_geom = [&](int i, int j) -> double {
        // FD gradients between nodes i,j (j=i±1) at node i's values for evaluation.
        const NodeEval& a = e[i];
        const NodeEval& b = e[j];
        const double dlnP = dln(a.oz.P, b.oz.P, a.r, b.r);
        const double dlnS = dln(a.Sigma, b.Sigma, a.r, b.r);
        // Q_adv (S11 Eq 29, CGS) = -(Ṁ/2π r_cm²)(P/Σ)[(Γ₁-1)dlnP - Γ₁ dlnΣ]
        const double r_cm = a.r * in.r_g;
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (a.oz.P / a.Sigma)
                          * ((kGamma1 - 1.0) * dlnP - kGamma1 * dlnS);   // [erg/cm²/s]
        // (2π r²/(Ṁ η₃))·Q_adv, rendered geometric/dimensionless:
        //   2π r_cm²/(Ṁ[g/s] η₃) [s·cm²/g] · Q_adv[erg/cm²/s=g/s³] = [cm²/s²];  /c² → dimensionless
        const double term = (2.0 * std::numbers::pi * r_cm * r_cm / (Mdot * kEta3)) * Qadv;
        return term / (c_cgs * c_cgs);
    };

    auto rhs_radial = [&](int i, int neighbor) -> double {
        const NodeEval& ei = e[i];
        const double D0 = calD0(ei);
        const double D0g = (std::abs(D0) > 1e-30) ? D0 : std::copysign(1e-30, D0 == 0 ? 1.0 : D0);
        const double Qadv_g = qadv_term_geom(i, neighbor);
        const double N1 = calN1(in, ei, Qadv_g);
        return (N1 / D0g) * (1.0 - ei.V * ei.V);
    };

    for (int i = 0; i < N - 1; ++i) {
        const double lnVi  = std::log(std::max(-e[i].V,   1e-300));   // V<0; use ln|V|
        const double lnVi1 = std::log(std::max(-e[i+1].V, 1e-300));
        const double dlnr  = std::log(r[i+1]) - std::log(r[i]);
        const double rhs_i  = rhs_radial(i,   i + 1);
        const double rhs_i1 = rhs_radial(i+1, i);
        R[2 * N + i] = (lnVi1 - lnVi) - 0.5 * dlnr * (rhs_i + rhs_i1);
    }

    // -----------------------------------------------------------------------
    // Group 4: energy ODE Q_vis = Q_rad + Q_adv (N-1 trapezoidal rows, §23).
    //   Q_vis = -(Ṁ/2π)(ℓ-ℓ_in)(dΩ/dr)(A^½Δ^½Γ/r³)
    //   Q_rad = 64 σ T_c⁴/(3 κ_R Σ)
    //   Q_adv = -(Ṁ/2π r²)(P/Σ)[(Γ₁-1)dlnP/dlnr - Γ₁ dlnΣ/dlnr]
    // Evaluate the residual at the interval midpoint trapezoidally: R = (G_i+G_{i+1})/2
    // where G = Q_vis - Q_rad - Q_adv, with dΩ/dr, dlnP, dlnΣ as FD across i,i+1.
    // All CGS [erg/cm²/s].
    // -----------------------------------------------------------------------
    auto Gbalance = [&](int i, int j) -> double {
        const NodeEval& a = e[i];
        const NodeEval& b = e[j];
        const double r_cm  = a.r * in.r_g;
        const double r_cm3 = r_cm * r_cm * r_cm;
        // dΩ/dr in CGS: Ω geometric → 1/s via c/r_g; r geometric → cm via r_g.
        const double dOmega_geom = (b.mech.Omega - a.mech.Omega) / (b.r - a.r); // [1/M²]
        const double dOmega_dr = dOmega_geom * (c_cgs / in.r_g) / in.r_g;        // [1/s/cm]
        // Q_vis = -(Ṁ/2π)(ℓ-ℓ_in)(dΩ/dr)(A^½Δ^½Γ/r³)  [erg/cm²/s]
        // Dimensional bookkeeping (all quantities below are in CGS unless noted):
        //   A^½~[M²], Δ^½~[M], r³~[M³]  →  geomfac ≡ A^½Δ^½/r³ is DIMENSIONLESS.
        //   Ṁ [g/s] × dl_cgs [cm²/s] × dOmega_dr [1/(s·cm)] × geomfac [1] × geomfac/r_g
        //   needs an extra [1/cm] to land at erg/cm²/s = g/s³.
        //   That [1/cm] comes from dividing the dimensionless geomfac by r_g [cm]:
        //     [g/s] × [cm²/s] × [1/(s·cm)] × (1/r_g)[1/cm] = g/s³ = erg/cm²/s.  ✓
        // Assembly: use geomfac/r_g as the net geometric factor.
        const double geomfac = a.mech.sqrtA * a.mech.sqrtDelta / (a.r * a.r * a.r); // dimensionless (A^{1/2}~M^2, Delta^{1/2}~M, /r^3~1/M^3)
        // (Ṁ/2π)(ℓ-ℓ_in): ℓ geometric → cm²/s via r_g·c.
        const double dl_cgs = (a.ell - ell_in) * in.r_g * c_cgs;                    // [cm²/s]
        const double Qvis = -(Mdot / (2.0 * std::numbers::pi)) * dl_cgs * dOmega_dr
                          * a.Gamma * (geomfac / in.r_g);  // [g/s]*[cm²/s]*[1/(s·cm)]*[1/cm] = erg/cm²/s
        // Q_rad:
        const double rho_mid = a.oz.rho_mid;
        const double kR = op.lookup_kappa_ross(rho_mid, a.Tc) + op.lookup_kappa_es(rho_mid, a.Tc);
        const double Qrad = 64.0 * sigma_SB * a.Tc * a.Tc * a.Tc * a.Tc
                          / (3.0 * std::max(kR, 1e-300) * a.Sigma);                  // [erg/cm²/s]
        // Q_adv:
        const double dlnP = dln(a.oz.P, b.oz.P, a.r, b.r);
        const double dlnS = dln(a.Sigma, b.Sigma, a.r, b.r);
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (a.oz.P / a.Sigma)
                          * ((kGamma1 - 1.0) * dlnP - kGamma1 * dlnS);               // [erg/cm²/s]
        return Qvis - Qrad - Qadv;
    };

    for (int i = 0; i < N - 1; ++i) {
        const double Gi  = Gbalance(i,   i + 1);
        const double Gi1 = Gbalance(i+1, i);
        R[3 * N - 1 + i] = 0.5 * (Gi + Gi1);
    }

    // -----------------------------------------------------------------------
    // Group 5: outer boundary conditions (2 rows).  The two ODE variables that
    // need an outer IC are ℓ (angular-momentum equation, treated as ODE in relaxation)
    // and T_c (energy ODE):
    //   row 4N-2:  ℓ(r_out) − ℓ_K(r_out)  (pins the outer angular-momentum BC to
    //              Keplerian; §23 outer BC: ℓ(r_out)=ℓ_K, since the disk is thin/cold
    //              at the outer edge and sub-sonic).
    //   row 4N-1:  T_c(N-1) − T_eff_thin(r_out)  (pins the energy ODE outer IC to
    //              the Novikov-Thorne effective temperature at the outer edge).
    // (V_out and Σ_out are determined by the algebraic mass & angular-momentum rows;
    //  ℓ_out and T_c,out are the two ODE outer ICs.  Together these yield a
    //  well-posed square system.)
    // -----------------------------------------------------------------------
    const int last = N - 1;
    R[4 * N - 2] = e[last].ell - ell_kepler(in.mass, in.spin, in.r_out);
    // T_eff_thin(r_out): rough Novikov-Thorne flux F=(3GMṀ/8πr³)(1-√(r_in/r)),
    // T_eff=(F/σ)^{1/4}.  Use CGS G,M and r in cm.
    {
        const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;  // M in g (from r_g=GM/c²)
        const double r_cm = in.r_out * in.r_g;
        const double F = (3.0 * G_cgs * M_cgs * Mdot / (8.0 * std::numbers::pi * r_cm * r_cm * r_cm))
                       * (1.0 - std::sqrt(in.r_in / in.r_out));
        const double T_eff = std::pow(std::max(F, 0.0) / sigma_SB, 0.25);
        R[4 * N - 1] = e[last].Tc - std::max(T_eff, kTFloor);
    }

    // -----------------------------------------------------------------------
    // Group 6: sonic-point regularity (2 rows, §23): 𝒟₀(r_s)=0 AND 𝒩₁(r_s)=0,
    // with (Σ,V,ℓ,T_c) linearly interpolated to r_s from the bracketing nodes.
    // -----------------------------------------------------------------------
    {
        // Locate the bracketing interval for r_s on the grid (clamp to ends).
        double rs = std::clamp(r_s, r[0], r[N - 1]);
        int k = 0;
        while (k < N - 2 && r[k + 1] < rs) ++k;     // r[k] <= rs <= r[k+1]
        const double rl = r[k], rh = r[k + 1];
        const double w = (rh > rl) ? (rs - rl) / (rh - rl) : 0.0;
        auto lerp = [&](double a, double b) { return a + (b - a) * w; };
        const double Sig_s = lerp(e[k].Sigma, e[k+1].Sigma);
        const double V_s   = lerp(e[k].V,     e[k+1].V);
        const double ell_s = lerp(e[k].ell,   e[k+1].ell);
        const double Tc_s  = lerp(e[k].Tc,    e[k+1].Tc);
        const NodeEval es = eval_node(in, op, rs, Sig_s, V_s, ell_s, Tc_s);
        // Q_adv at r_s via FD across the bracketing nodes (reuse qadv_term_geom
        // approximated with the bracketing pair, evaluated at es).
        const double dlnP = dln(e[k].oz.P, e[k+1].oz.P, e[k].r, e[k+1].r);
        const double dlnS = dln(e[k].Sigma, e[k+1].Sigma, e[k].r, e[k+1].r);
        const double r_cm = rs * in.r_g;
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (es.oz.P / es.Sigma)
                          * ((kGamma1 - 1.0) * dlnP - kGamma1 * dlnS);
        const double Qadv_g = (2.0 * std::numbers::pi * r_cm * r_cm / (Mdot * kEta3)) * Qadv
                            / (c_cgs * c_cgs);
        R[4 * N + 0] = calD0(es);
        R[4 * N + 1] = calN1(in, es, Qadv_g);
    }
}

// ---------------------------------------------------------------------------
// Radial solver stub (implemented in Task 5)
// ---------------------------------------------------------------------------
SlimDiskRadial solve_slim_disk_radial(const SlimDiskInputs& in, const OpacityLUTs& opacity) {
    (void)in; (void)opacity;
    SlimDiskRadial out;
    return out;  // not yet implemented (Task 5 wires the Newton relaxation)
}
} // namespace grrt
