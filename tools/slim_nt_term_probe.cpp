// ===========================================================================
// SLIM §23 ENERGY-TERM PROBE ON THE EXACT NT STATE  (diagnostic — DELETABLE)
// ---------------------------------------------------------------------------
// Pinpoints WHICH §23 energy term (Q_vis / Q_rad / Q_adv) breaks the
// slim -> Novikov-Thorne reduction at large radius, at a=0.9, f_Edd=0.02.
//
// Method (NO solver runs — pure term evaluation, runs in milliseconds):
//   1. Exact NT reference at each r: relativistic Page-Thorne one-face flux
//      F_PT(r) (Kerr circular-orbit E,L,Omega; cumulative integral from ISCO),
//      total dissipation F_NT = 2*F_PT [both faces], T_eff=(F_PT/sigma)^1/4.
//      Sigma_NT from the residual's OWN angular-momentum alpha-relation
//      (P_target bisection, identical to build_thin_disk_seed), T_c,NT from
//      the grey midplane T_c^4 = 0.75 T_eff^4 (tau+2/3), tau = kappa Sigma/2,
//      solved as a coupled (Sigma, T_c) fixed point.  Also T_c,slim from the
//      slim cooling law inverted at F_NT: T_c^4 = 3 kappa Sigma F_NT/(64 sigma).
//   2. Evaluate the slim Gbalance terms EXACTLY as src/slim_disk_radial.cpp
//      assembles them (same closure, same LUT kappa, same FD style) on that
//      NT state:  Q_vis (code form, geomfac/r_cm), Q_rad, Q_adv (+ pre-fix /r_g ref).
//   3. Also Q_vis with the geometric factor divided by the LOCAL radius
//      r_cm = r*r_g instead of the constant r_g ("rfix" hypothesis): the
//      Newtonian limit of the Page-Thorne dissipation REQUIRES the net
//      geometric factor to fall like 1/r (A^1/2 Delta^1/2/r^3 -> 1 is
//      dimensionless), so /r_g leaves Q_vis a factor r too large.
//   4. Report per-radius ratios to F_NT and a full factor-by-factor dump at
//      r=50 so the arithmetic can be checked by hand.
//
// Build:  cmake --build build --config Release --target slim-nt-term-probe
// Run:    build/Release/slim-nt-term-probe.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <numbers>

using namespace grrt;
using namespace grrt::slim_detail;

// ---------------------------------------------------------------------------
// Inputs: mirror tools/slim_benchmark_probe.cpp exactly.
// ---------------------------------------------------------------------------
static SlimDiskInputs make_inputs(double a, double f_Edd) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = a; in.alpha = 0.1; in.r_g = 1.48e6;
    in.r_out = 50.0; in.n_nodes = 48;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

// ---------------------------------------------------------------------------
// Kerr equatorial circular-orbit energetics (BPT72, geometric units).
// ---------------------------------------------------------------------------
static double E_circ(double M, double a, double r) {
    const double w = std::sqrt(M / (r*r*r)), aw = a * w;
    return (1.0 - 2.0*M/r + aw) / std::sqrt(1.0 - 3.0*M/r + 2.0*aw);
}
static double L_circ(double M, double a, double r) {
    const double w = std::sqrt(M / (r*r*r)), aw = a * w;
    return std::sqrt(M*r) * (1.0 - 2.0*aw + a*a/(r*r))
         / std::sqrt(1.0 - 3.0*M/r + 2.0*aw);
}
static double Omega_circ(double M, double a, double r) {
    const double sqM = std::sqrt(M);
    return sqM / (r*std::sqrt(r) + a*sqM);
}

// Page-Thorne flux from ONE face [erg/cm^2/s], exact relativistic NT:
//   F = (Mdot / 4 pi r) * f(r),  f = -Omega' (E-Omega L)^{-2} \int (E-Omega L) L' dr
// Newtonian limit check: f -> (3/2) M / r^2  =>  F -> 3 G M Mdot / (8 pi r_cm^3). OK
static double pt_flux_one_face(const SlimDiskInputs& in, double r) {
    using namespace constants;
    const double M = in.mass, a = in.spin;
    const double r_ms = isco_prograde(M, a);
    if (r <= r_ms) return 0.0;
    const double fd = 1e-6;
    auto EmOL = [&](double rr){ return E_circ(M,a,rr) - Omega_circ(M,a,rr)*L_circ(M,a,rr); };
    auto Lp   = [&](double rr){ return (L_circ(M,a,rr+fd) - L_circ(M,a,rr-fd)) / (2.0*fd); };
    // cumulative integral from r_ms (trapezoid, fine sub-grid; pure quadrature)
    const int NS = 4000;
    double I = 0.0, prev = EmOL(r_ms + fd) * Lp(r_ms + fd), prevr = r_ms + fd;
    for (int k = 1; k <= NS; ++k) {
        const double rr  = r_ms + fd + (r - r_ms - fd) * double(k) / NS;
        const double cur = EmOL(rr) * Lp(rr);
        I += 0.5 * (prev + cur) * (rr - prevr);
        prev = cur; prevr = rr;
    }
    const double dOm = (Omega_circ(M,a,r+fd) - Omega_circ(M,a,r-fd)) / (2.0*fd);
    const double den = EmOL(r);
    const double f = -dOm / (den*den) * I;                 // geometric, ~1/M
    // CGS: F = Mdot[g/s] * c^2 * f / (4 pi r r_g^2)
    return in.mdot * c_cgs * c_cgs * f / (4.0 * std::numbers::pi * r * in.r_g * in.r_g);
}

// Newtonian SS73 one-face flux [erg/cm^2/s] (cross-check only).
static double ss73_flux_one_face(const SlimDiskInputs& in, double r) {
    using namespace constants;
    const double r_ms = isco_prograde(in.mass, in.spin);
    const double GM = in.r_g * c_cgs * c_cgs;              // G M  [cm^3/s^2] (r_g = GM/c^2)
    const double r_cm = r * in.r_g;
    return (3.0 * GM * in.mdot / (8.0 * std::numbers::pi * r_cm*r_cm*r_cm))
         * (1.0 - std::sqrt(r_ms / r));
}

// ---------------------------------------------------------------------------
// NT state at radius r: Sigma from the residual's own alpha-relation,
// T_c from the grey midplane relation, coupled fixed point.
// ---------------------------------------------------------------------------
struct NTState {
    double r = 0.0;
    double F_tot = 0.0;      // 2 * F_PT  [erg/cm^2/s], total both-face dissipation
    double Teff = 0.0;       // (F_PT / sigma)^{1/4}
    double Sigma = 0.0, Tc = 0.0, Tc_slim = 0.0, Sigma_slim = 0.0;
    double P_target = 0.0, kappaR = 0.0, tau = 0.0, rho_mid = 0.0, H = 0.0;
    double ellK = 0.0, V = 0.0;
};

static NTState nt_state(const SlimDiskInputs& in, const OpacityLUTs& op, double r) {
    using namespace constants;
    NTState s; s.r = r;
    const double r_isco = isco_prograde(in.mass, in.spin);
    const double ell_in = ell_kepler(in.mass, in.spin, r_isco);
    const double F1 = pt_flux_one_face(in, r);
    s.F_tot = 2.0 * F1;
    s.Teff  = std::pow(std::max(F1, 0.0) / sigma_SB, 0.25);
    s.ellK  = ell_kepler(in.mass, in.spin, r);

    // P_target from the residual's angular-momentum balance (Gamma~1, ell=ell_K):
    //   (Mdot/2pi) (ell_K - ell_in) r_g c = (A^1/2 Delta^1/2 / r) r_g^2 alpha P
    const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
    const double sqrtA = std::sqrt(std::max(kerr_A(in.mass, in.spin, r), 0.0));
    const double dl_cgs = (s.ellK - ell_in) * in.r_g * c_cgs;
    const double geomlen = sqrtA * sqrtD / r;                       // [M^2]
    s.P_target = (in.mdot / (2.0 * std::numbers::pi)) * dl_cgs
               / std::max(geomlen * in.r_g * in.r_g * in.alpha, 1e-300);

    // Sigma(T_c) on the GAS branch: solve closure P(Sigma, T_c) = P_target.
    // P(Sigma) at fixed T is U-shaped (radiation term: H ~ 2 a T^4/(3 Sigma Omega^2)
    // as Sigma->0 makes P ~ T^8/Sigma blow up), so a naive monotone bisection can
    // land on the unphysical radiation branch.  Take the LARGEST root (the
    // increasing, gas-supported branch): walk lo downward from hi until P < P_target,
    // then bisect.  Returns -1 if no gas-branch root exists (past the radiation
    // ceiling: P(Sigma) > P_target for all Sigma).
    auto sigma_for_Tc = [&](double Tc_) -> double {
        auto Pof = [&](double Sig) { return one_zone_closure(Sig, Tc_, r, in, op).P; };
        // Locate the P(Sigma) minimum by log scan (U-shape: radiation branch falls,
        // gas branch rises).
        const double Slo = 1e-3, Shi = 1e12;
        const int NS = 220;
        double Smin = Slo, Pmin = 1e300;
        for (int k = 0; k <= NS; ++k) {
            const double Sg = Slo * std::pow(Shi / Slo, double(k) / NS);
            const double P = Pof(Sg);
            if (P < Pmin) { Pmin = P; Smin = Sg; }
        }
        if (Pmin > s.P_target) return -1.0;                 // no hydrostatic root at all
        if (Pof(Shi) > s.P_target) {
            // GAS-branch root (largest; P increasing on [Smin, Shi]) — preferred.
            double lo = Smin, hi = Shi;
            for (int b = 0; b < 80; ++b) {
                const double mid = std::sqrt(lo * hi);
                if (Pof(mid) < s.P_target) lo = mid; else hi = mid;
            }
            return std::sqrt(lo * hi);
        }
        // Fallback: RADIATION-branch root (P decreasing on [Slo, Smin]).
        double lo = Slo, hi = Smin;
        if (Pof(lo) < s.P_target) return -1.0;
        for (int b = 0; b < 80; ++b) {
            const double mid = std::sqrt(lo * hi);
            if (Pof(mid) > s.P_target) lo = mid; else hi = mid;
        }
        return std::sqrt(lo * hi);
    };

    // Coupled grey root: T_c^4 = 0.75 Teff^4 (tau + 2/3), tau = kappa Sigma(T_c)/2,
    // with Sigma(T_c) on the alpha-relation branch.  A damped fixed-point iteration
    // 2-cycles here (Sigma collapses at high T), so solve by SCAN + BISECTION on
    //   h(T) = ln[0.75 Teff^4 (tau(T)+2/3)] - 4 ln T   (first sign change upward).
    auto grey_h = [&](double T) -> double {
        const double Sig = sigma_for_Tc(T);
        if (Sig < 0.0) return -1e9;                         // past radiation ceiling
        const OneZoneState oz = one_zone_closure(Sig, T, r, in, op);
        const double kR = op.lookup_kappa_ross(oz.rho_mid, T) + op.lookup_kappa_es(oz.rho_mid, T);
        const double tau = 0.5 * kR * Sig;
        return std::log(0.75 * std::pow(s.Teff, 4.0) * (tau + 2.0/3.0)) - 4.0 * std::log(T);
    };
    auto first_root = [&](auto&& h) -> double {
        double Tlo = std::max(s.Teff, 1.0e4), Thi = 1.0e9;
        const int NSCAN = 240;
        double prevT = Tlo, prevh = h(Tlo);
        double rootlo = -1.0, roothi = -1.0;
        for (int k = 1; k <= NSCAN; ++k) {
            const double T = Tlo * std::pow(Thi / Tlo, double(k) / NSCAN);
            const double hv = h(T);
            if (prevh > 0.0 && hv <= 0.0) { rootlo = prevT; roothi = T; break; }
            prevT = T; prevh = hv;
        }
        if (rootlo < 0.0) return -1.0;                      // no sign change found
        for (int b = 0; b < 80; ++b) {
            const double Tm = std::sqrt(rootlo * roothi);
            if (h(Tm) > 0.0) rootlo = Tm; else roothi = Tm;
        }
        // Return the h>0 side: if the "root" is actually the radiation-ceiling
        // cliff (h jumps to -1e9 where the hydrostatic Sigma-root disappears),
        // rootlo is the last T with a VALID state; for a smooth root the interval
        // is ~1e-24 wide so the choice is irrelevant.
        return rootlo;
    };
    const double Tc = first_root(grey_h);
    s.Tc    = (Tc > 0.0) ? Tc : -1.0;
    s.Sigma = sigma_for_Tc(s.Tc > 0.0 ? s.Tc : std::max(s.Teff, 1e4));
    {
        const OneZoneState oz = one_zone_closure(s.Sigma, s.Tc, r, in, op);
        s.kappaR  = op.lookup_kappa_ross(oz.rho_mid, s.Tc) + op.lookup_kappa_es(oz.rho_mid, s.Tc);
        s.tau     = 0.5 * s.kappaR * s.Sigma;
        s.rho_mid = oz.rho_mid;
        s.H       = oz.H;
    }
    // Slim-consistent midplane temperature: invert the SLIM cooling law at F_NT,
    //   64 sigma T^4 / (3 kappa Sigma(T)) = F_tot,
    // again by scan + bisection on h(T) = ln Qrad(T) - ln F_tot (Qrad is monotone
    // increasing along the alpha-branch: T^4 up, Sigma down, Kramers kappa down).
    auto slim_h = [&](double T) -> double {
        const double Sig = sigma_for_Tc(T);
        if (Sig < 0.0) return -1e9;                         // past radiation ceiling
        const OneZoneState oz = one_zone_closure(Sig, T, r, in, op);
        const double kR = op.lookup_kappa_ross(oz.rho_mid, T) + op.lookup_kappa_es(oz.rho_mid, T);
        const double Qr = 64.0 * sigma_SB * T*T*T*T / (3.0 * std::max(kR, 1e-300) * Sig);
        return std::log(s.F_tot) - std::log(std::max(Qr, 1e-300));   // + below root, - above
    };
    const double Ts = first_root(slim_h);
    s.Tc_slim = (Ts > 0.0) ? Ts : -1.0;
    s.Sigma_slim = (Ts > 0.0) ? sigma_for_Tc(Ts) : -1.0;

    // V from mass conservation (identical to residual/seed):
    const double dn = 2.0 * std::numbers::pi * s.Sigma * sqrtD * in.r_g * c_cgs;
    double V = -1e-6;
    if (dn > 0.0) { const double X = -in.mdot / dn; V = X / std::sqrt(1.0 + X * X); }
    s.V = std::clamp(V, -kVCap, -1e-12);
    return s;
}

// ---------------------------------------------------------------------------
// Slim §23 terms evaluated on the NT state (assembly identical to Gbalance in
// src/slim_disk_radial.cpp; FD across two NT states at r*(1 -+ delta)).
// ---------------------------------------------------------------------------
struct SlimTerms {
    double Qvis_code = 0.0;   // geomfac / r_cm = /(r r_g) (the code's assembly, S09 Eq6×Eq4)
    double Qvis_old  = 0.0;   // geomfac / r_g             (PRE-FIX buggy assembly; before/after ref)
    double Qrad = 0.0, Qadv = 0.0;
    double dl_cgs = 0.0, dOmega_dr = 0.0, geomfac = 0.0, dlnP = 0.0, dlnS = 0.0;
    double Omega_geom = 0.0;
};

static SlimTerms slim_terms_on(const SlimDiskInputs& in, const OpacityLUTs& op,
                               const NTState& lo, const NTState& mid, const NTState& hi,
                               bool use_Tc_slim) {
    using namespace constants;
    SlimTerms t;
    const double r = mid.r, r_cm = r * in.r_g;
    const double r_isco = isco_prograde(in.mass, in.spin);
    const double ell_in = ell_kepler(in.mass, in.spin, r_isco);
    const double Tc  = use_Tc_slim ? mid.Tc_slim    : mid.Tc;
    const double Sig = use_Tc_slim ? mid.Sigma_slim : mid.Sigma;

    // dOmega/dr: Omega from ell via omega_from_ell (matches the residual), FD.
    const double Om_lo = omega_from_ell(in.mass, in.spin, lo.r,  lo.ellK);
    const double Om_hi = omega_from_ell(in.mass, in.spin, hi.r,  hi.ellK);
    t.Omega_geom = omega_from_ell(in.mass, in.spin, r, mid.ellK);
    const double dOmega_geom = (Om_hi - Om_lo) / (hi.r - lo.r);          // [1/M^2]
    t.dOmega_dr = dOmega_geom * (c_cgs / in.r_g) / in.r_g;               // [1/s/cm]

    const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
    const double sqrtA = std::sqrt(std::max(kerr_A(in.mass, in.spin, r), 0.0));
    t.geomfac = sqrtA / (std::max(sqrtD, 1e-30) * r);                     // dimensionless, A^½/(Δ^½r) (S09 Eq6×Eq4)
    t.dl_cgs  = (mid.ellK - ell_in) * in.r_g * c_cgs;                     // [cm^2/s]
    const double Gamma = 1.0 / std::sqrt(1.0 - mid.V * mid.V);

    // Q_vis exactly as Gbalance assembles it (geomfac / r_cm — LOCAL radius; S09 Eq6×Eq4):
    t.Qvis_code = -(in.mdot / (2.0 * std::numbers::pi)) * t.dl_cgs * t.dOmega_dr
                * Gamma * (t.geomfac / r_cm);
    // PRE-FIX assembly (constant r_g divisor) kept as the before/after reference:
    t.Qvis_old  = -(in.mdot / (2.0 * std::numbers::pi)) * t.dl_cgs * t.dOmega_dr
                * Gamma * (t.geomfac / in.r_g);

    // Q_rad exactly as Gbalance (LUT kappa at the NT midplane state):
    const OneZoneState oz = one_zone_closure(Sig, Tc, r, in, op);
    const double kR = op.lookup_kappa_ross(oz.rho_mid, Tc) + op.lookup_kappa_es(oz.rho_mid, Tc);
    t.Qrad = 64.0 * sigma_SB * Tc*Tc*Tc*Tc / (3.0 * std::max(kR, 1e-300) * Sig);

    // Q_adv exactly as Gbalance (FD ln-gradients across the NT neighbours):
    const double Tlo  = use_Tc_slim ? lo.Tc_slim    : lo.Tc;
    const double Thi  = use_Tc_slim ? hi.Tc_slim    : hi.Tc;
    const double Slo_ = use_Tc_slim ? lo.Sigma_slim : lo.Sigma;
    const double Shi_ = use_Tc_slim ? hi.Sigma_slim : hi.Sigma;
    const OneZoneState oz_lo = one_zone_closure(Slo_, Tlo, lo.r, in, op);
    const OneZoneState oz_hi = one_zone_closure(Shi_, Thi, hi.r, in, op);
    t.dlnP = (std::log(oz_hi.P) - std::log(oz_lo.P)) / (std::log(hi.r) - std::log(lo.r));
    t.dlnS = (std::log(Shi_)    - std::log(Slo_))    / (std::log(hi.r) - std::log(lo.r));
    // S11 Eq 29 one-zone bracket [η₃·dlnP − (1+η₃)·dlnΣ], η₃ = 1/(Γ₁−1) = 3/2
    // (flag #1 correction 2026-06-12; was the inverted [(Γ₁−1)dlnP − Γ₁dlnΣ]).
    const double eta3 = 1.0 / (kGamma1 - 1.0);
    t.Qadv = -(in.mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
           * (oz.P / Sig)
           * (eta3 * t.dlnP - (1.0 + eta3) * t.dlnS);
    return t;
}

int main() {
    using namespace constants;
    const double a = 0.9, f_Edd = 0.02;
    SlimDiskInputs in = make_inputs(a, f_Edd);
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    const double r_isco = isco_prograde(in.mass, in.spin);
    const double ell_in = ell_kepler(in.mass, in.spin, r_isco);
    std::printf("# slim-nt-term-probe  a=%.3f f_Edd=%.3f alpha=%.2f r_g=%.3e cm\n",
                a, f_Edd, in.alpha, in.r_g);
    std::printf("# Mdot=%.6e g/s   r_isco=%.6f M   ell_in=ell_K(isco)=%.6f\n",
                in.mdot, r_isco, ell_in);
    std::printf("# F_NT = 2*F_PT (TOTAL both-face dissipation); on a correct NT-reducing\n");
    std::printf("# solver Q_vis ~ Q_rad ~ F_NT and Q_adv ~ 0 at f_Edd=0.02.\n\n");

    const double radii[] = {3.0, 6.0, 10.0, 20.0, 35.0, 50.0};
    const double delta = 0.05;

    std::printf("%-6s %-11s %-11s | %-9s %-9s %-9s | %-9s | %-8s %-11s %-11s %-11s %-9s\n",
                "r[M]", "F_NT", "F_SSx2", "Qvis/F", "Qrad/F", "Qadv/F", "Qv_old/F",
                "kappaR", "Sigma_NT", "Tc_NT", "Teff_NT", "tau");
    std::vector<NTState> mids;
    for (double r : radii) {
        NTState lo  = nt_state(in, op, r * (1.0 - delta));
        NTState mid = nt_state(in, op, r);
        NTState hi  = nt_state(in, op, r * (1.0 + delta));
        mids.push_back(mid);
        SlimTerms t = slim_terms_on(in, op, lo, mid, hi, /*use_Tc_slim=*/false);
        const double F = mid.F_tot;
        std::printf("%-6.1f %-11.4e %-11.4e | %-9.3f %-9.3f %-9.2e | %-9.3f | %-8.3f %-11.4e %-11.4e %-11.4e %-9.3e\n",
                    r, F, 2.0 * ss73_flux_one_face(in, r),
                    t.Qvis_code / F, t.Qrad / F, t.Qadv / F, t.Qvis_old / F,
                    mid.kappaR, mid.Sigma, mid.Tc, mid.Teff, mid.tau);
    }

    // Same table with the SLIM-consistent midplane T_c (Q_rad == F_NT by
    // construction -> isolates the Q_vis question from the grey-vs-64/3
    // vertical-structure convention).
    std::printf("\n# With T_c = T_c,slim (64 sigma T^4/(3 kappa Sigma) = F_NT by construction):\n");
    std::printf("%-6s %-9s %-9s %-9s | %-9s | %-11s %-11s | %-12s %-12s\n",
                "r[M]", "Qvis/F", "Qrad/F", "Qadv/F", "Qv_old/F",
                "Tc_slim", "Tc_grey", "G_code/F", "G_old/F");
    for (size_t i = 0; i < std::size(radii); ++i) {
        const double r = radii[i];
        NTState lo  = nt_state(in, op, r * (1.0 - delta));
        NTState mid = mids[i];
        NTState hi  = nt_state(in, op, r * (1.0 + delta));
        SlimTerms t = slim_terms_on(in, op, lo, mid, hi, /*use_Tc_slim=*/true);
        const double F = mid.F_tot;
        const double G_code = t.Qvis_code - t.Qrad - t.Qadv;
        const double G_old  = t.Qvis_old  - t.Qrad - t.Qadv;
        std::printf("%-6.1f %-9.3f %-9.3f %-9.2e | %-9.3f | %-11.4e %-11.4e | %-12.4e %-12.4e\n",
                    r, t.Qvis_code / F, t.Qrad / F, t.Qadv / F, t.Qvis_old / F,
                    mid.Tc_slim, mid.Tc, G_code / F, G_old / F);
    }

    // ------------------------------------------------------------------
    // Factor-by-factor dump at r=50 (hand-checkable arithmetic).
    // ------------------------------------------------------------------
    {
        const double r = 50.0, r_cm = r * in.r_g;
        NTState lo  = nt_state(in, op, r * (1.0 - delta));
        NTState mid = nt_state(in, op, r);
        NTState hi  = nt_state(in, op, r * (1.0 + delta));
        SlimTerms t = slim_terms_on(in, op, lo, mid, hi, false);
        const double sqrtD = std::sqrt(kerr_delta(in.mass, in.spin, r));
        const double sqrtA = std::sqrt(kerr_A(in.mass, in.spin, r));
        std::printf("\n# ---- r=50 factor dump ----\n");
        std::printf("Mdot              = %.6e g/s\n", in.mdot);
        std::printf("ell_K(50)         = %.6f   ell_in = %.6f   (L_circ BPT72 = %.6f)\n",
                    mid.ellK, ell_in, L_circ(in.mass, in.spin, r));
        std::printf("dl_cgs            = (ell_K-ell_in)*r_g*c = %.6e cm^2/s\n", t.dl_cgs);
        std::printf("Omega(ell_K)      = %.6e [1/M]  (Omega_K = %.6e)\n",
                    t.Omega_geom, Omega_circ(in.mass, in.spin, r));
        std::printf("dOmega/dr (geom)  = %.6e [1/M^2] -> CGS %.6e [1/s/cm]\n",
                    t.dOmega_dr / ((c_cgs/in.r_g)/in.r_g), t.dOmega_dr);
        std::printf("sqrtA=%.6e [M^2]  sqrtD=%.6e [M]  r^3=%.6e [M^3]\n", sqrtA, sqrtD, r*r*r);
        std::printf("geomfac A^.5/(D^.5 r) = %.6f  (dimensionless; ->1 as r->inf)\n", t.geomfac);
        std::printf("CODE  divisor r_cm = %.4e cm -> Qvis_code = %.6e erg/cm^2/s\n", r_cm, t.Qvis_code);
        std::printf("OLD   divisor r_g  = %.4e cm -> Qvis_old  = %.6e erg/cm^2/s (pre-fix)\n", in.r_g, t.Qvis_old);
        std::printf("F_PT(one face)    = %.6e   F_NT(total) = %.6e   2*F_SS73 = %.6e\n",
                    0.5 * mid.F_tot, mid.F_tot, 2.0 * ss73_flux_one_face(in, r));
        std::printf("Qvis_code/F_NT    = %.4f      Qvis_old/F_NT = %.4f (pre-fix)\n",
                    t.Qvis_code / mid.F_tot, t.Qvis_old / mid.F_tot);
        std::printf("NT state: Sigma=%.5e g/cm^2  Tc_grey=%.5e K  Tc_slim=%.5e K  Teff=%.5e K\n",
                    mid.Sigma, mid.Tc, mid.Tc_slim, mid.Teff);
        std::printf("          rho_mid=%.5e g/cm^3  H=%.5e cm (H/r=%.4f)  P_target=%.5e erg/cm^2\n",
                    mid.rho_mid, mid.H, mid.H / r_cm, mid.P_target);
        std::printf("          kappaR=%.5f cm^2/g  tau=kappa*Sigma/2=%.5e\n", mid.kappaR, mid.tau);
        std::printf("Qrad(grey Tc)     = %.6e  (Qrad/F_NT = %.4f)\n", t.Qrad, t.Qrad / mid.F_tot);
        std::printf("Qadv              = %.6e  (Qadv/F_NT = %.4e; dlnP=%.4f dlnSig=%.4f)\n",
                    t.Qadv, t.Qadv / mid.F_tot, t.dlnP, t.dlnS);
        std::printf("V(mass cons.)     = %.6e c\n", mid.V);
    }
    std::printf("\n[slim-nt-term-probe] done.\n");
    return 0;
}
