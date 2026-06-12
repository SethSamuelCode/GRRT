// ===========================================================================
// SLIM-DISK BENCHMARK PROBE  (diagnostic — DELETABLE)
// ---------------------------------------------------------------------------
// Benchmarks the candidate near-Eddington slim-disk solution against ground
// truth.  Five tests (see the task spec):
//   (1) NT REDUCTION at f_Edd=0.02,0.05 (a=0.9): cold thin-disk seed converges;
//       compare converged Sigma/T_c/H/Q_rad against the in-house relativistic
//       Novikov-Thorne reference (built from the SAME residual physics with
//       Q_adv forced to 0 — i.e. the NT-balanced thin seed, plus an independent
//       NT relativistic-flux T_eff).  Per-radius % agreement + f_adv magnitude.
//   (2) THEORY at f_Edd=0.9: dump H/r(r), beta(r), f_adv(r), T_c/Sigma shapes.
//   (3) f_adv SIGN CHECK: per-node Q_adv, Q_rad, dlnP/dlnr, dlnSigma/dlnr so we
//       can read off WHY f_adv is negative (sign of the bracket vs the §23 def).
//   (4) RESOLUTION: re-run f_Edd=0.9 at N=48/96/150, locate the mid-disk glitch.
//   (5) BETTER SEED: thick-inner / thin-outer seed (H/r large only inside ~10
//       r_g, gas-dominated thin outer), relax, compare to the thick-outer root.
//
// Reuses the EXACT production machinery (solve_single_am = outer ell_in bracket
// + inner relax_structure + validity gate).  ONLY seeds / N / comparisons vary.
// NO residual-physics change.  Safety budget ON; tight per-solve wall caps.
//
// #includes slim_disk_radial.cpp + opacity.cpp directly (probe/test pattern).
//
// Build:  cmake --build build --config Release --target slim-benchmark-probe
// Run:    build/Release/slim-benchmark-probe.exe [test]   (test in 1..5; 0=all)
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numbers>

using namespace grrt;
using namespace grrt::slim_detail;

// ---------------------------------------------------------------------------
// inputs (mirror the other slim probes exactly: r_g=1.48e6, r_out=50, alpha=0.1)
// ---------------------------------------------------------------------------
static SlimDiskInputs make_inputs(double a, double f_Edd, int N, double wall_s) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = a; in.alpha = 0.1; in.r_g = 1.48e6;
    in.r_out = 50.0; in.n_nodes = N; in.max_iters = 800; in.tol = 1e-6;
    in.budget_wall_seconds = wall_s;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

struct SolveOut {
    bool conv = false, tripped = false;
    double wall = 0.0, merit = 0.0;
    GroupMags gm{};
    ValidityResult v{};
};
static SolveOut run_solve(const SlimDiskInputs& in, const OpacityLUTs& op,
                          std::vector<double>& U, double wall_s, bool require_N1 = true) {
    SolveOut so;
    SolveBudget budget; budget.wall_cap_s = wall_s; budget.start = std::chrono::steady_clock::now();
    g_budget = &budget;
    auto t0 = std::chrono::steady_clock::now();
    so.conv = solve_single_am(in, op, U, require_N1);
    auto t1 = std::chrono::steady_clock::now();
    so.wall = std::chrono::duration<double>(t1-t0).count();
    so.tripped = budget.tripped;
    g_budget = nullptr;
    std::vector<double> R; slim_radial_residual(U, in, op, R);
    so.merit = slim_scaled_residual_norm(U, R, in);
    so.gm = slim_group_mags(U, R, in);
    so.v = slim_validity_gate(in, op, U, require_N1);
    return so;
}

// Per-node thermodynamics helper (matches unpack_profile / residual definitions).
struct NodeTherm {
    double r, Sigma, Tc, V, Hr, beta, f_adv;
    double Qadv, Qrad, Qvis, dlnP, dlnS, P_over_Sig, rho_mid;
};
static std::vector<NodeTherm> therm_of(const SlimDiskInputs& in, const OpacityLUTs& op,
                                       const std::vector<double>& U) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    SlimDiskRadial out; unpack_profile(in, op, U, out);
    const double Mdot = in.mdot;
    auto dln = [&](double f_lo, double f_hi, double r_lo, double r_hi) {
        return (std::log(std::max(f_hi, 1e-300)) - std::log(std::max(f_lo, 1e-300)))
             / (std::log(r_hi) - std::log(r_lo));
    };
    std::vector<NodeTherm> v(N);
    for (int i = 0; i < N; ++i) {
        const double r = out.r[i], Sig = out.Sigma[i], Tc = out.Tc[i];
        const OneZoneState oz = one_zone_closure(std::max(Sig, kSigmaFloor),
                                                 std::max(Tc, kTFloor), r, in, op);
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const OneZoneState ozj = one_zone_closure(std::max(out.Sigma[j], kSigmaFloor),
                                                  std::max(out.Tc[j], kTFloor), out.r[j], in, op);
        const double dlnP = dln(oz.P, ozj.P, r, out.r[j]);
        const double dlnS = dln(Sig, out.Sigma[j], r, out.r[j]);
        const double r_cm = r * in.r_g;
        const double Qadv = -(Mdot / (2.0 * std::numbers::pi * r_cm * r_cm))
                          * (oz.P / std::max(Sig, kSigmaFloor))
                          * ((kGamma1 - 1.0) * dlnP - kGamma1 * dlnS);
        const double kR = op.lookup_kappa_ross(oz.rho_mid, Tc) + op.lookup_kappa_es(oz.rho_mid, Tc);
        const double Qrad = 64.0 * sigma_SB * Tc*Tc*Tc*Tc
                          / (3.0 * std::max(kR, 1e-300) * std::max(Sig, kSigmaFloor));
        // Slim's OWN Q_vis (Group-4 heating, CGS) — identical assembly to the residual
        // Gbalance: Q_vis = -(Mdot/2pi)(ell-ell_in)(dOmega/dr)(A^½Δ^½Γ/r³)/r_cm
        // (LOCAL r_cm = r·r_g divisor, i.e. A^½Δ^½Γ/r⁴ in geometric units; S11 13/23).
        const double Om_i = omega_from_ell(in.mass,in.spin,r,out.r[i]>0?U[4*i+2]:0.0);
        const double Om_j = omega_from_ell(in.mass,in.spin,out.r[j],U[4*j+2]);
        const double dOmega_geom = (Om_j - Om_i) / (out.r[j] - r);          // [1/M^2]
        const double dOmega_dr = dOmega_geom * (c_cgs / in.r_g) / in.r_g;   // [1/s/cm]
        const double sqrtA = std::sqrt(std::max(kerr_A(in.mass,in.spin,r),0.0));
        const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass,in.spin,r),0.0));
        const double geomfac = sqrtA*sqrtD/(r*r*r);
        const double Gam = 1.0/std::sqrt(1.0 - std::min(out.V[i]*out.V[i],0.999999));
        const double dl_cgs = (U[4*i+2] - U[4*N+0]) * in.r_g * c_cgs;
        const double Qvis = -(Mdot/(2.0*std::numbers::pi)) * dl_cgs * dOmega_dr
                          * Gam * (geomfac / r_cm);                          // [erg/cm^2/s]
        NodeTherm t;
        t.r = r; t.Sigma = Sig; t.Tc = Tc; t.V = out.V[i];
        t.Hr = out.H[i] / (r * in.r_g);
        t.beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
        t.f_adv = out.f_adv[i];
        t.Qadv = Qadv; t.Qrad = Qrad; t.Qvis = Qvis; t.dlnP = dlnP; t.dlnS = dlnS;
        t.P_over_Sig = oz.P / std::max(Sig, kSigmaFloor); t.rho_mid = oz.rho_mid;
        v[i] = t;
    }
    return v;
}

// ---------------------------------------------------------------------------
// Independent relativistic Novikov-Thorne reference (same Kerr circular-orbit
// energetics as VolumetricDisk::build_flux_lut / accretion_disk.cpp).  Returns
// the local DISSIPATED flux per face Q_vis,NT(r) [erg/cm^2/s] at absolute Mdot,
// so T_eff,NT = (Q_vis,NT/sigma_SB)^{1/4}.  Uses the SAME 3M/(8 pi r^3) ... form;
// the slim residual's Q_vis is the relativistic-ell version of this, so at low
// Mdot (f_adv->0, Q_rad=Q_vis) the slim Q_rad must reduce to this.
// ---------------------------------------------------------------------------
static void nt_flux_profile(const SlimDiskInputs& in, const std::vector<double>& rgrid,
                            std::vector<double>& Qvis_out /*[erg/cm^2/s]*/,
                            std::vector<double>& Teff_out /*[K]*/) {
    using namespace constants;
    const double M = in.mass, a = in.spin;
    const double r_isco = isco_prograde(M, a);
    auto omega_kepler = [&](double r){ return std::sqrt(M/(r*r*r)); };
    auto Omega = [&](double r){ const double w=omega_kepler(r); return w/(1.0+a*w); };
    auto E_circ = [&](double r){ const double w=omega_kepler(r), aw=a*w;
        return (1.0-2.0*M/r+aw)/std::sqrt(1.0-3.0*M/r+2.0*aw); };
    auto L_circ = [&](double r){ const double w=omega_kepler(r), aw=a*w;
        return std::sqrt(M*r)*(1.0-2.0*aw+a*a/(r*r))/std::sqrt(1.0-3.0*M/r+2.0*aw); };
    const double E_isco = E_circ(r_isco), L_isco = L_circ(r_isco);
    const double fd = 1e-6;
    // NT cumulative integral from r_isco outward (fine sub-grid for accuracy).
    auto integrand = [&](double r){
        const double Ep=(E_circ(r+fd)-E_circ(r-fd))/(2*fd);
        const double Lp=(L_circ(r+fd)-L_circ(r-fd))/(2*fd);
        return (E_circ(r)-E_isco)*Lp - (L_circ(r)-L_isco)*Ep;
    };
    auto flux_geom = [&](double r)->double{
        if (r <= r_isco) return 0.0;
        // integrate from r_isco to r
        const int NS = 4000;
        double I=0.0, prev=integrand(r_isco), prevr=r_isco;
        for (int k=1;k<=NS;++k){
            const double rr = r_isco + (r-r_isco)*k/NS;
            const double cur = integrand(rr);
            I += 0.5*(prev+cur)*(rr-prevr);
            prev=cur; prevr=rr;
        }
        const double Om=Omega(r), Er=E_circ(r), Lr=L_circ(r);
        const double dOm=(Omega(r+fd)-Omega(r-fd))/(2*fd);
        const double den=Er-Om*Lr;
        if (std::abs(den)<1e-20) return 0.0;
        // F = (3M/(8 pi r^3)) (1/(E-OmL)) (-dOm/dr) * I    [geometric, per unit Mdot]
        return (3.0*M/(8.0*std::numbers::pi*r*r*r))*(1.0/den)*(-dOm)*I;
    };
    // Convert geometric flux-per-Mdot to CGS [erg/cm^2/s]: F_geom has units 1/M^2
    // (per unit Mdot in geometric mass/M). With Mdot in g/s, multiply by
    // Mdot * c^2 / (4 pi r_g^2) is NOT it; we calibrate by dimension:
    //   Q_vis,NT = F_geom(r) * Mdot[g/s] * c_cgs^2 / r_g^2   [erg/cm^2/s].
    // (F_geom*Mdot has units [g/s / M^2]; *c^2 -> erg/s/M^2; /r_g^2 -> erg/s/cm^2.)
    const double conv = in.mdot * c_cgs * c_cgs / (in.r_g * in.r_g);
    Qvis_out.resize(rgrid.size());
    Teff_out.resize(rgrid.size());
    for (size_t i=0;i<rgrid.size();++i){
        const double Q = std::max(flux_geom(rgrid[i]) * conv, 0.0);
        Qvis_out[i] = Q;
        Teff_out[i] = std::pow(std::max(Q,0.0)/sigma_SB, 0.25);
    }
}

// ===========================================================================
// (1) NT REDUCTION at low f_Edd.
// ===========================================================================
static void test1_nt_reduction(const OpacityLUTs& op, double a, double f_Edd, double wall_s) {
    using namespace constants;
    const int N = 48;
    std::printf("\n############################################################\n");
    std::printf("#  (1) NT REDUCTION  a=%.3f  f_Edd=%.3f  N=%d\n", a, f_Edd, N);
    std::printf("############################################################\n");
    SlimDiskInputs in = make_inputs(a, f_Edd, N, wall_s);

    // Cold thin-disk (NT-balanced) seed -> this IS the NT reference profile by
    // construction (angular-momentum Sigma + energy T_c with Q_adv=0).
    std::vector<double> Useed = build_thin_disk_seed(in, op);
    std::vector<double> U = Useed;
    SolveOut so = run_solve(in, op, U, wall_s, /*require_N1=*/true);
    std::printf("  cold-seed solve: conv=%d tripped=%d merit=%.3e wall=%.1fs "
                "ell_in=%.6f r_s=%.4f r_isco=%.4f\n",
                (int)so.conv, (int)so.tripped, so.merit, so.wall, U[4*N+0], U[4*N+1], so.v.r_isco);
    if (!so.conv) {
        std::printf("  (cold seed did NOT converge at f_Edd=%.3f; trying require_N1=false)\n", f_Edd);
        U = Useed;
        so = run_solve(in, op, U, wall_s, /*require_N1=*/false);
        std::printf("  retry: conv=%d merit=%.3e\n", (int)so.conv, so.merit);
        if (!so.conv) { std::printf("  (still no convergence; abort test 1 at this f_Edd)\n"); return; }
    }

    // Converged + seed therm.
    std::vector<NodeTherm> tc = therm_of(in, op, U);          // converged slim
    // Independent NT-flux reference (geometric shape), normalized by matching its
    // total dissipation to the slim's OWN trusted Q_vis at a clean mid-disk node
    // (node 20, r~8-9) — this removes any absolute-constant ambiguity and tests
    // the SHAPE of Teff(r), which the reduction theorem fixes.
    std::vector<double> rgrid(N); for (int i=0;i<N;++i) rgrid[i]=tc[i].r;
    std::vector<double> Qvis_nt, Teff_nt; nt_flux_profile(in, rgrid, Qvis_nt, Teff_nt);
    const int iref = 20;
    const double norm = (Qvis_nt[iref]>0.0) ? tc[iref].Qvis / Qvis_nt[iref] : 1.0;
    std::printf("  [NT-flux ref normalized to slim Q_vis at node %d (r=%.2f): factor=%.3e]\n",
                iref, rgrid[iref], norm);

    // (A) RIGOROUS internal check: at low Mdot, slim Q_rad must equal slim Q_vis
    //     (f_adv->0).  Both are trusted CGS quantities from the residual itself.
    // (B) SHAPE check: Teff_slim=(Q_rad/sigma)^{1/4} vs normalized NT-flux Teff.
    std::printf("  %-3s %-8s | %-11s %-11s %-+8s | %-10s %-10s %-+7s | %-9s %-+9s\n",
                "i","r[M]","Qrad_slim","Qvis_slim","Qrad/Qvis-1%","Teff_slim","Teff_NTnrm","dTeff%","H/r","f_adv");
    double max_QrQv=0, maxdT=0, fadv_absmax=0;
    for (int i=0;i<N;++i){
        const double Teff_slim = std::pow(std::max(tc[i].Qrad,0.0)/sigma_SB,0.25);
        const double Teff_ntn  = std::pow(std::max(Qvis_nt[i]*norm,0.0)/sigma_SB,0.25);
        const double QrQv = (tc[i].Qvis>0)?100.0*(tc[i].Qrad/tc[i].Qvis-1.0):0.0;
        const double dT = (Teff_ntn>0)?100.0*(Teff_slim-Teff_ntn)/Teff_ntn:0.0;
        if (i>=4 && i<N-2){ max_QrQv=std::max(max_QrQv,std::abs(QrQv)); maxdT=std::max(maxdT,std::abs(dT));
                            fadv_absmax=std::max(fadv_absmax,std::abs(tc[i].f_adv)); }
        std::printf("  %-3d %-8.4f | %-11.3e %-11.3e %-+8.1f | %-10.3e %-10.3e %-+7.1f | %-9.4f %-+9.2e\n",
                    i, tc[i].r, tc[i].Qrad, tc[i].Qvis, QrQv, Teff_slim, Teff_ntn, dT,
                    tc[i].Hr, tc[i].f_adv);
    }
    std::printf("  >> bulk (nodes 4..N-3): max|Qrad/Qvis-1|=%.1f%%  max|dTeff_shape|=%.1f%%  max|f_adv|=%.2e\n",
                max_QrQv, maxdT, fadv_absmax);
    std::printf("  (Qrad=Qvis to <few%% AND f_adv~0  =>  slim reduces to NT energetics; "
                "dTeff_shape tests NT flux SHAPE.)\n");
    std::fflush(stdout);
}

// ===========================================================================
// (2)+(3) THEORY + f_adv SIGN CHECK at f_Edd=0.9 (uses the known slim-branch seed).
// ===========================================================================
// thick/advective upper-branch seed (same construction as slim_slimseed_probe).
static std::vector<double> build_slim_branch_seed(const SlimDiskInputs& in,
                                                  const OpacityLUTs& op,
                                                  double target_Hr, double sigma_mult, double Tc_floor) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    std::vector<double> U((size_t)4 * N + 2, 0.0);
    const double r_isco = isco_prograde(in.mass, in.spin);
    const double r_s = std::max(0.98 * r_isco, in.r_in * 1.001);
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    const double ell_in = ell_kepler(in.mass, in.spin, r_isco);
    auto Tc_for_H = [&](double Sig, double r, double H_target)->double{
        auto H_of=[&](double Tc_){return one_zone_closure(Sig,Tc_,r,in,op).H;};
        double lo=std::max(Tc_floor,1e3), hi=1e10;
        if(!(H_of(hi)>H_target)) return hi;
        if(H_of(lo)>H_target) return lo;
        for(int b=0;b<80;++b){const double mid=std::sqrt(lo*hi); if(H_of(mid)<H_target) lo=mid; else hi=mid;}
        return std::sqrt(lo*hi);
    };
    auto Vfrom=[&](double r,double Sig)->double{
        const double sqrtD=std::sqrt(std::max(kerr_delta(in.mass,in.spin,r),0.0));
        const double dn=2.0*std::numbers::pi*Sig*sqrtD*in.r_g*c_cgs;
        double V=-1e-6; if(dn>0.0){const double X=-in.mdot/dn; V=X/std::sqrt(1.0+X*X);}
        if(!(V<0.0)) V=-1e-6; return std::clamp(V,-kVCap,-1e-12);
    };
    for(int i=0;i<N;++i){
        const double t=(N==1)?0.0:double(i)/double(N-1);
        const double r=std::exp(lr0+(lr1-lr0)*t);
        const double Om_K=omega_k(in.mass,in.spin,r);
        const double Om_K_cgs=Om_K*(c_cgs/in.r_g);
        const double r_cm=r*in.r_g;
        const double Sig_slim=in.mdot/(2.0*std::numbers::pi*r_cm*r_cm*std::max(Om_K_cgs,1e-300));
        double Sig=std::max(sigma_mult*Sig_slim,1.0);
        double Tc=std::max(Tc_for_H(Sig,r,target_Hr*r_cm),Tc_floor);
        U[4*i+0]=Sig; U[4*i+1]=Vfrom(r,Sig);
        U[4*i+2]=ell_kepler(in.mass,in.spin,r); U[4*i+3]=Tc;
    }
    { const double r0=r_s; const double sqrtD0=std::sqrt(std::max(kerr_delta(in.mass,in.spin,r0),0.0));
      const double Tc0=U[3];
      auto mach=[&](double Sig_)->double{
          const double dn=2.0*std::numbers::pi*Sig_*sqrtD0*in.r_g*c_cgs;
          double V_=-1e-6; if(dn>0.0){const double X=-in.mdot/dn; V_=X/std::sqrt(1.0+X*X);}
          V_=std::clamp(V_,-kVCap,-1e-12);
          const OneZoneState oz=one_zone_closure(Sig_,Tc0,r0,in,op);
          return V_*V_-kGtilde1*(oz.P/Sig_)/(c_cgs*c_cgs);};
      double lo=1e-2,hi=1e12;
      if(mach(lo)>0.0&&mach(hi)<0.0){for(int b=0;b<80;++b){const double mid=std::sqrt(lo*hi); if(mach(mid)>0.0) lo=mid; else hi=mid;}
          const double Sig0=std::sqrt(lo*hi); U[0]=Sig0; U[1]=Vfrom(r0,Sig0);} }
    U[4*N+0]=ell_in; U[4*N+1]=r_s;
    return U;
}

// Solve the f_Edd=0.9 reference solution; returns converged U or empty.
static bool solve_ref_090(const OpacityLUTs& op, double a, int N, double wall_s,
                          std::vector<double>& U, SlimDiskInputs& in_out, SolveOut& so_out) {
    SlimDiskInputs in = make_inputs(a, 0.90, N, wall_s);
    struct V{double Hr,sig,Tcf;};
    const std::vector<V> order={{0.45,8.0,5e6},{0.50,10.0,1e6},{0.40,5.0,1e6},{0.60,20.0,1e7},{0.50,30.0,1e7},{0.30,3.0,1e6}};
    for(const auto& var:order){
        U = build_slim_branch_seed(in,op,var.Hr,var.sig,var.Tcf);
        SolveOut so = run_solve(in,op,U,wall_s,/*require_N1=*/true);
        std::printf("    seed H/r=%.2f sig=%.0f Tcf=%.0e -> conv=%d merit=%.3e wall=%.1fs\n",
                    var.Hr,var.sig,var.Tcf,(int)so.conv,so.merit,so.wall);
        if(so.conv){ in_out=in; so_out=so; return true; }
    }
    return false;
}

static void test2_3_theory(const OpacityLUTs& op, double a, double wall_s) {
    const int N=48;
    std::printf("\n############################################################\n");
    std::printf("#  (2)+(3) THEORY + f_adv SIGN CHECK  a=%.3f  f_Edd=0.90  N=%d\n", a, N);
    std::printf("############################################################\n");
    std::vector<double> U; SlimDiskInputs in; SolveOut so;
    if(!solve_ref_090(op,a,N,wall_s,U,in,so)){
        std::printf("  (no f_Edd=0.9 variant converged; cannot run theory test)\n"); return;
    }
    std::printf("  CONVERGED: merit=%.3e ell_in=%.6f r_s=%.4f r_isco=%.4f\n",
                so.merit, U[4*N+0], U[4*N+1], so.v.r_isco);
    std::vector<NodeTherm> t = therm_of(in,op,U);
    std::printf("  %-3s %-8s %-8s %-10s %-+10s | %-11s %-11s %-+9s | %-8s %-8s\n",
                "i","r[M]","H/r","beta","f_adv","Qadv","Qrad","brkt","dlnP","dlnSig");
    for(int i=0;i<N;++i){
        // §23 bracket sign: Qadv = -(Mdot/2pi r^2)(P/Sig)*BRK, BRK=(G1-1)dlnP - G1 dlnSig
        const double brk = (kGamma1-1.0)*t[i].dlnP - kGamma1*t[i].dlnS;
        std::printf("  %-3d %-8.4f %-8.4f %-10.3e %-+10.2e | %-11.3e %-11.3e %-+9.2e | %-+8.3f %-+8.3f\n",
                    i, t[i].r, t[i].Hr, t[i].beta, t[i].f_adv, t[i].Qadv, t[i].Qrad, brk,
                    t[i].dlnP, t[i].dlnS);
    }
    // Shape summary.
    double Hr_in=t[2].Hr, Hr_out=t[N-2].Hr, b_in=t[2].beta, b_out=t[N-2].beta;
    int nfneg=0,nfpos=0; double fmin=1e300,fmax=-1e300;
    bool Tc_mono=true, Sig_mono=true;
    for(int i=3;i<N-1;++i){
        if(t[i].f_adv<0) ++nfneg; else ++nfpos;
        fmin=std::min(fmin,t[i].f_adv); fmax=std::max(fmax,t[i].f_adv);
        if(t[i].Tc>t[i-1].Tc*1.02) Tc_mono=false;       // expect decreasing outward
        if(t[i].Sigma>t[i-1].Sigma*1.02) Sig_mono=false;
    }
    std::printf("  >> H/r: inner(node2,r=%.2f)=%.3f  outer(node%d,r=%.2f)=%.3f  -> %s\n",
                t[2].r,Hr_in, N-2,t[N-2].r,Hr_out, (Hr_in>Hr_out)?"DECREASES outward (theory OK)":"RISES outward (CONTRADICTS theory)");
    std::printf("  >> beta: inner=%.2e outer=%.2e -> %s\n",
                b_in,b_out, (b_out>b_in*3)?"RISES outward (theory OK)":"stays low / radiation-dominated (CONTRADICTS theory)");
    std::printf("  >> f_adv bulk: %d negative / %d positive nodes; range [%.2e, %.2e]\n", nfneg,nfpos,fmin,fmax);
    std::printf("  >> T_c decreasing outward: %s ; Sigma decreasing outward: %s\n",
                Tc_mono?"yes":"NO (non-monotone)", Sig_mono?"yes":"NO (non-monotone)");
    std::fflush(stdout);
}

// ===========================================================================
// (4) RESOLUTION: f_Edd=0.9 at N=48/96/150; locate the mid-disk glitch (r~15).
// ===========================================================================
static void test4_resolution(const OpacityLUTs& op, double a, double wall_s) {
    std::printf("\n############################################################\n");
    std::printf("#  (4) RESOLUTION TEST  a=%.3f  f_Edd=0.90  N in {48,96,150}\n", a);
    std::printf("############################################################\n");
    for(int N : {48,96,150}){
        std::vector<double> U; SlimDiskInputs in; SolveOut so;
        std::printf("\n-- N=%d --\n", N);
        if(!solve_ref_090(op,a,N,wall_s,U,in,so)){
            std::printf("  N=%d: no variant converged (within wall=%gs). SKIP.\n", N, wall_s); continue;
        }
        std::vector<NodeTherm> t = therm_of(in,op,U);
        // find max H/r and its radius; also the biggest node-to-node Sigma jump.
        int iHmax=0; double Hmax=0; for(int i=0;i<N;++i) if(t[i].Hr>Hmax){Hmax=t[i].Hr;iHmax=i;}
        int iJmax=1; double Jmax=1.0;
        for(int i=1;i<N;++i){const double jmp=std::max(t[i].Sigma/std::max(t[i-1].Sigma,1e-300),
                                                       t[i-1].Sigma/std::max(t[i].Sigma,1e-300));
                             if(jmp>Jmax){Jmax=jmp;iJmax=i;}}
        // f_adv extremes in the mid-disk
        int ifmin=0; double fmn=1e300; for(int i=2;i<N-1;++i) if(t[i].f_adv<fmn){fmn=t[i].f_adv;ifmin=i;}
        std::printf("  N=%d conv merit=%.3e: max H/r=%.3f at r=%.3f (node %d/%d) | "
                    "biggest Sigma jump=%.1fx at r=%.3f | min f_adv=%.2e at r=%.3f\n",
                    N, so.merit, Hmax, t[iHmax].r, iHmax, N, Jmax, t[iJmax].r, fmn, t[ifmin].r);
        // dump the few nodes nearest r=15 to see the glitch shape
        std::printf("    nodes near r=15: r  H/r  f_adv  Sigma  Tc\n");
        for(int i=0;i<N;++i){ if(t[i].r>=11.0 && t[i].r<=20.0)
            std::printf("      r=%-7.3f H/r=%-7.3f f_adv=%-+10.2e Sig=%-10.3e Tc=%-10.3e\n",
                        t[i].r,t[i].Hr,t[i].f_adv,t[i].Sigma,t[i].Tc); }
        std::fflush(stdout);
    }
}

// ===========================================================================
// (5) BETTER SEED: thick-inner / thin-outer.  H/r large only inside ~10 r_g,
//     transitioning to a gas-dominated thin outer disk.  Relax; compare.
// ===========================================================================
// Construction: inner region (r<r_knee) uses the thick slim seed (high Sigma,
// hot, H/r~target_in); outer region uses the NT thin-disk seed (gas-dominated,
// thin).  We BUILD both seeds and splice node-by-node at r_knee, then de-glitch
// V from mass conservation.  This is a physically-motivated alternate seed; the
// SAME relax machinery decides whether it converges to a cleaner root.
static std::vector<double> build_thick_inner_thin_outer_seed(
        const SlimDiskInputs& in, const OpacityLUTs& op, double r_knee, double inner_Hr) {
    using namespace constants;
    const int N = std::max(in.n_nodes, 4);
    // thin seed gives the NT (gas-dominated) outer structure on the SAME grid.
    std::vector<double> Uthin = build_thin_disk_seed(in, op);
    // thick seed gives the slim inner structure.
    std::vector<double> Uthick = build_slim_branch_seed(in, op, inner_Hr, 8.0, 5e6);
    // both seeds share r_s and grid (free-inner-node log grid from r_s); splice.
    const double r_s = Uthin[4*N+1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> U((size_t)4*N+2, 0.0);
    auto Vfrom=[&](double r,double Sig)->double{
        const double sqrtD=std::sqrt(std::max(kerr_delta(in.mass,in.spin,r),0.0));
        const double dn=2.0*std::numbers::pi*Sig*sqrtD*in.r_g*c_cgs;
        double V=-1e-6; if(dn>0.0){const double X=-in.mdot/dn; V=X/std::sqrt(1.0+X*X);}
        if(!(V<0.0)) V=-1e-6; return std::clamp(V,-kVCap,-1e-12);
    };
    for(int i=0;i<N;++i){
        const double t=double(i)/double(N-1);
        const double r=std::exp(lr0+(lr1-lr0)*t);
        // smooth blend over [0.5,1.5]*r_knee to avoid a hard splice glitch.
        double w; // weight on thick (inner) seed
        if (r <= 0.6*r_knee) w=1.0; else if (r >= 1.6*r_knee) w=0.0;
        else w = 0.5*(1.0+std::cos(std::numbers::pi*(r-0.6*r_knee)/(1.0*r_knee)));
        // geometric blend of Sigma, Tc (positive quantities).
        const double Sig = std::pow(std::max(Uthick[4*i+0],1e-30), w)*std::pow(std::max(Uthin[4*i+0],1e-30),1.0-w);
        const double Tc  = std::pow(std::max(Uthick[4*i+3],1.0),  w)*std::pow(std::max(Uthin[4*i+3],1.0), 1.0-w);
        U[4*i+0]=Sig; U[4*i+3]=Tc;
        U[4*i+2]=ell_kepler(in.mass,in.spin,r);
        U[4*i+1]=Vfrom(r,Sig);
    }
    // node-0 sonic override (Mach-1) at fixed Tc0.
    { const double r0=r_s; const double sqrtD0=std::sqrt(std::max(kerr_delta(in.mass,in.spin,r0),0.0));
      const double Tc0=U[3];
      auto mach=[&](double Sig_)->double{
          const double dn=2.0*std::numbers::pi*Sig_*sqrtD0*in.r_g*c_cgs;
          double V_=-1e-6; if(dn>0.0){const double X=-in.mdot/dn; V_=X/std::sqrt(1.0+X*X);}
          V_=std::clamp(V_,-kVCap,-1e-12);
          const OneZoneState oz=one_zone_closure(Sig_,Tc0,r0,in,op);
          return V_*V_-kGtilde1*(oz.P/Sig_)/(c_cgs*c_cgs);};
      double lo=1e-2,hi=1e12;
      if(mach(lo)>0.0&&mach(hi)<0.0){for(int b=0;b<80;++b){const double mid=std::sqrt(lo*hi); if(mach(mid)>0.0) lo=mid; else hi=mid;}
          const double Sig0=std::sqrt(lo*hi); U[0]=Sig0; U[1]=Vfrom(r0,Sig0);} }
    U[4*N+0]=Uthin[4*N+0]; U[4*N+1]=r_s;
    return U;
}

static void summarize(const char* tag, const SlimDiskInputs& in, const OpacityLUTs& op,
                      const std::vector<double>& U){
    const int N=std::max(in.n_nodes,4);
    std::vector<NodeTherm> t=therm_of(in,op,U);
    double Hr_in=t[2].Hr,Hr_out=t[N-2].Hr,b_in=t[2].beta,b_out=t[N-2].beta;
    double maxHr=0; for(int i=0;i<N;++i)maxHr=std::max(maxHr,t[i].Hr);
    std::printf("    [%s] H/r inner=%.3f outer=%.3f max=%.3f | beta inner=%.2e outer=%.2e | "
                "peakSig=%.3e peakTc=%.3e\n", tag,Hr_in,Hr_out,maxHr,b_in,b_out,
                [&]{double m=0;for(auto&x:t)m=std::max(m,x.Sigma);return m;}(),
                [&]{double m=0;for(auto&x:t)m=std::max(m,x.Tc);return m;}());
}

static void test5_better_seed(const OpacityLUTs& op, double a, double wall_s) {
    const int N=48;
    std::printf("\n############################################################\n");
    std::printf("#  (5) BETTER SEED: thick-inner / thin-outer  a=%.3f  f_Edd=0.90  N=%d\n", a, N);
    std::printf("############################################################\n");
    SlimDiskInputs in = make_inputs(a, 0.90, N, wall_s);
    for(double r_knee : {8.0, 10.0, 12.0}){
        for(double inner_Hr : {0.4, 0.5}){
            std::vector<double> U = build_thick_inner_thin_outer_seed(in,op,r_knee,inner_Hr);
            std::printf("\n-- thick-inner/thin-outer seed r_knee=%.0f inner_Hr=%.1f --\n", r_knee, inner_Hr);
            summarize("SEED", in, op, U);
            SolveOut so = run_solve(in,op,U,wall_s,/*require_N1=*/true);
            std::printf("    solve: conv=%d tripped=%d merit=%.3e wall=%.1fs ell_in=%.6f r_s=%.4f\n",
                        (int)so.conv,(int)so.tripped,so.merit,so.wall,U[4*N+0],U[4*N+1]);
            if(so.conv){
                summarize("CONVERGED", in, op, U);
                std::vector<NodeTherm> t=therm_of(in,op,U);
                double Hr_in=t[2].Hr,Hr_out=t[N-2].Hr,b_in=t[2].beta,b_out=t[N-2].beta;
                const bool cleaner = (Hr_in>Hr_out) && (b_out>b_in*3);
                std::printf("    >>> %s: H/r %s outward, beta %s outward\n",
                            cleaner?"CLEANER ROOT (thick-inner/thin-outer!)":"relaxed back to thick-outer-like",
                            (Hr_in>Hr_out)?"DECREASES":"RISES", (b_out>b_in*3)?"RISES":"stays low");
            }
            std::fflush(stdout);
        }
    }
}

int main(int argc, char** argv){
    int test = (argc>1)?std::atoi(argv[1]):0;   // 0 = all
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    const double a = 0.9;
    const double wall = 150.0;     // tight per-solve wall cap

    if(test==0||test==1){ test1_nt_reduction(op,a,0.02,wall); test1_nt_reduction(op,a,0.05,wall); }
    if(test==0||test==2||test==3) test2_3_theory(op,a,wall);
    if(test==0||test==4) test4_resolution(op,a,wall);
    if(test==0||test==5) test5_better_seed(op,a,wall);
    if(test==6) test5_better_seed(op,a,420.0);   // long-budget confirmatory single variant

    std::printf("\n[benchmark] done.\n");
    return 0;
}
