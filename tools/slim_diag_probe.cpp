// ===========================================================================
// TEMPORARY DIAGNOSTIC PROBE  (NOT a fix; safe to delete)
// ---------------------------------------------------------------------------
// Determines whether the inner Newton stall in the slim-disk transonic solver
// is a SEED/BASIN problem or a SOLVER-NOISE (FD-Jacobian) problem.
//
// It #includes slim_disk_radial.cpp and opacity.cpp DIRECTLY so it can reach
// the internal (anonymous-namespace) helpers: relax_structure,
// slim_numerical_jacobian, slim_radial_residual, the merit functions, eval_node,
// calD0, calN1, dense_solve, slim_group_scales.  Built as a standalone exe that
// does NOT link grrt (avoids duplicate-symbol clashes with the DLL copies).
//
// Build (added as target `slim-diag-probe` in CMakeLists.txt, marked TEMPORARY):
//   cmake --build build --config Release --target slim-diag-probe
//   build/Release/slim-diag-probe.exe
// ===========================================================================

// Neutralize the dll import/export decoration for this standalone TU.
#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace grrt;
using namespace grrt::slim_detail;

// We need the internal helpers. They live in anonymous namespaces inside
// slim_disk_radial.cpp, so code compiled as part of THIS translation unit (after
// the #include above) can see them only if it is in the same namespace scope.
// The anonymous-namespace symbols are visible at file scope of this TU. We add
// our probe code in `namespace grrt {}` below to reach the `grrt::` ones, and
// the anonymous-namespace ones are reachable directly by unqualified name.

namespace grrt {
namespace probe {

// Build the SlimDiskInputs for the a=0, low-f_Edd "essentially Novikov-Thorne"
// corner, mirroring the test harness conventions (r_g=1.48e6 ~ 10 Msun).
static SlimDiskInputs make_inputs_a0(double f_Edd, double& Mdot_Edd_out) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0;
    in.spin = 0.0;
    in.alpha = 0.1;
    in.r_g = 1.48e6;
    in.r_out = 50.0;
    in.n_nodes = 150;
    in.max_iters = 100;
    in.tol = 1e-6;
    // r_in: just outside prograde photon orbit. a=0 -> r_ph = 3M. ISCO=6M.
    const double a = 0.0;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    // Mdot_Edd matching the solver's own internal convention.
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    Mdot_Edd_out = Mdot_Edd;
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

// Recompute the active-merit at a state (full residual then reduced norm).
static double active_merit(const std::vector<double>& U, const SlimDiskInputs& in,
                           const OpacityLUTs& op) {
    std::vector<double> R;
    slim_radial_residual(U, in, op, R);
    return slim_scaled_residual_norm_active(U, R, in);
}

// ----------------------------------------------------------------------------
// A(i): does the easiest case converge at all?  Just run the full driver.
// ----------------------------------------------------------------------------
static void task_A_i(const OpacityLUTs& op) {
    std::printf("\n############################################################\n");
    std::printf("# A(i)  full solve_slim_disk_radial at a=0, f_Edd=0.02\n");
    std::printf("############################################################\n");
    double Mdot_Edd = 0;
    SlimDiskInputs in = make_inputs_a0(0.02, Mdot_Edd);
    std::printf("[probe] a=0 mdot=%.4e g/s (f_Edd=%.4f, Mdot_Edd=%.4e) r_in=%.4f r_out=%.1f N=%d\n",
                in.mdot, in.mdot / Mdot_Edd, Mdot_Edd, in.r_in, in.r_out, in.n_nodes);
    SlimDiskRadial s = solve_slim_disk_radial(in, op);
    std::printf("[probe] A(i) RESULT: converged=%d iters=%d final_residual=%.4e r_sonic=%.5f ell_in=%.6f\n",
                s.converged, s.iters, s.final_residual, s.r_sonic, s.ell_in);
}

// ----------------------------------------------------------------------------
// Build a near-TRUE NT seed and the corresponding ell_in / r_s.
// We use the code's OWN build_thin_disk_seed at the low f_Edd corner, which is
// constructed to satisfy the residual's own angular-momentum + energy balances
// at the thin (advection-negligible) limit -- i.e. a near-true NT seed.
// ----------------------------------------------------------------------------
static std::vector<double> near_true_seed(const SlimDiskInputs& in, const OpacityLUTs& op,
                                          double& ell_in_out) {
    std::vector<double> U = build_thin_disk_seed(in, op);
    ell_in_out = U[4 * std::max(in.n_nodes,4) + 0];
    return U;
}

// ----------------------------------------------------------------------------
// A(ii): seed near-true, run relax_structure directly, watch the merit floor.
// (SLIM_DIAG prints the inner trajectory; here we also report start/end merit.)
// ----------------------------------------------------------------------------
static void task_A_ii(const OpacityLUTs& op) {
    std::printf("\n############################################################\n");
    std::printf("# A(ii)  near-TRUE NT seed -> relax_structure (inner Newton)\n");
    std::printf("############################################################\n");
    double Mdot_Edd = 0;
    SlimDiskInputs in = make_inputs_a0(0.02, Mdot_Edd);
    double ell_in = 0;
    std::vector<double> U = near_true_seed(in, op, ell_in);
    const double m0 = active_merit(U, in, op);
    std::printf("[probe] A(ii) near-true seed: ell_in=%.6f r_s=%.5f start_active_merit=%.4e\n",
                ell_in, U[4*std::max(in.n_nodes,4)+1], m0);
    std::vector<double> Uw = U;
    const bool ok = relax_structure(in, op, ell_in, Uw);
    const double m1 = active_merit(Uw, in, op);
    std::printf("[probe] A(ii) RESULT: inner_converged=%d  start_merit=%.4e end_merit=%.4e (floor=1e-6)\n",
                ok, m0, m1);
}

// ----------------------------------------------------------------------------
// A(iii): Newton-step-downhill probe at the near-true point.
// Replicates the inner reduced Newton assembly (active row/col index sets,
// row+col scaling, LM with mu=0) to get the *pure* Newton direction, then sweeps
// lambda and reports the active merit at U + lambda*dU.
// Also: FD-step sensitivity of energy/regularity Jacobian columns at two steps.
// ----------------------------------------------------------------------------

// Build the reduced active Newton step dU (full-length, ell_in col zeroed),
// using the SAME scaling the inner solver uses, with LM damping lm_mu.
static std::vector<double> reduced_newton_step(const std::vector<double>& U,
                                               const SlimDiskInputs& in,
                                               const OpacityLUTs& op,
                                               double lm_mu,
                                               double fd_rel /*0 => use solver default*/) {
    const int N = std::max(in.n_nodes, 4);
    const int n = 4*N + 2;
    const int na = n - 1;
    std::vector<int> var(na), row(na);
    { int p=0; for (int j=0;j<n;++j) if (j!=4*N) var[p++]=j;
      p=0;     for (int j=0;j<n;++j) if (j!=4*N+1) row[p++]=j; }

    std::vector<double> R, J;
    slim_radial_residual(U, in, op, R);
    // Jacobian: either the solver default or a custom FD relative step.
    if (fd_rel <= 0.0) {
        slim_numerical_jacobian(U, in, op, J);
    } else {
        // custom-step FD jacobian (same per-variable floors as solver)
        J.assign((size_t)n*n, 0.0);
        double sSig=0,sV=0,sEll=0,sT=0;
        for (int i=0;i<N;++i){ sSig=std::max(sSig,std::abs(U[4*i+0])); sV=std::max(sV,std::abs(U[4*i+1]));
                               sEll=std::max(sEll,std::abs(U[4*i+2])); sT=std::max(sT,std::abs(U[4*i+3])); }
        const double fSig=1e-7*std::max(sSig,1e-30), fV=1e-7*std::max(sV,1e-30);
        const double fEll=1e-7*std::max(sEll,1e-30), fT=1e-7*std::max(sT,1e-30);
        const double fLin=1e-7*std::max(std::abs(U[4*N+0]),1e-30), fRs=1e-7*std::max(std::abs(U[4*N+1]),1e-30);
        std::vector<double> Up,Um,Rp,Rm;
        for (int j=0;j<n;++j){
            double af; if(j<4*N){ switch(j&3){case 0:af=fSig;break;case 1:af=fV;break;case 2:af=fEll;break;default:af=fT;} }
                       else af=(j==4*N)?fLin:fRs;
            const double delta=std::max(fd_rel*std::abs(U[j]),af);
            Up=U; Um=U; Up[j]+=delta; Um[j]-=delta;
            slim_radial_residual(Up,in,op,Rp); slim_radial_residual(Um,in,op,Rm);
            for (int r=0;r<n;++r) J[(size_t)r*n+j]=(Rp[r]-Rm[r])/(2.0*delta);
        }
    }

    // Row/col scaling identical to relax_structure.
    std::vector<double> cs(n), rs_inv(n);
    {
        double mSig=0,mV=0,mEll=0,mT=0;
        for (int i=0;i<N;++i){ mSig=std::max(mSig,std::abs(U[4*i+0])); mV=std::max(mV,std::abs(U[4*i+1]));
                               mEll=std::max(mEll,std::abs(U[4*i+2])); mT=std::max(mT,std::abs(U[4*i+3])); }
        mSig=std::max(mSig,1e-30); mV=std::max(mV,1e-30); mEll=std::max(mEll,1e-30); mT=std::max(mT,1.0);
        for (int i=0;i<N;++i){ cs[4*i+0]=mSig; cs[4*i+1]=mV; cs[4*i+2]=mEll; cs[4*i+3]=mT; }
        cs[4*N+0]=std::max(std::abs(U[4*N+0]),1e-30); cs[4*N+1]=std::max(std::abs(U[4*N+1]),1e-30);
        const GroupScales gs = slim_group_scales(U, in);
        auto sr=[&](int b,int e,double sc){ sc=std::max(sc,1e-300); for(int r=b;r<e;++r) rs_inv[r]=1.0/sc; };
        sr(0,N,gs.mass); sr(N,2*N,gs.ang); sr(2*N,3*N-1,gs.rad); sr(3*N-1,4*N-2,gs.ene);
        sr(4*N-2,4*N-1,gs.bc_ell); sr(4*N-1,4*N,gs.bc_T); sr(4*N,4*N+1,gs.reg_D0); sr(4*N+1,4*N+2,gs.reg_N1);
    }
    std::vector<double> Js((size_t)na*na,0.0), Rs(na,0.0);
    for (int a=0;a<na;++a){ const int ra=row[a]; Rs[a]=R[ra]*rs_inv[ra];
        for (int b=0;b<na;++b){ const int vb=var[b]; Js[(size_t)a*na+b]=J[(size_t)ra*n+vb]*rs_inv[ra]*cs[vb]; } }
    std::vector<double> JtJ((size_t)na*na,0.0), Jtr(na,0.0);
    for (int i=0;i<na;++i) for (int k=0;k<na;++k){ const double jik=Js[(size_t)k*na+i]; if(jik==0.0) continue;
        Jtr[i]+=jik*Rs[k]; for(int j=0;j<na;++j) JtJ[(size_t)i*na+j]+=jik*Js[(size_t)k*na+j]; }
    std::vector<double> A=JtJ, b(na);
    for (int i=0;i<na;++i) A[(size_t)i*na+i]+=lm_mu*std::max(JtJ[(size_t)i*na+i],1e-300);
    for (int i=0;i<na;++i) b[i]=-Jtr[i];
    std::vector<double> dU(n,0.0);
    if (!dense_solve(A,b,na)) { std::printf("[probe]   (newton solve singular at lm_mu=%.1e)\n", lm_mu); return dU; }
    for (int bb=0;bb<na;++bb) dU[var[bb]]=b[bb]*cs[var[bb]];
    dU[4*N+0]=0.0;
    return dU;
}

static void task_A_iii(const OpacityLUTs& op) {
    std::printf("\n############################################################\n");
    std::printf("# A(iii)  Newton-step-downhill probe at the near-true point\n");
    std::printf("############################################################\n");
    double Mdot_Edd = 0;
    SlimDiskInputs in = make_inputs_a0(0.02, Mdot_Edd);
    const int N = std::max(in.n_nodes, 4);
    double ell_in = 0;
    std::vector<double> U = near_true_seed(in, op, ell_in);
    U[4*N+0] = ell_in;  // pin

    const double m0 = active_merit(U, in, op);
    std::printf("[probe] A(iii) at NEAR-TRUE seed merit=%.4e ell_in=%.6f r_s=%.5f\n", m0, ell_in, U[4*N+1]);

    // ---- lambda sweep on the pure Newton direction (lm_mu near 0). ----
    for (double lm_mu : {0.0, 1e-3}) {
        std::vector<double> dU = reduced_newton_step(U, in, op, lm_mu, /*fd_rel*/0.0);
        std::printf("[probe] --- lambda sweep, lm_mu=%.1e (Newton direction) ---\n", lm_mu);
        bool any_downhill = false;
        for (double lam : {1.0, 0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6}) {
            std::vector<double> Ut = U;
            bool physical = true;
            for (int i=0;i<(int)Ut.size();++i) Ut[i] += lam*dU[i];
            for (int i=0;i<N && physical;++i){ if (Ut[4*i+0]<=0.0||Ut[4*i+3]<=0.0||std::abs(Ut[4*i+1])>=1.0) physical=false; }
            const double rs = Ut[4*N+1];
            if (!(rs>in.r_in && rs<in.r_out)) physical=false;
            double m = physical ? active_merit(Ut, in, op) : NAN;
            const char* tag = (physical && m < m0) ? "  <-- DOWNHILL" : (physical?"":"  (non-physical)");
            std::printf("[probe]   lambda=%.0e  merit=%.6e  (dmerit=%+.3e)%s\n",
                        lam, m, physical?(m-m0):NAN, tag);
            if (physical && m < m0) any_downhill = true;
        }
        std::printf("[probe]   => ANY lambda downhill? %s\n", any_downhill ? "YES" : "NO (step not downhill)");
    }

    // ---- FD-step sensitivity of energy/regularity Jacobian columns. ----
    // Build two FD jacobians (rel step 1e-6 and 1e-4) and compare a few columns of
    // the energy rows (3N-1 .. 4N-3) and regularity rows (4N, 4N+1).
    std::printf("[probe] --- FD-step sensitivity of energy/reg Jacobian rows ---\n");
    const int n = 4*N+2;
    auto build_J = [&](double fd_rel){
        std::vector<double> J((size_t)n*n,0.0);
        double sSig=0,sV=0,sEll=0,sT=0;
        for (int i=0;i<N;++i){ sSig=std::max(sSig,std::abs(U[4*i+0])); sV=std::max(sV,std::abs(U[4*i+1]));
                               sEll=std::max(sEll,std::abs(U[4*i+2])); sT=std::max(sT,std::abs(U[4*i+3])); }
        const double fSig=1e-7*std::max(sSig,1e-30), fV=1e-7*std::max(sV,1e-30);
        const double fEll=1e-7*std::max(sEll,1e-30), fT=1e-7*std::max(sT,1e-30);
        const double fLin=1e-7*std::max(std::abs(U[4*N+0]),1e-30), fRs=1e-7*std::max(std::abs(U[4*N+1]),1e-30);
        std::vector<double> Up,Um,Rp,Rm;
        for (int j=0;j<n;++j){
            double af; if(j<4*N){ switch(j&3){case 0:af=fSig;break;case 1:af=fV;break;case 2:af=fEll;break;default:af=fT;} }
                       else af=(j==4*N)?fLin:fRs;
            const double delta=std::max(fd_rel*std::abs(U[j]),af);
            Up=U; Um=U; Up[j]+=delta; Um[j]-=delta;
            slim_radial_residual(Up,in,op,Rp); slim_radial_residual(Um,in,op,Rm);
            for (int r=0;r<n;++r) J[(size_t)r*n+j]=(Rp[r]-Rm[r])/(2.0*delta);
        }
        return J;
    };
    std::vector<double> Ja = build_J(1e-6);
    std::vector<double> Jb = build_J(1e-4);
    // Compare: for a handful of representative rows, find max relative difference
    // across columns and a couple of sample entries.
    auto cmp_row = [&](const char* label, int rrow){
        double maxreldiff=0; int worstcol=-1; double va=0, vb=0;
        for (int j=0;j<n;++j){
            const double a=Ja[(size_t)rrow*n+j], b=Jb[(size_t)rrow*n+j];
            const double scale=std::max(std::abs(a),std::abs(b));
            if (scale < 1e-300) continue;
            const double rd=std::abs(a-b)/scale;
            if (rd>maxreldiff){ maxreldiff=rd; worstcol=j; va=a; vb=b; }
        }
        std::printf("[probe]   %s row=%d: max FD-step reldiff=%.3e at col=%d (J@1e-6=%.4e, J@1e-4=%.4e)\n",
                    label, rrow, maxreldiff, worstcol, va, vb);
    };
    cmp_row("energy(first)", 3*N-1);
    cmp_row("energy(mid)  ", 3*N-1 + (N-1)/2);
    cmp_row("reg_D0       ", 4*N+0);
    cmp_row("reg_N1       ", 4*N+1);
    // Also a mass row (well-conditioned algebraic) as a control.
    cmp_row("mass(ctrl)   ", 0);
    cmp_row("ang(ctrl)    ", N);
}

// Helper: at a given state, sweep lm_mu, build the damped step, do a fine
// 1-D line minimization along it, and report the best one-step merit reduction.
static void best_step_at(const char* tag, const std::vector<double>& U,
                         const SlimDiskInputs& in, const OpacityLUTs& op) {
    const int N = std::max(in.n_nodes, 4);
    const double m0 = active_merit(U, in, op);
    std::printf("[probe] %s merit=%.6e  best one-step reduction over lm_mu x line-min:\n", tag, m0);
    double global_best = m0; double best_mu = 0, best_lam = 0;
    for (double lm_mu : {1e-12, 1e-9, 1e-6, 1e-3, 1e-1, 1e0, 1e1, 1e2, 1e3}) {
        std::vector<double> dU = reduced_newton_step(U, in, op, lm_mu, 0.0);
        double best_m = m0, best_l = 0;
        for (double lam = 1.0; lam >= 1e-8; lam *= 0.5) {
            std::vector<double> Ut = U; bool physical = true;
            for (int i=0;i<(int)Ut.size();++i) Ut[i] += lam*dU[i];
            for (int i=0;i<N && physical;++i){ if (Ut[4*i+0]<=0.0||Ut[4*i+3]<=0.0||std::abs(Ut[4*i+1])>=1.0) physical=false; }
            const double rs=Ut[4*N+1]; if(!(rs>in.r_in&&rs<in.r_out)) physical=false;
            if (!physical) continue;
            const double m = active_merit(Ut, in, op);
            if (m < best_m) { best_m = m; best_l = lam; }
        }
        std::printf("[probe]    lm_mu=%.0e: best merit=%.6e (dmerit=%+.3e) at lambda=%.0e\n",
                    lm_mu, best_m, best_m-m0, best_l);
        if (best_m < global_best) { global_best = best_m; best_mu = lm_mu; best_lam = best_l; }
    }
    std::printf("[probe]    => GLOBAL best one-step: merit=%.6e (factor %.3f of start) at lm_mu=%.0e lambda=%.0e\n",
                global_best, global_best/std::max(m0,1e-300), best_mu, best_lam);
}

// ----------------------------------------------------------------------------
// A(iv): characterize the STALL point.  Run relax to the stall, then ask: is
// there ANY one-step (over the full lm_mu x line-search grid the solver itself
// explores) that materially reduces the merit?  If the best achievable one-step
// reduction is ~0, the stall is a genuine local min of the merit landscape (the
// solver can't do better with its own machinery).  We also report which groups
// dominate, and re-run with a big max_iters to see if it is a slow crawl.
// ----------------------------------------------------------------------------
static void task_A_iv(const OpacityLUTs& op) {
    std::printf("\n############################################################\n");
    std::printf("# A(iv)  stall-point characterization (a=0, f_Edd=0.02)\n");
    std::printf("############################################################\n");
    double Mdot_Edd = 0;
    SlimDiskInputs in = make_inputs_a0(0.02, Mdot_Edd);
    double ell_in = 0;
    std::vector<double> U = near_true_seed(in, op, ell_in);
    std::vector<double> Ustall = U;
    relax_structure(in, op, ell_in, Ustall);   // leaves U at the stall point
    best_step_at("A(iv) at-stall:", Ustall, in, op);

    // Is it a slow crawl?  Re-run relax from the same seed with a huge max_iters.
    SlimDiskInputs in_big = in; in_big.max_iters = 2000;
    std::vector<double> Ubig = U;
    const bool ok_big = relax_structure(in_big, op, ell_in, Ubig);
    const double m_big = active_merit(Ubig, in_big, op);
    std::printf("[probe] A(iv) relax with max_iters=2000: converged=%d end_merit=%.6e\n", ok_big, m_big);
}

// ----------------------------------------------------------------------------
// A(v): the task's referenced HARD case a=0.998, f_Edd~0.3, to confirm the
// ene/reg-dominated stall signature (the easy a=0 case stalls rad/mass-dominated;
// the controller's report described ene/reg).  Just run the full driver under
// SLIM_DIAG so the [INNER] lines show the dominant stuck groups.
// ----------------------------------------------------------------------------
static void task_A_v(const OpacityLUTs& op) {
    std::printf("\n############################################################\n");
    std::printf("# A(v)  hard case a=0.998 f_Edd~0.3 (confirm ene/reg stall)\n");
    std::printf("############################################################\n");
    using namespace constants;
    SlimDiskInputs in{};
    in.mass=1.0; in.spin=0.998; in.alpha=0.1; in.r_g=1.48e6;
    const double a=0.998;
    const double r_ph = 2.0*(1.0+std::cos((2.0/3.0)*std::acos(-a)));
    in.r_in=r_ph+0.02; in.r_out=50.0; in.n_nodes=150; in.max_iters=100; in.tol=1e-6;
    in.mdot=4.0e17;
    SlimDiskRadial s = solve_slim_disk_radial(in, op);
    std::printf("[probe] A(v) RESULT: converged=%d iters=%d final_residual=%.4e\n",
                s.converged, s.iters, s.final_residual);
}

} // namespace probe
} // namespace grrt

int main() {
    using namespace grrt;
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    grrt::probe::task_A_i(op);
    grrt::probe::task_A_ii(op);
    grrt::probe::task_A_iii(op);
    grrt::probe::task_A_iv(op);
    grrt::probe::task_A_v(op);
    std::printf("\n[probe] done.\n");
    return 0;
}
