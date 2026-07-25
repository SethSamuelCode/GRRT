// ===========================================================================
// SLIM FULL-18 FEASIBILITY @ n_z=256  (DIAGNOSTIC — DELETABLE)
// ---------------------------------------------------------------------------
// "Run the whole thing at n_z=256." For ALL 18 radial nodes at the f_Edd=0.001,
// a=0.9 base rung, does each column solve at its DEMANDED Σ (thin/NT seed Σ, with
// T_c on the f_adv≈0 manifold) at high VERTICAL resolution n_z=256?  Counts how
// many of 18 are genuinely feasible once the column is well-resolved — the
// complete base-rung picture (node 9 turned out to be a coarse-measurement
// artifact; this checks whether 4/10/others are too, and isolates the true walls).
//
// Build:  cmake --build build --config Release --target slim-full256-probe
// Run:    build/Release/slim-full256-probe.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include "../src/slim_disk_radial.cpp"
#include "../src/slim_disk_coupled.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <numbers>
#include <algorithm>
#include <chrono>
#include <io.h>
#include <fcntl.h>

using namespace grrt;
using namespace grrt::slim_coupled_detail;

static int g_saved=-1;
static void mute(){ std::fflush(stdout); g_saved=_dup(_fileno(stdout)); int n=_open("NUL",_O_WRONLY); _dup2(n,_fileno(stdout)); _close(n); }
static void unmute(){ std::fflush(stdout); if(g_saved>=0){_dup2(g_saved,_fileno(stdout)); _close(g_saved); g_saved=-1;} }

static double mdot_from_fEdd(const SlimDiskInputs& in, double f) {
    using namespace constants;
    const double M=in.mass*in.r_g*c_cgs*c_cgs/G_cgs;
    const double L=4.0*std::numbers::pi*G_cgs*M*c_cgs/0.34;
    return f*10.0*L/(c_cgs*c_cgs);
}

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);
    const auto t0=std::chrono::steady_clock::now();
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    const int Nreq = (argc>1)?std::atoi(argv[1]):18;   // radial rings
    const int NZ   = (argc>2)?std::atoi(argv[2]):256;  // vertical points

    SlimDiskInputs in{};
    in.mass=1.0; in.spin=0.9; in.alpha=0.1; in.r_g=1.48e6; in.r_out=50.0; in.n_nodes=Nreq; in.tol=1e-8;
    in.r_in=0.5*slim_detail::isco_prograde(in.mass,in.spin);
    in.mdot=mdot_from_fEdd(in,1e-3);

    std::vector<double> U = build_thin_disk_seed(in, op);
    const int N=std::max(in.n_nodes,4);
    const double r_s=U[4*N+1];
    const double lr0=std::log(r_s), lr1=std::log(in.r_out);

    std::printf("# slim-full256-probe : all 18 nodes, feasibility at DEMANDED Σ, n_z=%d\n", NZ);
    std::printf("#   a=%.3f f_Edd=0.001  r_s=%.4f\n", in.spin, r_s);
    std::printf("%-4s %-9s %-12s %-12s %-9s\n","i","r[M]","Sigma_dem","Tc_manif","solves?");

    int nfeas=0;
    for (int i=0;i<N;++i){
        const double t=double(i)/double(N-1);
        const double ri=std::exp(lr0+(lr1-lr0)*t);
        const int j=(i+1<N)?i+1:i-1;
        const double tj=double(j)/double(N-1);
        const double rj=std::exp(lr0+(lr1-lr0)*tj);
        const double Omi=slim_detail::omega_from_ell(in.mass,in.spin,ri,U[4*i+2]);
        const double Omj=slim_detail::omega_from_ell(in.mass,in.spin,rj,U[4*j+2]);
        const double shear=std::max(shear_cgs(in,ri,Omi,rj,Omj),1e-300);
        const double omz=std::max(omega_perp_cgs(in,ri),1e-300);
        const double Sig=std::max(U[4*i+0],1e2);
        const double Tc_thin=std::max(U[4*i+3],1.0);
        const double rho0=std::max(slim_detail::one_zone_closure(Sig,Tc_thin,ri,in,op).rho_mid,1e-30);

        // Manifold T_c at (Σ, geometry), then feasibility solve at demanded Σ, n_z=256.
        mute();
        ColumnCoupledInputs cm{}; cm.Sigma_target=Sig; cm.Tc=Tc_thin; cm.shear=shear; cm.omega_z=omz;
        cm.alpha=in.alpha; cm.rho_mid_guess=rho0; cm.n_nodes=NZ; cm.max_iters=300; cm.tol=1e-8; cm.Teff_guess=0.0;
        std::vector<double> Uc; double Tc_manif=Tc_thin;
        if (build_coupled_seed(cm, op, Uc) || build_coupled_seed_advective(cm, op, Uc))
            Tc_manif=std::max(Uc[2],1.0);

        ColumnCoupledInputs ci{}; ci.Sigma_target=Sig; ci.Tc=Tc_manif; ci.shear=shear; ci.omega_z=omz;
        ci.alpha=in.alpha; ci.rho_mid_guess=rho0; ci.n_nodes=NZ; ci.max_iters=300; ci.tol=1e-8; ci.Teff_guess=0.0;
        const bool ok = solve_column_coupled(ci, op, nullptr).converged;
        unmute();
        if (ok) ++nfeas;
        std::printf("%-4d %-9.4f %-12.4e %-12.4e %-9s\n", i, ri, Sig, Tc_manif, ok?"YES":"no");
    }
    std::printf("\n-> feasible at n_z=%d : %d / %d\n", NZ, nfeas, N);
    std::printf("wall %.1f s\nDONE\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());
    return 0;
}
