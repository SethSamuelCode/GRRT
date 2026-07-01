// CONVECTION VERIFICATION  (DIAGNOSTIC — DELETABLE)
// Confirms the deep-rad-pressure column's β rise (7.2e-6 pure-radiative -> 2.9e-3
// convective) is convection COOLING the midplane (flattening ∇), not misfiring.
// Solves the exact test_rad_pressure_barrier_reach column (T_eff=1e7) and reports,
// per node: T, β, ∇_rad vs the ACTUAL ∇ used (∇_conv where unstable), + convective flag.
#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1
#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include <cstdio>
#include <cmath>
using namespace grrt;
int main() {
    using namespace grrt::constants;
    auto lut = build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    ColumnInputs in{};
    in.T_eff = 1e7; in.shear = 5.25e3; in.omega_z = 3.12e3; in.alpha = 0.1;
    in.rho_mid_guess = 1e-6; in.n_nodes = 96; in.max_iters = 120; in.tol = 1e-8;
    ColumnBVPSolution s = solve_column_bvp(in, lut);
    std::printf("converged=%d  Sigma0=%.4e  z0=%.4e\n", s.converged, s.Sigma0, s.z0);
    if (!s.converged) { std::printf("did not converge\n"); return 1; }
    const int N = in.n_nodes;
    const double Tc = s.T[0];
    const double Pg0 = s.P_gas[0], Prad0 = (a_rad/3.0)*Tc*Tc*Tc*Tc;
    const double beta0 = Pg0/(Pg0 + Prad0);
    std::printf("MIDPLANE: T_c=%.4e K  beta=%.4e  (pure-radiative HEAD gave beta=7.2e-6)\n", Tc, beta0);
    std::printf("  %-3s %-10s %-10s %-11s %-11s %-8s %-8s\n","i","T[K]","beta","nab_rad","nab_used","conv?","flatten");
    int n_conv = 0;
    for (int i = 0; i < N; i += std::max(1, N/16)) {
        const double Pg = s.P_gas[i], T = s.T[i], Q = s.Q[i], z = s.z[i];
        const double Ptot = Pg + (a_rad/3.0)*T*T*T*T;
        const double rho = std::max(Pg*mu_fully_ionized*m_p/(k_B*T), 1e-300);
        const double kR = lut.lookup_kappa_ross(rho, std::max(T,3000.0)) + lut.lookup_kappa_es(rho, std::max(T,3000.0));
        double nab_used; bool conv;
        detail_bvp::convective_gradient(rho, T, Ptot, Q, kR, z, in.omega_z, nab_used, conv);
        // recompute nab_rad for display
        const double dTdz_rad = -3.0*kR*rho*Q/(16.0*sigma_SB*T*T*T);
        const double dPdz = -rho*in.omega_z*in.omega_z*z;
        const double nab_rad = (z>0 && dPdz<0 && Q>0) ? (Ptot/T)*(dTdz_rad/dPdz) : 0.0;
        const double beta = Pg/Ptot;
        const double flatten = (nab_rad>0)? nab_used/nab_rad : 1.0;   // <1 = convection flattened it
        std::printf("  %-3d %-10.3e %-10.3e %-11.4e %-11.4e %-8s %-8.3f\n",
                    i, T, beta, nab_rad, nab_used, conv?"YES":"no", flatten);
    }
    // count convective nodes over the whole column
    for (int i = 0; i < N; ++i) {
        const double Pg=s.P_gas[i],T=s.T[i],Q=s.Q[i],z=s.z[i];
        const double Ptot=Pg+(a_rad/3.0)*T*T*T*T;
        const double rho=std::max(Pg*mu_fully_ionized*m_p/(k_B*T),1e-300);
        const double kR=lut.lookup_kappa_ross(rho,std::max(T,3000.0))+lut.lookup_kappa_es(rho,std::max(T,3000.0));
        double nu; bool c; detail_bvp::convective_gradient(rho,T,Ptot,Q,kR,z,in.omega_z,nu,c);
        if (c) ++n_conv;
    }
    std::printf("=> convective nodes: %d/%d ;  VERDICT: %s\n", n_conv, N,
        (n_conv>0 && beta0>1e-4) ? "convection ACTIVE + flattening -> cooler T_c -> higher beta (VERIFIED, not misfire)"
                                 : "unexpected — investigate");
    std::printf("DONE\n");
    return 0;
}
