// TEMPORARY slim-disk radial verification probe (spin-walk continuation).
// Configurable via argv so one binary covers the verification matrix:
//   slim-full-target-probe [spin] [f_Edd] [n_nodes]
// Defaults: spin=0.998, f_Edd=0.9, n_nodes=48 (small enough that the dense-FD
// inner Jacobian is tractable; the full-resolution N=150 is intractable until the
// analytic Jacobian lands).  Reports converged, r_sonic/r_isco, f_adv range,
// H/r range, V<0 / Sigma>0, and final_residual.  Run under SLIM_DIAG to watch
// the [SPINWALK] / [OUTER] / [INNER] ladders.  Safe to delete; registered in
// CMakeLists as slim-full-target-probe.
#include "grrt/scene/slim_disk_radial.h"
#include "grrt/color/opacity.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

static double isco_prograde_a(double a) {
    const double Z1 = 1.0 + std::cbrt(1.0 - a*a) * (std::cbrt(1.0 + a) + std::cbrt(1.0 - a));
    const double Z2 = std::sqrt(3.0*a*a + Z1*Z1);
    return 3.0 + Z2 - std::sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0*Z2));
}

int main(int argc, char** argv) {
    double spin   = (argc > 1) ? std::atof(argv[1]) : 0.998;
    double f_Edd  = (argc > 2) ? std::atof(argv[2]) : 0.9;
    int    nnodes = (argc > 3) ? std::atoi(argv[3]) : 48;
    double wall_s = (argc > 4) ? std::atof(argv[4]) : 14.0 * 60.0;  // wall-clock budget [s]

    grrt::SlimDiskInputs in{};
    in.mass = 1.0;
    in.spin = spin;
    in.alpha = 0.1;
    in.r_g = 1.48e6;                 // ~10 M_sun
    const double a = in.spin;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    in.r_out = 50.0;
    in.n_nodes = nnodes;
    in.max_iters = 100;              // public knob; internal continuation uses its own budget
    in.tol = 1e-6;
    // Belt-and-suspenders external guard (the solver's internal default is the same
    // ~15 min); keep it explicit so a stalled probe self-aborts honestly.
    in.budget_wall_seconds = wall_s;

    // f_Edd in the solver's internal Mdot_Edd = 10 L_Edd/c^2 convention.
    using namespace grrt::constants;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * 3.14159265358979323846 * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;
    std::printf("=== PROBE spin=%.4f f_Edd=%.3f n_nodes=%d ===\n", spin, f_Edd, nnodes);
    std::printf("Mdot_Edd=%.4e  in.mdot(%.3f f_Edd)=%.4e\n", Mdot_Edd, f_Edd, in.mdot);

    auto lut = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    auto s = grrt::solve_slim_disk_radial(in, lut);

    const double r_isco = isco_prograde_a(a);
    std::printf("converged=%d iters=%d final_residual=%.4e\n",
                s.converged, s.iters, s.final_residual);
    std::printf("r_sonic=%.5f  r_isco=%.5f  sonic<isco=%d\n",
                s.r_sonic, r_isco, (int)(s.r_sonic < r_isco));
    if (!s.converged) { std::printf("NOT CONVERGED\n"); return 1; }

    double fadv_lo=1e300, fadv_hi=-1e300, hr_lo=1e300, hr_hi=-1e300;
    bool all_inflow=true, all_pos=true;
    for (size_t i = 0; i < s.r.size(); ++i) {
        fadv_lo = std::min(fadv_lo, s.f_adv[i]);
        fadv_hi = std::max(fadv_hi, s.f_adv[i]);
        const double hr = s.H[i] / (s.r[i] * in.r_g);   // H[cm] / (r[M]*r_g[cm/M])
        hr_lo = std::min(hr_lo, hr);
        hr_hi = std::max(hr_hi, hr);
        if (!(s.V[i] < 0.0)) all_inflow = false;
        if (!(s.Sigma[i] > 0.0)) all_pos = false;
    }
    std::printf("V<0 everywhere=%d  Sigma>0 everywhere=%d\n", all_inflow, all_pos);
    std::printf("f_adv in [%.4e, %.4e]\n", fadv_lo, fadv_hi);
    std::printf("H/r   in [%.4e, %.4e]\n", hr_lo, hr_hi);
    std::printf("f_adv[inner=0]=%.4e  f_adv[outer=last]=%.4e\n",
                s.f_adv.front(), s.f_adv.back());
    std::printf("H/r[inner=0]=%.4e  H/r[outer=last]=%.4e\n",
                s.H.front()/(s.r.front()*in.r_g), s.H.back()/(s.r.back()*in.r_g));
    return 0;
}
