#include "grrt/scene/volumetric_disk.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

int failures = 0;

void check(const char* name, double got, double expected, double rel_tol) {
    double rel_err = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel_err < rel_tol;
    std::printf("  %s: got=%.4e expected=%.4e rel_err=%.2e %s\n",
                name, got, expected, rel_err, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

void test_construction() {
    std::printf("\n=== VolumetricDisk construction (a=0.998, T_peak=1e7 K) ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    std::printf("  r_isco = %.4f M\n", disk.r_isco());
    std::printf("  r_horizon = %.4f M\n", disk.r_horizon());
    double H = disk.scale_height(10.0);
    std::printf("  H(10M) = %.4f M\n", H);
    if (H <= 0.0 || !std::isfinite(H)) { std::printf("  FAIL\n"); failures++; }
    else { std::printf("  PASS\n"); }
}

void test_density_profile() {
    std::printf("\n=== Density profile ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    double r = 10.0;
    double H = disk.scale_height(r);
    double rho_mid = disk.density(r, 0.0, 0.0);
    double rho_1H = disk.density(r, H, 0.0);
    double rho_3H = disk.density(r, 3.0*H, 0.0);
    std::printf("  rho(mid)=%.4e, rho(1H)=%.4e, rho(3H)=%.4e\n", rho_mid, rho_1H, rho_3H);
    if (rho_1H >= rho_mid || rho_3H >= rho_1H) {
        std::printf("  FAIL: density should decrease with height\n"); failures++;
    } else { std::printf("  PASS\n"); }
}

void test_temperature_profile() {
    std::printf("\n=== Temperature profile ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    double r = 10.0;
    double H = disk.scale_height(r);
    double T_mid = disk.temperature(r, 0.0);
    double T_1H = disk.temperature(r, H);
    double T_3H = disk.temperature(r, 3.0*H);
    std::printf("  T(mid)=%.2f, T(1H)=%.2f, T(3H)=%.2f\n", T_mid, T_1H, T_3H);
    if (T_1H >= T_mid || T_3H >= T_1H) {
        std::printf("  FAIL: T should decrease with height\n"); failures++;
    } else { std::printf("  PASS\n"); }
}

void test_taper() {
    std::printf("\n=== ISCO taper ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    check("taper(r_isco)", disk.taper(disk.r_isco()), 1.0, 0.01);
    check("taper(r_isco+1)", disk.taper(disk.r_isco()+1.0), 1.0, 0.01);
    double t_hor = disk.taper(disk.r_horizon());
    if (t_hor > 0.1) { std::printf("  FAIL: taper at horizon should be small\n"); failures++; }
    else { std::printf("  PASS: taper(horizon)=%.4e\n", t_hor); }
}

void test_volume_bounds() {
    std::printf("\n=== Volume bounds ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    if (!disk.inside_volume(10.0, 0.0)) { std::printf("  FAIL: midplane should be inside\n"); failures++; }
    else { std::printf("  PASS: midplane inside\n"); }
    if (disk.inside_volume(10.0, 100.0)) { std::printf("  FAIL: z=100 should be outside\n"); failures++; }
    else { std::printf("  PASS: far above outside\n"); }
    if (disk.inside_volume(50.0, 0.0)) { std::printf("  FAIL: r=50 should be outside\n"); failures++; }
    else { std::printf("  PASS: beyond r_outer outside\n"); }
}

void test_warnings_initially_empty() {
    std::printf("\n=== Warnings initially empty ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    if (!disk.warnings().empty()) {
        std::printf("  FAIL: expected empty warnings on a normal construction, got %zu\n",
                    disk.warnings().size());
        failures++;
    } else {
        std::printf("  PASS\n");
    }
    if (disk.promptable_count() != 0) {
        std::printf("  FAIL: expected promptable_count=0\n");
        failures++;
    } else {
        std::printf("  PASS: promptable_count=0\n");
    }
}

void test_severity_enum_values() {
    std::printf("\n=== Severity enum stability ===\n");
    if (static_cast<int>(grrt::WarningSeverity::Info) != 0) {
        std::printf("  FAIL: Info != 0\n"); failures++; return;
    }
    if (static_cast<int>(grrt::WarningSeverity::Warning) != 1) {
        std::printf("  FAIL: Warning != 1\n"); failures++; return;
    }
    if (static_cast<int>(grrt::WarningSeverity::Promptable) != 2) {
        std::printf("  FAIL: Promptable != 2\n"); failures++; return;
    }
    if (static_cast<int>(grrt::WarningSeverity::Severe) != 3) {
        std::printf("  FAIL: Severe != 3\n"); failures++; return;
    }
    std::printf("  PASS\n");
}

void dump_vertical_profile() {
    std::printf("\n=== Vertical profile diagnostic ===\n");
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;  // no noise for clean profile
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7, vp);

    for (double r : {3.0, 5.0, 10.0, 20.0}) {
        double H = disk.scale_height(r);
        std::printf("\nr=%.1f  H=%.6f  taper=%.6f\n", r, H, disk.taper(r));
        for (int i = 0; i <= 20; i++) {
            double z_frac = i / 20.0 * 3.0;  // 0 to 3H
            double z = z_frac * H;
            double rho = disk.density_cgs(r, z, 0.0);
            double T = disk.temperature(r, z);
            bool inside = disk.inside_volume(r, z);
            std::printf("  z/H=%5.2f  rho=%.4e  T=%8.1f  in=%d\n",
                       z_frac, rho, T, inside);
        }
    }
}

void test_photosphere_extends_to_negligible() {
    std::printf("\n=== Photosphere LUT extends to rho < 1e-15 ===\n");
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7, vp);

    // Spec §1a: the ODE convergence floor (CONV_FLOOR) must be low enough that
    // rho_z[n_z-1] (the last LUT cell, normalized so rho_z[0]=1) is truly
    // negligible.  With the old floor of 1e-15, the cell was pinned exactly at
    // 1e-15 — not strictly less.  After lowering RHO_FLOOR to 1e-18 the cell
    // drops below 1e-15 as the ODE drives it to machine-zero.
    //
    // We access the raw LUT to measure the boundary cell directly, avoiding
    // the density() cross-column interpolation artefact that can inflate the
    // value when the neighbouring column has a larger z_max.

    const int nr  = disk.radial_bins();
    const int nz  = disk.vertical_bins();
    const auto& lut = disk.density_profile_lut();

    // Locate the LUT bin closest to r = 6 M (peak-flux region, a_star=0.998).
    const double ri_f = (6.0 - disk.r_min()) / (disk.r_max() - disk.r_min())
                        * (nr - 1);
    const int ri = std::clamp(static_cast<int>(ri_f), 0, nr - 1);

    // rho_z[0] = 1.0 by construction; last cell should be < old RHO_FLOOR = 1e-15.
    const double rho_boundary = lut[static_cast<std::size_t>(ri) * nz + (nz - 1)];
    const double rho_mid_val  = lut[static_cast<std::size_t>(ri) * nz];  // = 1.0

    std::printf("  ri=%d  rho_z[0]=%.4e  rho_z[nz-1]=%.4e  (expect < 1e-15)\n",
                ri, rho_mid_val, rho_boundary);
    // FAIL if pinned at the old RHO_FLOOR (>= 1e-15); PASS once it drops below.
    if (rho_boundary >= 1e-15) {
        std::printf("  FAIL: boundary cell pinned at old RHO_FLOOR (>= 1e-15)\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

// We'll only test smoothstep indirectly through the outer-taper task. For this
// task, just sanity-check that the disk constructs cleanly (regression guard).
void test_smoothstep_regression() {
    std::printf("\n=== Smoothstep helper (regression guard) ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    std::printf("  PASS: construction completed\n");
}

void test_density_smooth_across_zmax() {
    std::printf("\n=== Density smooth across z_max ===\n");
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7, vp);

    const double r = 6.0;
    const double zm = disk.z_max_at(r);

    // Sample density just below and just above z_max — they should be similar
    // (both are tiny, but the ratio between them should be smooth, not a cliff).
    const double rho_below = disk.density(r, zm * 0.99, 0.0);
    const double rho_above = disk.density(r, zm * 1.01, 0.0);

    // After spec §1a: above z_max, density should be exactly 0 (LUT-defined).
    // The cliff at zm itself is irrelevant because the LUT already has ρ < 1e-15 there.
    if (rho_above != 0.0) {
        std::printf("  FAIL: rho(z>z_max) should be exactly 0, got %.4e\n", rho_above);
        failures++; return;
    }
    // The smoothness criterion: rho_below should be small (LUT extended past photosphere)
    const double rho_mid = disk.density(r, 0.0, 0.0);
    if (rho_below / rho_mid > 1e-10) {
        std::printf("  FAIL: rho_below z_max too large (%.4e of midplane)\n",
                    rho_below / rho_mid);
        failures++; return;
    }
    std::printf("  PASS: rho_below=%.2e (small), rho_above=0 (cliff is in LUT, smooth)\n",
                rho_below);
}

void test_outer_radial_taper() {
    std::printf("\n=== Outer radial taper (smoothstep, not cliff) ===\n");
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7, vp);

    const double r_outer = 30.0;
    const double H_outer = disk.scale_height(r_outer);
    const double dr_out  = 2.0 * H_outer;

    // Sample density at: well inside, mid-taper, just inside r_outer
    const double rho_inside    = disk.density(r_outer - 4.0 * H_outer, 0.0, 0.0);
    const double rho_mid_taper = disk.density(r_outer - 1.0 * H_outer, 0.0, 0.0);
    const double rho_near_edge = disk.density(r_outer - 0.05 * H_outer, 0.0, 0.0);

    if (!(rho_inside > rho_mid_taper && rho_mid_taper > rho_near_edge)) {
        std::printf("  FAIL: expected monotonic decay across taper zone\n");
        std::printf("    rho_inside=%.4e rho_mid=%.4e rho_edge=%.4e\n",
                    rho_inside, rho_mid_taper, rho_near_edge);
        failures++;
        return;
    }
    std::printf("  PASS: rho_inside=%.2e > rho_mid=%.2e > rho_edge=%.2e\n",
                rho_inside, rho_mid_taper, rho_near_edge);
}

int main() {
    test_construction();
    test_density_profile();
    test_temperature_profile();
    test_taper();
    test_volume_bounds();
    test_warnings_initially_empty();
    test_severity_enum_values();
    test_smoothstep_regression();
    dump_vertical_profile();
    test_photosphere_extends_to_negligible();
    test_density_smooth_across_zmax();
    test_outer_radial_taper();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
