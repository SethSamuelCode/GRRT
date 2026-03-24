#include "grrt/scene/volumetric_disk.h"
#include <cstdio>
#include <cmath>

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

int main() {
    test_construction();
    test_density_profile();
    test_temperature_profile();
    test_taper();
    test_volume_bounds();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
