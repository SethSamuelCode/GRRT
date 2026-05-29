// tools/dump_disk_lut.cpp
//
// Standalone diagnostic for the banding investigation: construct the volumetric
// disk with the banding-scene structural parameters and dump its radial +
// vertical-profile LUTs to CSV, so density "holes" (silently floored columns)
// can be read straight off a file.
//
// Build target: dump-disk-lut.  Usage: dump-disk-lut [out.csv]
//
// Columns:
//   ri            radial bin index (0 .. n_r-1)
//   r             Boyer-Lindquist radius [M]
//   H             scale height H(r) [M]
//   z_max         vertical envelope z_max(r) [M]
//   rho_mid_norm  normalized midplane density (rho_mid_lut_[ri])
//   rho_mid_cgs   midplane density in CGS (= rho_mid_norm * rho_scale)
//   prof_mid      vertical profile at z=0           (== 1.0 by construction)
//   prof_0p1zmax  vertical profile at z = 0.1*z_max (~near midplane)
//   prof_0p5zmax  vertical profile at z = 0.5*z_max
//   prof_top      vertical profile at the top bin   (z ~ z_max)
//
// A radial density hole shows as an anomalously tiny rho_mid_cgs and/or a
// vertical profile that has collapsed to ~RHO_FLOOR (1e-18) away from the
// midplane while neighbouring radii are normal.

#include "grrt/scene/volumetric_disk.h"

#include <cstdio>

int main(int argc, char** argv) {
    using namespace grrt;

    // Banding-scene structural params. Defaults already match: alpha=0.1,
    // seed=42, tau_mid=100. turbulence/seed do not affect the base LUTs.
    VolumetricParams vp;
    VolumetricDisk disk(1.0 /*M*/, 0.998 /*a*/, 20.0 /*r_outer*/, 1e7 /*T_peak*/, vp);

    const int    n_r       = disk.radial_bins();
    const int    n_z       = disk.vertical_bins();
    const double r_min     = disk.r_min();
    const double r_max     = disk.r_max();
    const double rho_scale = disk.rho_scale();
    const auto&  H_lut     = disk.scale_height_lut();
    const auto&  zmax_lut  = disk.z_max_lut();
    const auto&  rho_mid   = disk.rho_mid_lut();
    const auto&  prof      = disk.density_profile_lut();  // n_r * n_z, row-major

    const char* path = (argc > 1) ? argv[1] : "disk_lut_dump.csv";
    std::FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); return 1; }

    std::fprintf(f, "ri,r,H,z_max,rho_mid_norm,rho_mid_cgs,"
                    "prof_mid,prof_0p1zmax,prof_0p5zmax,prof_top\n");
    for (int ri = 0; ri < n_r; ++ri) {
        const double r        = r_min + (r_max - r_min) * ri / (n_r - 1);
        const double rmid_cgs = rho_mid[ri] * rho_scale;
        const double pm = prof[static_cast<size_t>(ri) * n_z + 0];
        const double p1 = prof[static_cast<size_t>(ri) * n_z + n_z / 10];
        const double p5 = prof[static_cast<size_t>(ri) * n_z + n_z / 2];
        const double pt = prof[static_cast<size_t>(ri) * n_z + (n_z - 1)];
        std::fprintf(f,
            "%d,%.5f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
            ri, r, H_lut[ri], zmax_lut[ri], rho_mid[ri], rmid_cgs, pm, p1, p5, pt);
    }
    std::fclose(f);

    std::printf("Wrote %d radial bins x %d vertical to %s (rho_scale=%.4e)\n",
                n_r, n_z, path, rho_scale);
    return 0;
}
