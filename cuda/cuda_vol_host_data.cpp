/// @file cuda_vol_host_data.cpp
/// @brief Host-only extraction of VolumetricDisk data for CUDA upload.
///
/// Compiled by MSVC (not nvcc), so C++20 features in transitive includes
/// (e.g. designated initializers in atomic_data.h) are handled correctly.

#include "cuda_vol_host_data.h"
#include "grrt/scene/volumetric_disk.h"

VolDiskHostData build_vol_disk_host_data(double mass, double spin,
                                          double r_outer, double peak_temperature,
                                          double alpha, double turbulence,
                                          unsigned int seed) {
    grrt::VolumetricParams vp;
    vp.alpha = alpha;
    vp.turbulence = turbulence;
    vp.seed = seed;

    grrt::VolumetricDisk disk(mass, spin, r_outer, peak_temperature, vp);

    VolDiskHostData data{};
    data.r_isco      = disk.r_isco();
    data.r_horizon   = disk.r_horizon();
    data.taper_width = disk.taper_width();
    data.E_isco      = disk.E_isco();
    data.L_isco      = disk.L_isco();
    data.rho_scale   = disk.rho_scale();
    data.turbulence  = turbulence;
    data.r_min       = disk.r_min();
    data.r_max       = disk.r_max();
    data.n_r         = disk.radial_bins();
    data.n_z         = disk.vertical_bins();

    const auto& olut = disk.opacity_luts();
    data.opacity_n_nu       = olut.n_nu;
    data.opacity_n_rho      = olut.n_rho;
    data.opacity_n_T        = olut.n_T;
    data.opacity_log_nu_min  = olut.log_nu_min;
    data.opacity_log_nu_max  = olut.log_nu_max;
    data.opacity_log_rho_min = olut.log_rho_min;
    data.opacity_log_rho_max = olut.log_rho_max;
    data.opacity_log_T_min   = olut.log_T_min;
    data.opacity_log_T_max   = olut.log_T_max;

    data.H_lut           = disk.scale_height_lut();
    data.rho_mid_lut     = disk.rho_mid_lut();
    data.rho_profile_lut = disk.density_profile_lut();
    data.T_profile_lut   = disk.temperature_profile_lut();
    data.kappa_abs_lut   = olut.kappa_abs_lut;
    data.kappa_es_lut    = olut.kappa_es_lut;
    data.kappa_ross_lut  = olut.kappa_ross_lut;
    data.mu_lut          = olut.mu_lut;

    const auto& perm = disk.noise().permutation_table();
    data.perm_table.assign(perm.begin(), perm.end());

    return data;
}
