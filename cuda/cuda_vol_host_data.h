#ifndef CUDA_VOL_HOST_DATA_H
#define CUDA_VOL_HOST_DATA_H

/// @file cuda_vol_host_data.h
/// @brief Plain-C struct holding extracted VolumetricDisk data for CUDA upload.
///
/// This exists because nvcc (in C++17 mode) cannot parse atomic_data.h's
/// designated initializers. The extraction is done in a regular .cpp file
/// compiled by MSVC, then passed to cuda_backend.cu as a plain struct.

#include <vector>

struct VolDiskHostData {
    // Scalar parameters
    double r_isco;
    double r_horizon;
    double taper_width;
    double E_isco;
    double L_isco;
    double rho_scale;
    double turbulence;
    double noise_scale;
    double r_min;
    double r_max;
    int n_r;
    int n_z;

    // Opacity grid parameters
    int opacity_n_nu, opacity_n_rho, opacity_n_T;
    double opacity_log_nu_min, opacity_log_nu_max;
    double opacity_log_rho_min, opacity_log_rho_max;
    double opacity_log_T_min, opacity_log_T_max;

    // LUT data (copied from VolumetricDisk)
    std::vector<double> H_lut;
    std::vector<double> rho_mid_lut;
    std::vector<double> rho_profile_lut;
    std::vector<double> T_profile_lut;
    std::vector<double> kappa_abs_lut;
    std::vector<double> kappa_es_lut;
    std::vector<double> kappa_ross_lut;
    std::vector<double> mu_lut;

    // Noise permutation table (512 ints)
    std::vector<int> perm_table;
};

/// @brief Build a VolumetricDisk on the CPU and extract all data needed for CUDA upload.
///
/// This function is compiled by MSVC (not nvcc), so it can safely include
/// headers that use C++20 features like designated initializers.
VolDiskHostData build_vol_disk_host_data(double mass, double spin,
                                          double r_outer, double peak_temperature,
                                          double alpha, double turbulence,
                                          unsigned int seed);

#endif // CUDA_VOL_HOST_DATA_H
