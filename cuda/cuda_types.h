#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

#include "cuda_math.h"
#include "cuda_metric.h"

namespace cuda {

constexpr int MAX_SPECTRUM_ENTRIES = 1000;
constexpr int MAX_FLUX_LUT_ENTRIES = 500;
constexpr int MAX_STARS = 5000;

struct Star {
    double theta;
    double phi;
    double brightness;
};

struct RenderParams {
    // Image dimensions
    int width;
    int height;

    // Metric
    MetricType metric_type;
    double mass;
    double spin;  // Dimensional spin a = spin_param * mass

    // Observer
    double observer_r;
    double observer_theta;
    double observer_phi;

    // Camera tetrad (precomputed on host)
    Vec4 cam_position;
    Vec4 cam_e0;  // timelike
    Vec4 cam_e1;  // right
    Vec4 cam_e2;  // up
    Vec4 cam_e3;  // forward
    double fov;

    // Accretion disk
    int disk_enabled;
    double disk_r_inner;
    double disk_r_outer;
    double disk_peak_temperature;
    double disk_flux_max;
    double disk_flux_r_min;
    double disk_flux_r_max;
    int disk_flux_lut_size;

    // Spectrum LUT params
    double spectrum_t_min;
    double spectrum_t_max;
    int spectrum_num_entries;

    // Celestial sphere
    int background_type;  // 0=black, 1=stars
    int num_stars;
    double star_angular_tolerance;

    // Integrator
    double integrator_tolerance;
    int integrator_max_steps;
    double r_escape;
    double horizon_epsilon;
};

} // namespace cuda

#endif
