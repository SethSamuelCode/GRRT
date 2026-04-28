#ifndef GRRT_TYPES_H
#define GRRT_TYPES_H

typedef enum {
    GRRT_METRIC_SCHWARZSCHILD = 0,
    GRRT_METRIC_KERR = 1
} GRRTMetricType;

typedef enum {
    GRRT_BACKEND_CPU = 0,
    GRRT_BACKEND_CUDA = 1
} GRRTBackend;

typedef enum {
    GRRT_BG_BLACK = 0,
    GRRT_BG_STARS = 1,
    GRRT_BG_TEXTURE = 2
} GRRTBackgroundType;

typedef struct {
    int width;
    int height;

    GRRTMetricType metric_type;
    double mass;
    double spin;

    double observer_r;
    double observer_theta;
    double observer_phi;
    double fov;

    double cam_yaw;
    double cam_pitch;
    double cam_roll;

    int disk_enabled;
    double disk_inner;
    double disk_outer;
    double disk_temperature;

    int disk_volumetric;        /* 0 = thin disk (default), 1 = volumetric */
    double disk_alpha;          /* Shakura-Sunyaev viscosity (default 0.1) */
    double disk_turbulence;     /* Noise amplitude (default 0.4) */
    int disk_seed;              /* Noise seed (default 42) */
    double disk_noise_scale;    /* Noise feature size in M (0 = auto, default 0) */
    int disk_noise_octaves;     /* fBm octave count (default 2) */

    /* New fields for boundary-smoothing spec — all 0 = use VolumetricParams defaults */
    double disk_noise_compressive_b;
    double disk_noise_correlation_length_factor;
    double disk_outer_taper_width;
    double disk_plunging_h_decay_exponent;
    int    disk_bins_per_h;
    int    disk_bins_per_gradient;
    double disk_target_lut_eps;
    int    disk_min_n_r;
    int    disk_min_n_z;
    int    disk_max_n_r;
    int    disk_max_n_z;
    int    disk_refine_num_frequencies;
    int    disk_force;          /* 1 = skip prompt and proceed on warnings */
    int    disk_strict;         /* 1 = abort on any Promptable/Severe warning */

    double mass_solar;          /* BH mass in solar masses (0 = use disk_temperature directly) */
    double eddington_fraction;  /* Accretion rate as fraction of Eddington (0 = use disk_temperature) */

    GRRTBackgroundType background_type;
    const char* background_texture_path;

    double integrator_tolerance;
    int integrator_max_steps;

    int samples_per_pixel;
    int thread_count;
    GRRTBackend backend;

    int num_frequency_bins;          /* 0 = legacy RGB mode (default) */
    const double* frequency_bins_hz; /* Array of observer-frame frequencies in Hz */
} GRRTParams;

typedef struct GRRTContext GRRTContext;

#endif
