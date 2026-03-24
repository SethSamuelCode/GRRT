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

    GRRTBackgroundType background_type;
    const char* background_texture_path;

    double integrator_tolerance;
    int integrator_max_steps;

    int samples_per_pixel;
    int thread_count;
    GRRTBackend backend;
} GRRTParams;

typedef struct GRRTContext GRRTContext;

#endif
