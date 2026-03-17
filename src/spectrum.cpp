#include "grrt/color/spectrum.h"
#include <cmath>
#include <algorithm>

namespace grrt {

// Physical constants
static constexpr double h_planck = 6.62607015e-34;  // J·s
static constexpr double c_light = 2.99792458e8;     // m/s
static constexpr double k_boltz = 1.380649e-23;     // J/K
static constexpr double sigma_sb = 5.670374419e-8;  // W/(m²·K⁴)

// CIE 1931 2° observer color matching functions
// 81 entries from 380nm to 780nm in 5nm steps
// Each row: {x_bar, y_bar, z_bar}
static constexpr double cie_data[][3] = {
    {0.0014,0.0000,0.0065}, {0.0022,0.0001,0.0105}, {0.0042,0.0001,0.0201},
    {0.0076,0.0002,0.0362}, {0.0143,0.0004,0.0679}, {0.0232,0.0006,0.1102},
    {0.0435,0.0012,0.2074}, {0.0776,0.0022,0.3713}, {0.1344,0.0040,0.6456},
    {0.2148,0.0073,1.0391}, {0.2839,0.0116,1.3856}, {0.3285,0.0168,1.6230},
    {0.3483,0.0230,1.7471}, {0.3481,0.0298,1.7826}, {0.3362,0.0380,1.7721},
    {0.3187,0.0480,1.7441}, {0.2908,0.0600,1.6692}, {0.2511,0.0739,1.5281},
    {0.1954,0.0910,1.2876}, {0.1421,0.1126,1.0419}, {0.0956,0.1390,0.8130},
    {0.0580,0.1693,0.6162}, {0.0320,0.2080,0.4652}, {0.0147,0.2586,0.3533},
    {0.0049,0.3230,0.2720}, {0.0024,0.4073,0.2123}, {0.0093,0.5030,0.1582},
    {0.0291,0.6082,0.1117}, {0.0633,0.7100,0.0782}, {0.1096,0.7932,0.0573},
    {0.1655,0.8620,0.0422}, {0.2257,0.9149,0.0298}, {0.2904,0.9540,0.0203},
    {0.3597,0.9803,0.0134}, {0.4334,0.9950,0.0087}, {0.5121,1.0002,0.0057},
    {0.5945,0.9950,0.0039}, {0.6784,0.9786,0.0027}, {0.7621,0.9520,0.0021},
    {0.8425,0.9154,0.0018}, {0.9163,0.8700,0.0017}, {0.9786,0.8163,0.0014},
    {1.0263,0.7570,0.0011}, {1.0567,0.6949,0.0010}, {1.0622,0.6310,0.0008},
    {1.0456,0.5668,0.0006}, {1.0026,0.5030,0.0003}, {0.9384,0.4412,0.0002},
    {0.8544,0.3810,0.0002}, {0.7514,0.3210,0.0001}, {0.6424,0.2650,0.0000},
    {0.5419,0.2170,0.0000}, {0.4479,0.1750,0.0000}, {0.3608,0.1382,0.0000},
    {0.2835,0.1070,0.0000}, {0.2187,0.0816,0.0000}, {0.1649,0.0610,0.0000},
    {0.1212,0.0446,0.0000}, {0.0874,0.0320,0.0000}, {0.0636,0.0232,0.0000},
    {0.0468,0.0170,0.0000}, {0.0329,0.0119,0.0000}, {0.0227,0.0082,0.0000},
    {0.0158,0.0057,0.0000}, {0.0114,0.0041,0.0000}, {0.0081,0.0029,0.0000},
    {0.0058,0.0021,0.0000}, {0.0041,0.0015,0.0000}, {0.0029,0.0010,0.0000},
    {0.0020,0.0007,0.0000}, {0.0014,0.0005,0.0000}, {0.0010,0.0004,0.0000},
    {0.0007,0.0002,0.0000}, {0.0005,0.0002,0.0000}, {0.0003,0.0001,0.0000},
    {0.0002,0.0001,0.0000}, {0.0002,0.0001,0.0000}, {0.0001,0.0000,0.0000},
    {0.0001,0.0000,0.0000}, {0.0001,0.0000,0.0000}, {0.0000,0.0000,0.0000},
};
static constexpr int cie_count = 81;
static constexpr double cie_lambda_min = 380e-9;
static constexpr double cie_lambda_step = 5e-9;

double SpectrumLUT::planck(double lambda, double T) {
    double x = h_planck * c_light / (lambda * k_boltz * T);
    if (x > 500.0) return 0.0;
    return (2.0 * h_planck * c_light * c_light) /
           (lambda * lambda * lambda * lambda * lambda * (std::exp(x) - 1.0));
}

Vec3 SpectrumLUT::blackbody_to_rgb(double T) const {
    double X = 0.0, Y = 0.0, Z = 0.0;
    for (int i = 0; i < cie_count; ++i) {
        double lambda = cie_lambda_min + i * cie_lambda_step;
        double B = planck(lambda, T);
        X += B * cie_data[i][0] * cie_lambda_step;
        Y += B * cie_data[i][1] * cie_lambda_step;
        Z += B * cie_data[i][2] * cie_lambda_step;
    }

    // XYZ → linear sRGB (IEC 61966-2-1)
    double R =  3.2406 * X - 1.5372 * Y - 0.4986 * Z;
    double G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
    double B_val =  0.0557 * X - 0.2040 * Y + 1.0570 * Z;

    R = std::max(R, 0.0);
    G = std::max(G, 0.0);
    B_val = std::max(B_val, 0.0);

    return {{R, G, B_val}};
}

SpectrumLUT::SpectrumLUT(double t_min, double t_max, int num_entries)
    : t_min_(t_min), t_max_(t_max), num_entries_(num_entries),
      color_lut_(num_entries), luminosity_lut_(num_entries) {

    double lum_max = sigma_sb * t_max * t_max * t_max * t_max;

    for (int i = 0; i < num_entries; ++i) {
        double t = t_min + (t_max - t_min) * i / (num_entries - 1);

        Vec3 rgb = blackbody_to_rgb(t);
        double max_c = rgb.max_component();
        if (max_c > 0.0) {
            rgb = rgb * (1.0 / max_c);
        }
        color_lut_[i] = rgb;

        luminosity_lut_[i] = (sigma_sb * t * t * t * t) / lum_max;
    }
}

Vec3 SpectrumLUT::temperature_to_color(double temperature) const {
    temperature = std::clamp(temperature, t_min_, t_max_);

    double frac = (temperature - t_min_) / (t_max_ - t_min_) * (num_entries_ - 1);
    int idx = static_cast<int>(frac);
    double t = frac - idx;

    if (idx >= num_entries_ - 1) {
        return color_lut_[num_entries_ - 1] * luminosity_lut_[num_entries_ - 1];
    }

    Vec3 color = color_lut_[idx] * (1.0 - t) + color_lut_[idx + 1] * t;
    double lum = luminosity_lut_[idx] * (1.0 - t) + luminosity_lut_[idx + 1] * t;

    return color * lum;
}

} // namespace grrt
