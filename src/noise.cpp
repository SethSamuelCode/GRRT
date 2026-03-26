#include "grrt/math/noise.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace grrt {

// Gradient vectors for 3D simplex noise.
// These 12 vectors point to the midpoints of the 12 edges of a cube,
// ensuring an even distribution of gradient directions with no preferred axis.
static constexpr double grad3[12][3] = {
    { 1, 1, 0}, {-1, 1, 0}, { 1,-1, 0}, {-1,-1, 0},
    { 1, 0, 1}, {-1, 0, 1}, { 1, 0,-1}, {-1, 0,-1},
    { 0, 1, 1}, { 0,-1, 1}, { 0, 1,-1}, { 0,-1,-1}
};

/// Dot product of a gradient vector with an offset vector.
static double dot3(const double g[3], double x, double y, double z) {
    return g[0]*x + g[1]*y + g[2]*z;
}

SimplexNoise3D::SimplexNoise3D(uint32_t seed) {
    // Fill [0..255] then Fisher-Yates shuffle with the given seed.
    std::array<int, 256> base;
    std::iota(base.begin(), base.end(), 0);
    std::mt19937 rng(seed);
    for (int i = 255; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(base[i], base[dist(rng)]);
    }
    // Duplicate into the 512-entry table so index wrapping never needs masking.
    for (int i = 0; i < 256; i++) {
        perm_[i] = perm_[i + 256] = base[i];
    }
}

double SimplexNoise3D::evaluate(double x, double y, double z) const {
    // Skewing and unskewing factors for 3D simplex noise.
    // F3: skew factor — maps the simplex grid to a cubic grid.
    // G3: unskew factor — maps back.
    constexpr double F3 = 1.0 / 3.0;
    constexpr double G3 = 1.0 / 6.0;

    // Skew the input space to determine which simplex cell we are in.
    double s = (x + y + z) * F3;
    int i = static_cast<int>(std::floor(x + s));
    int j = static_cast<int>(std::floor(y + s));
    int k = static_cast<int>(std::floor(z + s));

    // Unskew the cell origin back to (x,y,z) space.
    double t = (i + j + k) * G3;
    double X0 = i - t, Y0 = j - t, Z0 = k - t;
    // Offset from cell origin to the evaluation point.
    double x0 = x - X0, y0 = y - Y0, z0 = z - Z0;

    // For 3D, the simplex shape is a tetrahedron.
    // Determine which simplex we are in by ranking (x0, y0, z0).
    int i1, j1, k1; // second corner offsets (middle vertex 1)
    int i2, j2, k2; // third corner offsets  (middle vertex 2)
    if (x0 >= y0) {
        if (y0 >= z0)      { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else               { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0)       { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0)  { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else               { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    // Offsets for the remaining three corners in (x,y,z) coordinates.
    double x1 = x0 - i1 + G3,     y1 = y0 - j1 + G3,     z1 = z0 - k1 + G3;
    double x2 = x0 - i2 + 2*G3,   y2 = y0 - j2 + 2*G3,   z2 = z0 - k2 + 2*G3;
    double x3 = x0 - 1  + 3*G3,   y3 = y0 - 1  + 3*G3,   z3 = z0 - 1  + 3*G3;

    // Hash the integer coordinates for gradient lookup.
    int ii = i & 255, jj = j & 255, kk = k & 255;

    // Contribution from one simplex corner.
    // The radial falloff kernel (0.6 - r^2)^4 ensures C1 continuity and
    // naturally truncates to zero at the boundary of each simplex's influence.
    auto contrib = [&](double cx, double cy, double cz, int gi) -> double {
        double t0 = 0.6 - cx*cx - cy*cy - cz*cz;
        if (t0 < 0.0) return 0.0;
        t0 *= t0;
        return t0 * t0 * dot3(grad3[gi % 12], cx, cy, cz);
    };

    double n0 = contrib(x0, y0, z0, perm_[ii       + perm_[jj       + perm_[kk      ]]]);
    double n1 = contrib(x1, y1, z1, perm_[ii+i1    + perm_[jj+j1    + perm_[kk+k1   ]]]);
    double n2 = contrib(x2, y2, z2, perm_[ii+i2    + perm_[jj+j2    + perm_[kk+k2   ]]]);
    double n3 = contrib(x3, y3, z3, perm_[ii+1     + perm_[jj+1     + perm_[kk+1    ]]]);

    // Scale to approximately [-1, 1].
    return 32.0 * (n0 + n1 + n2 + n3);
}

double SimplexNoise3D::evaluate_turbulent(double x, double y, double z) const {
    // Two-octave turbulence: base frequency + 3× harmonic at half amplitude.
    // Mimics the turbulent density fluctuations expected in a magnetised
    // accretion disk at scales well below the disk scale height.
    return evaluate(x, y, z) + 0.5 * evaluate(x * 3.0, y * 3.0, z * 3.0);
}

double SimplexNoise3D::evaluate_fbm(double x, double y, double z, int octaves) const {
    double result = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    for (int i = 0; i < octaves; ++i) {
        result += amplitude * evaluate(x * frequency, y * frequency, z * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return result;
}

} // namespace grrt
