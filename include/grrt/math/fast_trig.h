#ifndef GRRT_FAST_TRIG_H
#define GRRT_FAST_TRIG_H

/// Fast sincos: degree-11 sin / degree-10 cos minimax polynomials.
/// Max error ~1e-12 over all inputs (well within Hamiltonian budget of 1e-10).
/// Uses Cody-Waite extended-precision range reduction to [-pi/4, pi/4],
/// then evaluates both sin and cos of the reduced argument simultaneously.
///
/// Coefficients from Cephes (S. Moshier), public domain.

#include <cmath>
#include <cstdint>

namespace grrt {

#ifdef _MSC_VER
__forceinline
#else
__attribute__((always_inline)) inline
#endif
void fast_sincos(double x, double& s, double& c) {
    // Extended-precision pi/4 for Cody-Waite reduction
    constexpr double DP1 = 7.85398125648498535156e-1;
    constexpr double DP2 = 3.77489470793079817668e-8;
    constexpr double DP3 = 2.69515142907905952645e-15;
    constexpr double FOPI = 1.27323954473516268615;  // 4/pi

    // sin(r) = r * (1 + r^2 * S(r^2)), degree 11
    constexpr double S0 = -1.66666666666666324348e-01;
    constexpr double S1 =  8.33333333332248946124e-03;
    constexpr double S2 = -1.98412698298579493134e-04;
    constexpr double S3 =  2.75573137070700676789e-06;
    constexpr double S4 = -2.50507602534068634195e-08;
    constexpr double S5 =  1.58969099521155010221e-10;

    // cos(r) = 1 - r^2/2 + r^4 * C(r^2), degree 10
    constexpr double C0 =  4.16666666666666019037e-02;
    constexpr double C1 = -1.38888888888741095749e-03;
    constexpr double C2 =  2.48015872894767294178e-05;
    constexpr double C3 = -2.75573143513906633035e-07;
    constexpr double C4 =  2.08757232129817482790e-09;
    constexpr double C5 = -1.13596475577881948265e-11;

    // Take absolute value; track sign separately
    double ax = std::abs(x);
    int sign_s = (x < 0.0) ? -1 : 1;
    int sign_c = 1;

    // Compute octant: j = nearest integer to (|x| * 4/pi)
    int j = static_cast<int>(ax * FOPI);
    // Make j even (map to quadrant)
    j = (j + 1) & ~1;
    double y = static_cast<double>(j);

    // Reduce x to [-pi/4, pi/4] using extended precision
    ax = ((ax - y * DP1) - y * DP2) - y * DP3;

    // Adjust signs based on octant
    int octant = j & 7;
    if (octant > 3) { sign_s = -sign_s; sign_c = -sign_c; octant -= 4; }
    if (octant > 1) { sign_c = -sign_c; }

    // Squared reduced argument
    double z = ax * ax;

    // Evaluate both polynomials (Horner form)
    double ps = ((((S5 * z + S4) * z + S3) * z + S2) * z + S1) * z + S0;
    double pc = (((((C5 * z + C4) * z + C3) * z + C2) * z + C1) * z + C0);

    // sin(r) = r + r^3 * ps,  cos(r) = 1 - z/2 + z^2 * pc
    double sin_r = ax + ax * z * ps;
    double cos_r = 1.0 - 0.5 * z + z * z * pc;

    // Swap sin/cos for odd octants
    if (octant == 1 || octant == 2) {
        s = sign_s * cos_r;
        c = sign_c * sin_r;
    } else {
        s = sign_s * sin_r;
        c = sign_c * cos_r;
    }
}

} // namespace grrt

#endif
