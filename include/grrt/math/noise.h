#ifndef GRRT_NOISE_H
#define GRRT_NOISE_H

#include "grrt_export.h"
#include <array>
#include <cstdint>

namespace grrt {

/// 3D Simplex noise generator.
///
/// Implements Ken Perlin's simplex noise algorithm in three dimensions.
/// The simplex noise function subdivides space into simplices (tetrahedra in 3D)
/// rather than the hypercubes used by classic Perlin noise, giving O(n^2)
/// complexity instead of O(2^n) and fewer directional artifacts.
///
/// Output range: approximately [-1, 1].
/// The permutation table is seeded with a user-supplied 32-bit integer so
/// different seeds produce statistically independent noise fields — useful
/// for layering independent turbulence contributions in the accretion disk
/// density model.
class GRRT_EXPORT SimplexNoise3D {
public:
    /// Construct a noise generator seeded with \p seed.
    /// @param seed 32-bit seed; different values yield independent noise fields.
    explicit SimplexNoise3D(uint32_t seed = 42);

    /// Evaluate 3D simplex noise at position (x, y, z).
    /// @return Value in approximately [-1, 1].
    double evaluate(double x, double y, double z) const;

    /// Evaluate turbulent (multi-octave) noise at (x, y, z).
    /// Combines the base frequency with a second octave at 3× frequency
    /// and half amplitude, giving a rougher, more cloud-like appearance.
    /// @return Value in approximately [-1.5, 1.5].
    double evaluate_turbulent(double x, double y, double z) const;

    /// Read-only access to the internal permutation table (512 entries).
    const std::array<int, 512>& permutation_table() const { return perm_; }

private:
    std::array<int, 512> perm_;
};

} // namespace grrt

#endif // GRRT_NOISE_H
