// src/sobol_sampler.cpp
//
// Owen-scrambled Sobol 2D sub-pixel sampler.
//
// Direction numbers:
//   - Dimension 0: trivial Sobol (van der Corput in base 2),
//                  V[0][k] = 1 << (31 - k).
//   - Dimension 1: Sobol with primitive polynomial x + 1.
//                  V[1][0] = 1 << 31; V[1][k] = V[1][k-1] ^ (V[1][k-1] >> 1).
//                  Equivalently: V[1][k] = m_{k+1} << (31 - k), where
//                  m_1 = 1, m_k = (2 * m_{k-1}) ^ m_{k-1}.
//
// Owen scramble: Burley 2020, "Practical Hash-Based Owen Scrambling," JCGT 9(4).
// Same form used by pbrt-v4's nested_uniform_scramble.
//
// References:
//   - Joe & Kuo, "Constructing Sobol' sequences with better two-dimensional
//     projections," SIAM J. Sci. Comput. 30(5), 2008.
//   - Burley, "Practical Hash-Based Owen Scrambling," JCGT 9(4), 2020.

#include "grrt/render/sobol_sampler.h"

#include <array>
#include <cstdint>

namespace grrt {

namespace {

// Direction numbers for the first two Sobol dimensions, computed at compile time.
struct DirectionTables {
    std::array<uint32_t, 32> v0{};
    std::array<uint32_t, 32> v1{};
};

constexpr DirectionTables make_direction_tables() {
    DirectionTables d{};
    for (int k = 0; k < 32; ++k) {
        d.v0[k] = static_cast<uint32_t>(1u) << (31 - k);
    }
    d.v1[0] = static_cast<uint32_t>(1u) << 31;
    for (int k = 1; k < 32; ++k) {
        d.v1[k] = d.v1[k - 1] ^ (d.v1[k - 1] >> 1);
    }
    return d;
}

constexpr DirectionTables kDirs = make_direction_tables();

// Generate the i-th 32-bit Sobol point in dimension `dim` (0 or 1) by
// gray-code traversal of the direction-number table. Caller passes a
// non-negative `i`; the unsigned conversion makes shifts well-defined
// regardless. The `k < 32` guard prevents a future signed/wider param
// type change from accidentally OOB-reading the 32-element direction table.
uint32_t sobol_point_1d(int i, int dim) {
    const uint32_t* V = (dim == 0) ? kDirs.v0.data() : kDirs.v1.data();
    uint32_t result = 0;
    uint32_t idx = static_cast<uint32_t>(i) ^ (static_cast<uint32_t>(i) >> 1);  // gray code
    int k = 0;
    while (idx && k < 32) {
        if (idx & 1u) result ^= V[k];
        idx >>= 1;
        ++k;
    }
    return result;
}

// Burley 2020, Listing 3. Hash-based Owen scramble. Permutes a 32-bit
// integer using a per-pixel seed, breaking correlations between adjacent
// pixels' sample sequences.
uint32_t burley_scramble(uint32_t x, uint32_t seed) {
    x = x ^ (x * 0x3d20adeau);
    x += seed;
    x *= (seed >> 16) | 1u;
    x = x ^ (x * 0x05526c56u);
    x = x ^ (x * 0x53a22864u);
    return x;
}

// Per-pixel scramble seed. Same splitmix-style 32-bit finalizer as
// pixel_hash in renderer.cpp. Different `dim` arguments give different
// seeds so x and y dimensions are independently scrambled.
uint32_t scramble_seed(int pixel_x, int pixel_y, int dim) {
    uint32_t h = static_cast<uint32_t>(pixel_x) * 73856093u
               ^ static_cast<uint32_t>(pixel_y) * 19349663u
               ^ static_cast<uint32_t>(dim)     * 83492791u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

} // anonymous namespace

SobolSample sobol_sample_2d(int pixel_x, int pixel_y, int sample_index) {
    const uint32_t seed_x = scramble_seed(pixel_x, pixel_y, 0);
    const uint32_t seed_y = scramble_seed(pixel_x, pixel_y, 1);

    const uint32_t sx_raw = sobol_point_1d(sample_index, 0);
    const uint32_t sy_raw = sobol_point_1d(sample_index, 1);

    const uint32_t sx = burley_scramble(sx_raw, seed_x);
    const uint32_t sy = burley_scramble(sy_raw, seed_y);

    constexpr double inv_2_32 = 1.0 / 4294967296.0;  // 2^-32
    return { static_cast<double>(sx) * inv_2_32,
             static_cast<double>(sy) * inv_2_32 };
}

} // namespace grrt
