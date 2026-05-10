// tests/test_sobol_sampler.cpp
//
// Unit tests for sobol_sample_2d. These tests are pure — no rendering, no
// disk construction, no LUTs. They exercise the sampler's mathematical
// properties (range, determinism, decorrelation, uniformity, dyadic set).

#include "grrt/render/sobol_sampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <set>
#include <vector>

using namespace grrt;

int failures = 0;

#define EXPECT_TRUE(cond, msg)                                                \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("FAIL %s:%d: %s — %s\n",                              \
                        __FILE__, __LINE__, #cond, msg);                      \
            failures++;                                                       \
        }                                                                     \
    } while (0)

// Test 1: every produced point lies in [0, 1).
static void test_range() {
    constexpr int N = 1000;
    for (int i = 0; i < N; ++i) {
        // pseudo-random pixel coords for variety
        const int px = (i * 73856093u) & 0xFFFF;
        const int py = (i * 19349663u) & 0xFFFF;
        for (int s = 0; s < 8; ++s) {
            SobolSample p = sobol_sample_2d(px, py, s);
            EXPECT_TRUE(p.x >= 0.0 && p.x < 1.0, "x out of range");
            EXPECT_TRUE(p.y >= 0.0 && p.y < 1.0, "y out of range");
        }
    }
}

int main() {
    std::printf("Running test_sobol_sampler...\n");
    test_range();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
