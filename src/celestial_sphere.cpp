#include "grrt/scene/celestial_sphere.h"
#include <cmath>
#include <algorithm>
#include <numbers>
#include <random>

namespace grrt {

CelestialSphere::CelestialSphere(int num_stars, unsigned int seed)
    : grid_(grid_theta * grid_phi) {

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    stars_.reserve(num_stars);
    for (int i = 0; i < num_stars; ++i) {
        Star s;
        s.theta = std::acos(1.0 - 2.0 * uniform(rng));
        s.phi = 2.0 * std::numbers::pi * uniform(rng);

        // Power-law brightness: many dim, few bright
        double u = uniform(rng);
        s.brightness = 0.1 * std::pow(std::max(u, 0.001), -2.5);
        s.brightness = std::min(s.brightness, 50.0);

        stars_.push_back(s);
    }

    build_grid();
}

int CelestialSphere::grid_index(int t_bin, int p_bin) const {
    t_bin = std::clamp(t_bin, 0, grid_theta - 1);
    p_bin = ((p_bin % grid_phi) + grid_phi) % grid_phi;
    return t_bin * grid_phi + p_bin;
}

void CelestialSphere::build_grid() {
    for (int i = 0; i < static_cast<int>(stars_.size()); ++i) {
        int t_bin = static_cast<int>(stars_[i].theta * grid_theta / std::numbers::pi);
        int p_bin = static_cast<int>((stars_[i].phi + std::numbers::pi) * grid_phi / (2.0 * std::numbers::pi));
        t_bin = std::clamp(t_bin, 0, grid_theta - 1);
        p_bin = std::clamp(p_bin, 0, grid_phi - 1);
        grid_[grid_index(t_bin, p_bin)].push_back(i);
    }
}

Vec3 CelestialSphere::sample(const Vec4& position) const {
    double theta = position[2];
    double phi = std::fmod(position[3], 2.0 * std::numbers::pi);
    if (phi > std::numbers::pi) phi -= 2.0 * std::numbers::pi;
    if (phi < -std::numbers::pi) phi += 2.0 * std::numbers::pi;

    int t_bin = static_cast<int>(theta * grid_theta / std::numbers::pi);
    int p_bin = static_cast<int>((phi + std::numbers::pi) * grid_phi / (2.0 * std::numbers::pi));

    for (int dt = -1; dt <= 1; ++dt) {
        for (int dp = -1; dp <= 1; ++dp) {
            int idx = grid_index(t_bin + dt, p_bin + dp);
            for (int si : grid_[idx]) {
                const Star& s = stars_[si];
                double dtheta = theta - s.theta;
                double dphi = phi - s.phi;
                if (dphi > std::numbers::pi) dphi -= 2.0 * std::numbers::pi;
                if (dphi < -std::numbers::pi) dphi += 2.0 * std::numbers::pi;
                double sin_t = std::sin(theta);
                double ang_dist2 = dtheta * dtheta + dphi * dphi * sin_t * sin_t;

                if (ang_dist2 < angular_tolerance_ * angular_tolerance_) {
                    double b = s.brightness;
                    return {{b, b, b}};
                }
            }
        }
    }

    return {};
}

} // namespace grrt
