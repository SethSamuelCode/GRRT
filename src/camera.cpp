#include "grrt/camera/camera.h"
#include <cmath>

namespace grrt {

// Metric inner product: <u, v> = g_μν u^μ v^ν
static double metric_dot(const Matrix4& g, const Vec4& u, const Vec4& v) {
    double sum = 0.0;
    for (int mu = 0; mu < 4; ++mu)
        for (int nu = 0; nu < 4; ++nu)
            sum += g.m[mu][nu] * u[mu] * v[nu];
    return sum;
}

// Normalize a vector: v / sqrt(|<v,v>|)
static Vec4 metric_normalize(const Matrix4& g, const Vec4& v) {
    double norm2 = metric_dot(g, v, v);
    double norm = std::sqrt(std::abs(norm2));
    return v * (1.0 / norm);
}

// Project out: v - <v, e>/<e, e> * e
static Vec4 project_out(const Matrix4& g, const Vec4& v, const Vec4& e) {
    double ve = metric_dot(g, v, e);
    double ee = metric_dot(g, e, e);
    return v - e * (ve / ee);
}

Camera::Camera(const Metric& metric, double r_obs, double theta_obs, double phi_obs,
               double fov, int width, int height)
    : metric_(metric), fov_(fov), width_(width), height_(height) {
    position_[0] = 0.0;       // t
    position_[1] = r_obs;     // r
    position_[2] = theta_obs; // θ
    position_[3] = phi_obs;   // φ
    build_tetrad();
}

void Camera::build_tetrad() {
    Matrix4 g = metric_.g_lower(position_);

    // e_0: static observer 4-velocity u^μ = (1/√(-g_tt), 0, 0, 0)
    double g_tt = g.m[0][0];
    e0_ = Vec4{{1.0 / std::sqrt(-g_tt), 0.0, 0.0, 0.0}};

    // e_3 (forward = radially inward = -r direction)
    Vec4 r_dir = {{0.0, -1.0, 0.0, 0.0}};
    Vec4 v3 = project_out(g, r_dir, e0_);
    e3_ = metric_normalize(g, v3);

    // e_2 (up = -θ direction, since θ increases downward from pole)
    Vec4 theta_dir = {{0.0, 0.0, -1.0, 0.0}};
    Vec4 v2 = project_out(g, theta_dir, e0_);
    v2 = project_out(g, v2, e3_);
    e2_ = metric_normalize(g, v2);

    // e_1 (right = φ direction)
    Vec4 phi_dir = {{0.0, 0.0, 0.0, 1.0}};
    Vec4 v1 = project_out(g, phi_dir, e0_);
    v1 = project_out(g, v1, e3_);
    v1 = project_out(g, v1, e2_);
    e1_ = metric_normalize(g, v1);
}

GeodesicState Camera::ray_for_pixel(int i, int j) const {
    // Screen angles
    double alpha = (static_cast<double>(i) - width_ / 2.0) * fov_ / width_;
    double beta = (static_cast<double>(j) - height_ / 2.0) * fov_ / width_;

    // Local 3-direction in tetrad frame
    double ca = std::cos(beta) * std::sin(alpha);
    double cb = std::sin(beta);
    double cc = std::cos(beta) * std::cos(alpha);

    // d^μ = -ca * e1 - cb * e2 + cc * e3  (unit spatial vector in tetrad)
    Vec4 d;
    for (int mu = 0; mu < 4; ++mu) {
        d[mu] = -ca * e1_[mu] - cb * e2_[mu] + cc * e3_[mu];
    }

    // Null 4-momentum (contravariant): p^μ = -e0 + d
    Vec4 p_contra;
    for (int mu = 0; mu < 4; ++mu) {
        p_contra[mu] = -e0_[mu] + d[mu];
    }

    // Lower index: p_μ = g_μν p^ν
    Matrix4 g = metric_.g_lower(position_);
    Vec4 p_cov = g.contract(p_contra);

    return {position_, p_cov};
}

} // namespace grrt
