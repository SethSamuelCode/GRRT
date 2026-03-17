#include "grrt/scene/accretion_disk.h"
#include <cmath>
#include <algorithm>
#include <numbers>

namespace grrt {

AccretionDisk::AccretionDisk(double mass, double spin, double r_isco,
                             double r_outer, double peak_temperature, int flux_lut_size)
    : mass_(mass), spin_(spin),
      r_inner_(r_isco),
      r_outer_(r_outer),
      peak_temperature_(peak_temperature),
      flux_lut_size_(flux_lut_size) {
    build_flux_lut();
}

double AccretionDisk::omega_kepler(double r) const {
    return std::sqrt(mass_ / (r * r * r));
}

double AccretionDisk::Omega(double r) const {
    double w = omega_kepler(r);
    return w / (1.0 + spin_ * w);
}

double AccretionDisk::E_circ(double r) const {
    double M = mass_;
    double a = spin_;
    double w = omega_kepler(r);
    double aw = a * w;
    return (1.0 - 2.0 * M / r + aw) / std::sqrt(1.0 - 3.0 * M / r + 2.0 * aw);
}

double AccretionDisk::L_circ(double r) const {
    double M = mass_;
    double a = spin_;
    double w = omega_kepler(r);
    double aw = a * w;
    return std::sqrt(M * r) * (1.0 - 2.0 * aw + a * a / (r * r))
           / std::sqrt(1.0 - 3.0 * M / r + 2.0 * aw);
}

void AccretionDisk::build_flux_lut() {
    flux_r_min_ = r_inner_;
    flux_r_max_ = r_outer_;
    flux_lut_.resize(flux_lut_size_);

    double E_isco = E_circ(r_inner_);
    double L_isco = L_circ(r_inner_);

    flux_max_ = 0.0;
    double I_cumulative = 0.0;
    constexpr double fd_eps = 1e-6;
    double prev_integrand = 0.0;

    for (int i = 0; i < flux_lut_size_; ++i) {
        double r = r_inner_ + (r_outer_ - r_inner_) * i / (flux_lut_size_ - 1);

        if (i == 0) {
            flux_lut_[i] = 0.0;
            continue;
        }

        double E_prime = (E_circ(r + fd_eps) - E_circ(r - fd_eps)) / (2.0 * fd_eps);
        double L_prime = (L_circ(r + fd_eps) - L_circ(r - fd_eps)) / (2.0 * fd_eps);

        double integrand = (E_circ(r) - E_isco) * L_prime - (L_circ(r) - L_isco) * E_prime;

        double dr = (r_outer_ - r_inner_) / (flux_lut_size_ - 1);
        I_cumulative += 0.5 * (prev_integrand + integrand) * dr;
        prev_integrand = integrand;

        double Om = Omega(r);
        double E_r = E_circ(r);
        double L_r = L_circ(r);
        double dOmega_dr = (Omega(r + fd_eps) - Omega(r - fd_eps)) / (2.0 * fd_eps);

        double denominator = E_r - Om * L_r;
        if (std::abs(denominator) < 1e-20) {
            flux_lut_[i] = 0.0;
            continue;
        }

        double F = (3.0 * mass_ / (8.0 * std::numbers::pi * r * r * r))
                   * (1.0 / denominator) * (-dOmega_dr) * I_cumulative;

        flux_lut_[i] = std::max(F, 0.0);
        if (flux_lut_[i] > flux_max_) {
            flux_max_ = flux_lut_[i];
        }
    }
}

double AccretionDisk::flux(double r) const {
    if (r <= r_inner_ || r >= r_outer_ || flux_max_ <= 0.0) return 0.0;

    double frac = (r - flux_r_min_) / (flux_r_max_ - flux_r_min_) * (flux_lut_size_ - 1);
    int idx = static_cast<int>(frac);
    double t = frac - idx;

    if (idx >= flux_lut_size_ - 1) return flux_lut_[flux_lut_size_ - 1];
    return flux_lut_[idx] * (1.0 - t) + flux_lut_[idx + 1] * t;
}

double AccretionDisk::temperature(double r) const {
    double F = flux(r);
    if (F <= 0.0 || flux_max_ <= 0.0) return 0.0;
    return peak_temperature_ * std::pow(F / flux_max_, 0.25);
}

double AccretionDisk::redshift(double r_cross, const Vec4& p, double observer_r) const {
    double M = mass_;
    double a = spin_;

    // Observer: static at r_obs
    double u_t_obs = 1.0 / std::sqrt(1.0 - 2.0 * M / observer_r);
    double pu_obs = p[0] * u_t_obs;

    // Emitter: circular orbit at r_cross
    double w = omega_kepler(r_cross);
    double aw = a * w;
    double u_t_emit = 1.0 / std::sqrt(1.0 - 3.0 * M / r_cross + 2.0 * aw);
    double u_phi_emit = Omega(r_cross) * u_t_emit;
    double pu_emit = p[0] * u_t_emit + p[3] * u_phi_emit;

    if (std::abs(pu_obs) < 1e-30) return 1.0;
    return pu_emit / pu_obs;
}

Vec3 AccretionDisk::emission(double r_cross, const Vec4& p_cross,
                             double observer_r, const SpectrumLUT& spectrum) const {
    double T = temperature(r_cross);
    if (T <= 0.0) return {};

    double g = redshift(r_cross, p_cross, observer_r);

    double T_obs = g * T;
    if (T_obs < 100.0) return {};

    Vec3 color = spectrum.temperature_to_color(T_obs);
    double g3 = g * g * g;
    return color * g3;
}

} // namespace grrt
