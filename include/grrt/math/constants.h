#ifndef GRRT_CONSTANTS_H
#define GRRT_CONSTANTS_H

// Physical constants in CGS units
// Used by volumetric disk opacity and thermodynamics.
// The geodesic integrator remains in pure geometric units (G=c=1).

namespace grrt::constants {

// Fundamental
inline constexpr double k_B        = 1.380649e-16;    // erg/K, Boltzmann
inline constexpr double sigma_SB   = 5.670374e-5;     // erg/(cm^2 s K^4), Stefan-Boltzmann
inline constexpr double m_p        = 1.672622e-24;     // g, proton mass
inline constexpr double m_e        = 9.10938e-28;      // g, electron mass
inline constexpr double c_cgs      = 2.997924e10;      // cm/s, speed of light
inline constexpr double h_planck   = 6.626070e-27;     // erg*s, Planck constant
inline constexpr double G_cgs      = 6.674e-8;         // cm^3/(g s^2), gravitational constant
inline constexpr double M_sun      = 1.989e33;         // g, solar mass
inline constexpr double sigma_T    = 6.652e-25;        // cm^2, Thomson cross-section
inline constexpr double eV_to_erg  = 1.602e-12;        // erg/eV
inline constexpr double Ry         = 2.180e-11;        // erg, Rydberg energy (13.6 eV)

// Derived
inline constexpr double gamma_E    = 1.7811;           // exp(Euler-Mascheroni), for Gaunt factor
inline constexpr double C_ff       = 3.69e8;           // CGS, free-free coefficient (R&L 5.18a)
inline constexpr double a_rad      = 4.0 * sigma_SB / c_cgs;  // erg/(cm^3 K^4), radiation constant

// Composition
inline constexpr double X_hydrogen = 0.70;             // Hydrogen mass fraction
inline constexpr double Y_helium   = 0.28;             // Helium mass fraction
inline constexpr double Z_metal    = 0.02;             // Metal mass fraction (solar)

// Gas
inline constexpr double gamma_gas  = 5.0 / 3.0;        // Adiabatic index (ideal monatomic)
inline constexpr double mu_fully_ionized = 0.6;         // Mean molecular weight reference

} // namespace grrt::constants

#endif
