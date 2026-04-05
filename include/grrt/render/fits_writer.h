#ifndef GRRT_FITS_WRITER_H
#define GRRT_FITS_WRITER_H

#include "grrt_export.h"
#include <string>
#include <vector>

namespace grrt {

/// Render and simulation metadata to embed in FITS header keywords.
struct FITSMetadata {
    double spin            = 0.0;    ///< Dimensionless Kerr spin parameter a/M
    double mass            = 1.0;    ///< Black hole mass in geometrized units
    double observer_r      = 50.0;   ///< Observer radial coordinate (M)
    double observer_theta  = 1.396;  ///< Observer polar angle (rad), ~80 deg
    double fov             = 1.047;  ///< Camera field of view (rad), ~60 deg
    int    samples_per_pixel = 1;    ///< Anti-aliasing samples per pixel
};

/// Write a spectral data cube to a FITS file.
///
/// Data layout: data[(j * width + i) * num_bins + k] = intensity at pixel (i,j), bin k.
/// Units are erg/s/cm²/Hz/sr (specific intensity).
///
/// Output FITS axes:
///   NAXIS1 = width   (RA-like, fastest varying in FITS)
///   NAXIS2 = height  (Dec-like)
///   NAXIS3 = num_bins (frequency, slowest varying in FITS)
///   BITPIX = -64 (IEEE 754 double precision, big-endian)
///
/// WCS keywords are written for the frequency axis using CTYPE3='FREQ'.
/// If frequency_bins_hz has uniform spacing, CDELT3 encodes the step;
/// otherwise individual FREQnnn keywords are written for each bin centre.
///
/// @throws std::runtime_error if the output file cannot be opened.
GRRT_EXPORT void write_fits(const std::string&         path,
                            const double*              data,
                            int                        width,
                            int                        height,
                            int                        num_bins,
                            const std::vector<double>& frequency_bins_hz,
                            const FITSMetadata&        metadata);

} // namespace grrt

#endif // GRRT_FITS_WRITER_H
