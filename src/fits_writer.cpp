/// @file fits_writer.cpp
/// @brief Minimal dependency-free FITS spectral data cube writer.
///
/// FITS format reference: NOST 100-2.0 (NASA/OSSA).
/// Header: sequence of 80-character cards grouped into 2880-byte blocks.
/// Data:   raw big-endian IEEE 754 doubles, also padded to 2880-byte blocks.

#include "grrt/render/fits_writer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace grrt {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

constexpr int CARD_LEN  = 80;   // characters per header card
constexpr int BLOCK_LEN = 2880; // bytes per FITS block
constexpr int CARDS_PER_BLOCK = BLOCK_LEN / CARD_LEN; // 36

/// Pad or truncate a string to exactly `len` characters using spaces.
std::string pad_to(std::string s, std::size_t len, char fill = ' ') {
    if (s.size() >= len) return s.substr(0, len);
    s.append(len - s.size(), fill);
    return s;
}

/// Format one 80-character FITS header card.
/// keyword : max 8 chars, left-justified.
/// value   : pre-formatted value string, right-justified in columns 11-30.
/// comment : optional free text after " / ".
std::string fits_card(std::string_view keyword,
                      std::string_view value,
                      std::string_view comment = "") {
    // Columns 1-8:  keyword (left-justified, space-padded)
    // Column 9:     '='
    // Column 10:    ' '
    // Columns 11-30: value (right-justified in 20-char field)
    // Columns 31-80: " / comment" (optional)
    std::string kw = pad_to(std::string(keyword), 8);
    // value field: right-justify in 20 chars
    std::string val = std::string(value);
    if (val.size() < 20) val = std::string(20 - val.size(), ' ') + val;

    std::string card = kw + "= " + val;
    if (!comment.empty()) {
        card += " / ";
        card += comment;
    }
    return pad_to(card, CARD_LEN);
}

/// Card for a boolean value ('T' or 'F').
std::string fits_card_bool(std::string_view keyword, bool value,
                           std::string_view comment = "") {
    return fits_card(keyword, value ? "T" : "F", comment);
}

/// Card for an integer value.
std::string fits_card_int(std::string_view keyword, long long value,
                          std::string_view comment = "") {
    return fits_card(keyword, std::format("{}", value), comment);
}

/// Card for a double value (uses G20.12 style via std::format).
std::string fits_card_double(std::string_view keyword, double value,
                             std::string_view comment = "") {
    // Use exponential notation with enough precision for a double.
    // FITS standard allows free format in value field; we use 18 significant digits.
    std::string s = std::format("{:.12E}", value);
    return fits_card(keyword, s, comment);
}

/// Card for a string value (FITS strings are enclosed in single quotes,
/// left-justified, padded to at least 8 chars inside the quotes).
std::string fits_card_string(std::string_view keyword, std::string_view value,
                             std::string_view comment = "") {
    // Inner string must be at least 8 characters wide per FITS standard.
    std::string inner = std::string(value);
    if (inner.size() < 8) inner.append(8 - inner.size(), ' ');
    // Truncate to stay within value field (18 chars inside quotes + 2 quotes = 20)
    if (inner.size() > 18) inner = inner.substr(0, 18);
    std::string quoted = "'" + inner + "'";
    return fits_card(keyword, quoted, comment);
}

/// Write a double as 8 big-endian bytes to the stream.
/// FITS mandates big-endian (network byte order) for all binary data.
void write_double_be(std::ofstream& out, double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    uint8_t bytes[8];
    for (int i = 7; i >= 0; --i) {
        bytes[7 - i] = static_cast<uint8_t>(bits >> (i * 8));
    }
    out.write(reinterpret_cast<const char*>(bytes), 8);
}

/// Return true if the frequency bins have uniform linear spacing (within 0.01%).
bool is_uniform_linear(const std::vector<double>& bins) {
    if (bins.size() < 2) return true;
    const double step = bins[1] - bins[0];
    if (step == 0.0) return false;
    for (std::size_t i = 2; i < bins.size(); ++i) {
        const double delta = (bins[i] - bins[i - 1]) - step;
        if (std::abs(delta / step) > 1e-4) return false;
    }
    return true;
}

/// Return true if the frequency bins have uniform log10 spacing (within 0.01%).
bool is_uniform_log(const std::vector<double>& bins) {
    if (bins.size() < 2) return true;
    if (bins[0] <= 0.0 || bins[1] <= 0.0) return false;
    const double log_step = std::log10(bins[1]) - std::log10(bins[0]);
    if (log_step == 0.0) return false;
    for (std::size_t i = 2; i < bins.size(); ++i) {
        if (bins[i] <= 0.0) return false;
        const double delta = (std::log10(bins[i]) - std::log10(bins[i - 1])) - log_step;
        if (std::abs(delta / log_step) > 1e-4) return false;
    }
    return true;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void write_fits(const std::string&         path,
                const double*              data,
                int                        width,
                int                        height,
                int                        num_bins,
                const std::vector<double>& frequency_bins_hz,
                const FITSMetadata&        metadata) {

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            std::format("grrt::write_fits: cannot open '{}' for writing", path));
    }

    // -----------------------------------------------------------------------
    // Build the header cards
    // -----------------------------------------------------------------------
    std::vector<std::string> cards;
    cards.reserve(64);

    // Mandatory structural keywords
    cards.push_back(fits_card_bool("SIMPLE",  true,  "conforms to FITS standard"));
    cards.push_back(fits_card_int ("BITPIX",  -64,   "IEEE 754 double precision"));
    cards.push_back(fits_card_int ("NAXIS",   3,     "number of array dimensions"));
    cards.push_back(fits_card_int ("NAXIS1",  width,    "pixels along RA axis"));
    cards.push_back(fits_card_int ("NAXIS2",  height,   "pixels along Dec axis"));
    cards.push_back(fits_card_int ("NAXIS3",  num_bins, "number of frequency bins"));

    // Physical units
    cards.push_back(fits_card_string("BUNIT", "erg/s/cm2/Hz/sr",
                                     "specific intensity"));

    // WCS for frequency axis (NAXIS3)
    if (!frequency_bins_hz.empty()) {
        if (is_uniform_linear(frequency_bins_hz) && frequency_bins_hz.size() >= 2) {
            // Linear frequency axis
            cards.push_back(fits_card_string("CTYPE3", "FREQ", "frequency"));
            cards.push_back(fits_card_string("CUNIT3", "Hz",   "frequency unit"));
            cards.push_back(fits_card_double("CRPIX3", 1.0, "reference pixel"));
            cards.push_back(fits_card_double("CRVAL3", frequency_bins_hz.front(),
                                             "frequency at ref pixel (Hz)"));
            const double step = frequency_bins_hz[1] - frequency_bins_hz[0];
            cards.push_back(fits_card_double("CDELT3", step,
                                             "frequency step (Hz)"));
        } else if (is_uniform_log(frequency_bins_hz) && frequency_bins_hz.size() >= 2) {
            // Log-spaced: axis represents log10(freq in Hz).
            // DS9 will label slices with the log10 values; the actual Hz
            // values are also stored as FREQnnnn keywords for astropy.
            const double log_min = std::log10(frequency_bins_hz.front());
            const double log_step = std::log10(frequency_bins_hz[1])
                                  - std::log10(frequency_bins_hz[0]);
            cards.push_back(fits_card_string("CTYPE3", "FREQ-LOG",
                                             "log10(frequency/Hz)"));
            cards.push_back(fits_card_string("CUNIT3", "log(Hz)",
                                             "log10 of frequency in Hz"));
            cards.push_back(fits_card_double("CRPIX3", 1.0, "reference pixel"));
            cards.push_back(fits_card_double("CRVAL3", log_min,
                                             "log10(freq) at ref pixel"));
            cards.push_back(fits_card_double("CDELT3", log_step,
                                             "log10(freq) step per pixel"));
            // Also store actual Hz values for programmatic access
            for (int k = 0; k < num_bins && k < static_cast<int>(frequency_bins_hz.size()); ++k) {
                std::string kw = std::format("FREQ{:04d}", k + 1);
                cards.push_back(fits_card_double(kw, frequency_bins_hz[k],
                                                 std::format("freq bin {} (Hz)", k + 1)));
            }
        } else {
            // Non-uniform, non-log: use slice index as axis, store Hz as keywords
            cards.push_back(fits_card_string("CTYPE3", "FREQ-TAB",
                                             "frequency (see FREQnnnn keys)"));
            cards.push_back(fits_card_string("CUNIT3", "Hz",
                                             "frequency unit"));
            cards.push_back(fits_card_double("CRPIX3", 1.0, "reference pixel"));
            cards.push_back(fits_card_double("CRVAL3", 1.0, "slice index"));
            cards.push_back(fits_card_double("CDELT3", 1.0, "slice index step"));
            for (int k = 0; k < num_bins && k < static_cast<int>(frequency_bins_hz.size()); ++k) {
                std::string kw = std::format("FREQ{:04d}", k + 1);
                cards.push_back(fits_card_double(kw, frequency_bins_hz[k],
                                                 std::format("freq bin {} (Hz)", k + 1)));
            }
        }
    }

    // Render simulation metadata
    cards.push_back(fits_card_double("SPIN",   metadata.spin,
                                     "Kerr spin parameter a/M"));
    cards.push_back(fits_card_double("MASS",   metadata.mass,
                                     "black hole mass (geometrized units)"));
    cards.push_back(fits_card_double("OBS_R",  metadata.observer_r,
                                     "observer radial coordinate (M)"));
    cards.push_back(fits_card_double("OBS_TH", metadata.observer_theta,
                                     "observer polar angle (rad)"));
    cards.push_back(fits_card_double("FOV",    metadata.fov,
                                     "camera field of view (rad)"));
    cards.push_back(fits_card_int   ("SPP",    metadata.samples_per_pixel,
                                     "samples per pixel (anti-aliasing)"));

    // Provenance
    cards.push_back(fits_card_string("ORIGIN", "grrt",
                                     "GR ray tracer (github: gr_ray_tracer)"));

    // END card — exactly 80 spaces after "END"
    cards.push_back(pad_to("END", CARD_LEN));

    // -----------------------------------------------------------------------
    // Write header blocks (pad to multiple of 2880 bytes with spaces)
    // -----------------------------------------------------------------------
    const std::size_t num_cards = cards.size();
    const std::size_t full_blocks =
        (num_cards + CARDS_PER_BLOCK - 1) / CARDS_PER_BLOCK;
    const std::size_t total_header_cards = full_blocks * CARDS_PER_BLOCK;

    for (const auto& card : cards) {
        out.write(card.data(), CARD_LEN);
    }
    // Pad remaining cards in the last block with spaces
    const std::string blank_card(CARD_LEN, ' ');
    for (std::size_t i = num_cards; i < total_header_cards; ++i) {
        out.write(blank_card.data(), CARD_LEN);
    }

    // -----------------------------------------------------------------------
    // Write data in FITS axis order: k (NAXIS3) outer, j (NAXIS2) middle,
    // i (NAXIS1) inner.  Source layout: data[(j*width + i)*num_bins + k].
    // -----------------------------------------------------------------------
    // FITS pixel (1,1) is bottom-left; renderer row 0 is top.
    // Write rows in reverse order so the image is right-side-up in DS9.
    for (int k = 0; k < num_bins; ++k) {
        for (int j = height - 1; j >= 0; --j) {
            for (int i = 0; i < width; ++i) {
                const double v = data[(j * width + i) * num_bins + k];
                write_double_be(out, v);
            }
        }
    }

    // Pad data section to multiple of 2880 bytes with zeros
    const std::size_t data_bytes =
        static_cast<std::size_t>(width) * height * num_bins * 8;
    const std::size_t remainder = data_bytes % BLOCK_LEN;
    if (remainder != 0) {
        const std::size_t pad = BLOCK_LEN - remainder;
        const std::vector<char> zeros(pad, '\0');
        out.write(zeros.data(), static_cast<std::streamsize>(pad));
    }

    out.flush();
    if (!out) {
        throw std::runtime_error(
            std::format("grrt::write_fits: write error on '{}'", path));
    }
}

} // namespace grrt
