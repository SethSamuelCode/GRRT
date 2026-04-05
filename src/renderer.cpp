#include "grrt/render/renderer.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/scene/celestial_sphere.h"
#include "grrt/color/spectrum.h"
#include "grrt/geodesic/geodesic_tracer.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <vector>

namespace grrt {

// Simple hash for deterministic per-pixel jitter (no external RNG state needed)
static double pixel_hash(int i, int j, int s, int channel) {
    uint32_t h = static_cast<uint32_t>(i * 73856093u ^ j * 19349663u ^ s * 83492791u ^ channel * 45678917u);
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return (h & 0xFFFFu) / 65536.0;
}

Renderer::Renderer(const Camera& camera, const GeodesicTracer& tracer,
                   const AccretionDisk* disk, const CelestialSphere* sphere,
                   const SpectrumLUT* spectrum, const ToneMapper& tonemapper,
                   int samples_per_pixel)
    : camera_(camera), tracer_(tracer), disk_(disk), sphere_(sphere),
      spectrum_(spectrum), tonemapper_(tonemapper),
      spp_(samples_per_pixel < 1 ? 1 : samples_per_pixel) {}

void Renderer::render(float* framebuffer, int width, int height,
                      ProgressCallback progress_cb) const {
    // Stratified jittered sampling: divide pixel into sqrt(spp) x sqrt(spp) grid,
    // jitter within each cell. For non-square spp, use the closest square.
    const int grid = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(spp_))));
    const int actual_spp = grid * grid;
    const double inv_spp = 1.0 / actual_spp;
    const double cell = 1.0 / grid;

    const int total_pixels = width * height;
    std::atomic<int> pixels_done{0};

    #pragma omp parallel for schedule(guided)
    for (int pixel = 0; pixel < total_pixels; ++pixel) {
        const int j = pixel / width;
        const int i = pixel % width;

        Vec3 accum;
        for (int sy = 0; sy < grid; ++sy) {
            for (int sx = 0; sx < grid; ++sx) {
                const int s = sy * grid + sx;
                // Stratified jitter: sample within sub-cell
                const double jx = pixel_hash(i, j, s, 0);
                const double jy = pixel_hash(i, j, s, 1);
                const double px = i + (sx + jx) * cell;
                const double py = j + (sy + jy) * cell;

                GeodesicState state = camera_.ray_for_pixel(px, py);
                TraceResult result = tracer_.trace(state, disk_, spectrum_);

                Vec3 color = result.accumulated_color;
                if (result.termination == RayTermination::Escaped && sphere_) {
                    color += sphere_->sample(result.final_position);
                }
                accum = accum + color;
            }
        }

        const int idx = pixel * 4;
        framebuffer[idx + 0] = static_cast<float>(accum[0] * inv_spp);
        framebuffer[idx + 1] = static_cast<float>(accum[1] * inv_spp);
        framebuffer[idx + 2] = static_cast<float>(accum[2] * inv_spp);
        framebuffer[idx + 3] = 1.0f;

        if (progress_cb) {
            const int done = ++pixels_done;
            if (done % width == 0)
                progress_cb(static_cast<float>(done) / static_cast<float>(total_pixels));
        }
    }

    if (progress_cb) progress_cb(1.0f);
}

void Renderer::render_spectral(double* spectral_buffer, int width, int height,
                                const std::vector<double>& frequency_bins,
                                ProgressCallback progress_cb) const {
    const int num_bins = static_cast<int>(frequency_bins.size());
    const int grid = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(spp_))));
    const int actual_spp = grid * grid;
    const double inv_spp = 1.0 / actual_spp;
    const double cell = 1.0 / grid;

    const int total_pixels = width * height;
    std::atomic<int> pixels_done{0};

    #pragma omp parallel
    {
        std::vector<double> accum(num_bins, 0.0);

        #pragma omp for schedule(guided)
        for (int pixel = 0; pixel < total_pixels; ++pixel) {
            const int j = pixel / width;
            const int i = pixel % width;

            std::fill(accum.begin(), accum.end(), 0.0);

            for (int sy = 0; sy < grid; ++sy) {
                for (int sx = 0; sx < grid; ++sx) {
                    const int s = sy * grid + sx;
                    const double jx = pixel_hash(i, j, s, 0);
                    const double jy = pixel_hash(i, j, s, 1);
                    const double px = i + (sx + jx) * cell;
                    const double py = j + (sy + jy) * cell;

                    GeodesicState state = camera_.ray_for_pixel(px, py);
                    SpectralTraceResult result = tracer_.trace_spectral(state, frequency_bins);

                    for (int k = 0; k < num_bins; ++k) {
                        accum[k] += result.spectral_intensity[k];
                    }
                }
            }

            const int base = pixel * num_bins;
            for (int k = 0; k < num_bins; ++k) {
                spectral_buffer[base + k] = accum[k] * inv_spp;
            }

            if (progress_cb) {
                const int done = ++pixels_done;
                if (done % width == 0)
                    progress_cb(static_cast<float>(done) / static_cast<float>(total_pixels));
            }
        }
    }

    if (progress_cb) progress_cb(1.0f);
}

void Renderer::render_spectral_streaming(int width, int height,
                                          const std::vector<double>& frequency_bins,
                                          RowCallback row_cb,
                                          ProgressCallback progress_cb) const {
    const int num_bins  = static_cast<int>(frequency_bins.size());
    const int grid      = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(spp_))));
    const int actual_spp = grid * grid;
    const double inv_spp = 1.0 / actual_spp;
    const double cell    = 1.0 / grid;

    int rows_done = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < height; ++j) {
        // Per-thread accum and row output buffers — stack-allocated relative
        // to the OMP thread, so no false sharing between threads.
        std::vector<double> accum(num_bins, 0.0);
        std::vector<double> row_buf(static_cast<std::size_t>(width) * num_bins, 0.0);

        for (int i = 0; i < width; ++i) {
            std::fill(accum.begin(), accum.end(), 0.0);

            for (int sy = 0; sy < grid; ++sy) {
                for (int sx = 0; sx < grid; ++sx) {
                    const int s = sy * grid + sx;
                    const double jx = pixel_hash(i, j, s, 0);
                    const double jy = pixel_hash(i, j, s, 1);
                    const double px = i + (sx + jx) * cell;
                    const double py = j + (sy + jy) * cell;

                    GeodesicState state = camera_.ray_for_pixel(px, py);
                    SpectralTraceResult result = tracer_.trace_spectral(state, frequency_bins);

                    for (int k = 0; k < num_bins; ++k) {
                        accum[k] += result.spectral_intensity[k];
                    }
                }
            }

            for (int k = 0; k < num_bins; ++k) {
                row_buf[i * num_bins + k] = accum[k] * inv_spp;
            }
        }

        // row_cb handles its own serialisation (FITSStreamWriter::write_row
        // is mutex-protected), so we do not need omp critical here.
        row_cb(j, row_buf.data());

        if (progress_cb) {
            int done;
            #pragma omp critical
            { done = ++rows_done; }
            progress_cb(static_cast<float>(done) / static_cast<float>(height));
        }
    }

    if (progress_cb) progress_cb(1.0f);
}

} // namespace grrt
