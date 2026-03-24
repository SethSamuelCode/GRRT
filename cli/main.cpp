#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "grrt/api.h"
#include <print>
#include <vector>
#include <cmath>
#include <string>
#include <cstring>
#include <numbers>

static void print_usage() {
    std::println("Usage: grrt-cli [options]");
    std::println("  --width N             Image width (default: 1024)");
    std::println("  --height N            Image height (default: 1024)");
    std::println("  --metric TYPE         schwarzschild | kerr (default: kerr)");
    std::println("  --mass M              Black hole mass (default: 1.0)");
    std::println("  --spin A              Spin parameter a/M (default: 0.998)");
    std::println("  --observer-r R        Observer radius (default: 50)");
    std::println("  --observer-theta T    Observer polar angle, degrees (default: 80)");
    std::println("  --observer-phi P      Observer azimuthal angle, degrees (default: 0)");
    std::println("  --fov F               Field of view, degrees (default: 90)");
    std::println("  --disk on|off         Accretion disk (default: on)");
    std::println("  --disk-outer R        Disk outer radius (default: 20)");
    std::println("  --disk-temp T         Peak disk temperature in K (default: 1e7)");
    std::println("  --disk-volumetric     Enable volumetric disk model");
    std::println("  --disk-alpha A        Shakura-Sunyaev viscosity (default: 0.1)");
    std::println("  --disk-turbulence T   Noise amplitude (default: 0.4)");
    std::println("  --disk-seed N         Noise seed (default: 42)");
    std::println("  --background TYPE     black | stars (default: stars)");
    std::println("  --max-steps N         Max integration steps (default: 10000)");
    std::println("  --tolerance T         Integrator tolerance (default: 1e-8)");
    std::println("  --threads N           CPU threads, 0=auto (default: 0)");
    std::println("  --backend TYPE        cpu | cuda (default: cpu)");
    std::println("  --validate            Render on both backends, compare results");
    std::println("  --output NAME         Output base name (default: output)");
    std::println("                        Produces NAME.png, NAME.hdr, NAME_linear.hdr");
    std::println("  --help                Show this help");
}

static double deg_to_rad(double deg) { return deg * std::numbers::pi / 180.0; }

int main(int argc, char* argv[]) {
    // Defaults
    GRRTParams params{};
    params.width = 1024;
    params.height = 1024;
    params.metric_type = GRRT_METRIC_KERR;
    params.mass = 1.0;
    params.spin = 0.998;
    params.observer_r = 50.0;
    params.observer_theta = deg_to_rad(80.0);
    params.observer_phi = 0.0;
    params.fov = deg_to_rad(90.0);
    params.integrator_max_steps = 10000;
    params.integrator_tolerance = 1e-8;
    params.disk_enabled = 1;
    params.disk_inner = 0.0;
    params.disk_outer = 20.0;
    params.disk_temperature = 1e7;
    params.background_type = GRRT_BG_STARS;
    params.thread_count = 0;
    params.disk_volumetric = 0;
    params.disk_alpha = 0.1;
    params.disk_turbulence = 0.4;
    params.disk_seed = 42;

    std::string output_name = "output";
    std::string backend_str = "cpu";
    bool validate = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0; };
        auto next = [&]() -> const char* {
            if (i + 1 < argc) return argv[++i];
            std::println(stderr, "Missing value for {}", argv[i]);
            return nullptr;
        };

        if (arg("--help") || arg("-h")) {
            print_usage();
            return 0;
        } else if (arg("--width")) {
            if (auto v = next()) params.width = std::atoi(v);
        } else if (arg("--height")) {
            if (auto v = next()) params.height = std::atoi(v);
        } else if (arg("--metric")) {
            if (auto v = next()) {
                if (std::strcmp(v, "schwarzschild") == 0) params.metric_type = GRRT_METRIC_SCHWARZSCHILD;
                else if (std::strcmp(v, "kerr") == 0) params.metric_type = GRRT_METRIC_KERR;
                else { std::println(stderr, "Unknown metric: {}", v); return 1; }
            }
        } else if (arg("--mass")) {
            if (auto v = next()) params.mass = std::atof(v);
        } else if (arg("--spin")) {
            if (auto v = next()) params.spin = std::atof(v);
        } else if (arg("--observer-r")) {
            if (auto v = next()) params.observer_r = std::atof(v);
        } else if (arg("--observer-theta")) {
            if (auto v = next()) params.observer_theta = deg_to_rad(std::atof(v));
        } else if (arg("--observer-phi")) {
            if (auto v = next()) params.observer_phi = deg_to_rad(std::atof(v));
        } else if (arg("--fov")) {
            if (auto v = next()) params.fov = deg_to_rad(std::atof(v));
        } else if (arg("--disk")) {
            if (auto v = next()) params.disk_enabled = (std::strcmp(v, "on") == 0) ? 1 : 0;
        } else if (arg("--disk-outer")) {
            if (auto v = next()) params.disk_outer = std::atof(v);
        } else if (arg("--disk-temp")) {
            if (auto v = next()) params.disk_temperature = std::atof(v);
        } else if (arg("--disk-volumetric")) {
            params.disk_volumetric = 1;
        } else if (arg("--disk-alpha")) {
            if (auto v = next()) params.disk_alpha = std::atof(v);
        } else if (arg("--disk-turbulence")) {
            if (auto v = next()) params.disk_turbulence = std::atof(v);
        } else if (arg("--disk-seed")) {
            if (auto v = next()) params.disk_seed = std::atoi(v);
        } else if (arg("--background")) {
            if (auto v = next()) {
                if (std::strcmp(v, "black") == 0) params.background_type = GRRT_BG_BLACK;
                else if (std::strcmp(v, "stars") == 0) params.background_type = GRRT_BG_STARS;
                else { std::println(stderr, "Unknown background: {}", v); return 1; }
            }
        } else if (arg("--max-steps")) {
            if (auto v = next()) params.integrator_max_steps = std::atoi(v);
        } else if (arg("--tolerance")) {
            if (auto v = next()) params.integrator_tolerance = std::atof(v);
        } else if (arg("--threads")) {
            if (auto v = next()) params.thread_count = std::atoi(v);
        } else if (arg("--backend")) {
            if (auto v = next()) backend_str = v;
        } else if (arg("--validate")) {
            validate = true;
        } else if (arg("--output")) {
            if (auto v = next()) output_name = v;
        } else {
            std::println(stderr, "Unknown argument: {}", argv[i]);
            print_usage();
            return 1;
        }
    }

    // Set backend
    if (backend_str == "cuda") {
        params.backend = GRRT_BACKEND_CUDA;
    } else {
        params.backend = GRRT_BACKEND_CPU;
    }

    // Validation mode: render on both backends, compare, and exit
    if (validate) {
        std::println("gr-raytracer validation mode");
        std::println("============================");

        GRRTParams cpu_params = params;
        cpu_params.backend = GRRT_BACKEND_CPU;
        GRRTContext* cpu_ctx = grrt_create(&cpu_params);
        if (!cpu_ctx) {
            std::println(stderr, "Failed to create CPU context");
            return 1;
        }
        std::vector<float> cpu_fb(params.width * params.height * 4);
        grrt_render(cpu_ctx, cpu_fb.data());

        GRRTParams cuda_params = params;
        cuda_params.backend = GRRT_BACKEND_CUDA;
        GRRTContext* cuda_ctx = grrt_create(&cuda_params);
        if (!cuda_ctx) {
            std::println(stderr, "Failed to create CUDA context: {}",
                         grrt_last_error() ? grrt_last_error() : "unknown");
            grrt_destroy(cpu_ctx);
            return 1;
        }
        std::vector<float> cuda_fb(params.width * params.height * 4);
        grrt_render(cuda_ctx, cuda_fb.data());

        double max_rel_err = 0.0;
        int num_pixels = params.width * params.height;
        int cpu_nonzero = 0, cuda_nonzero = 0, both_nonzero = 0;
        int boundary_mismatches = 0;  // pixels nonzero on one side only

        for (int p = 0; p < num_pixels; ++p) {
            double cpu_lum = cpu_fb[p*4+0] + cpu_fb[p*4+1] + cpu_fb[p*4+2];
            double cuda_lum = cuda_fb[p*4+0] + cuda_fb[p*4+1] + cuda_fb[p*4+2];
            if (cpu_lum > 0.0) cpu_nonzero++;
            if (cuda_lum > 0.0) cuda_nonzero++;
            if (cpu_lum > 0.0 && cuda_lum > 0.0) both_nonzero++;

            // Skip boundary mismatches (one side zero, other nonzero) —
            // these are from FP-divergent geodesics at star tolerance edges
            bool one_zero = (cpu_lum < 1e-10) != (cuda_lum < 1e-10);
            if (one_zero) { boundary_mismatches++; continue; }

            for (int c = 0; c < 3; ++c) {
                double denom = std::max(std::abs(cpu_fb[p*4+c]), std::abs(cuda_fb[p*4+c]));
                if (denom > 1e-10) {
                    double rel = std::abs(cpu_fb[p*4+c] - cuda_fb[p*4+c]) / denom;
                    if (rel > max_rel_err) max_rel_err = rel;
                }
            }
        }

        std::println("Validation results:");
        std::println("  CPU nonzero pixels: {}, CUDA nonzero: {}, both: {}",
                     cpu_nonzero, cuda_nonzero, both_nonzero);
        std::println("  Boundary mismatches (FP edge): {}", boundary_mismatches);
        std::println("  Max relative error (shared pixels): {:.2e}", max_rel_err);

        // CPU/GPU FP divergence causes ~1-2% relative error in disk emission
        // and occasional boundary star mismatches. Both are expected.
        bool pass = max_rel_err < 0.05 && boundary_mismatches < num_pixels / 50;
        std::println("  Result: {}", pass ? "PASS" : "FAIL");

        grrt_destroy(cpu_ctx);
        grrt_destroy(cuda_ctx);
        return pass ? 0 : 1;
    }

    std::string path_png = output_name + ".png";
    std::string path_hdr = output_name + ".hdr";
    std::string path_linear = output_name + "_linear.hdr";

    std::println("gr-raytracer CLI");
    std::println("================");

    GRRTContext* ctx = grrt_create(&params);
    if (!ctx) {
        std::println(stderr, "Failed to create render context");
        return 1;
    }

    // Render to linear HDR
    std::vector<float> framebuffer(params.width * params.height * 4);
    int result = grrt_render(ctx, framebuffer.data());

    if (result == 0) {
        // 1. Raw linear HDR (for Blender / programmatic use)
        {
            std::vector<float> hdr_rgb(params.width * params.height * 3);
            for (int i = 0; i < params.width * params.height; ++i) {
                hdr_rgb[i * 3 + 0] = framebuffer[i * 4 + 0];
                hdr_rgb[i * 3 + 1] = framebuffer[i * 4 + 1];
                hdr_rgb[i * 3 + 2] = framebuffer[i * 4 + 2];
            }
            stbi_write_hdr(path_linear.c_str(), params.width, params.height, 3, hdr_rgb.data());
            std::println("Saved {}", path_linear);
        }

        // 2. Normalized HDR (for darktable / post-processing)
        {
            double log_sum = 0.0;
            int lit = 0;
            for (int i = 0; i < params.width * params.height; ++i) {
                int idx = i * 4;
                double L = 0.2126 * framebuffer[idx] + 0.7152 * framebuffer[idx+1]
                         + 0.0722 * framebuffer[idx+2];
                if (L > 1e-10) {
                    log_sum += std::log(1e-6 + L);
                    ++lit;
                }
            }
            double L_avg = (lit > 0) ? std::exp(log_sum / lit) : 1.0;
            float scale = static_cast<float>(0.18 / L_avg);

            std::vector<float> hdr_rgb(params.width * params.height * 3);
            for (int i = 0; i < params.width * params.height; ++i) {
                hdr_rgb[i * 3 + 0] = framebuffer[i * 4 + 0] * scale;
                hdr_rgb[i * 3 + 1] = framebuffer[i * 4 + 1] * scale;
                hdr_rgb[i * 3 + 2] = framebuffer[i * 4 + 2] * scale;
            }
            stbi_write_hdr(path_hdr.c_str(), params.width, params.height, 3, hdr_rgb.data());
            std::println("Saved {}", path_hdr);
        }

        // 3. Tone-mapped PNG (quick preview)
        grrt_tonemap(framebuffer.data(), params.width, params.height);

        std::vector<unsigned char> pixels(params.width * params.height * 4);
        for (int i = 0; i < params.width * params.height * 4; ++i) {
            float v = framebuffer[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            pixels[i] = static_cast<unsigned char>(v * 255.0f);
        }

        stbi_write_png(path_png.c_str(), params.width, params.height, 4,
                       pixels.data(), params.width * 4);
        std::println("Saved {}", path_png);
    } else {
        std::println(stderr, "Render failed: {}", grrt_error(ctx));
    }

    grrt_destroy(ctx);
    return result;
}
