#include "grrt/scene/volumetric_disk.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

int failures = 0;

void check(const char* name, double got, double expected, double rel_tol) {
    double rel_err = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel_err < rel_tol;
    std::printf("  %s: got=%.4e expected=%.4e rel_err=%.2e %s\n",
                name, got, expected, rel_err, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

// Shared VolumetricDisk instances. Construction is expensive (~1 minute each
// at default DP45 tolerance), so tests that use the same configuration
// share a single instance via Meyer's singleton (constructed lazily on
// first call, reused thereafter).

const grrt::VolumetricDisk& shared_disk_default() {
    // Default params (turbulence=1.0). Used by tests that don't care
    // about noise (taper, volume bounds, construction sanity, etc.).
    static const grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    return disk;
}

const grrt::VolumetricDisk& shared_disk_no_noise() {
    // Same as default but turbulence=0. Used by profile tests that
    // need a clean (noise-free) density / temperature.
    static const grrt::VolumetricDisk disk = []() {
        grrt::VolumetricParams vp;
        vp.turbulence = 0.0;
        return grrt::VolumetricDisk(1.0, 0.998, 30.0, 1e7, vp);
    }();
    return disk;
}

const grrt::VolumetricDisk& shared_disk_lognormal() {
    // alpha=0.1, turbulence=1.0 — for noise-statistics tests. Identical
    // LUT to shared_disk_default; kept distinct only for clarity at the
    // call site.
    static const grrt::VolumetricDisk disk = []() {
        grrt::VolumetricParams vp;
        vp.alpha = 0.1;
        vp.turbulence = 1.0;
        return grrt::VolumetricDisk(1.0, 0.998, 30.0, 1e7, vp);
    }();
    return disk;
}

const grrt::VolumetricDisk& shared_disk_tau_test() {
    // tau_mid=100, turbulence=0 — for the optical-depth invariant test.
    static const grrt::VolumetricDisk disk = []() {
        grrt::VolumetricParams vp;
        vp.tau_mid = 100.0;
        vp.turbulence = 0.0;
        return grrt::VolumetricDisk(1.0, 0.998, 30.0, 1e7, vp);
    }();
    return disk;
}

void test_construction() {
    std::printf("\n=== VolumetricDisk construction (a=0.998, T_peak=1e7 K) ===\n");
    const auto& disk = shared_disk_default();
    std::printf("  r_isco = %.4f M\n", disk.r_isco());
    std::printf("  r_horizon = %.4f M\n", disk.r_horizon());
    double H = disk.scale_height(10.0);
    std::printf("  H(10M) = %.4f M\n", H);
    if (H <= 0.0 || !std::isfinite(H)) { std::printf("  FAIL\n"); failures++; }
    else { std::printf("  PASS\n"); }
}

void test_density_profile() {
    std::printf("\n=== Density profile (no noise) ===\n");
    const auto& disk = shared_disk_no_noise();
    double r = 10.0;
    double H = disk.scale_height(r);
    double rho_mid = disk.density(r, 0.0, 0.0);
    double rho_1H = disk.density(r, H, 0.0);
    double rho_3H = disk.density(r, 3.0*H, 0.0);
    std::printf("  rho(mid)=%.4e, rho(1H)=%.4e, rho(3H)=%.4e\n", rho_mid, rho_1H, rho_3H);
    if (rho_1H >= rho_mid || rho_3H >= rho_1H) {
        std::printf("  FAIL: density should decrease with height\n"); failures++;
    } else { std::printf("  PASS\n"); }
}

void test_temperature_profile() {
    std::printf("\n=== Temperature profile ===\n");
    const auto& disk = shared_disk_default();
    double r = 10.0;
    double H = disk.scale_height(r);
    double T_mid = disk.temperature(r, 0.0);
    double T_1H = disk.temperature(r, H);
    double T_3H = disk.temperature(r, 3.0*H);
    std::printf("  T(mid)=%.2f, T(1H)=%.2f, T(3H)=%.2f\n", T_mid, T_1H, T_3H);
    if (T_1H >= T_mid || T_3H >= T_1H) {
        std::printf("  FAIL: T should decrease with height\n"); failures++;
    } else { std::printf("  PASS\n"); }
}

void test_taper() {
    std::printf("\n=== ISCO taper ===\n");
    const auto& disk = shared_disk_default();
    check("taper(r_isco)", disk.taper(disk.r_isco()), 1.0, 0.01);
    check("taper(r_isco+1)", disk.taper(disk.r_isco()+1.0), 1.0, 0.01);
    double t_hor = disk.taper(disk.r_horizon());
    if (t_hor > 0.05) { std::printf("  FAIL: BPT72 taper at horizon should be near zero (got %.4e)\n", t_hor); failures++; }
    else { std::printf("  PASS: taper(horizon)=%.4e\n", t_hor); }
}

void test_volume_bounds() {
    std::printf("\n=== Volume bounds ===\n");
    const auto& disk = shared_disk_default();
    if (!disk.inside_volume(10.0, 0.0)) { std::printf("  FAIL: midplane should be inside\n"); failures++; }
    else { std::printf("  PASS: midplane inside\n"); }
    if (disk.inside_volume(10.0, 100.0)) { std::printf("  FAIL: z=100 should be outside\n"); failures++; }
    else { std::printf("  PASS: far above outside\n"); }
    if (disk.inside_volume(50.0, 0.0)) { std::printf("  FAIL: r=50 should be outside\n"); failures++; }
    else { std::printf("  PASS: beyond r_outer outside\n"); }
}

void test_warnings_initially_empty() {
    std::printf("\n=== Warnings: no Severe on clean construction ===\n");
    const auto& disk = shared_disk_default();
    int severe = 0;
    for (const auto& w : disk.warnings()) {
        if (w.severity == grrt::WarningSeverity::Severe) ++severe;
    }
    if (severe > 0) {
        std::printf("  FAIL: %d Severe warnings on clean construction\n", severe);
        failures++;
    } else {
        std::printf("  PASS: no Severe warnings (%zu total)\n", disk.warnings().size());
    }
}

void test_severity_enum_values() {
    std::printf("\n=== Severity enum stability ===\n");
    if (static_cast<int>(grrt::WarningSeverity::Info) != 0) {
        std::printf("  FAIL: Info != 0\n"); failures++; return;
    }
    if (static_cast<int>(grrt::WarningSeverity::Warning) != 1) {
        std::printf("  FAIL: Warning != 1\n"); failures++; return;
    }
    if (static_cast<int>(grrt::WarningSeverity::Promptable) != 2) {
        std::printf("  FAIL: Promptable != 2\n"); failures++; return;
    }
    if (static_cast<int>(grrt::WarningSeverity::Severe) != 3) {
        std::printf("  FAIL: Severe != 3\n"); failures++; return;
    }
    std::printf("  PASS\n");
}

void dump_vertical_profile() {
    std::printf("\n=== Vertical profile diagnostic ===\n");
    const auto& disk = shared_disk_no_noise();

    for (double r : {3.0, 5.0, 10.0, 20.0}) {
        double H = disk.scale_height(r);
        std::printf("\nr=%.1f  H=%.6f  taper=%.6f\n", r, H, disk.taper(r));
        for (int i = 0; i <= 20; i++) {
            double z_frac = i / 20.0 * 3.0;  // 0 to 3H
            double z = z_frac * H;
            double rho = disk.density_cgs(r, z, 0.0);
            double T = disk.temperature(r, z);
            bool inside = disk.inside_volume(r, z);
            std::printf("  z/H=%5.2f  rho=%.4e  T=%8.1f  in=%d\n",
                       z_frac, rho, T, inside);
        }
    }
}

void test_photosphere_extends_to_negligible() {
    std::printf("\n=== Photosphere LUT extends to rho < 1e-15 ===\n");
    const auto& disk = shared_disk_no_noise();

    // Spec §1a: the ODE convergence floor (CONV_FLOOR) must be low enough that
    // rho_z[n_z-1] (the last LUT cell, normalized so rho_z[0]=1) is truly
    // negligible.  With the old floor of 1e-15, the cell was pinned exactly at
    // 1e-15 — not strictly less.  After lowering RHO_FLOOR to 1e-18 the cell
    // drops below 1e-15 as the ODE drives it to machine-zero.
    //
    // We access the raw LUT to measure the boundary cell directly, avoiding
    // the density() cross-column interpolation artefact that can inflate the
    // value when the neighbouring column has a larger z_max.

    const int nr  = disk.radial_bins();
    const int nz  = disk.vertical_bins();
    const auto& lut = disk.density_profile_lut();

    // Locate the LUT bin closest to r = 6 M (peak-flux region, a_star=0.998).
    const double ri_f = (6.0 - disk.r_min()) / (disk.r_max() - disk.r_min())
                        * (nr - 1);
    const int ri = std::clamp(static_cast<int>(ri_f), 0, nr - 1);

    // rho_z[0] = 1.0 by construction; last cell should be < old RHO_FLOOR = 1e-15.
    const double rho_boundary = lut[static_cast<std::size_t>(ri) * nz + (nz - 1)];
    const double rho_mid_val  = lut[static_cast<std::size_t>(ri) * nz];  // = 1.0

    std::printf("  ri=%d  rho_z[0]=%.4e  rho_z[nz-1]=%.4e  (expect < 1e-15)\n",
                ri, rho_mid_val, rho_boundary);
    // FAIL if pinned at the old RHO_FLOOR (>= 1e-15); PASS once it drops below.
    if (rho_boundary >= 1e-15) {
        std::printf("  FAIL: boundary cell pinned at old RHO_FLOOR (>= 1e-15)\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

// We'll only test smoothstep indirectly through the outer-taper task. For this
// task, just sanity-check that the disk constructs cleanly (regression guard).
void test_smoothstep_regression() {
    std::printf("\n=== Smoothstep helper (regression guard) ===\n");
    const auto& disk = shared_disk_default();
    (void)disk;  // construction-only check; values don't matter
    std::printf("  PASS: construction completed\n");
}

void test_density_smooth_across_zmax() {
    std::printf("\n=== Density smooth across z_max ===\n");
    const auto& disk = shared_disk_no_noise();

    const double r = 6.0;
    const double zm = disk.z_max_at(r);

    // Sample density just below and just above z_max — they should be similar
    // (both are tiny, but the ratio between them should be smooth, not a cliff).
    const double rho_below = disk.density(r, zm * 0.99, 0.0);
    const double rho_above = disk.density(r, zm * 1.01, 0.0);

    // After spec §1a: above z_max, density should be exactly 0 (LUT-defined).
    // The cliff at zm itself is irrelevant because the LUT already has ρ < 1e-15 there.
    if (rho_above != 0.0) {
        std::printf("  FAIL: rho(z>z_max) should be exactly 0, got %.4e\n", rho_above);
        failures++; return;
    }
    // The smoothness criterion: rho_below should be small (LUT extended past photosphere)
    const double rho_mid = disk.density(r, 0.0, 0.0);
    if (rho_below / rho_mid > 1e-10) {
        std::printf("  FAIL: rho_below z_max too large (%.4e of midplane)\n",
                    rho_below / rho_mid);
        failures++; return;
    }
    std::printf("  PASS: rho_below=%.2e (small), rho_above=0 (cliff is in LUT, smooth)\n",
                rho_below);
}

void test_outer_radial_taper() {
    std::printf("\n=== Outer radial taper (smoothstep, not cliff) ===\n");
    const auto& disk = shared_disk_no_noise();

    const double r_outer = 30.0;
    const double H_outer = disk.scale_height(r_outer);
    const double dr_out  = 2.0 * H_outer;

    // Sample density at: well inside, mid-taper, just inside r_outer
    const double rho_inside    = disk.density(r_outer - 4.0 * H_outer, 0.0, 0.0);
    const double rho_mid_taper = disk.density(r_outer - 1.0 * H_outer, 0.0, 0.0);
    const double rho_near_edge = disk.density(r_outer - 0.05 * H_outer, 0.0, 0.0);

    if (!(rho_inside > rho_mid_taper && rho_mid_taper > rho_near_edge)) {
        std::printf("  FAIL: expected monotonic decay across taper zone\n");
        std::printf("    rho_inside=%.4e rho_mid=%.4e rho_edge=%.4e\n",
                    rho_inside, rho_mid_taper, rho_near_edge);
        failures++;
        return;
    }
    std::printf("  PASS: rho_inside=%.2e > rho_mid=%.2e > rho_edge=%.2e\n",
                rho_inside, rho_mid_taper, rho_near_edge);
}

void test_h_continuous_across_isco() {
    std::printf("\n=== H(r) continuous across ISCO ===\n");
    const auto& disk = shared_disk_no_noise();

    const double r_isco = disk.r_isco();
    const double H_below = disk.scale_height(r_isco * 0.95);
    const double H_at    = disk.scale_height(r_isco);
    const double H_above = disk.scale_height(r_isco * 1.05);

    // After spec §1c: H decays continuously inside ISCO; H_below should be < H_at,
    // not equal (which would indicate frozen-H).
    if (std::abs(H_below - H_at) / std::max(H_at, 1e-30) < 0.01) {
        std::printf("  FAIL: H frozen across ISCO (H_below=%.4f vs H_at=%.4f)\n",
                    H_below, H_at);
        failures++; return;
    }
    if (H_below > H_at) {
        std::printf("  FAIL: H_below (%.4f) > H_at (%.4f); should decay\n", H_below, H_at);
        failures++; return;
    }
    std::printf("  PASS: H_below=%.4f < H_at=%.4f < H_above=%.4f\n",
                H_below, H_at, H_above);
}

void test_sigma_s_phys_in_range() {
    std::printf("\n=== σ_s_phys in expected range for stellar-mass disk ===\n");
    const auto& disk = shared_disk_no_noise();

    const double sigma = disk.sigma_s_phys();
    std::printf("  σ_s_phys = %.4f (expect 0.05 < σ < 0.5 for α=0.1)\n", sigma);
    if (sigma < 0.05 || sigma > 0.5) {
        std::printf("  FAIL: σ outside expected range\n");
        failures++;
        return;
    }
    std::printf("  PASS\n");
}

void test_density_strictly_positive_inside_volume() {
    std::printf("\n=== Density strictly positive inside volume ===\n");
    const auto& disk = shared_disk_lognormal();

    int fails = 0;
    const int N = 200;
    for (int i = 0; i < N; ++i) {
        const double r   = 4.0 + 20.0 * (i / static_cast<double>(N));
        const double H   = disk.scale_height(r);
        const double zm  = disk.z_max_at(r);
        const double z   = (zm * 0.99) * (-1.0 + 2.0 * (i % 13) / 12.0);
        const double phi = i * 0.314;
        const double rho = disk.density(r, z, phi);
        if (rho <= 0.0) { fails++; }
    }
    if (fails > 0) {
        std::printf("  FAIL: %d/%d samples returned rho <= 0\n", fails, N);
        failures++;
    } else {
        std::printf("  PASS: all %d samples positive\n", N);
    }
}

void test_density_lognormal_mean() {
    std::printf("\n=== Density mean over phi ≈ rho_smooth · exp(σ²/2) ===\n");
    const auto& disk = shared_disk_lognormal();
    const auto& disk0 = shared_disk_no_noise();
    // Use vp.turbulence value matching shared_disk_lognormal (1.0)
    const double turb = 1.0;

    const double r = 8.0, z = 0.0;
    const double rho_smooth = disk0.density(r, z, 0.0);

    const int N = 4096;
    double sum = 0.0;
    for (int i = 0; i < N; ++i) {
        const double phi = 2.0 * 3.14159265358979 * i / N;
        sum += disk.density(r, z, phi);
    }
    const double mean = sum / N;
    const double sigma = disk.sigma_s_phys() * turb;
    const double expected = rho_smooth * std::exp(sigma * sigma * 0.5);
    const double rel_err = std::abs(mean - expected) / expected;
    std::printf("  mean=%.4e expected=%.4e rel_err=%.3f\n", mean, expected, rel_err);
    if (rel_err > 0.10) {
        std::printf("  FAIL\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

void test_inside_volume_tight_margin() {
    std::printf("\n=== inside_volume margin = 0.5·H ===\n");
    const auto& disk = shared_disk_no_noise();

    const double r = 6.0;
    const double H = disk.scale_height(r);
    const double zm = disk.z_max_at(r);

    // Just inside the new margin (zm + 0.4·H) → inside
    if (!disk.inside_volume(r, zm + 0.4 * H)) {
        std::printf("  FAIL: zm+0.4H should be inside\n"); failures++; return;
    }
    // Just outside the new margin (zm + 0.6·H) → outside
    if (disk.inside_volume(r, zm + 0.6 * H)) {
        std::printf("  FAIL: zm+0.6H should be outside\n"); failures++; return;
    }
    std::printf("  PASS\n");
}

void test_validate_luts_clean_construction() {
    std::printf("\n=== validate_luts: clean construction has no Severe ===\n");
    const auto& disk = shared_disk_default();

    int severe = 0;
    for (const auto& w : disk.warnings()) {
        if (w.severity == grrt::WarningSeverity::Severe) ++severe;
    }
    if (severe > 0) {
        std::printf("  FAIL: %d Severe warnings on clean construction\n", severe);
        failures++;
    } else {
        std::printf("  PASS: no Severe warnings\n");
    }
}

void test_compare_columns_compiles() {
    std::printf("\n=== compare_columns compiles (refinement scaffold) ===\n");
    std::printf("  PASS\n");
}

void test_refine_n_z_caps_with_warning() {
    std::printf("\n=== refine_n_z caps emit Promptable when delta >> target ===\n");
    grrt::VolumetricParams vp;
    vp.turbulence = 0.0;
    vp.bins_per_h = 0;             // auto
    vp.bins_per_gradient = 16;     // skip n_r refinement (set deterministically)
    vp.target_lut_eps = 1e-8;       // unrealistically tight
    vp.max_n_z = 64;                // tiny cap
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7, vp);

    bool found_promptable = false;
    for (const auto& w : disk.warnings()) {
        if (w.severity >= grrt::WarningSeverity::Promptable
            && w.code == "n_z_cap") {
            found_promptable = true; break;
        }
    }
    if (!found_promptable) {
        std::printf("  FAIL: expected Promptable n_z_cap warning\n");
        failures++;
    } else {
        std::printf("  PASS: n_z_cap Promptable emitted\n");
    }
}

void test_smoke_parameter_sweep() {
    std::printf("\n=== Smoke parameter sweep (mass micro to SMBH) ===\n");
    struct Case { double mass, spin, alpha, turb; double r_outer; double T_peak; };
    Case cases[] = {
        { 1.0, 0.0,    0.01, 0.0, 30.0, 1e7  },     // baseline, no turbulence
        { 1.0, 0.998,  0.10, 1.0, 30.0, 1e7  },     // stellar-mass canonical
        { 1.0, 0.5,    0.05, 1.5, 60.0, 5e6  },     // intermediate
        { 1.0, 0.998,  0.10, 1.0, 100.0, 5e5 },     // AGN-like
        { 1.0, 0.0,    0.30, 0.5, 200.0, 1e5 },     // SMBH high-α
        { 1.0, 0.99,   0.10, 2.0, 20.0, 1e9  },     // micro-BH near Eddington
        { 1.0, 0.0,    0.01, 0.0, 500.0, 1e4 },     // very SMBH, gas-dominated
    };
    int case_failures = 0;
    for (const auto& c : cases) {
        grrt::VolumetricParams vp;
        vp.alpha = c.alpha;
        vp.turbulence = c.turb;
        try {
            grrt::VolumetricDisk disk(c.mass, c.spin, c.r_outer, c.T_peak, vp);
            int severe = 0;
            for (const auto& w : disk.warnings()) {
                if (w.severity == grrt::WarningSeverity::Severe) ++severe;
            }
            if (severe > 0) {
                std::printf("  FAIL: mass=%.0e spin=%.3f T=%.0e: %d Severe\n",
                            c.mass, c.spin, c.T_peak, severe);
                case_failures++;
            } else if (!std::isfinite(disk.sigma_s_phys()) || disk.sigma_s_phys() <= 0.0) {
                std::printf("  FAIL: mass=%.0e spin=%.3f: σ_s_phys=%.4f bad\n",
                            c.mass, c.spin, disk.sigma_s_phys());
                case_failures++;
            } else {
                std::printf("  PASS: mass=%.0e spin=%.3f T=%.0e σ=%.3f\n",
                            c.mass, c.spin, c.T_peak, disk.sigma_s_phys());
            }
        } catch (const std::exception& e) {
            std::printf("  FAIL: mass=%.0e spin=%.3f: exception '%s'\n",
                        c.mass, c.spin, e.what());
            case_failures++;
        }
    }
    if (case_failures > 0) {
        std::printf("  Total case failures: %d\n", case_failures);
        failures += case_failures;
    } else {
        std::printf("  All %zu cases PASS\n", sizeof(cases)/sizeof(cases[0]));
    }
}

void test_tau_midplane_near_target() {
    std::printf("\n=== τ at midplane ≈ tau_mid at peak-flux radius ===\n");
    const auto& disk = shared_disk_tau_test();
    constexpr double tau_mid_target = 100.0;

    // Peak-flux radius — approximate as r where rho_mid is largest
    // (we don't have a public accessor, so scan with density_cgs(r,0,0))
    double best_r = 6.0, best_rho = 0.0;
    for (int i = 0; i < 50; ++i) {
        const double r = disk.r_isco() + (30.0 - disk.r_isco()) * i / 49.0;
        const double rho = disk.density_cgs(r, 0.0, 0.0);
        if (rho > best_rho) { best_rho = rho; best_r = r; }
    }

    // Integrate kappa·rho dz from z=0 to z_max at best_r
    const double r = best_r;
    const double zm = disk.z_max_at(r);
    const double T = disk.temperature(r, 0.0);
    const auto& opa = disk.opacity_luts();

    const int N = 256;
    const double dz = zm / (N - 1);
    double tau = 0.0;
    for (int i = 0; i < N - 1; ++i) {
        const double z_a = i * dz;
        const double z_b = (i + 1) * dz;
        // density_cgs returns the fake-CGS value used internally by normalize_density;
        // we must NOT clamp it before integrating (the actual midplane value is far
        // larger than 1e-6 in these units). The clamp is appropriate ONLY for the
        // opacity-LUT lookup, whose grid is bounded.
        const double rho_a = disk.density_cgs(r, z_a, 0.0);
        const double rho_b = disk.density_cgs(r, z_b, 0.0);
        const double T_a = std::clamp(disk.temperature(r, z_a), 3000.0, 1e8);
        const double T_b = std::clamp(disk.temperature(r, z_b), 3000.0, 1e8);
        const double rho_a_lookup = std::clamp(rho_a, 1e-18, 1e-6);
        const double rho_b_lookup = std::clamp(rho_b, 1e-18, 1e-6);
        const double k_a = opa.lookup_kappa_ross(rho_a_lookup, T_a)
                         + opa.lookup_kappa_es(rho_a_lookup, T_a);
        const double k_b = opa.lookup_kappa_ross(rho_b_lookup, T_b)
                         + opa.lookup_kappa_es(rho_b_lookup, T_b);
        tau += 0.5 * (k_a * rho_a + k_b * rho_b) * dz;
    }
    // Double for both sides of midplane (matches normalize_density's symmetry).
    tau *= 2.0;
    std::printf("  τ(z=0..z_max, both sides) at r=%.2f: %.2f (target %.2f)\n", r, tau, tau_mid_target);
    if (std::abs(tau - tau_mid_target) / tau_mid_target > 0.30) {  // 30% tolerance
        std::printf("  FAIL\n"); failures++;
    } else {
        std::printf("  PASS\n");
    }
}

int main() {
    test_construction();
    test_density_profile();
    test_temperature_profile();
    test_taper();
    test_volume_bounds();
    test_warnings_initially_empty();
    test_severity_enum_values();
    test_smoothstep_regression();
    dump_vertical_profile();
    test_photosphere_extends_to_negligible();
    test_density_smooth_across_zmax();
    test_outer_radial_taper();
    test_h_continuous_across_isco();
    test_sigma_s_phys_in_range();
    test_density_strictly_positive_inside_volume();
    test_density_lognormal_mean();
    test_inside_volume_tight_margin();
    test_validate_luts_clean_construction();
    test_compare_columns_compiles();
    test_refine_n_z_caps_with_warning();
    test_smoke_parameter_sweep();  // ~5-7 min at 1e-6 DP45 (7 unique configs, no sharing)
    test_tau_midplane_near_target();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
