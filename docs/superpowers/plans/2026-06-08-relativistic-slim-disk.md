# Relativistic Slim-Disk Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a relativistic transonic slim-disk subsystem so GRRT renders near-Eddington (`f_Edd ≈ 0.9`) disks — the radiation-pressure-dominated inner disk that the thin-disk `α(total-P)` model cannot solve (Lightman-Eardley fold). Advection both removes the fold and captures the puffed, advective inner disk; the model reduces exactly to Novikov-Thorne at low `Ṁ`.

**Architecture:** A global Newton-**relaxation** transonic radial solve (Kerr; sonic-point regularity as an interior BC with `(ℓ_in, r_s)` as unknowns; seeded from the thin disk and continued up in `Ṁ`) coupled **self-consistently (2D iteration)** with the existing grey per-column vertical BVP (heating reduced by `1/(1+f_adv)`). The slim-disk 4-velocity feeds the raymarch redshift; the sonic point is the structural inner edge.

**Tech stack:** C++23, CMake/MSVC, OpenMP. Reuses the column-BVP Newton/warm-start/homotopy infrastructure (`src/disk_column_bvp.cpp`) and the `OpacityLUTs`. **Verified equations: `docs/superpowers/references/disk-physics-formulas.md` §22 (+ §20, §21, traps #9–12). Spec: `docs/superpowers/specs/2026-06-08-relativistic-slim-disk-design.md`. Check physics there — do not re-derive.**

---

## Context the implementer must know

- **This is a from-scratch transonic solver.** Unlike the BVP-wiring plan (which wired an existing solver), Phase 1 builds the radial solver new. The proven pattern: design the state vector → residual → Jacobian → Newton loop incrementally with a test at each step, exactly as the column BVP (`plans/2026-06-04-disk-column-bvp-solver.md`) was built.
- **All physics is verified in §22** — every equation (mass/momentum/angular-momentum/energy, the sonic-point regularity, the Kerr factors `𝒞,𝒟,ℋ`, `Ω_⊥`, the `f_adv` definition) is there with sources and convention traps. **Reference §22; do not derive from memory.** The four traps (#9 integrated α-stress; #10 transonic eigenvalue / sonic point inside ISCO; #11 `(1+f_adv)^{-1}` convention; #12 `Ṁ_Edd` factor) are load-bearing.
- **Units:** §22 equations are geometric (`G=c=1`, `M`=length unit). CGS quantities (Σ, T, opacity) convert via `r_g` (§1, §19). Keep the radial solver's internal units consistent (recommend geometric for the dynamics, CGS for thermodynamics, converting at the EOS/opacity boundary as the column BVP does).
- **The relaxation reuses the column-BVP toolkit:** `dense_solve` (Gaussian elimination), the scaled-residual merit, the damped line search, and the homotopy-continuation idea (`bootstrap_column`). Mirror them.
- **Workflow constraints (carry forward, non-negotiable):** NEVER run `git commit` — hand the message to the human. Subagents: **sonnet or opus only, never haiku.** Present every reviewer recommendation with a take and WAIT for the human's call before fixing.

## Test-coverage requirements (spin & Eddington)

**Canonical render config: `a ≈ 0.998` (near-extremal Kerr) and `f_Edd ≈ 0.9` (near-Eddington)** — this is what GRRT actually renders, so the solver and integration tests must prove it works *there*, not just on easy cases. Across the convergence (Task 5), continuation (Task 9), benchmark (Task 10), and integration (Phase 6) tasks:
- **Spin:** include a **near-extremal `a = 0.998`** case alongside `a = 0.9` (with `a = 0` as the Schwarzschild / thin-disk anchor). Near-extremal is harder — the ISCO and sonic point sit close to the horizon and frame-dragging is strong, so the solver must be exercised there.
- **Eddington:** cover **`f_Edd ∈ {0.9, 0.95, 1.0}`** (the render regime + the graceful upper edge), plus a low `f_Edd ≤ 0.05` case for the thin-disk reduction. At `f_Edd = 1.0` the grey-diffusion approximation may strain — emit the documented caveat rather than failing.

The easy cases (`a=0`, low `f_Edd`) stay as reduction/anchor checks; the near-extremal + near-Eddington cases are the ones that prove the model works where it's used.

## File structure

- `include/grrt/scene/slim_disk_radial.h` / `src/slim_disk_radial.cpp` — the standalone transonic radial solver (Phase 1).
- `tests/test_slim_disk_radial.cpp` — its tests (`test-slim-disk-radial` target).
- `include/grrt/scene/disk_column_bvp.h` / `src/disk_column_bvp.cpp` — gains the `f_adv` heating factor (Phase 2).
- `include/grrt/scene/slim_disk.h` / `src/slim_disk.cpp` — the 2D coupling driver + `slim_disk_velocity` (Phases 3–4).
- `src/volumetric_disk.cpp` / `.h` — `compute_radial_structure` replaced; LUT honest-CGS switch; wiring (Phase 5).
- `src/geodesic_tracer.cpp` — redshift call-site switch to `slim_disk_velocity` (Phase 4).
- `tests/test_volumetric.cpp`, `tools/dump_disk_lut.cpp` — integration (Phase 6).

## Phase overview

1. **Standalone transonic radial solver** (this plan, full detail) — Kerr radial relaxation, sonic-point regularity, `Ṁ`-continuation; tested vs the thin-disk limit + Sądowski 2011 benchmark. *Largest piece; self-contained and testable without the rest.*
2. **`f_adv` vertical coupling** — `1/(1+f_adv)` heating in the column BVP; energy-closure + hot-column-converges tests.
3. **Self-consistent 2D iteration** — couple radial ↔ vertical; replace the Phase-1 one-zone closure with real vertical moments; fixed-point test.
4. **Kinematics / inner edge** — `slim_disk_velocity`; sonic point as inner edge; supersonic-plunge layer; raymarch redshift switch.
5. **Wire into `VolumetricDisk` + LUT** — replace `compute_radial_structure`; honest-CGS + log-density LUT switch; retire in-domain BPT72; validation.
6. **Integration sweep** — `f_Edd=0.9` construction + edge-on render; thin-disk regression; before/after to the user.

> Phases 2–6 are specified as task roadmaps at the end. They get fleshed to full TDD detail (like Phase 1) as each is reached, because their exact code depends on the realized Phase-1 interface. **Execute Phase 1 first.**

---

# PHASE 1 — Standalone transonic radial slim-disk solver

**Scope note:** Phase 1 uses a **one-zone (height-integrated) vertical closure** (`P, H, c_s, S` from `Σ, T_c` via algebraic relations) so the radial solver is self-contained and testable now. Phase 3 replaces that closure with the real coupled vertical BVP. `f_adv` is computed from the radial entropy gradient throughout.

### Task 1: Interface scaffold + build target

**Files:**
- Create: `include/grrt/scene/slim_disk_radial.h`
- Create: `src/slim_disk_radial.cpp`
- Create: `tests/test_slim_disk_radial.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the header**

```cpp
#ifndef GRRT_SLIM_DISK_RADIAL_H
#define GRRT_SLIM_DISK_RADIAL_H
#include "grrt/color/opacity.h"
#include "grrt_export.h"
#include <vector>
namespace grrt {

/// Inputs for the relativistic transonic slim-disk radial solve.
/// Geometric mechanics (G=c=1, M sets the scale); CGS thermodynamics via r_g.
struct SlimDiskInputs {
    double mass = 1.0;      ///< M (geometric)
    double spin = 0.0;      ///< a, |a|<M
    double mdot = 0.0;      ///< accretion rate Mdot [g/s]
    double alpha = 0.1;     ///< Shakura-Sunyaev viscosity
    double r_g = 0.0;       ///< gravitational radius [cm] (geometric->cm)
    double r_in = 0.0;      ///< inner edge of the grid [M] (>= horizon)
    double r_out = 50.0;    ///< outer edge [M]
    int    n_nodes = 400;
    int    max_iters = 100;
    double tol = 1e-8;
};

/// Converged transonic radial structure. Index 0 = inner edge, back = outer.
struct SlimDiskRadial {
    std::vector<double> r;       ///< radius [M]
    std::vector<double> Sigma;   ///< surface density [g/cm^2]
    std::vector<double> V;       ///< radial velocity (corotating frame), <0 = inflow
    std::vector<double> Omega;   ///< orbital angular velocity [1/s]
    std::vector<double> Tc;      ///< midplane temperature [K]
    std::vector<double> H;       ///< scale height [cm]
    std::vector<double> f_adv;   ///< advected fraction Q_adv/Q_vis
    double ell_in = 0.0;         ///< inner specific angular momentum (eigenvalue)
    double r_sonic = 0.0;        ///< sonic radius [M]
    bool   converged = false;
    int    iters = 0;
    double final_residual = 0.0;
};

/// Solve the relativistic transonic slim-disk radial structure (§22).
GRRT_EXPORT SlimDiskRadial solve_slim_disk_radial(const SlimDiskInputs& in,
                                                  const OpacityLUTs& opacity);
} // namespace grrt
#endif
```

- [ ] **Step 2: Stub the solver** in `src/slim_disk_radial.cpp` (`return SlimDiskRadial{};` with `converged=false`), include the header.

- [ ] **Step 3: Add the CMake target** in `CMakeLists.txt` after `test-column-bvp`:
```cmake
add_executable(test-slim-disk-radial tests/test_slim_disk_radial.cpp)
target_link_libraries(test-slim-disk-radial PRIVATE grrt)
```

- [ ] **Step 4: Scaffold test** (`tests/test_slim_disk_radial.cpp`): a `failures` int, a `check()` helper (copy from `test_column_bvp.cpp`), a `test_links_and_returns()` that constructs `SlimDiskInputs`, calls the solver, and asserts it returns. `main()` calls it.

- [ ] **Step 5: Build & run** `cmake --build build --config Release --target test-slim-disk-radial && ./build/Release/test-slim-disk-radial` → links, runs.

- [ ] **Step 6: Commit** (hand to human): `feat(slim-disk): scaffold transonic radial solver interface`

### Task 2: Kerr relativistic factors

**Files:** Modify `src/slim_disk_radial.cpp` (anonymous-namespace free functions); Test `tests/test_slim_disk_radial.cpp`.

- [ ] **Step 1: Failing test** — assert the §22 Kerr factors at known points:
```cpp
static void test_kerr_factors() {
    using namespace grrt::slim_detail;  // expose via a test hook or friend
    // Schwarzschild (a=0): Omega_K = M^{1/2}/r^{3/2}; C = 1-3M/r; D = 1-2M/r; H = 1.
    check("Omega_K Schw r=10", omega_k(1.0, 0.0, 10.0), std::pow(10.0,-1.5), 1e-12);
    check("C Schw r=6 (ISCO)", calC(1.0,0.0,6.0), 1.0-3.0/6.0, 1e-12);
    check("D Schw r=10", calD(1.0,0.0,10.0), 1.0-2.0/10.0, 1e-12);
    check("H Schw", calH(1.0,0.0,10.0), 1.0, 1e-12);
}
```
(Expose the factor functions to the test via a small `slim_detail` namespace or `*_test` hooks, mirroring the column BVP's test hooks.)

- [ ] **Step 2: Implement** `omega_k(M,a,r)`, `calC`, `calD`, `calH`, the vertical epicyclic `omega_perp2(M,a,r) = (M/r^3)*(calH/calC)`, `Delta`, `A_kerr` — copy the exact forms from §22. Cross-check `omega_perp2` against the existing `VolumetricDisk::omega_z_sq` (they must agree — same physics).

- [ ] **Step 3: Run → pass. Step 4: Commit** `feat(slim-disk): Kerr relativistic factors (verified vs §22)`.

### Task 3: One-zone vertical closure (height-integrated EOS)

**Files:** Modify `src/slim_disk_radial.cpp`; Test.

Provides, given `(Σ, T_c)` and `r`: midplane pressure `p_mid`, sound speed `c_s`, scale height `H = c_s/Ω_⊥`, vertically-integrated pressure `P = (2/√(2π))·... ` (use `P ≈ Σ c_s²` integrated form; precise factor per §22/Sądowski one-zone), midplane density `ρ_mid = Σ/(2H)`, and the specific entropy `S` (gas+radiation) needed for `Q_adv`. Uses the gas+radiation EOS (§10) and `OpacityLUTs::lookup_mu`.

- [ ] **Step 1: Failing test** — gas-dominated limit: `H = c_s,gas/Ω_⊥` with `c_s,gas² = k_B T/(μ m_p)`; check `H`, `ρ_mid = Σ/(2H)`, and `P_gas = ρ_mid c_s,gas²` round-trip for chosen `(Σ, T_c, r)`.
- [ ] **Step 2: Implement** the closure struct/functions; radiation pressure `aT⁴/3` included; entropy `S = S_gas + S_rad` (standard forms — record in §22 if not present). **Trap #9:** `P` is the vertically *integrated* pressure used in the α-stress.
- [ ] **Step 3: Run → pass. Step 4: Commit** `feat(slim-disk): one-zone vertical closure for the radial solve`.

### Task 4: Radial residual (4 conservation laws + regularity BC)

**Files:** Modify `src/slim_disk_radial.cpp`; Test.

State vector `U`: per node `(Σ_i, V_i, ℓ_i, T_{c,i})` × N, plus globals `(ℓ_in, r_s)` → length `4N+2`. Residual rows: the four §22 conservation laws discretized (trapezoidal between nodes, mirroring the column BVP), plus the boundary/regularity rows:
- Outer BC: match the thin-disk/Keplerian state at `r_out` (`Ω→Ω_K`, `V` from mass conservation, `ℓ→ℓ_K`).
- **Regularity at the sonic point:** `𝒩(r_s)=0` AND `𝒟₀(r_s)=0` (§22) — two rows pinning `(ℓ_in, r_s)`.
- Mass conservation closes `Σ`–`V`; angular momentum closes `ℓ`; energy(+advection) closes `T_c`; `f_adv` computed from the entropy gradient.

- [ ] **Step 1: Failing test** — `residual length == 4N+2`, all finite, on a thin-disk seed (Task 8's seed builder).
- [ ] **Step 2: Implement** `slim_radial_residual(U, in, op, R)` using the Task-2 factors and Task-3 closure; the §22 equations. **Traps #10, #11.**
- [ ] **Step 3: Run → pass. Step 4: Commit** `feat(slim-disk): transonic radial residual + sonic-point regularity (§22)`.

### Task 5: Newton relaxation solve

**Files:** Modify `src/slim_disk_radial.cpp`; Test.

Reuse the column-BVP Newton machinery: numerical (and later analytic) Jacobian, `dense_solve`, scaled-residual merit, damped line search. Unknowns include `(ℓ_in, r_s)`.

- [ ] **Step 1: Failing test** — `test_converges_midmdot`: a mid-`Ṁ` case (`f_Edd≈0.3`) converges (`converged==true`, residual < tol) at **both `a=0.9` and the near-extremal `a=0.998`** (the render spin). Harder / near-Eddington cases come via the Task-9 continuation.
- [ ] **Step 2: Implement** the Newton loop (numerical Jacobian first, per the column-BVP build order; analytic Jacobian is a later speed task). Honest fallback on non-convergence (`converged=false`, no fabricated profile).
- [ ] **Step 3: Run → pass. Step 4: Commit** `feat(slim-disk): Newton relaxation solve (numerical Jacobian)`.

### Task 6: Sonic-point regularity test

- [ ] **Step 1:** assert on the converged solution: `𝒩(r_s)→0` and `𝒟₀(r_s)→0` (both below tol), `V(r)` smooth through `r_s` (no kink — bounded `dV/dr`), and `r_s < r_isco` (trap #10). **Step 2: Commit** `test(slim-disk): sonic-point regularity`.

### Task 7: Conservation invariants

- [ ] **Step 1:** on the converged solution assert `Ṁ = −2πΣΔ^½V/√(1−V²)` constant in `r` (within tol); the angular-momentum and energy(+advection) balances hold pointwise. **Step 2: Commit** `test(slim-disk): mass / angular-momentum / energy conservation`.

### Task 8: Thin-disk limit (the superset guarantee)

**Files:** Test + a thin-disk seed builder used by Tasks 4–9.

- [ ] **Step 1:** `test_thin_disk_limit`: at `f_Edd = 0.02`, assert `f_adv(r) → 0` (max `< 1e-2`) and `Σ(r), Ω(r), T_eff(r) = (Q_rad/σ)^¼` match the **existing Novikov-Thorne** result (`VolumetricDisk::build_flux_lut` / `compute_radial_structure` at the same `M,a,Ṁ`) to a few %.
- [ ] **Step 2:** implement the NT-seed builder (`build_thin_disk_seed`) — the analytic thin-disk `Σ,V,ℓ,T_c` used as the relaxation seed.
- [ ] **Step 3: Run → pass. Step 4: Commit** `test(slim-disk): reduces to Novikov-Thorne at low Mdot`.

### Task 9: Ṁ-continuation to f_Edd = 0.9

- [ ] **Step 1:** `test_mdot_continuation`: seed at `f_Edd=0.02` (thin-disk seed), then **continue up** in `Ṁ` (warm-starting each step from the last converged solution) through **`f_Edd ∈ {0.9, 0.95, 1.0}`**; assert convergence at every step and at each target (`converged`, `f_adv` significant and growing with `f_Edd`, `H/r` elevated ~0.1–0.2 at 0.9 and higher toward 1.0). Run the continuation for **both `a=0.9` and the near-extremal `a=0.998`** (the render spin). Mirrors `bootstrap_column`'s homotopy. At `f_Edd=1.0` (the graceful edge) emit the grey-diffusion caveat if it strains, rather than failing; above 1.0 is out of scope.
- [ ] **Step 2:** implement the continuation driver (a thin wrapper that ramps `mdot` with warm starts). **Step 3: Run → pass. Step 4: Commit** `feat(slim-disk): Mdot-continuation seeding from the thin disk`.

### Task 10: Literature benchmark

- [ ] **Step 1:** `test_sadowski_benchmark`: for published Sądowski 2009/2011 cases (`M=10 M_sun, α=0.1, Ṁ` near Eddington), assert `H/r(r)`, `f_adv(r)`, `Σ(r)` match the paper's figures within a stated tolerance (digitize 3–4 points; record source + values in the test comment). **Include a high-spin case (`a≈0.9` or `a≈0.998`, whichever the paper provides figures for)** in addition to `a=0`, since the render spin is near-extremal. This is the external-accuracy check.
- [ ] **Step 2: Commit** `test(slim-disk): benchmark vs Sadowski 2011`.

**Phase 1 closeout:** full `test-slim-disk-radial` green; sonic regularity + conservation + thin-disk limit + `f_Edd=0.9` continuation + literature benchmark all pass. Dispatch a code-quality review over the phase before moving on.

---

# PHASES 2–6 — task roadmaps (detailed to full TDD when reached)

### Phase 2 — `f_adv` vertical coupling
- **2.1** Add an `f_adv` field to `ColumnInputs`; in `node_deriv`, divide the viscous-heating term by `(1+f_adv)` (trap #11). Default `f_adv=0` → existing behavior unchanged (regression: `test-column-bvp` still green).
- **2.2** Re-derive the affected analytic-Jacobian block (the `dQ` row gains the `1/(1+f_adv)` factor); cross-check stays exact.
- **2.3** Test: energy closes to `(1−f_adv)·Q_vis` per face; a hot radiation-dominated column with a realistic `f_adv` (~0.2) **converges at the strict gate** — the partial of the fold fix.
- **Files:** `disk_column_bvp.{h,cpp}`, `test_column_bvp.cpp`. **Interface:** `ColumnInputs.f_adv`.

### Phase 3 — self-consistent 2D iteration
- **3.1** New `src/slim_disk.cpp`: the driver coupling `solve_slim_disk_radial` ↔ the per-column vertical BVP. Replace Phase-1's one-zone closure with the real vertical moments (the column returns pressure/density moments → radial EOS coefficients).
- **3.2** Outer fixed-point loop: relax radial → solve columns (warm-started march + homotopy) → update moments → repeat until `Σ,H,f_adv` converge. Damp if needed.
- **3.3** Test: 2D fixed point converges; the `f_Edd=0.9` disk's full structure is self-consistent (radial `f_adv` matches the columns' realized advected fraction).
- **Files:** `slim_disk.{h,cpp}`, `test_slim_disk.cpp`.

### Phase 4 — kinematics / inner edge
- **4.1** `slim_disk_velocity(r, ut, ur, uphi)`: build the 4-velocity from `V(r), Ω(r)` + the Kerr metric. Test: `u·u=−1`; `u^r<0`; reduces to `circular_velocity` (`u^r→0`) as `f_Edd→0`.
- **4.2** Sonic point as the structural inner edge; a thin supersonic-plunge layer (sonic point → horizon) with velocity continuity (slim-disk `u^μ` → geodesic plunge).
- **4.3** Switch the raymarch redshift call sites (`geodesic_tracer.cpp` `sample_integrand`, `raymarch_volumetric_spectral`) to `slim_disk_velocity` in the slim-disk domain — **additive** (keep `circular_velocity`/`plunging_velocity` for the thin-disk/fallback path). Stepping logic untouched.
- **Files:** `slim_disk.{h,cpp}`, `geodesic_tracer.cpp`.

### Phase 5 — wire into `VolumetricDisk` + LUT
- **5.1** Replace `compute_radial_structure` with the slim-disk 2D solve producing the radial inputs; `compute_vertical_profiles` consumes `f_adv(r)`.
- **5.2** Land the deferred **honest-CGS + log-density LUT switch** (BVP-wiring plan Task 4): `rho_scale_≡1`, absolute `ρ_mid`, log-density encoding, the `r_g` optical-depth factor in both raymarch integrands. (The BVP-wiring plan's Tasks 4–9 are superseded by Phases 5–6 here.)
- **5.3** Retire the in-domain BPT72 plunging path; validation: Toomre Q, opacity-range guard, the Promptable-truncate / Severe-hole fallback policy.
- **Files:** `volumetric_disk.{h,cpp}`, `geodesic_tracer.cpp`, `test_volumetric.cpp`.

### Phase 6 — integration sweep
- **6.1** Construction at the **render config (`a=0.998`, `f_Edd=0.9`)** and across **`f_Edd ∈ {0.9, 0.95, 1.0}`**: converged, non-collapsed, thick inner disk; `test_hot_inner_disk_columns_converge` passes at the strict gate (the success signal flips green). `f_Edd=1.0` may emit the grey-diffusion caveat (graceful edge).
- **6.2** Edge-on render at the render config (`--spin 0.998 --observer-theta 80 --fov 30 --eddington-fraction 0.9 --samples 30`): band-free, puffed inner disk, radial-infall Doppler visible. Surface before/after to the human.
- **6.3** Thin-disk regression: a low-`f_Edd` render matches the pre-slim-disk thin-disk render (superset guarantee). Construction-time check (cached). `dump-disk-lut` health.
- **Files:** `test_volumetric.cpp`, `tools/dump_disk_lut.cpp`.

---

## Self-review (author)
- **Spec coverage:** §4 architecture → Phases 1/3/4; §6 physics → §22 (verified) referenced throughout; §7 relaxation (locked) → Phase 1 Tasks 4–5,9; §8 interface → Phases 4–5; §9 testing → Tasks 6–10 + Phases 3/6; §12 phasing → the six phases. The thin-disk superset (§2/§11) → Tasks 8 + Phase 6.3.
- **Convention traps** (§22 #9–12) are called out at the tasks that touch them (3, 4, Phase 2).
- **Type consistency:** `SlimDiskInputs`/`SlimDiskRadial`, `solve_slim_disk_radial`, `ColumnInputs.f_adv`, `slim_disk_velocity` used consistently across phases.
- **Granularity note:** Phases 2–6 are roadmaps by design (their code depends on Phase 1's realized interface); each is elaborated to full TDD detail when reached. Phase 1 is execution-ready.
