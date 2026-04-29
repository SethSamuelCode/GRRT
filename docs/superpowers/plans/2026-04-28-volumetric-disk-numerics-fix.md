# Volumetric Disk Numerics Fix — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 3 remaining failing tests in `test-volumetric` and reduce refinement cap-binding by widening the plunging-region taper and fixing a units mismatch in the tau test.

**Architecture:** Three independent surgical changes:
1. Widen the BPT-style plunging-region taper from `(r_isco - r_horizon)/3` to the full plunging region width (factor=1.0), exposed as a new `plunging_taper_width_factor` parameter.
2. Plumb the new field through `VolumetricParams`, `GRRTParams`, `api.cpp`, and `cli/main.cpp` as `--disk-plunging-taper-width F`.
3. Fix the tau-midplane test to use `density_cgs()` instead of `density()` (the latter returns geometric-scaled units).

After landing, measure refinement convergence behavior (Phase 1 checkpoint) to decide if a Phase 2 (adaptive RK45 in `solve_column`) follow-up is needed.

**Tech Stack:** C++23, CMake, OpenMP, plain C++ test executables (no external test framework), Windows VS2022 + Linux GCC/Clang.

**Reference spec:** `docs/superpowers/specs/2026-04-28-volumetric-disk-numerics-fix-design.md` — read before starting.

**Build commands** used throughout:
- Configure (Windows VS2022, first time only): `cmake -B build -G "Visual Studio 17 2022"`
- Build: `cmake --build build --config Release`
- Run tests (Windows): `./build/Release/test-volumetric`
- Run tests (Linux): substitute `./build/test-volumetric`

The plan below uses the Windows form; substitute as needed for Linux.

---

## File Structure

Files modified, each with one clear responsibility:

| File | Responsibility |
|------|----------------|
| `include/grrt/scene/volumetric_disk.h` | Type definitions for the C++ volumetric disk model |
| `include/grrt/types.h` | C-API parameter struct (DLL boundary) |
| `src/volumetric_disk.cpp` | VolumetricDisk constructor, in particular the line that sets `taper_width_` |
| `src/api.cpp` | C-API → C++ parameter mapping in `grrt_create` |
| `cli/main.cpp` | CLI argument parsing and `print_usage` text |
| `tests/test_volumetric.cpp` | Test functions, in particular `test_tau_midplane_near_target` |

No new files. No new abstractions. ~15 lines total across 6 files, distributed across 3 commits + 1 measurement checkpoint.

---

## Task 1: Add `plunging_taper_width_factor` parameter and apply in constructor

**Goal:** Make the plunging-region taper width configurable and default to spanning the full plunging region. This is the core fix that resolves `test_density_profile` and `test_density_smooth_across_zmax`.

**Files:**
- Modify: `include/grrt/scene/volumetric_disk.h` (add field to `VolumetricParams`)
- Modify: `include/grrt/types.h` (add field to `GRRTParams`)
- Modify: `src/volumetric_disk.cpp` (constructor `taper_width_` line)
- Modify: `src/api.cpp` (pass-through in `grrt_create`)

- [ ] **Step 1: Run the two currently-failing tests to confirm baseline state**

Run: `./build/Release/test-volumetric`

Expected: see these two `FAIL` lines in the output:
```
=== Density profile (no noise) ===
... FAIL: density should decrease with height
=== Density smooth across z_max ===
... FAIL: rho_below z_max too large (~e-XX of midplane)
```

(The third failure `test_tau_midplane_near_target` is fixed in Task 3, ignore for now.)

- [ ] **Step 2: Add the field to `VolumetricParams`**

In `include/grrt/scene/volumetric_disk.h`, find the `VolumetricParams` struct. After the existing `// --- Smooth volumetric envelope (NEW) ---` block (it currently contains `outer_taper_width` and `plunging_h_decay_exponent`), insert this new field at the end of that block (before the LUT-sizing block):

```cpp
    double plunging_taper_width_factor = 1.0;  ///< Plunging-region taper width as fraction of (r_isco - r_horizon)
```

- [ ] **Step 3: Add the matching field to `GRRTParams`**

In `include/grrt/types.h`, find the `GRRTParams` struct. After the existing `disk_plunging_h_decay_exponent` line, insert:

```c
    double disk_plunging_taper_width_factor;  /* 0 = use VolumetricParams default (1.0) */
```

Place it immediately after `disk_plunging_h_decay_exponent` so the related plunging-region knobs stay grouped.

- [ ] **Step 4: Apply the new field in the constructor**

In `src/volumetric_disk.cpp`, find the line in the `VolumetricDisk` constructor that sets `taper_width_`. It currently reads:

```cpp
    taper_width_ = (r_isco_ - r_horizon_) / 3.0;
```

Replace with:

```cpp
    taper_width_ = params_.plunging_taper_width_factor * (r_isco_ - r_horizon_);
```

This is the only line that changes the LUT construction itself.

- [ ] **Step 5: Pass through in `api.cpp`**

In `src/api.cpp`, locate the block in `grrt_create` that maps `GRRTParams` fields onto `VolumetricParams`. After the existing `if (params->disk_plunging_h_decay_exponent > 0.0)` block, insert:

```cpp
            if (params->disk_plunging_taper_width_factor > 0.0)
                vp.plunging_taper_width_factor = params->disk_plunging_taper_width_factor;
```

The `> 0.0` guard preserves backward compatibility: zero-initialized `GRRTParams` (the documented convention) gets the new default of 1.0.

- [ ] **Step 6: Build**

Run: `cmake --build build --config Release`

Expected: clean build, no errors. The pre-existing `D9025: overriding '/Ob2' with '/Ob3'` MSVC warning is unrelated and should still be the only warning.

- [ ] **Step 7: Run the test suite**

Run: `./build/Release/test-volumetric`

Expected: `test_density_profile` and `test_density_smooth_across_zmax` now PASS. The `test_tau_midplane_near_target` failure remains (fixed in Task 3). The total failure count should drop from 3 to 1.

If those two tests still fail, halt and investigate before proceeding. Report back with the actual rho values printed by each test.

- [ ] **Step 8: Verify the `h_jump` Promptable warning is gone on healthy construction**

Construct a default disk and check warnings. Add this temporary observation to confirm the fix worked:

Run: `./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 --disk-volumetric --mass-solar 10 --eddington-fraction 0.1 --output post_task1 --width 64 --height 64 --force 2>&1 | grep -E "h_jump|n_z_cap|n_r_cap|nested_refine"`

Expected: NO line containing `h_jump`. Cap warnings (`n_z_cap`, `n_r_cap`) may still appear — that's expected at this stage and addressed by Task 4's measurement.

- [ ] **Step 9: Commit**

```bash
git add include/grrt/scene/volumetric_disk.h include/grrt/types.h src/volumetric_disk.cpp src/api.cpp
git commit -m "feat(volumetric): widen plunging-region taper to full width by default"
```

The commit message reflects the behavior change: the default taper width has changed from `(r_isco-r_horizon)/3` to `(r_isco-r_horizon)`. Existing volumetric renders will look slightly different (smoother near the inner edge), which is the intended fix.

---

## Task 2: Add `--disk-plunging-taper-width` CLI flag

**Goal:** Expose the new parameter from `grrt-cli` so it can be tuned without recompiling.

**Files:**
- Modify: `cli/main.cpp` (argument parsing + `print_usage`)

- [ ] **Step 1: Add the flag parser**

In `cli/main.cpp`, find the argument-parsing loop (where flags like `--disk-alpha` and `--disk-turbulence` are parsed). After the existing block that parses `--disk-plunging-h-decay` (or whichever existing `--disk-*` flag is alphabetically/structurally adjacent), insert a parallel block:

```cpp
        } else if (arg("--disk-plunging-taper-width")) {
            params.disk_plunging_taper_width_factor = std::stod(argv[++i]);
```

The `arg(...)` helper and `std::stod(argv[++i])` pattern match the existing convention. Match the surrounding indentation exactly.

- [ ] **Step 2: Add the help text**

In `cli/main.cpp`, find `print_usage()` (the function that prints `--help` text). Find the line documenting `--disk-plunging-h-decay` (or whichever existing disk knob is structurally closest). Add immediately after it:

```cpp
    std::println("  --disk-plunging-taper-width F  Plunging-region taper as fraction of (r_isco - r_horizon) (default: 1.0)");
```

If the surrounding code uses `std::printf` instead of `std::println`, match that style. (Check the surrounding lines.)

- [ ] **Step 3: Build**

Run: `cmake --build build --config Release`

Expected: clean build. Only `cli/main.cpp` should recompile.

- [ ] **Step 4: Verify the flag is documented**

Run: `./build/Release/grrt-cli --help 2>&1 | grep -i taper`

Expected: see a line containing `--disk-plunging-taper-width F  Plunging-region taper as fraction of (r_isco - r_horizon) (default: 1.0)`.

- [ ] **Step 5: Smoke test the flag end-to-end**

Run: `./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 --disk-volumetric --mass-solar 10 --eddington-fraction 0.1 --disk-plunging-taper-width 0.5 --output smoke_taper05 --width 64 --height 64 --force 2>&1 | tail -10`

Expected: render completes (produces `smoke_taper05.png`). No parse error. Construction log shows the disk built normally.

Try the value `2.0` as well to make sure non-default values are accepted:

Run: `./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 --disk-volumetric --mass-solar 10 --eddington-fraction 0.1 --disk-plunging-taper-width 2.0 --output smoke_taper2 --width 64 --height 64 --force 2>&1 | tail -3`

Expected: render completes, no errors.

- [ ] **Step 6: Commit**

```bash
git add cli/main.cpp
git commit -m "feat(cli): add --disk-plunging-taper-width flag"
```

---

## Task 3: Fix `test_tau_midplane_near_target` units mismatch

**Goal:** Make the tau-midplane test pass by switching from `density()` (geometric-scaled units) to `density_cgs()` (CGS, which is what the integration formula needs).

**Files:**
- Modify: `tests/test_volumetric.cpp` (`test_tau_midplane_near_target` function only)

- [ ] **Step 1: Confirm the test currently fails**

Run: `./build/Release/test-volumetric 2>&1 | grep -A1 "tau_mid at peak-flux"`

Expected: the test prints something like `τ(z=0..z_max) at r=2.41: 0.00 (target 100.00)` followed by `FAIL`. Record the actual `r` and `τ` values printed.

- [ ] **Step 2: Read the current test body**

Read `tests/test_volumetric.cpp` from the line `void test_tau_midplane_near_target()` to its closing `}`. Identify three calls to `disk.density(r, z, 0.0)`:
1. One inside the peak-radius scan loop (sets `best_rho`)
2. One that reads `rho_a` inside the integration loop
3. One that reads `rho_b` inside the integration loop

Also note the two clamp expressions: `std::clamp(disk.density(...), 1e-30, 1e-3)` for `rho_a` and `rho_b`.

- [ ] **Step 3: Replace `density()` with `density_cgs()` in all three sites and widen the clamps**

In `tests/test_volumetric.cpp`, edit the `test_tau_midplane_near_target` function body:

The peak-radius scan line, currently:
```cpp
        const double rho = disk.density(r, 0.0, 0.0);
```
becomes:
```cpp
        const double rho = disk.density_cgs(r, 0.0, 0.0);
```

The two integration-loop lines, currently:
```cpp
        const double rho_a = std::clamp(disk.density(r, z_a, 0.0), 1e-30, 1e-3);
        const double rho_b = std::clamp(disk.density(r, z_b, 0.0), 1e-30, 1e-3);
```
become:
```cpp
        const double rho_a = std::clamp(disk.density_cgs(r, z_a, 0.0), 1e-18, 1e-6);
        const double rho_b = std::clamp(disk.density_cgs(r, z_b, 0.0), 1e-18, 1e-6);
```

The clamp range `[1e-18, 1e-6]` matches the bounds the opacity LUTs were built with (see `OpacityLUTs` construction in `src/volumetric_disk.cpp`).

The rest of the function body (kappa lookups, integration formula, comparison to `vp.tau_mid`) stays identical.

- [ ] **Step 4: Build**

Run: `cmake --build build --config Release`

Expected: clean build. Only `test_volumetric.cpp` should recompile.

- [ ] **Step 5: Run the test**

Run: `./build/Release/test-volumetric 2>&1 | grep -A1 "tau_mid at peak-flux"`

Expected: PASS. The printed `τ` should be within 30% of `tau_mid=100`, i.e., somewhere in `[70, 130]`.

If the test still fails despite the units fix, the printed `τ` value tells us the magnitude:
- `τ` in `[70, 130]` → PASS, proceed.
- `τ` in `[10, 70]` or `[130, 1000]` → wrong order of magnitude. This indicates a real bug in `normalize_density`'s peak-radius detection (the test's scan and `normalize_density`'s internal `peak_idx` are picking different radii). Halt and report; the fix is out of scope for this plan.
- `τ` near 0 or `nan` → something else is wrong. Halt and report.

- [ ] **Step 6: Run all tests to confirm overall test suite is now green**

Run: `./build/Release/test-volumetric 2>&1 | tail -5`

Expected: `=== 0 failures ===` (or fewer than 3 if other regressions appear — investigate any new failures before proceeding).

- [ ] **Step 7: Commit**

```bash
git add tests/test_volumetric.cpp
git commit -m "test(volumetric): use density_cgs in tau-midplane integration"
```

---

## Task 4: Phase 1 checkpoint — measure refinement behavior

**Goal:** Determine whether the cap-binding warnings (`n_z_cap`, `n_r_cap`, `nested_refine_no_fixed_point`) still fire after Tasks 1-3. The result decides whether a Phase 2 plan (adaptive RK45 in `solve_column`) is needed.

**Files:** None modified. This is observational only.

- [ ] **Step 1: Render a canonical case and capture the construction log**

Run:

```bash
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 \
    --disk-volumetric --mass-solar 10 --eddington-fraction 0.1 \
    --output checkpoint_phase1 --width 256 --height 256 --force \
    2>&1 | tee phase1_log.txt | tail -20
```

Expected: render completes. The full log is captured in `phase1_log.txt`.

- [ ] **Step 2: Extract refinement and warning lines**

Run:

```bash
grep -E "Refining LUT sizing|Refinement done|h_jump|n_z_cap|n_r_cap|nested_refine|σ_s_phys|Construction complete" phase1_log.txt
```

Record all output. Expect to see lines like:
```
[VolumetricDisk] Refining LUT sizing (n_r=256, n_z=64 initial)...
[VolumetricDisk] Refinement done: n_r=X, n_z=Y
[VolumetricDisk] σ_s_phys = ...
[VolumetricDisk] Construction complete. ...
```

If any `WARNING [n_z_cap]` or `WARNING [n_r_cap]` lines appear, record them too.

- [ ] **Step 3: Render a second case (high-spin AGN-like) for cross-check**

Run:

```bash
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 \
    --disk-volumetric --mass-solar 1e8 --eddington-fraction 0.1 --disk-outer 100 \
    --output checkpoint_smbh --width 64 --height 64 --force \
    2>&1 | grep -E "Refinement done|h_jump|n_z_cap|n_r_cap"
```

Expected: a `Refinement done` line. Record it.

- [ ] **Step 4: Decision**

Inspect the two construction logs. Apply this decision matrix:

| Observation | Action |
|-------------|--------|
| `Refinement done: n_r=X, n_z=Y` with `X << 4096` AND `Y << 1024` AND no cap warnings | Phase 1 is sufficient. STOP — no Phase 2 needed. |
| Cap warnings (`n_z_cap` or `n_r_cap`) still fire on either case | ODE oscillation near photosphere is the residual cause. Phase 2 (adaptive RK45 in `solve_column`) is needed. |
| `nested_refine_no_fixed_point` warning fires | The two refinement loops aren't converging together. Investigate whether `compare_columns` weighting needs tuning. Out of scope; flag for follow-up brainstorm. |

- [ ] **Step 5: Document the result**

Append a short note to `docs/superpowers/specs/2026-04-28-volumetric-disk-numerics-fix-design.md` under a new `## Phase 1 Checkpoint Results` section, recording:
- The `Refinement done: n_r=X, n_z=Y` values for both renders.
- Whether cap warnings fired.
- The decision (Phase 2 needed: yes/no, with one-line justification).

This makes the next planning session straightforward.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-04-28-volumetric-disk-numerics-fix-design.md
git commit -m "docs(spec): record Phase 1 checkpoint results"
```

If Phase 2 is needed, the next step is a fresh brainstorm session for the RK45 follow-up. If not, the branch is ready for merge consideration.

---

## Self-Review Notes (for the plan author)

**Spec coverage:**
- Spec §Components.1 (taper widening) → Task 1 Steps 2, 4 ✓
- Spec §Components.2 (parameter plumbing — VolumetricParams, GRRTParams, api.cpp, CLI) → Task 1 Steps 2, 3, 5 + Task 2 ✓
- Spec §Components.3 (tau test units fix) → Task 3 ✓
- Spec §Validation (3 tests pass, h_jump retires) → Task 1 Steps 7-8 + Task 3 Step 5 ✓
- Spec §Checkpoint decision point → Task 4 ✓
- Spec §Future work (RK45, normalize_density investigation) → Task 4 Step 4's decision matrix ✓

**Placeholder scan:** No TBDs, TODOs, or "implement later" markers. Every step shows actual code or actual commands.

**Type consistency:** Field name is `plunging_taper_width_factor` (in `VolumetricParams`) and `disk_plunging_taper_width_factor` (in `GRRTParams`). Both are `double`. Used consistently across Task 1 Steps 2, 3, 5 and Task 2 Step 1.

**Estimated implementation surface:** ~15 lines changed across 6 files, distributed as:
- Task 1: ~6 lines (4 files)
- Task 2: ~3 lines (1 file)
- Task 3: ~3 lines (1 file)
- Task 4: documentation only
