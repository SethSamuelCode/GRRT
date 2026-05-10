# Future optimization: switch disk-step-entry substep integrator to plain RK4

**Status:** noted, not implemented. Ranked low-priority.
**Origin:** brainstorm for `2026-05-10-disk-step-entry-design.md` on 2026-05-10.

## Idea

Inside the disk-step-entry helper (Tier C subdivision), each substep currently uses `RK4::step_kerr_rkdp45(metric, state, dl).y5` — fixed-step Dormand-Prince RK4(5), the same integrator family the main trace loop uses. This costs **6 derivative evaluations per substep**.

The optimization: switch substepping to plain `RK4::step_kerr(metric, state, dl)` — fixed-step RK4. **4 derivative evaluations per substep**, ~33% per-call savings.

## Why it's not free

- RK4 fixed-step is 4th-order accurate; DP45 fixed-step is 5th-order. At the same `dl`, RK4 substep mid-points lie on a slightly different trajectory than DP45 would compute.
- This *can* introduce an integrator-mismatch artifact class: the helper subdivides with RK4 and detects a mid-point as inside the disk, but when the main loop's adaptive DP45 actually traces that same parameter range, it lands on a slightly different trajectory and the entry point shifts. The wedge bug class isn't reintroduced (Tier A still gates correctness), but renders may show non-trivial differences between the two integrator choices.
- For the wedge-fix purpose, 4th-order substep accuracy is sufficient — the predicate cares about side-of-disk, not 5th-decimal-place trajectory. No expected behavioral regression on the original wedge repro.

## Estimated gain

- Tier C subdivision is the only place this matters. Tier A handles the no-bug fast path; Tier B is endpoint+pad arithmetic only.
- Tier C fires rarely — only for grazing trajectories near the photon ring or LUT-cliff edges. Estimate **<1% of pixel-steps** at typical FOV/disk parameters.
- Within a Tier C invocation, depth_limit ≈ 4–8 substeps. Each substep saves 2 derivative evals → **per-Tier-C savings ≈ 8–16 derivative evals**.
- Per-render speedup at the levels where Tier C is rare: **<0.4% of total render time**.

For most production renders this is below noise. For very-narrow-FOV or supermassive scenes where Tier C fires at >5% of pixels, this could climb to a few percent.

## Why it was *not* taken

For the disk-step-entry spec, we chose to keep the substep integrator aligned with the main loop's integrator family:

- Spec §5.1 explicitly calls for "the same integrator the main loop uses — no duplicated integration code, identical numerical behavior."
- Eliminates an integrator-mismatch failure mode: helper and main loop always agree on the geodesic trajectory at any resolution.
- Makes the design auditable: there is one geodesic integrator family in the codebase (DP45), and every place we step through a geodesic uses it.

The estimated savings (<0.4% typical, <few% worst-case) is below the value of "single source of truth on integration."

## When to revisit

- Profiling shows `disk_step_entry::subdivide` accounts for >5% of render time, **and**
- The render in question is exercising Tier C heavily (check `tracer.substep_invocation_count() / total_steps`).

## Sketch of a validation plan (when revisited)

1. Render the wedge repro with both integrators:
   ```
   ./grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output ref_dp45.png --force --fov 30
   # switch substep to step_kerr, rebuild
   ./grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output rk4.png --force --fov 30
   ```
2. Diff the images. Acceptance: per-pixel L1 below visual-perception threshold (e.g. ≤2 LSB on 8-bit channels). Wedges remain absent in both.
3. Check `test_no_horizontal_bands` still passes without recalibration.
4. Render a hot-Tier-C scene (very narrow FOV, supermassive params) and measure realized speedup. Confirm against the <0.4%–few% estimate.

If both pass with measurable speedup, consider the swap. Otherwise leave the more-conservative DP45 substep in place.
