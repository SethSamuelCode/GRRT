# Slim-disk handoff — 2026-07-26

**Read this first.** Supersedes the 06-30 audit/seed-pivot handoff. Goal unchanged: converge a *physical* coupled slim disk at **a=0.9, f_Edd=0.9**, then wire it into the render for a near-Eddington black-hole image.

---

## ⚠️ SUPERSEDED IN PART — read "2026-07-26 (later): the blocker was a DEAD JACOBIAN" below first.

The "blocker is PERFORMANCE" framing in this doc was **wrong**. It was diagnosed against a
structurally singular Jacobian. Sections below are kept for history; the corrected state is
in the new section at the end.

## TL;DR (as written earlier — premise now corrected)

The entire prior arc was "get a feasible seed + get the coupled relax to *start iterating*." **Done.** At the base rung (f_Edd=0.001, a=0.9, n_z=256): seed is **18/18 feasible**, and `relax_coupled` **iterates** (2 outer Newton iters last run — the "0 inner-iters" wall is gone). It "failed" only on a **wall-clock budget**, not divergence/infeasibility.

**Then-believed blocker:** each outer iteration is **~1h at n_z=256** (was 1.8h; OpenMP got 1.85×). Dominated by the reduced Jacobian's full-FD on (ℓ, ℓ_in, r_s) re-solving all columns.

**THE open question at the time:** is the relax actually *converging*?

---

## In-flight run — RESULT IN, AND IT IS INVALID

The merit-trajectory run completed: `scratchpad/merit_traj.txt` (task `bni3berjc`). It produced
exactly **one** usable line before the wall budget tripped:

```
[COUPLED] it=0 merit=3.009e+00 maxrel=6.25e-02 mu=3.1e+00 r_s=2.2745
[COUPLED] it=1 BUDGET EXCEEDED -> abort      [9668.4s, 2 inner-iters]
```

**Do not draw conclusions from it.** It measured a solver whose Jacobian had 20 identically-zero
columns (see below), i.e. only 54 of 74 unknowns were being driven. The `merit`, `mu=3.1`, and
`maxrel` values describe a crippled solve, not the physics. **Re-run on the fixed Jacobian
before concluding anything about convergence.**

---

## What's committed (recent → older)
- `9e3b1a9` perf: **OpenMP** parallelize per-node column solves (bit-identical gate, race fixed `vector<bool>`→`char`, ~1.85×) + **every-iter merit print** (`relax_coupled` ~L1044, under `SLIM_DIAG`) + diagnostic probes + walk probe n_z=256.
- `3133979` docs: specs+plans (multi-start, advective seed-Tc, stretched grid).
- `0cc6fb2` feat: **advective seed-T_c** (`build_coupled_seed_advective`, f_adv-laddered) — the fix that got node 10 feasible → 18/18 at n_z=256.
- `2971bcf` feat: **multi-start** the 2-D bring-up in `solve_column_coupled` (fixes basin-miss nodes).
- `0cc6fb2`..`d6236aa` also: convection-blind coupled Jacobian fix; earlier convection (#13).

## SHELVED — do NOT retry
**Stretched vertical grid** as a feasibility enabler. Full write-up: `docs/superpowers/slim-disk-stretched-grid-shelved-2026-07-24.md`; memory `project_stretched_grid_shelved.md`. It's NON-MONOTONIC in n_z (16/18@96 → 14/18@128) because **base-rung feasibility is solver-reachability-bound, not resolution-bound**. Reliable feasible seed = **n_z=256 uniform**. (The machinery was correct + a per-column accuracy win, but the *purpose* was wrong.)

## The next lever (IF merit descends): ~12× parallelism
OpenMP-per-node gave only 1.85× (load imbalance: one dominant high-Σ column per residual eval bounds it). The real win: **parallelize the ~40 independent full-FD residual evals** (the Jacobian's ℓ/ℓ_in/r_s columns) across threads — needs a **per-thread `ColumnCache`** (avoids the slot race). That's ~min(40,cores)× on the bottleneck → ~10 min/iter → convergence in ~1-2h. Also consider: reduce the FD Jacobian cost (are the full-FD ℓ columns necessary, or do frozen-FD+Schur suffice? — flagged as provisional).

## Key facts / config
- Base rung: a=0.9, f_Edd=0.001, N=18 radial, n_z=256 vertical. `slim_coupled_walk_probe` Phase-1 tries the ladder; rung-1 (0.001) is the one that's 18/18 feasible and iterates; higher rungs fail fast (cold seeds not 18/18).
- Cost per outer iter ≈ residual (18 col solves) + reduced Jacobian (~40 full-FD evals × 18 col solves ≈ 720 col solves) at n_z=256. Jacobian dominates.
- Wall-clock budget: `slim_disk_radial.cpp` ~L414-443 (`kDefaultWallSeconds`, `in.budget_wall_seconds`); raise for longer diagnostic runs.
- Merit is printed under `SLIM_DIAG=1` (`kDiag`).

## Render path (for wiring later — NO redesign needed)
`volumetric_disk.cpp` calls `solve_column_bvp` (L1007/1016/1085) and IS in the render path (`api.cpp`→`geodesic_tracer`→`disk_step_entry`→`volumetric_disk`). Interface is `ColumnInputs`→`ColumnBVPSolution`. To render the converged slim disk: tabulate T_eff(r), H(r) from the converged relax → feed the Scene/volumetric disk. Branch `fix/volumetric-ring` is the right home (H(r) drives the thick ring). User will handle "saving the outputs" once the relax converges.

## Workflow constraints (STANDING)
- **NEVER `git commit`** — hand the message to the user (EXCEPT when the user explicitly says "commit"). Present every reviewer/subagent rec with your take and WAIT.
- **One change at a time** (user's explicit ask — avoids tail-chasing).
- **Verify load-bearing claims: opus + Wolfram; fable is RESTORED (2026-07-11) as a second oracle. NEVER sonnet for physics.**
- Subagent-driven for implementation; keep gates honest (convergence ≠ physical; FD-oracle is the model-independent Jacobian check).
- Doc-first for formula edits (`references/disk-physics-formulas.md` §22-24).

## Resume checklist (SUPERSEDED — see the corrected checklist at the end)

---

# 2026-07-26 (later): the blocker was a DEAD JACOBIAN, not performance

## What was actually wrong

`slim_coupled_residual`'s out-param is **`infeasible`** (`true` = a column FAILED;
`src/slim_disk_coupled.cpp:516`, set false at :520, true on failure; cf.
`tools/slim_omp_gate_probe.cpp:166` doing `return !infeas;`).

Both Jacobian builders bound it to variables named `f0`/`fp`/`fm` and tested them as if
`true` meant **feasible**. At any feasible point all three are `false`, so every branch of
the difference selector fell through to `d = 0.0`.

**Result:** all **20 full-FD columns** of the reduced Jacobian — ℓ_i at each of the 18 nodes,
plus **ℓ_in and r_s, the transonic eigenvalue pair** — were identically zero. The entire
`slim_coupled_numerical_jacobian` was zero. The Newton solver was driving 54 of 74 unknowns;
ℓ/ℓ_in/r_s moved only via LM damping, never via a gradient.

## Why no gate caught it
- `test-slim-coupled-jacobian` perturb-resolves **only** the Σ_i and T_c,i columns
  (`tests/test_slim_coupled_jacobian.cpp:409-410` → `check_col(4*i+0)`, `check_col(4*i+3)`).
  Those come from the frozen-FD + Schur path, which never touches these flags. The ℓ columns
  were **never gated**.
- The numerical-Jacobian "oracle" carried the **same inversion**, so an analytic-vs-numerical
  comparison would have compared 0 against 0 — a vacuously passing check.
- **Lesson:** two independent-looking gates shared one blind spot. A gate that doesn't cover a
  column proves nothing about that column.

## Fixed (commit `3c001ad`)
Renamed `f0/fp/fm` → `inf0/infp/infm`, **initialized `true`** (= "no usable residual for this
side" — critical, because a skipped one-sided side has an EMPTY R buffer that the central
branch must never index), inverted every test, both sites.

**Measured A/B vs the pre-fix build**, feasible base point (a=0.9, N=12, n_z=96, f_Edd=1e-3):

| | pre-fix | post-fix |
|---|---|---|
| non-zero full-FD columns | **0 / 14** | **14 / 14** |
| whole-J non-zeros | 204 / 2500 | 366 / 2500 |
| max·\|J\| | 7.5e29 | 2.5e33 |

`r_s` now touches 48/50 rows (it rescales the whole radial grid); `ℓ_in` touches 24.

## Performance actually achieved this session
| commit | change | effect |
|---|---|---|
| `f569eb0` | base-seed full-FD columns from a cache snapshot | prerequisite; order-independent |
| `5ffcac2` | per-column OpenMP over the full-FD loop | whole-J **1.82× → 2.85×** |
| `3c001ad` | polarity fix + one-sided FD (default ON) | one-sided now live: warm **34.8s → 17.6s (1.99×)** |

Probe grid, warm: single-thread central **97.9s** → parallel one-sided **17.6s ≈ 5.6×**;
≈ **3×** vs where the code stood at session start. `SLIM_FD_ONESIDED=0` restores central
differencing without a rebuild (first A/B to run if convergence stalls).

One-sided justification: the FD error budget is dominated by **column-solver noise**
(`copt.tol/h` ≈ 1e-2), identical for both schemes; truncation only rises O(h²)≈1e-12 →
O(h)≈1e-6. Measured one-sided-vs-central difference on the ℓ columns: **max 5.4e-6**
per-column norm.

## MISATTRIBUTION to avoid repeating
The `LU pivot ratio: raw = 9.664e+12` lines in the merit log are from
`src/disk_column_coupled.cpp:1102` — the **per-column 2-D solver's** conditioning. They are
**not** the reduced radial Jacobian's. No instrument prints the radial J's conditioning. Do
not cite those numbers as evidence about the radial solve.

## Corrected resume checklist
1. **Re-run the base rung on the fixed Jacobian** (`SLIM_DIAG=1`, raise the wall budget at
   `slim_disk_radial.cpp` ~L414 — it trips at ~2.7h; est. ~54 min/iter now). Read `merit=`
   across iterations. *This is the first trustworthy measurement of whether the relax converges.*
2. Expect **different convergence behavior, not marginally different** — LM damping tuned
   against a singular J may need revisiting.
3. If it converges → verify physical (H/r, β, f_adv, transonic V) → walk f_Edd up by
   continuation toward 0.9 → wire T_eff(r)/H(r) into the volumetric-disk render.
4. If it still stalls → A/B `SLIM_FD_ONESIDED=0` first (cheap), then consider extending
   `test-slim-coupled-jacobian` to gate the ℓ/ℓ_in/r_s columns, then the deeper levers
   (analytic ℓ sensitivity to kill full-FD entirely; graph-colored FD; Broyden).
5. **Do NOT** pursue a coarser-n_z Jacobian: base-rung feasibility is n_z-sensitive (15/18 at
   n_z=96 vs 18/18 at 256), so a coarse-J would silently see infeasible columns and corrupt
   the direction. (User's call, and correct.)
