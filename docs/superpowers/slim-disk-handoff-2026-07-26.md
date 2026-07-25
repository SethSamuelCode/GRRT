# Slim-disk handoff — 2026-07-26

**Read this first.** Supersedes the 06-30 audit/seed-pivot handoff. Goal unchanged: converge a *physical* coupled slim disk at **a=0.9, f_Edd=0.9**, then wire it into the render for a near-Eddington black-hole image.

---

## TL;DR — the feasibility battle is WON; the blocker is now PERFORMANCE

The entire prior arc was "get a feasible seed + get the coupled relax to *start iterating*." **Done.** At the base rung (f_Edd=0.001, a=0.9, n_z=256): seed is **18/18 feasible**, and `relax_coupled` **iterates** (2 outer Newton iters last run — the "0 inner-iters" wall is gone). It "failed" only on a **wall-clock budget**, not divergence/infeasibility.

**New blocker:** each outer iteration is **~1h at n_z=256** (was 1.8h; OpenMP got 1.85×). Dominated by the reduced Jacobian's full-FD on (ℓ, ℓ_in, r_s) re-solving all columns.

**THE open question (a run is IN FLIGHT answering it):** is the relax actually *converging*? See "In-flight run" below.

---

## In-flight run (CHECK THIS FIRST)

A background run is computing the **merit trajectory**: `SLIM_DIAG=1 OMP_NUM_THREADS=12 slim-coupled-walk-probe` → `scratchpad/merit_traj.txt` (also task `bni3berjc.output`). It runs the base rung with **every-outer-iteration merit printing** (just added). It's multi-hour and will outlive the session that launched it.

**When it finishes / has output:** `grep "\[COUPLED\] it=" merit_traj.txt`. Read the `merit=` column across iters:
- **Descending** → the relax is CONVERGING (just slow). → build the ~12× parallelism (below), drive it to convergence, then walk f_Edd up.
- **Flat / rising** → a CONVERGENCE problem, not speed. Different fix (re-diagnose the coupled residual/Jacobian at the base state; the reduced Jacobian's full-FD provisionality is a suspect — see Explore notes in this session).

This single result decides the next move. Do not build more speed before reading it.

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

## Resume checklist
1. Read `merit_traj.txt` — is the base-rung merit descending?
2. If YES → spec the per-thread-cache parallelism (~12×) → converge the base rung → verify physical (H/r, β, f_adv, transonic V) → walk f_Edd up (continuation) toward 0.9.
3. If NO → re-diagnose the coupled relax convergence at the base state (Jacobian provisionality / residual balance).
4. Once a rung converges → wire T_eff(r)/H(r) into the volumetric-disk render → first slim-disk image → push f_Edd up.
