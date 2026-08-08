# Per-column parallelization of the coupled reduced Jacobian — design

**Date:** 2026-07-26
**Branch:** `fix/volumetric-ring`
**Status:** approved design → ready for implementation plan

## Goal

Cut the wall-clock cost of one outer Newton iteration of the coupled slim-disk
relax (`relax_coupled`) by ~8–12× so we can finally read the **merit trajectory**
across iterations and drive the base rung (a=0.9, f_Edd=0.001, n_z=256) to
convergence. Today one outer iteration costs ~2.7 h at n_z=256 and the wall
budget trips after a **single** merit point (`it=0 merit=3.009 maxrel=6.25e-2`),
so descending-vs-flat is unanswerable. Speed is on the critical path in *both*
branches of that decision (converge if descending; run cheap diagnostics if flat).

## The bottleneck (measured)

`slim_coupled_reduced_jacobian` (`src/slim_disk_coupled.cpp:666`) has three parts:

1. **Base column solve** (~L694) — already OpenMP-parallel over nodes.
2. **Frozen-FD** columns for Σ_i, V_i, T_c,i (~L728–740) — cheap; no column solves.
3. **Full-FD** columns for ℓ_i, ℓ_in, r_s (~L748–771) — **the ~65% cost.** A
   *serial* loop over N+2 ≈ 20 columns; each column calls `slim_coupled_residual`
   twice, and each such call re-solves all N columns. The only parallelism today
   is the inner per-node loop inside `slim_coupled_residual`, which tops out at
   ~1.85× because one dominant high-Σ node starves the other threads.

**Why per-column parallelism wins:** every full-FD *column* task solves *all* N
nodes — including the one fat node. So all ~20 tasks have ~equal cost → near-ideal
load balance across ~12 threads → ~min(threads, 20)× ≈ the handoff's ~12×. The
imbalance that capped per-node parallelism is amortized across every column task.

## Approach (chosen: A — per-column OpenMP)

Parallelize the **outer full-FD loop over columns**. Confined to one function;
exact Jacobian preserved; keeps the inner per-node `parallel for` (which runs
serially under the outer region — OpenMP nesting is off by default).

Rejected alternatives:
- **B — flatten to a ~40-task pool over (column × side) residual evals.** Marginal
  extra balance (A's tasks are already near-equal-cost) for a bigger diff
  (splitting the one-sided-fallback logic across tasks, reassembling J columns).
  **Strictly-later option; A does not foreclose it** — B is an incremental
  refactor on top of A's clean per-column loop, to be reached for only if
  profiling shows a straggler.
- **C — thread the inner per-node loop harder.** Dead end: this is exactly what
  already caps at 1.85× because one node genuinely dominates.

## Seed / correctness policy (decided: snapshot base-seed → bit-identical)

The current serial full-FD loop has an **accidental serial dependency**: each
column's residual solve warm-starts from the `ColumnCache` state left by the
*previous* column (the shared `cache` is mutated as a side effect of
`slim_coupled_residual`). Parallelizing breaks that ordering.

**Fix:** after the base solve, snapshot the converged cache
(`const ColumnCache base_snap = cache;`). Every full-FD column warm-starts from
that *same* snapshot. This makes the parallel output **bit-identical to a
base-seeded serial reference** — a strong, thread-order-independent gate — and is
arguably more correct (one consistent linearization point). It shifts today's
serial numbers microscopically (different warm-start seed → same root to solver
tolerance), which the FD-oracle (Gate 2) confirms is physics-neutral.

## Design (Approach A)

**Architecture:** one localized rewrite of the full-FD block
(`src/slim_disk_coupled.cpp:742–771`). Unchanged: the function signature, the `J`
layout (`J[row*n + col]`), all callers, the frozen-FD block, the analytic Schur
block, and the infeasible-side sentinel / one-sided-fallback semantics.

**Data flow:**
1. After the base solve (~L706), take `const ColumnCache base_snap = cache;`.
2. Compute the one-sided anchor `R0`/`f0` **once, serially**, from a *throwaway
   copy* of `base_snap` (so the snapshot stays pristine).
3. Build the column index list
   `fd_cols = {4i+2 : i∈[0,N)} ∪ {4N+0, 4N+1}` (the ℓ_i, ℓ_in, r_s columns).
4. `#pragma omp parallel for schedule(dynamic)` over `fd_cols`. **Each iteration
   declares thread-local** `Up, Um, Rp, Rm`, `bool fp, fm`, and
   `ColumnCache task_cache = base_snap;`. It runs the identical central-difference
   / one-sided-fallback logic present today and writes only its own `J[:, col]`.

**Why race-free:** no shared mutable state (each task owns its cache + scratch);
distinct `col` per task → distinct `J` columns written. Identical seed for every
column → result independent of thread count / schedule.

**Error handling:** unchanged. A fully-infeasible base still `return false`s
*before* the parallel loop (the base-solve `base_infeasible` reduction at ~L707).
Inside a task, an infeasible perturbed side is handled by the existing
sentinel-aware one-sided fallback against `R0`/`f0`; both-sides-infeasible → 0.0
entry, exactly as today.

## Testing — two gates + perf sanity

**Gate 1 — bit-identical (reuse `tools/slim_omp_gate_probe.cpp`).** Its existing
test (3) already builds `J` at `omp_set_num_threads(1)` vs `max_threads` and
requires `max-rel ≤ 1e-12` (bit-identical), plus test (4) determinism
(parallel-vs-parallel) and test (5) timing/speedup. After the change: 1-thread runs
the new per-column loop serialized + base-seeded; max-thread runs it parallel +
base-seeded; base-seeding makes them bit-identical. **Acceptance: tests (2)–(4)
PASS (max-rel ≤ 1e-12), test (5) speedup ≥ ~4× at the probe grid** (N=12, n_z=96;
the property is grid-independent, larger speedup expected at n_z=256).

**Gate 2 — FD-oracle (reduced-Jacobian regression).** The existing
`tests/test_slim_coupled_jacobian.cpp` gate (`test-slim-coupled-jacobian`)
perturb-resolves the FD oracle and compares it to `slim_coupled_reduced_jacobian`
at a feasible synthetic point, requiring per-column scaled 2-norm **< 1e-3**. It
builds the *full* reduced `J` — including the base-seeded ℓ/ℓ_in/r_s full-FD
columns — so it confirms the refactor did not break the reduced-J assembly. Must
still PASS after the change.

Note: the ℓ/ℓ_in/r_s columns are themselves the perturb-resolve oracle (no
independent analytic form), so base-seeding can only shift them by the column
solver's root-reproducibility (≈ its tolerance). Task 1 quantifies that shift
directly (serial base-seeded J vs the pre-change serial J, expect max-rel ≲ 1e-6),
making the "microscopic, physics-neutral" claim a measured fact rather than an
assumption.

**Perf sanity (informational).** Time one `slim_coupled_reduced_jacobian` at
n_z=256, 1 vs 12 threads — expect ≈ 8–12×. May be added as an optional larger-grid
timing line in the gate probe; not an acceptance gate (grid-independent property
is already gated at n_z=96).

## Non-goals / out of scope

- Approach B (per-side task pool) — later, only if profiling demands it.
- Any change to the physics, the residual, the frozen-FD or Schur blocks, the
  column solver, or the relax driver's outer logic.
- Walking f_Edd up / wiring the render — separate downstream steps after the
  base rung converges.

## Files touched

- `src/slim_disk_coupled.cpp` — rewrite the full-FD block in
  `slim_coupled_reduced_jacobian` (~L742–771); add the `base_snap` snapshot after
  the base solve.
- `tools/slim_omp_gate_probe.cpp` — reuse as-is for Gate 1; optionally add an
  n_z=256 timing line. No new test TU required.

## Workflow constraints (standing)

- **NEVER `git commit`** — hand the message to the user; wait for their call.
- **One change at a time.**
- Subagent-driven implementation; TDD; keep the gates honest.
- Verify load-bearing claims with opus + Wolfram; fable is a second oracle; never
  sonnet for physics. (This change is a parallelization refactor, not a physics
  edit — the bit-identical gate is the primary correctness oracle here.)
