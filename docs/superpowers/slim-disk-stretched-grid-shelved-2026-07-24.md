# SHELVED: stretched vertical grid as a feasibility enabler (2026-07-24)

**TL;DR — do NOT re-attempt a stretched/clustered vertical grid to fix base-rung column *feasibility*.** It was built, gated-green, and measured. It does **not** deliver cheap 18/18 feasibility, and it makes feasibility **non-monotonic in n_z** (worse at higher n_z). Root cause: the base-rung wall is **solver-reachability-bound, not vertical-resolution-bound** — a grid change cannot fix it. Reverted (kept only as uncommitted learning). Spec/plan committed for the record: `docs/superpowers/specs/2026-07-24-stretched-vertical-grid-design.md`, `.../plans/2026-07-24-stretched-vertical-grid.md`.

---

## What it was and why we tried it

After the advective seed-T_c fix, base-rung feasibility (a=0.9, f_Edd=0.001) was **18/18 at n_z=256 uniform** but only **15/18 at n_z=96**, with n_z=256 ~7× slower per pass — too slow to run the coupled relax. Hypothesis: the 3 holdouts (nodes 3,4,9 — highest-Σ, radiation-pressure-dominated inner columns) fail because a *uniform* n_z=96 grid under-resolves their steep **photosphere**. A grid clustered toward the surface (q=1) should give n_z=256-quality feasibility at ~n_z=96 cost — the intended speed enabler for the relax.

## What was actually built (and is CORRECT)

- `column_q_grid(N, stretch)`: tanh map, dense at the surface; `stretch<=0` ⇒ uniform. (Note: the first formula in the plan clustered the *wrong* way — toward the midplane — and was corrected to `q[i]=tanh(stretch·u)/tanh(stretch)`.)
- Per-interval `dq_i = q[i+1]-q[i]` threaded through the trapezoidal ODE rows of `column_residual`, `analytic_jacobian`, `coupled_column_residual`, `coupled_column_jacobian`. Quadratures (Σ,τ,η3,η4) and BCs were already adaptive — untouched.
- Gates that PASSED: **Gate 1** uniform-recovery `stretch=0` → bit-identical (rel = 0). **Gate 2 (bvp)** analytic-vs-FD Jacobian on a stretched grid = 0.0. bvp physics moved *toward* continuum (f_adv heating ~25× closer; midplane β corrected).

So the *machinery* is sound. The failure is about *what it buys*, not correctness.

## Why it failed — the measurements

**(1) It doesn't reach 18/18, and it's NON-MONOTONIC in n_z** (the decisive tell):

| grid | feasible @ base rung |
|---|---|
| n_z=96 uniform | 15/18 |
| n_z=96 stretch=1.0 | **16/18** (rescued 3,4; **broke 12**; 9 still fails) |
| n_z=128 stretch=1.0 | **14/18** ⬇ (worse: 4,5,6,9 fail) |
| n_z=160 stretch=1.0 | timed out (>10 min) |
| n_z=256 uniform | 18/18 |

Uniform is **monotonic** (6→15→18 as n_z 24→96→256). Stretched is **non-monotonic** (96→128 got *worse*). A genuine resolution improvement is monotone in point count; a non-monotone result means the variable that actually gates feasibility is **not resolution** — it's the column solver's ability to *reach* the converged root from its seed, which the clustered grid perturbs (conditioning + basin) in node-dependent, non-monotone ways. This is the same "failures move when you re-grid ⇒ solver-basin, not physics" signature seen earlier in the base-rung diagnosis.

**(2) A hard accuracy ceiling on the stretch amount.** The coupled FD-Jacobian gate (<4e-4) fails for `stretch≥1.5` (2.5→4.6e-3). Diagnosed: NOT a dq-threading bug (bvp gate = 0.0; the offending entry is a base-block `∂R_T/∂T` radiative-diffusion partial, base-block mismatch identical to coupled). It is the pre-existing **inherited bilinear-LUT opacity-gradient Jacobian floor** (3.27e-4 uniform), amplified ~14× because surface-clustering makes the *midplane* intervals large and the opacity-partial error enters as `half_dq·Δ(dT_dT)`. So the safe stretch ceiling is ~1.0 — too mild to resolve the hardest nodes, and even 1.0 coarsens the midplane enough to *break a previously-feasible node* (12).

**Net:** clustering the surface (to help high-Σ photospheres) necessarily coarsens the midplane (hurting the Jacobian + other nodes). At the max stretch the Jacobian tolerates, the win is marginal and unstable. There is no free lunch here.

## The lesson (so we don't repeat it)

1. **Base-rung column feasibility is solver-reachability-bound, not resolution-bound.** Confirmed twice now (grid-move non-reproducibility + stretched-grid non-monotonicity). Do not throw *resolution* (uniform n_z, stretched grids, adaptive meshes) at it expecting a clean win. Throw **solver robustness** at it (multi-start seeding — already done; warm-starting; better continuation), or change the approach.
2. **The reliable feasible-seed config is n_z=256 uniform** (monotone, dependable, just slow). If the relax needs a cheaper feasible seed, the lever is solver-side, not grid-side.
3. **Don't couple an accuracy change to a feasibility goal.** The stretched grid is a legitimate *per-column accuracy* improvement in isolation, but it was justified as a *feasibility* enabler — which it isn't. If ever revisited, revisit it as a **pure accuracy task, decoupled**, and only after the **opacity-gradient Jacobian floor** is addressed (that floor, not the grid, is what caps the usable stretch).
4. **Non-monotonicity in a refinement parameter is a red flag** that you're not refining the thing that actually gates the outcome. Treat it as a signal to re-diagnose the binding constraint, not to tune harder.

## Status
Reverted to HEAD (uniform grid). Machinery not committed. Spec + plan committed as the design record. Next: run the coupled relax at **n_z=256 uniform** (the reliable feasible seed) to answer the real open question — *does the coupled Newton actually converge?* — which no grid change addresses.
