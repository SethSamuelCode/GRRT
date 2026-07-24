# Robust Multi-Start for the Coupled Column Solve (design)

**Date:** 2026-07-12
**Status:** DESIGN — awaiting user review before writing-plans.
**Scope discipline:** ONE change only. No stretched grid, no lever C, no relax changes, no `build_coupled_seed` gate changes. Just multi-start the 2-D bring-up inside `solve_column_coupled`.

---

## Problem (evidence-grounded)

At the f_Edd=0.001, a=0.9 base rung, ~2 of 18 (and 3 of 32) radial nodes' columns fail to solve at their demanded Σ. Proven **not physical**: the failing radii **move when the radial grid is re-gridded** (N=18 fails at r≈3.9,14.0; N=32 fails at r≈6.2,11.2,12.4 — each grid's failures are feasible in the other). So it is a **solver-basin miss**, not a capacity wall.

**Mechanism (confirmed in code, `disk_column_coupled.cpp:482-504`):** the **f_adv=0** grey column's Σ0 is a *saturating* function of T_eff (ceiling a few×10³ g/cm²). Inner nodes demand Σ ~1.3–1.6×10⁴ ≫ ceiling → **no f_adv=0 root exists**; holding that Σ **requires f_adv≠0**. The f_adv-freeing **2-D bring-up** (`build_coupled_seed_2d`, called at `solve_column_coupled` ~line 986) is the intended path, but from its single default starting guess it misses the basin at scattered nodes.

## The one change

Add a **multi-start retry of the cold-seed 2-D bring-up inside `solve_column_coupled`**: when the primary solve (built from the default 2-D bring-up guess) fails to converge, retry the cold-seed build + affine-invariant Newton from a small **spread of starting guesses** — varying `Teff_guess`, the initial `f_adv`, and `rho_mid_guess` — and accept the first converged (lowest-merit) result. Placed as a fallback **after** the primary Newton and **before** the existing Σ-continuation, mirroring the existing internal-fallback pattern.

- **Transparent to every caller.** The relax reaches the column solve via `eval_node_coupled → solve_column_coupled` directly (`slim_disk_coupled.cpp:293`); a retry inside is invisible to `relax_coupled`. Probes and seed calibration benefit identically. No call-site changes anywhere.
- **Always-on, cost only on failure.** The retry triggers only when the primary solve fails (rare — the healthy majority never enters it), so no slowdown on good nodes. (User decision: always-on, not flag-gated — a column that needs multi-start is exactly the case we always want handled.)
- **Does NOT change the pinned targets.** Σ and T_c stay pinned; the spread is only over the *initial guess* for the free unknowns (T_eff, f_adv, ρ_mid). Same problem, better basin entry — so it cannot silently "solve a different column."

### The seed spread (productionized from the proven probe)

`tools/slim_nz_refine_probe.cpp` PART-3 already demonstrated that a spread converges columns the single guess misses. Productionize a compact version — the implementer confirms exact knobs against `build_coupled_seed_2d`'s signature, but the intent:
- `Teff_guess ∈ {default, default×0.5, ×2, ×0.25, ×4}` (mirrors `build_coupled_seed`'s existing multiplier ladder).
- initial `f_adv ∈ {0, 0.5, 2.0}` (the freeing that high-Σ nodes need).
- `rho_mid_guess ∈ {default, default×0.3, ×3}`.
Iterate outer-product-ish but **short-circuit on first convergence**; cap total retries (e.g. ≤12, matching the probe) so a truly infeasible node fails fast, not slowly.

## Isolated success test (the whole point of one-at-a-time)

With **nothing else changed**, rebuild and re-run `slim-full256-probe`:
- **`./slim-full256-probe.exe` → 18/18** (was 16/18), and
- **`./slim-full256-probe.exe 32 256` → 32/32** (was 29/32).

Same seeds, same demanded Σ, same n_z — the *only* variable is the multi-start. Clean attribution.

**Honest caveat on the test:** `full256` pins `T_c = Tc_manif` (from `build_coupled_seed`, which is *stale* at the very nodes that fail, since f_adv=0 has no root there). If a node stays infeasible because the *pinned T_c itself* is unreachable (not a basin miss), that's a **different** one-thing (seed-T_c on the f_adv-freed manifold) and we address it next — separately. We do not bundle it here. If full256 does not reach 18/18, its per-node output tells us exactly which nodes and whether it's basin (multi-start's job) or pinned-T_c (next change).

## Non-goals
- No stretched vertical grid (deferred — speed, not correctness).
- No change to `build_coupled_seed`'s honest Σ0-match gate (it is correct — f_adv=0 genuinely has no root there).
- No relax run yet (next change, after feasibility is verified in isolation).

## Testing / validation gates (TDD)
1. **Unit — multi-start converges a known basin-miss node.** Construct a column at an inner-node geometry + demanded Σ that fails from the default guess today; assert `solve_column_coupled` now converges (converged==true), at the *same* pinned (Σ, T_c). *Write first.*
2. **Regression — healthy nodes bit-identical.** A node that already converges must return the same converged column (multi-start not entered) — existing `test-column-coupled` suite stays green, no perturbation of the passing path.
3. **Isolated integration — `slim-full256-probe` → 18/18 and 32/32.** The deliverable gate.

## Workflow
Never `git commit` — hand the message over. TDD, gate 1 first. Present every reviewer rec & WAIT. One change, one commit. Convergence ≠ physical, but here the target *is* convergence-robustness and the gate is objective (feasible count).
