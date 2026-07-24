# Advective Seed-T_c for High-Σ Nodes (design)

**Date:** 2026-07-24
**Status:** DESIGN — awaiting user review before writing-plans.
**Scope discipline:** ONE change. No stretched grid, no relax changes, no touching `build_coupled_seed`'s hot-path use or its Σ0-match gate. Just: derive a *feasible* seed T_c on the f_adv-freed manifold for nodes where f_adv=0 has no root, and wire it into the seed-T_c derivation.

---

## Problem (evidence-grounded)

After the multi-start fix, `slim-full256-probe` is **17/18** (n_z=256). The lone holdout is **node 10 (r=14.0, Σ=1.293e4)**, pinned at a stale **T_c=3.2e6** — the cold thin-seed value kept because `build_coupled_seed` (f_adv=0) found no root there. Its feasible neighbours sit at T_c ~7e6–1.7e7. No column exists at (Σ=1.29e4, T_c=3.2e6) for any f_adv, so both the pinned solve and the multi-start (which cannot change the pin) fail. **This is a seed-T_c problem, not a basin miss.**

**Root cause (confirmed):** `build_coupled_seed` (`disk_column_coupled.cpp:430`) pins f_adv=0 and, on the saturating f_adv=0 Σ0 ceiling, honestly returns false (its Σ0-match gate). `build_coupled_seed_2d` (line 532) **pins T_c** (residual `T_c/T_c_target−1`), so it cannot *supply* a T_c — it needs one. The callers (`calibrate_seed_to_manifold` in the walk probe; the `Tc_manif` step in `slim_full256_probe`) then fall back to the stale thin T_c → infeasible pin.

## The one change

Add a reusable **f_adv-laddered seed builder** that produces a *feasible* seed column (and hence a reachable T_c) for advective nodes, and wire it in as the fallback where `build_coupled_seed` currently fails.

### Component 1 — `build_coupled_seed_advective` (new, `src/disk_column_coupled.cpp`)
Same recipe as `build_coupled_seed` — secant on T_eff to match Σ0(T_eff)=Σ_target — but run at a small ladder of **f_adv > 0**, reading the resulting midplane T_c off the converged column (T_c is an OUTPUT, never pinned):
```
for fa in {0.5, 1.0, 2.0, 4.0}:
    secant T_eff s.t. Σ0(base_inputs_from(in, T_eff, fa)) == Σ_target   # reuse the existing secant
    if Σ0-matched (same 0.30 band as build_coupled_seed):
        pack U exactly like build_coupled_seed but with U[4N+3] = fa (not 0),
        U[4N+2] = T_eff, midplane T_c = column.T.front();  return true
return false
```
- Reuse the exact secant/Σ0-match logic of `build_coupled_seed` (prefer factoring the shared secant into a small `secant_Teff_at_fadv(in, op, fa, U)` helper that both call; duplication is acceptable only if factoring is messy — do NOT alter `build_coupled_seed`'s behavior at fa=0).
- First f_adv that Σ0-matches wins (smallest advection that reaches the demand — the physically-minimal advective seed).
- Ladder cap = 4 tries; returns false if none (genuinely no advective column at that Σ — a real obstruction, reported honestly).

### Component 2 — wire it into the seed-T_c derivation (the two callers)
Replace the single `build_coupled_seed(...)` call with `build_coupled_seed(...) || build_coupled_seed_advective(...)` at:
- `tools/slim_full256_probe.cpp` — the `Tc_manif` step (so the isolated test sees it).
- `tools/slim_coupled_walk_probe.cpp` — `calibrate_seed_to_manifold` (so the relax seed becomes 18/18-feasible; the real prize, tested next).
The read is identical: use `Uc[2]` as the seed T_c. No other caller changes.

`build_coupled_seed`'s own hot-path use inside `solve_column_coupled` (line 995) is **NOT** touched — it stays the f_adv=0 seed there; the merit selection already handles the rest. Contract of `build_coupled_seed` unchanged.

## Isolated success test (clean attribution)

Nothing else changes; rebuild + re-run:
- `slim-full256-probe` → **18/18** (was 17/18), and
- `slim-full256-probe 32 256` → **32/32** (was 31/32 with multi-start; confirm).

The only variable is the advective seed-T_c fallback. If node 10 stays infeasible, its per-node line (Σ, the new T_c, converged?) tells us whether `build_coupled_seed_advective` found no advective column (real obstruction → investigate) or the pinned solve still misses — do NOT loosen anything.

## Testing / validation gates (TDD)
1. **Unit — advective seed converges a high-Σ node.** At node 10's geometry + demanded Σ (thin-seed values), `build_coupled_seed` returns false (f_adv=0 saturates) but `build_coupled_seed_advective` returns true with a T_c in the neighbours' band (~5e6–2e7) and f_adv>0, AND `solve_column_coupled` at (Σ, that T_c) converges. *Write first.*
2. **Regression — f_adv=0 nodes unchanged.** A node where `build_coupled_seed` already succeeds must be untouched (advective builder not consulted). Existing `test-column-coupled` suite stays green; `build_coupled_seed` byte-identical.
3. **Isolated integration — full256 → 18/18 and 32/32.** The deliverable gate (controller runs it).

## Non-goals
- No relax run yet (next: seed the walk from the now-18/18 calibration and watch it iterate — the real integration test).
- No change to `build_coupled_seed` or its Σ0-match gate, no change to `solve_column_coupled`'s cold path or the multi-start.
- No stretched grid.

## Workflow
Never `git commit` — hand the message over. TDD, gate 1 first. Present every reviewer rec & WAIT. One change, one commit.
