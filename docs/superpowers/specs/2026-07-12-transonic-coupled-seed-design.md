# Lever C — Transonic-Σ Seed for the Coupled Relax (design)

**Date:** 2026-07-12
**Status:** DESIGN — awaiting user review before writing-plans.
**Context docs:** `slim-disk-handoff-2026-06-30-AUDIT-and-seed-pivot.md`, `references/disk-physics-formulas.md` §22 (transonic eqns), `references/papers/` (Sądowski 2009/2011).

---

## Goal

Give the **coupled relax** an initial (Σ, V) state whose surface density is **within vertical-column capacity at every node**, by reseating Σ onto the **transonic branch** (|V|=c_s at the sonic point, mass conservation elsewhere) instead of the thin-disk profile. This unblocks the base rung of the f_Edd continuation ladder (currently dies at "0 inner-iters" because ~6/18 columns can't hold the thin-seed Σ) **and** serves as the decisive test of whether the split architecture can reach a self-consistent inner disk.

## The correction that defines scope

The coupled relax seeds from **`build_thin_disk_seed` + `calibrate_seed_to_manifold`** (see `slim_coupled_walk_probe.cpp` Phase 1), NOT from `build_slim_disk_seed`. So lever C is a **new sibling seed builder for the coupled path**; it does not touch `build_slim_disk_seed` (radial-solver only, one guarded call site) or `build_thin_disk_seed`. Zero existing callers disturbed.

## What lever C is / is not

- **Is:** a *seed* (initial guess). The coupled relax owns the final state and is free to move Σ.
- **Is not:** a solver, and not a fallback closure (every node stays on the full convective column BVP — no unphysical hybrid).
- **Decisive-test property:** seed transonically-low → relax starts feasible → watch it. Converges near the seeded Σ ⇒ the high thin-seed Σ was a **seed artifact** (disk physical, split architecture fine). Relax drives Σ back into the capacity wall (0-inner-iters returns / merit stalls as Σ climbs) ⇒ the demand is **physical** ⇒ that is the evidence that justifies the monolithic Sądowski rebuild.

---

## Architecture

New function (in `src/slim_disk_coupled.cpp` or a small new TU, exported for probes):

```
std::vector<double> build_transonic_coupled_seed(const SlimDiskInputs& in,
                                                 const OpacityLUTs& op,
                                                 const ColumnOpts& copt);
```

Returns the same packed radial state layout the coupled relax consumes (per-node Σ, V, ℓ, T_c + tail r_s/ℓ_in), so it is a **drop-in replacement** for `build_thin_disk_seed(...)` in the coupled seed path.

### Algorithm (per the verified transonic physics)

1. **Base grid + ℓ, r_s.** Reuse `build_thin_disk_seed` for the log-r grid, ℓ(r) profile, and the f_Edd-aware r_s / ℓ_in guesses (lines 931–932 style). Node 0 = sonic point r_s.
2. **Sonic anchor.** At r_s, get c_s from `one_zone_closure(Σ, T_c, r_s).c_s` (CGS). Set |V(r_s)| = c_s/c (Mach 1). Compute Σ(r_s) from the **verified mass-conservation inverse**:
   `Σ = Ṁ·√(1−V²) / (2π|V|·√Δ·r_g·c)`.
3. **Transonic V(r) outward (r > r_s).** Prescribe |V(r)| declining smoothly from c_s/c at r_s to the thin-disk subsonic |V| at r_out — a monotone log-r interpolation in |V| (NOT a full ODE integration; see Fork below). Σ(r) from the same inverse at each node.
4. **Capacity guard.** For each node, if Σ(r) still exceeds `η · Σ0_capacity(r)` (η≈0.9, capacity = max converged column Σ0 at that geometry, computed at the relax's n_z), **clamp** Σ down to η·capacity and recompute |V| from `Vfrom`. Guarantees 18/18 feasible by construction. Log every clamped node (no silent capping).
5. **T_c on the feasible manifold.** For each node's (now-feasible) Σ, set T_c via `build_coupled_seed` / the f_adv≈0 manifold (same as `calibrate_seed_to_manifold`).
6. **Return** the packed state.

### The one real fork — V(r) construction sophistication

- **(b, RECOMMENDED) Smooth prescription + capacity guard.** Step 3 as above. Rationale: it is a *seed*; the relax refines it. YAGNI — do not integrate the stiff neighbor-coupled transonic ODE for a starting guess. The capacity guard, not the ODE fidelity, is what guarantees the relax can start. Cheapest path to the decisive test.
- **(a) Full one-zone transonic ODE integration** outward from r_s (assemble a loop over `calD0`/`calN1`). More faithful to Sądowski's trial-solution construction, materially more work, and its extra fidelity is wasted on a seed the relax will move. Reserve for later *only if* (b)+guard proves too far from the basin.

Recommendation: **build (b)+guard now.** If the relax won't converge from it, escalating to (a) is a scoped follow-up.

---

## Testing / validation gates (TDD)

1. **Feasibility gate (the point of the lever):** at a=0.9, f_Edd=0.001, n_z=96, every one of the 18 seed nodes' columns solves (`solve_column_coupled` converged 18/18). *Write this first* — it is the direct pass/fail.
2. **Transonic-anchor gate:** |V(r_s)| = c_s/c to tolerance; Σ(r_s) matches the mass-conservation inverse; |V| monotone-declining outward; Σ within η·capacity at every node.
3. **Mass-conservation round-trip:** `Vfrom(r, Σ(r))` reproduces the seeded V(r) to ~1e-10 (the inverse is exact).
4. **No-silent-cap:** every capacity-clamped node is logged with (r, Σ_transonic, Σ_clamped, capacity).
5. **Integration test (the deliverable):** re-run `slim_coupled_walk_probe` seeded from `build_transonic_coupled_seed` at n_z=96. Honest outcome:
   - base rung **converges** (Newton iterates, merit → tol) ⇒ seed artifact confirmed; proceed to walk toward f_Edd=0.9;
   - base rung **fails with Σ climbing back into the wall** ⇒ physical demand ⇒ trigger the monolithic-rebuild decision.
   Either is a valid, informative result — do not force convergence, do not loosen a gate.

---

## Non-goals / limits

- Unblocks the **base rung**, not the whole POC (f_Edd=0.9). Walking up is separate.
- Does not guarantee convergence — it guarantees a *feasible, transonic-shaped start*, then reports honestly.
- No new physics formula (the mass-conservation inverse is the algebraic inverse of `Vfrom`, verified; the capacity guard is numerical). So no §-doc edit required; the inverse is recorded here.

## Workflow

Never `git commit` — hand the message over. TDD, gate 1 first. Present every reviewer rec & WAIT. Convergence ≠ physical; a clean honest stall is a valid outcome. Fable available for a second-opinion pass on the integration-test verdict.
