# Slim-disk handoff — AUDIT + SEED PIVOT (2026-06-30, later same day)

**Read this AFTER `slim-disk-handoff-2026-06-30.md`.** That doc ended at "run the Σ↔V check." This doc captures what the Σ↔V check + a full first-principles/literature audit of the radial solver found, and the resulting **pivot in approach**. The headline: **the way to f_Edd=0.9 is the global Sądowski slim seed → COUPLED relax, NOT arclength continuation through the fold (a confirmed dead end).**

---

## STATUS UPDATE (2026-07-02) — component B (convection) DONE; next = A + real coupling test

**Component B (MLT convection) is BUILT, VERIFIED, and COMMITTED** (plan `plans/2026-07-01-convective-column.md`, Tasks 1–4):
- `grrt::detail_bvp::{nabla_ad, c_p_gas_rad, mlt_solve_y, convective_gradient}` in `disk_column_bvp.h` (all §24 formulas, opus+Wolfram-verified; δ sign-trap resolved; τ_ml→0 radiative-limit guard Wolfram-verified).
- Wired into `node_deriv`; analytic Jacobian extended with convective `dT` partials (local-FD of `node_deriv`, **FD-oracle clean 0.0**). Pure-radiative reduction **bit-identical** (NT gate green). Two pure-radiative gates recalibrated to the convective model (verified via `slim-convection-verify-probe`: the deep-rad-pressure column is 94/96 convective, ∇ flattened 0.40→0.25, midplane cooled, β 7e-6→3e-3 — the documented convective signature).
- **Task 5 (Σ0-capacity-lift success metric): INCONCLUSIVE via proxy.** `slim-sonic-sigma-probe` is NOT a valid tool for the convective column — it computes c_s from the *pure-radiative* `one_zone_closure`, which returns garbage (cs/c~0.08) exactly where convection is active (high T_eff). On trustworthy (radiative-regime) rows col/req at r_s stays ~0.64 (unchanged), but that's where convection is OFF — uninformative. **The capacity lift is neither confirmed nor refuted.** USER DECISION (2026-07-02): skip the proxy; let the **real coupled relax** answer it (option 2).

**NEXT = component A (transonic-V seed) → wire `relax_coupled` with convective columns → run at f_Edd=0.9 (the real integration test).** Component A: `build_slim_disk_seed`'s Σ is 10–100× too high (subsonic V); re-derive (Σ,V) onto the transonic branch (|V|→c_s at r_s, declining outward; Σ from mass conservation). Then seed `relax_coupled` from it with the now-convective columns and see if the inner disk is feasible / the relax converges at f_Edd=0.9. That is the ground-truth answer to "do convection + a good seed unblock f_Edd=0.9". Component A needs its own brainstorm→spec→plan (fresh work).

---

## 0. (SUPERSEDED by the status update above) THE ORIGINAL DECISION: (A) transonic-V seed + (B) convection #13

**Reach a physical (a=0.9, f_Edd=0.9) disk via TWO committed pieces:**
- **(A) Transonic-V seed fix** — `build_slim_disk_seed` produces Σ **10–100× too high** because its inner V is far too subsonic (it thickened the shape but kept thin-disk-like high-Σ/low-V). Re-derive (Σ,V) onto the transonic branch (|V|→c_s at r_s, declining outward; back-derive Σ from mass conservation) → Σ drops to the true ~1e4. **Dominant, tractable lever.**
- **(B) Convection (#13), GUARANTEED** — at f_Edd=0.9 the inner disk is **at the pure-radiative column's capacity edge** (rigorous sonic-Σ test §2a: col/req=0.64 at r_s, ~1.1–1.6 just outside; converged capacity ~1e4–1.3e4 ≈ self-consistent sonic Σ ~1e4). Literature is explicit: convection sets in as flux→Eddington; the inner near-Eddington slim disk IS convective (Sądowski 2011, mixing-length). Convection raises the column Σ0 capacity (~2×, toward the literature f_F≈0.94) which (i) closes the marginal inner gap and (ii) makes A's feasibility forgiving. User decision: convection is REQUIRED for a physical near-Eddington render anyway — build it.

**Recommended order:** B first (column-level, independently gated by f_F→0.94 + capacity↑), then A (seed targets the convective capacity), then couple `relax_coupled` from the fixed seed with convective columns → converge at 0.9, verify physical. Full design in §5.

---

## 1. THE AUDIT VERDICT — the transonic physics is CORRECT (no rotten formula)

A full audit of the radial solver against first principles + S09/S11/AF13 (via the externally-verified `references/disk-physics-formulas.md` §22–23 + a code-trace) found the transonic machinery is **textbook-correct Sądowski**:

| Required physics (S11 / ref §22) | Code | Status |
|---|---|---|
| ℓ_in is an **eigenvalue** via sonic regularity 𝒩₁(r_s)=0 | outer bracket on `g(ℓ_in)=R[4N+1]=𝒩₁(r_s)` (`solve_outer_bracket`) | ✓ |
| r_s **found, not prescribed**, via 𝒟₀(r_s)=0 (Mach 1) | inner Newton unknown `U[4N+1]`, driven by `R[4N]=𝒟₀` | ✓ (empirical r_s fit = initial guess only) |
| Radial-momentum transonic ODE dlnV/dlnr=(𝒩₁/𝒟₀)(1−V²) | Group 3 + L'Hôpital at the sonic node 0 | ✓ |
| Domain [r_s, r_out] subsonic branch | node 0 = sonic point | ✓ matches S11 |
| Verified formula set (Q_adv bracket, full Γ, metric factor, η₃(β)) | all §23 corrections in | ✓ |

**Consequence:** the repeated "moving frontier" was never a wrong equation. Every prior narrow probe found a true-but-narrow fact downstream of a sound foundation. The difficulty is **robustness + seeding**, a more tractable class.

**The known pure-radiative-vs-Sądowski-convective difference (f_F≈0.42, §23) does NOT block f_Edd=0.9** — it lives in the vertical closure (internal T_c/H); emitted flux/T_eff are correct by energy balance. Not a reason we can't reach 0.9.

---

## 2. THE REACH TESTS — both radial paths currently fail at the f_Edd≈0.11 fold

Empirical, this session (a=0.9, target f_Edd=0.9):
- **`solve_slim_disk_radial`** (spin-walk + Ṁ-ladder): 146,724 inner iters, 14 min wall-budget exhausted, **NOT CONVERGED**. (`slim-full-target-probe 0.9 0.9 32`.)
- **`solve_slim_disk_arclength`** (dedicated fold-crosser): traces the branch, **max f_Edd = 0.104**, "crossed f_Edd=0.11 = NO", **fold detected = YES**, Ṁ̇ sign oscillating around the turning point. (`slim-arclength-probe`.)

So the handoff/memory claim "solve_slim_disk_radial reaches f_Edd=0.9" is **NOT currently reproducible at a=0.9.** (Either regressed since #150, or #150's "0.9" was the torus now rejected by the tightened f_adv gate / changed by #11/#12 physics.)

---

## 3. THE PIVOT — arclength-through-the-fold is a CONFIRMED dead end (from 06-13 history)

`slim-disk-handoff-2026-06-13.md` §2 (verbatim), on `solve_slim_disk_arclength`:
> "hardened 2026-06-13 (committed): secant-based tangent orientation (the textbook fix), fold-aware `ds` shrink, f_adv validity gate in the corrector. Rounds the fold cleanly to 0.143 on physical states; **does NOT ride the unstable middle to the slim branch (Sądowski doesn't either — use the global seed, not arclength).**"

The f_Edd≈0.11 fold leads onto the **radiation-pressure Lightman–Eardley unstable middle branch** — a numerical saddle the corrector *dithers* on. **The corrector was already hardened (secant tangent, fold-aware ds, f_adv gate) and the conclusion was: do not continue through the fold; jump to the slim upper branch via the global seed.** The 06-13 §0 resume action set the path: build the proper global Sądowski slim seed (became `build_slim_disk_seed`, tasks #147/#150).

**Important correction to a stale assumption:** "Lever 1 = un-nest the eigenvalue" is a NON-issue — the **arclength corrector already solves ℓ_in jointly** (full augmented 4N+3 Newton, both regularity rows kept, no outer bracket; `slim_disk_radial.cpp` ~3388–3398). Un-nesting is not the missing piece. The nested outer bracket only exists in the *simple-ladder* `relax_structure` path, which is not the fold vehicle.

---

## 4. WHY THE PIECES WERE NEVER JOINED (the actual gap)

The full arc, grounded:
1. Global slim seed built (#150) → reached a one-zone f_Edd=0.9 disk.
2. Diagnostic A (#153): that one-zone 0.9 is a **torus** — a one-zone *closure* inadequacy → motivated the vertical-BVP coupling.
3. The coupling was built & validated (C1–C5, Schur Jacobian, NT gate — see the 06-30 handoff) but **can't start from the thin-disk seed** (columns can't hold the thin-disk Σ).
4. **Σ↔V check (this session):** the demanded thin-disk Σ is a *seed artifact*. The **self-consistent slim Σ FITS the column.** Proven two independent ways:
   - `slim-sigma-v-probe`: the V required to carry Ṁ through the column's Σ0 capacity is sub-light by 3–5 orders of magnitude at every node/f_Edd (≤1.6e-3 c even at f_Edd=0.9) → no superluminal/forced demand → NOT convection (#13).
   - `slim-edd09-dump` (converged one-zone slim profile at f_Edd=0.05): inner Σ ≈ 8e3–2e4 (vs thin-disk seed ~1e5 at the same f_Edd), transonic inner V (−1.4e-3 c at ISCO) — its Σ sits right at column capacity.

**So the two completed workstreams — the global slim seed and the vertical-BVP coupling — were never connected.** Connecting them is the path.

---

## 5. THE DESIGN — global slim seed → coupled relax

**Goal:** converge a *physical* (a=0.9, f_Edd=0.9) disk by seeding `relax_coupled` from `build_slim_disk_seed` instead of `build_thin_disk_seed`. No fold traversal. The column closure prevents the torus; the slim seed gives the transonic shape + column-feasible Σ.

- **Task 1 — Reality-check the slim seed at 0.9.** Dump `build_slim_disk_seed(a=0.9, f_Edd=0.9)` raw: inner-peaked H/r? β→1 outward? sonic point inside ISCO? Σ ≤ column capacity (~2e4)? Decides everything; cheap. (Running now.)
- **Task 2 — Wire `relax_coupled` to seed from the slim seed** (+ the existing `calibrate_seed_to_manifold` T_c recal), attempt convergence at f_Edd=0.9.
- **Task 3 — If it stalls, harden the *coupled* relax** at this regime (reuse the Ruiz/affine-invariant machinery already built for the column) and/or improve the slim seed shape per Sądowski 2009 §3. **NOT arclength.**
- **Task 4 — Verify physical**: literature-match H/r, β, f_adv profiles; tightened f_adv>−1 gate; NT reduction at low Ṁ still green; convergence ≠ physical (judge with independent checks).

---

## 6. STATE / FILES / UNCOMMITTED

- **Uncommitted (this session, diagnostics):** `tools/slim_sigma_v_probe.cpp` (the Σ↔V check) + its `CMakeLists.txt` target; this handoff doc. Hand the user a commit message; never `git commit`.
- **Key code:** `src/slim_disk_radial.cpp` — `build_slim_disk_seed` (~902, the global slim seed, Sądowski §3 shape), `build_thin_disk_seed` (~596), `relax_structure` (~2640, nested-ℓ_in inner), `arclength_corrector` (~3423, un-nested), `slim_analytic_jacobian` (~1564, ℓ_in column at 1992). `src/slim_disk_coupled.cpp` — `relax_coupled`, `ColumnOpts::n_z=24` (still 3× too coarse; bump to ~96–128 deliberately — does NOT unblock but is a real correctness fix). `src/disk_column_coupled.cpp` — the augmented column + honest Σ0 seed gate (461c394).
- **Probes:** `slim-full-target-probe [a] [f_Edd] [N]` (production reach), `slim-arclength-probe` (fold trace), `slim-edd09-dump` (one-zone slim profile), `slim-sigma-v-probe` (Σ↔V), `slim-coupled-hard-probe HARD_NODE=-1` (per-node column feasibility), `slim-coupled-nz-probe` (Σ0 resolution).
- **Docs:** `references/disk-physics-formulas.md` §22–23 (verified transonic eqns + trap checklist #10 = the eigenvalue), `slim-disk-handoff-2026-06-13.md` (the arclength-abandon + global-seed decision), the 06-30 handoff (coupling state).

---

## 7. WORKFLOW CONSTRAINTS (non-negotiable)
- **Never `git commit`** — hand the message over. **Present every recommendation with a take and WAIT.** **Doc-first** for formula edits.
- **Verify load-bearing claims (opus + Wolfram, NOT sonnet).** The user's consistency-check instinct has caught real bugs/stale-narratives repeatedly — honor it.
- Gates green; don't tune to pass; **convergence ≠ physical**; honest fallback.
- The 3-way split (incorrect-model / physically-unstable / genuinely-impossible) is the governing epistemic theme — distinguish them with independent checks.
