# Branch-consistent column seeding — design

**Date:** 2026-08-09
**Branch:** `fix/volumetric-ring`
**Status:** approved design → ready for implementation plan

## Goal

Stop the coupled slim-disk relax from seeding a radial node onto the **wrong thermal branch**.
Two small, targeted changes. The acceptance test is that the base rung's angular-momentum
residual at node 9 collapses and merit descends past its current 2.758 floor.

This is deliberately the *cheap* experiment: it distinguishes "branch selection was the whole
problem" from "there is something else at node 9 too", before we commit to the larger
tabulated-vertical-structure redesign.

## Evidence (measured, not inferred)

From `slim-multiroot-probe` (`982ffef`) at the base rung (a=0.9, f_Edd=1e-3, N=18, n_z=256),
run against a live checkpoint:

- Nodes 8, 9 and 10 **each admit two roots** on the seed manifold: a cool gas-dominated branch
  (β ≈ 0.71–0.85) and a hot radiation-dominated one (β ≈ 0.035–0.098). The fold sits at β ≈ 0.33.
- **Nodes 8 and 10 sit on their cool roots. Node 9 does not** — it is the outlier carrying ~97%
  of the residual in the angular-momentum rows (`ang` = 26.0 of merit 3.06; `mass` = 4.4e-4,
  `reg_D0` = 0.024, `reg_N1` = 1.6e-6 — i.e. mass conservation and sonic regularity are *fine*).
- Node 9's live T_c sits **within 2% of the fold nose**, where ∂Σ₀/∂T_eff = 0 and the closure
  Jacobian degenerates — so Newton has no usable gradient there. That is the stall.
- Its live state back-solves to **f_adv = −0.224** (negative ⇒ off the physical advective manifold),
  and its live P is 7–11× its neighbours' — *intermediate* between branches, consistent with
  sitting at the fold rather than cleanly on the hot branch.
- **Mechanism, demonstrated:** `estimate_Teff_guess ∝ T_c`, so a hot incoming T_c seeds above the
  fold and re-selects the hot branch. Node 8's production ladder rung `Teff × 4` converges on its
  hot root (T_c = 2.2336e7) — first-to-converge, no branch criterion.
- **Method note:** a *cold* T_eff scan finds only ONE root; `solve_column_bvp` fails to converge
  across a window straddling the fold. The upper branch is only reachable by warm-marching.

Literature (`docs/superpowers/references/slim-disk-literature-2026-08.md`): branch selection is
**structural** — the sonic-point regularity condition is the eigenvalue of a two-point BVP, and the
branch is fixed by **continuity from the outer boundary inward**. Local multiplicity at a node is
*expected and is not itself an error*. The documented failure mode "passes the sonic radius but
follows an improper branch" is precisely ours. The prescribed remedy is **continuation in r**, not
per-node multi-start.

## SCOPE CORRECTION (important) — with one proven exception

An earlier draft proposed also adding a branch guard to `solve_column_coupled`'s cold multi-start
fallback. **That is unnecessary.** That path varies only *starting guesses* on a mutable copy while
Σ_target and T_c stay pinned (`src/disk_column_coupled.cpp:1166-1171` documents this explicitly),
and the probe's Stage D confirmed the coupled solve is **single-valued at pinned (Σ, T_c)**: 18 of
21 starting-guess combos — including all 13 production combos — converged to the same root.

**Proof that T_c cannot be re-seeded during evaluation** (verified in code, 2026-08-09):
- All four consumers take `const std::vector<double>& U` — `eval_node_coupled` (:213),
  `slim_coupled_residual` (:513), `slim_coupled_residual_frozen` (:621),
  `slim_coupled_reduced_jacobian` (:667). `T_c = U[4i+3]` is compiler-enforced read-only there.
- The only seed builder that writes `U[4*i+3]` is `build_transonic_coupled_seed`
  (`src/slim_disk_coupled.cpp:199`, inside the function starting :155) — the **shelved lever-C
  builder, which grep confirms is never called**. Dead code.

**THE EXCEPTION — `deglitch_sigma_outliers` DOES overwrite T_c mid-relax.** Called from
`src/slim_disk_coupled.cpp:1105` after every accepted step, it replaces an outlier node's Σ, T_c and
V with the local log-median over a ±3-node window:
```cpp
const double Tnew = std::exp(local_median(i, 3));
U[4*i+0] = std::max(Snew, kSigmaFloor);
U[4*i+3] = std::max(Tnew, kTFloor);        // T_c replaced outright
U[4*i+1] = Vfrom(i, U[4*i+0]);
```
This is branch-**repairing**, not branch-flipping — it moves an outlier *toward* its neighbours. But
it is a discontinuous T_c write outside the Newton, so "the branch is chosen only at seeding" is
true of *evaluation* but not of the full iteration.

**It never fires for node 9 because it is keyed on the wrong variable:** the trigger is Σ deviating
by more than `kOutlierFac = 8.0`, and node 9's Σ is only ~1.5× off the local median while its **T_c
is 2.33× off**.

### Fix 0 (fast experiment, runs first)
Extend the outlier trigger to fire on **T_c** deviation as well as Σ (suggested
`kOutlierFacT = 2.0`; a legitimate node sits within ~1.2× of the window median given the ~0.182
ln-r spacing, while node 9 is at 2.33×). This reuses machinery that is already written and running,
and answers the decisive question — *does forcing node 9 onto its neighbours' branch unstick the
relax?* — faster than anything else.

**Treat it as an experiment, not the fix.** Replacing a Newton unknown with a median is blunt, and
the audit already flagged that repeated firing near convergence means it is fighting the Newton. Its
value is the answer, not the mechanism.

### Ordering (and why Fix 2 is NOT first)
**Fix 0 → Fix 1 → Fix 2.** `SIGMA_SEED_BAND` controls *scatter, not branch*: `build_coupled_seed`
secants on T_eff to match Σ₀, and Σ₀(T_eff) is single-peaked, so for any Σ_target below the nose
there are **two exact roots**. The *starting guess* picks the side; the *band* only sets how close
the secant must get. The probe's measured ±10% spread was **within the cool basin**, while the
branches differ by ~2×. Fix 2 first would be a wasted run.

---

## Fix 1 — r-continuation seeding (replaces the blind multi-start ladder)

**Site:** `calibrate_seed_to_manifold` in `tools/slim_coupled_walk_probe.cpp` (~L69-97).

**Current behaviour:** a serial loop over nodes, each seeding independently from the thin-disk
seed's own `U[4i+3]`, then `build_coupled_seed(...) || build_coupled_seed_advective(...)`. Nothing
couples node *i*'s seed to node *i−1*'s result, so each node independently draws a branch.

**New behaviour:** march **inward, from the outer node to the inner** (i = N−1 → 0), seeding each
node's `Teff_guess` and seed `T_c` from the **previously converged neighbour's** values rather than
from the thin-disk seed.

Direction matters: the outer disc is the trustworthy anchor — low Σ, low T, firmly gas-dominated,
single-valued, far from the fold. Marching inward propagates that unambiguous branch choice toward
the inner nodes where multiplicity appears. This mirrors the literature's "continuity from the outer
boundary inward".

The first (outermost) node has no predecessor and keeps the current seeding path unchanged.

If a node fails to seed from its neighbour's value, fall back to the existing independent path for
that node only (so we never lose feasibility we already have), and **report** which nodes fell back
— a node that needs the fallback is exactly where a branch flip can still occur.

## Fix 2 — `SIGMA_SEED_BAND` acceptance ladder

**Site:** `src/disk_column_coupled.cpp:505` (`build_coupled_seed`) and `:538`
(`build_coupled_seed_advective`). Both use `constexpr double SIGMA_SEED_BAND = 0.30;`.

**Problem:** a seed is accepted when its Σ₀ matches the target within **30%**. The probe measured
the consequence: node 9's returned "manifold T_c" scattered over **8.78e6–9.82e6 (±10%) across
seeds *within the correct basin***. That is node-to-node noise injected straight into the initial
state, independent of the branch problem.

**Change:** try a **tightening ladder** — attempt acceptance at 0.05, then 0.10, then 0.30 — and
take the tightest band that yields a converged seed.

**Why a ladder and not simply a tighter constant:** 0.30 is the value that won 18/18 seed
feasibility at n_z=256, and that was hard-fought. Tightening it blindly risks losing feasibility at
the high-Σ inner nodes. The ladder gets the best available accuracy per node while preserving the
existing behaviour as the last rung, so feasibility cannot regress by construction.

Report the achieved band per node so we can see how much of the profile actually tightened.

---

## Acceptance gates

**Must not regress (existing gates):**
- `slim-omp-gate-probe`: tests (2)/(3)/(4) = `0.000e+00` PASS (parallel==serial, deterministic).
- `test-slim-coupled-jacobian`: 0 failures.
- Seed calibration must still report **18/18** manifold-set and solve-feasible.

**The actual acceptance test** — a base-rung run (`SLIM_DIAG=1`, `SLIM_CHECKPOINT=…`,
`OMP_NUM_THREADS=12`), reading the instrumentation added in `3b8b8d6`:
1. **Node 9's seeded T_c comes out on the cool branch** (~9e6, not ~1.7e7). Checkable from the
   `[calib]` output and the `it=000` checkpoint *without waiting for the relax* — this is the fast
   signal and should be checked first.
2. **`ang` collapses.** It is currently 2.60e+01 with its max pinned at row 27 (= node 9) every
   iteration. A large drop, and the max moving off node 9, is the success signature.
3. **Merit descends past 2.758.** The previous run asymptoted there over 6 iterations
   (3.062 → 2.886 → 2.799 → 2.761 → 2.759 → 2.758).

**Honest partial-success criterion:** node 9's Σ is *also* anomalous (1.488e4 vs neighbours 9587
and 1.279e4), so even on the cool branch its P stays ~2× its neighbours'. Expect the pressure
excess to fall from ~27× to ~2×, which should remove the wall Newton is hitting — but full
convergence additionally requires Newton to relax Σ₉ down. **Branch selection fixing the seed but
not reaching `merit < kMeritFloor = 5e-3` is a partial success and a useful result**, not a failure:
it would localise the remaining error to Σ rather than to branch choice.

## Non-goals (explicitly out of scope)

- **The vertical-structure table.** Next step, contingent on what this measures.
- **Extending r_out beyond 50M.** At f_Edd=1e-3 advection is negligible and the NT outer boundary
  condition is valid at 50M. It becomes invalid as f_Edd→1 (Sądowski's convective zone alone reaches
  300M at 0.1 Ṁ_Edd), so the move to r_out=1000M / 25 nodes is deferred to when we walk f_Edd up,
  and gets measured on its own.
- **Any branch guard in `solve_column_coupled`** — see the scope correction above.
- **The energy-equation Q_adv discrepancy.** Our `Gbalance` uses the two-term one-zone bracket
  while `calN1_coupled` carries all four terms of Sądowski Eq. (29) (including `η₃·dlnη₃/dr` and
  `Ω⊥²η₄·dlnη₄/dr`). The documented justification for two terms — "the one-zone radial model treats
  η-moments as constant ⇒ their gradients vanish" — no longer holds now that the vertical BVP
  supplies them per node. Recorded here as a **known discrepancy requiring a derivation pass**, not
  fixed in this change. It is secondary (`ene` = 0.334 vs `ang` = 26.0).
- **α(r) from arXiv:2603.10997**, and the graduated ≥0.3 L_Edd validity warning. Both recorded in
  the literature review for later.

## Risks

1. **Feasibility regression.** Both fixes touch the path that won 18/18. Both are designed with the
   current behaviour as an explicit fallback rung, and 18/18 is a hard gate.
2. **Inward marching could propagate a *bad* choice** as readily as a good one if the outer anchor
   is itself wrong. Mitigated at f_Edd=1e-3, where the outer disc is unambiguous; revisit when r_out
   moves.
3. **This may not be sufficient.** See the partial-success criterion — that outcome is still
   informative and is the reason for doing the cheap experiment first.

## Workflow constraints (standing)

- **NEVER `git commit`** — hand the message over and wait.
- One change at a time; present every reviewer recommendation with a take and wait for the call.
- Subagent-driven implementation; doc-first for any formula edit.
