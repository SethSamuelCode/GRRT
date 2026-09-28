# Stall diagnosis on RunPod — design

**Date:** 2026-09-28
**Branch:** `fix/volumetric-ring` (HEAD `e7198f4`)
**Status:** DRAFT for approval — nothing here is implemented or committed
**Author:** Hermes (with Seth), following the standing workflow constraints in
`slim-disk-handoff-2026-07-26.md` (no commits without Seth, one change at a time,
instrumentation-only changes gated bit-identical, doc-first for formula edits)

## Goal

Find and fix the mechanism pinning the base-rung relax (a=0.9, f_Edd=1e-3, N=18,
n_z=256) at merit ≈ 1.113 (kMeritFloor = 5e-3), then walk f_Edd toward 0.9.
Compute-heavy steps run on a RunPod pod (Seth's account, MCP control plane + SSH);
the laptop is for code reading, edits, and gates only.

## State at HEAD (measured, from `e7198f4` + handoff)

- Fix 0 (de-glitcher T_c trigger, kOutlierFacT=2.0) fired once on node 9 as designed:
  merit floor 2.758 → ~1.113, ang L2 2.60e+01 → 8.94, ang max moved off node 9.
  Seed still 18/18. Gates unchanged.
- NOT converged. Remaining residual: broad, with spikes at **node 16** (r≈41.7M,
  the known outer-disk β/f_adv sawtooth, task #140: "mesh/convergence artifact")
  and **node 0** (sonic point, `rad`).
- Stall triad from the 2026-08-09 trajectory (pre-Fix-0): μ doubling per iteration,
  maxrel collapsing, r_s frozen at 2.2744–2.2745 despite a live Jacobian column.

### Derived: ang still owns the merit (arithmetic on the recorded numbers)

merit² = Σ_groups L2²/n with n = 4N+2 = 74. At the recorded floor:
ang L2 = 8.94 → ang²/n = 1.080; merit² = 1.239. **ang ≈ 87% of merit²** — the
"second mechanism" is still dominantly the angular-momentum rows, now concentrated
at the outer boundary (node 16) instead of node 9. To be confirmed against the full
per-group table from run E1 below (we only have ang from the commit message).

## Root-cause hypotheses (ranked by evidence, each with a discriminating test)

- **H1 — Outer-boundary mesh artifact (task #140 sawtooth).** The ang spike sits at
  node 16 of 17 on a log grid ending at r_out=50M; #140 was independently observed at
  f_Edd=0.05 as a β/f_adv sawtooth and called a mesh/convergence artifact. Candidate
  mechanisms: even-odd decoupling of the centered ℓ differencing on the stretched grid,
  or the outer BC stencil interacting with the last interior node.
  *Discriminates:* E2 mesh/domain sweep — if the spike tracks the mesh (stays at
  second-to-last node, changes amplitude with N, moves with r_out) it is discretization;
  if it pins to r≈41.7M regardless, it is physical/BC-validity.
- **H2 — `slim_fadv_ok` grid mismatch corrupting the line search (checklist #7, CONFIRMED
  in code).** `slim_disk_radial.cpp:2526` builds its rgrid from `in.r_in` (=0.5·r_ISCO)
  while the state U lives on [r_s, r_out] (r_s≈0.98·r_ISCO; cf. `slim_disk_coupled.cpp`
  L104 and walk-probe L104). Inner nodes are evaluated at ~half their true radius
  (Qadv carries 1/r² ⇒ ~4× error at node 0), and this gate REJECTS line-search steps
  (`slim_disk_coupled.cpp:1077`). A mis-evaluated gate near the sonic point can pin
  exactly the node-0 `rad` spike and truncate λ.
  *Discriminates:* E0a rejection-count instrumentation (observability only), then E0b
  one-line fix (use r_s = U[4N+1]) + A/B.
- **H3 — r_s FD step too small (audit finding #4).** h=1e-5 absolute vs r_s≈2.27
  (4.4e-6 relative) against column-solver noise ⇒ noisy gradient for the transonic
  eigenvalue ⇒ LM distrusts the model ⇒ μ climbs ⇒ stall. Handoff notes this is
  path-only (changes convergence path, not the root).
  *Discriminates:* E1's reg_D0/reg_N1/rad-at-node-0 magnitudes; then FD-step A/B
  (1e-5→1e-3 relative at `:720`, ℓ 1e-6→1e-4 at `:729`) if and only if those groups
  are load-bearing.
- **H4 — Energy-equation discrepancy (Gbalance two-term vs calN1_coupled four-term,
  spec §Non-goals).** Was secondary (ene=0.334 vs ang=26). As ang falls, ene's share
  grows automatically.
  *Discriminates:* E1 ene L2 at the new floor. If it becomes dominant after an ang fix,
  do the derivation pass (doc-first, §22-24 formulas).
- **H5 — Seeding still branch-inconsistent (spec Fix 1, not yet implemented).**
  Fix 0 repaired node 9 mid-relax; seeds may still land on mixed branches elsewhere.
  Weaker now: the ang max moved OFF node 9 and the remaining spikes are at the
  boundaries, which inward r-marching does not specifically target (the outer anchor
  keeps the existing path by design).
  *Discriminates:* E1's it=0 per-group table + checkpoint branch census (β per node).

Note on spec ordering: the approved 2026-08-09 spec ordered Fix 0 → Fix 1 → Fix 2.
Fix 0's result falsified its premise that node-9 branch selection was the whole
problem, so this plan re-orders to evidence-first (E1/E2 before committing to Fix 1).
Fix 1/Fix 2 remain in the queue as branch hygiene once the boundary mechanism is
identified. Flagging the deviation explicitly for Seth's call.

## Phase 2 execution plan

### Step 0 — laptop, no pod (cheap, correctness-first)

- **E0a (instrumentation-only):** count and print fadv-gate rejections per outer
  iteration (λ before/after, which nodes failed, absmax f_adv). No computed value may
  change; gate with slim-omp-gate-probe bit-identity + test-slim-coupled-jacobian.
- **E0b (single fix):** `slim_fadv_ok` uses r_s from U[4N+1] (signature gains r_s or
  reads U — implementation detail). Same gates. A/B on the N=12, n_z=96 fast point
  (the A/B rig from 3c001ad) before spending any pod time.

### Step 1 — pod bring-up (RunPod, MCP + SSH)

- Template: Ubuntu 24.04 base (ships GCC 13; C++23 OK) or any CUDA template +
  `apt install g++-13`. No GPU needed — the solver is CPU/OpenMP.
- Sizing: high-vCPU CPU pod if available in the region (target 32–64 vCPU, ≥64 GB RAM,
  20 GB disk). If CPU-only instances are unavailable/pricier, the cheapest GPU pod with
  ≥32 vCPU. Query list-cpu-types/list-gpu-types + get-capacity for price before create;
  Seth approves the hourly rate once, then start/stop is automatic.
- Verify: `./build.sh`, run gates (slim-omp-gate-probe, test-slim-coupled-jacobian,
  test-slim-disk-radial), then measure one outer iteration at
  OMP_NUM_THREADS ∈ {16, 32, 64} to pick the cost-efficient setting (load imbalance
  caps scaling; measure, don't assume).
- Long runs inside tmux on the pod; SLIM_DIAG=1, SLIM_CHECKPOINT=<prefix>, logs
  rsync'd back; pod STOPPED (not terminated while state matters) after each campaign;
  terminated when the campaign closes.

### Step 2 — evidence runs

- **E1 (the 12 h question, now hours):** base-rung relax, **N=18 — no lower** (see below),
  n_z=256, fixed Jacobian + Fix 0 + E0a/E0b, ~8–10 outer iterations or merit < 5e-3.
  Output: full per-group L2/max table per iteration, fadv-rejection counts, checkpoints,
  μ/maxrel trajectory. Reads which group pins merit and whether the stall triad persists.
  **N stays 18:** seed/solver failure locations are grid-dependent (proven in
  `specs/2026-07-12-robust-column-multistart-design.md`: N=18 fails at r≈3.9,14.0 while
  N=32 fails at r≈6.2,11.2,12.4, each grid's failures feasible in the other). Diagnosing
  a mesh artifact (H1) at reduced N would measure a different mesh's artifact. The Aug-09
  handoff item-4 suggestion ("feasibility is driven by n_z, not N") predates no test and is
  superseded by this evidence. N=12/n_z=96 remains valid ONLY as the established fast A/B
  rig for Jacobian-structure gates (as used in 3c001ad), never for convergence diagnosis.
- **E2 (mesh/domain sweep, config-only, runs parallel to E1 if vCPU allows):**
  N ∈ {12, 18, 24} × r_out ∈ {50M, 65M, 100M} at f_Edd=1e-3, seeded identically;
  2–3 iterations each suffice to read the node-16 spike's mesh dependence. Also
  SLIM_FD_ONESIDED=0 A/B on one point (free, env-only).

### Step 3 — decision gate (with Seth)

One targeted fix, chosen by the evidence:
- ang@node-16 dominant + mesh-tracking ⇒ outer-boundary stencil/BC change (design
  pass first: what does #140's sawtooth mode look like — read checkpoint β/f_adv
  profiles at nodes 13–17; consider one-sided ℓ differencing at the last node, or
  a smoother outer BC).
- rad@node-0 / reg_D0 load-bearing ⇒ H3 FD-step fix (path-only; pair with E1 rerun).
- fadv rejections frequent at inner nodes ⇒ already addressed by E0b; measure the delta.
- ene dominant after ang falls ⇒ H4 derivation pass (doc-first, formula review with
  Seth's oracles per standing constraint).
- seed branches mixed at it=0 ⇒ spec Fix 1 (r-continuation seeding), then Fix 2.

### Step 4 — convergence → continuation → render

On merit < 5e-3 at base rung: verify physical (H/r, β, f_adv, transonic V profile);
then f_Edd ladder 1e-3 → 0.01 → 0.1 → 0.3 → 0.9 with warm starts (higher rungs
currently fail on cold seeds — warm-start from the previous rung's checkpoint;
r_out growth to ~1000M/25 nodes enters at f_Edd ≳ 0.1 per the spec's deferred note,
measured on its own). Tabulate T_eff(r), H(r) → volumetric-disk render on this branch.

## Budget & safety

- Pod spending: start/stop via MCP around every campaign; nothing runs unattended
  without a wall budget (kDefaultWallSeconds / budget_wall_seconds already exist).
- Every code change: one at a time, gates green, commit messages handed to Seth.
- Physics claims that are load-bearing for a fix get flagged for Seth's oracle
  verification (standing constraint), with the relevant papers from
  ~/Projects/slimdisk (Sadowski 2011 = arXiv:1006.4309 is the architectural basis;
  branch-selection doctrine from the 2026-08 lit review §3).

## Open questions for Seth

1. Pod spend cap per campaign (e.g. $10/$25)? CPU-instance OK if cheaper than GPU pods?
2. Approve re-ordering (E1/E2 before spec Fix 1)?
3. ~~N=12 for parts of the diagnostic loop~~ — RESOLVED 2026-09-28: rejected on Seth's
   flag; grid-dependence evidence (2026-07-12 multistart spec) makes reduced-N diagnosis
   invalid for a suspected mesh artifact. N=18 for all physics runs.
4. Is there an existing task-list file where #140 lives, so the sawtooth history can be
   linked? (June handoffs reference it; I haven't found the tracker itself.)
