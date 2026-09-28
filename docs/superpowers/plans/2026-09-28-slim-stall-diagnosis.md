# Slim-disk stall diagnosis — implementation plan

**Date:** 2026-09-28
**Branch:** `feat/slim-disk` (created from `fix/volumetric-ring` @ e7198f4; volumetric branch untouched)
**Design:** `docs/superpowers/specs/2026-09-28-stall-diagnosis-runpod-design.md` (APPROVED by Seth 2026-09-28)
**Status:** DRAFT for approval — not started

## Goal

Instrument and fix the confirmed `slim_fadv_ok` grid bug, then run the instrumented
base-rung relax (E1) and a mesh/domain sweep (E2) on a RunPod CPU pod to identify the
mechanism pinning merit ≈ 1.113 (kMeritFloor 5e-3) at a=0.9, f_Edd=1e-3, N=18, n_z=256.

## Global constraints (standing, from handoff-2026-07-26 + Seth 2026-09-28)

- NO `git commit` without explicit instruction — hand commit messages to Seth.
- One change at a time; all gates green before the next change.
- Instrumentation (Task 1) must be OBSERVABILITY-ONLY: no computed value changes;
  verified by `slim-omp-gate-probe` (parallel == serial bit-identical).
- Physics diagnosis at N=18 / n_z=256 ONLY (grid-dependence evidence, 2026-07-12
  multistart spec). N=12/n_z=96 is the fast A/B rig for structure gates only.
- RunPod spend ≤ $10 per campaign; pod stopped between campaigns.
- Doc-first for any formula edit (references/disk-physics-formulas.md §22–24).
- Load-bearing physics claims flagged for Seth's oracle verification — **Seth runs
  opus/fable himself via his Claude Code subscription; NEVER invoke those models from
  this harness. Wolfram checks Hermes may run directly.**

## File structure

| File | Role |
|---|---|
| `src/slim_disk_coupled.cpp` | Task 1: line-search rejection counters + diag print (E0a) |
| `src/slim_disk_radial.cpp` | Task 2: `slim_fadv_ok` grid-origin fix at :2523–2531 (E0b) |
| `tools/slim_coupled_walk_probe.cpp` | Task 1: env knobs (harness-only) |
| `docs/superpowers/specs/2026-09-28-stall-diagnosis-runpod-design.md` | findings get appended per task |

Gate binaries (existing cmake targets): `test-slim-disk-radial`, `test-slim-jacobian`,
`test-slim-coupled-jacobian`, `slim-omp-gate-probe`, `test-column-coupled`.
Build: `./build.sh` (Release, binaries land in `build/`, no suffix on Linux).

---

## Task 1 — E0a: line-search instrumentation + walk-probe env knobs

### 1a. Rejection counters (`src/slim_disk_coupled.cpp`, line search ~:1069–1084)

Declare three counters + accepted-λ tracker at the top of the OUTER iteration scope
(same scope as `F_old`/`pred`; reset per outer iteration — they accumulate across LM
retries within the iteration). Then transform the λ-loop minimally:

BEFORE (exact, :1069–1084):
```cpp
bool physical=false; double F_new=F_old; bool infeas_try=false;
for (int ls=0; ls<40; ++ls) {
    Utry.assign(U.begin(),U.end());
    for (int i=0;i<n;++i) Utry[i]+=lam*dU[i];
    physical=true;
    for (int i=0;i<N&&physical;++i){ const double S=Utry[4*i+0],V=Utry[4*i+1],T=Utry[4*i+3];
        if (S<=0.0||T<=0.0||!(V<0.0)) physical=false; }
    if (physical){ const double rs=Utry[4*N+1]; if(!(rs>in.r_in&&rs<in.r_out)) physical=false; }
    if (physical && !slim_fadv_ok(in, op, Utry, 50.0)) physical=false;
    if (physical){
        merit_try=eval_merit(Utry,Rtry,infeas_try);
        if (infeas_try || !std::isfinite(merit_try)) { physical=false; }  // a column failed
        else { F_new=merit_to_F(merit_try); break; }
    }
    lam*=0.5;
}
```

AFTER (changes marked `// E0a`):
```cpp
bool physical=false; double F_new=F_old; bool infeas_try=false;
for (int ls=0; ls<40; ++ls) {
    Utry.assign(U.begin(),U.end());
    for (int i=0;i<n;++i) Utry[i]+=lam*dU[i];
    physical=true;
    for (int i=0;i<N&&physical;++i){ const double S=Utry[4*i+0],V=Utry[4*i+1],T=Utry[4*i+3];
        if (S<=0.0||T<=0.0||!(V<0.0)) physical=false; }
    if (physical){ const double rs=Utry[4*N+1]; if(!(rs>in.r_in&&rs<in.r_out)) physical=false; }
    if (!physical) ++n_rj_phys;                                                        // E0a
    if (physical && !slim_fadv_ok(in, op, Utry, 50.0)) { physical=false; ++n_rj_fadv; } // E0a
    if (physical){
        merit_try=eval_merit(Utry,Rtry,infeas_try);
        if (infeas_try || !std::isfinite(merit_try)) { physical=false; ++n_rj_infeas; } // E0a
        else { F_new=merit_to_F(merit_try); lam_acc=lam; break; }                       // E0a
    }
    lam*=0.5;
}
```
with, at outer-iteration scope: `int n_rj_phys=0, n_rj_fadv=0, n_rj_infeas=0; double lam_acc=0.0;`

Extend the kDiag print (:1114–1116) — one added printf AFTER the existing one:
```cpp
if (kDiag) std::printf("[COUPLED] it=%d ls_rej phys/fadv/infeas=%d/%d/%d lam_acc=%.3e\n",
                       it, n_rj_phys, n_rj_fadv, n_rj_infeas, lam_acc);
```

### 1b. Walk-probe env knobs (`tools/slim_coupled_walk_probe.cpp`, harness-only)

After `in_base` construction (~:290–296) and `copt.n_z = 256` (~:299):
```cpp
if (const char* e = std::getenv("SLIM_WALK_N"))    in_base.n_nodes = std::atoi(e);   // E0a2
if (const char* e = std::getenv("SLIM_WALK_ROUT")) in_base.r_out   = std::atof(e);   // E0a2
if (const char* e = std::getenv("SLIM_WALK_NZ"))   copt.n_z        = std::atoi(e);   // E0a2
```
Replace `const double kWallPerAttempt = 43200.0;` (~:318):
```cpp
double kWallPerAttempt = 43200.0;                                                     // E0a2
if (const char* e = std::getenv("SLIM_WALK_WALL_S")) kWallPerAttempt = std::atof(e);  // E0a2
```
In PHASE 1 loop (~:330):
```cpp
const bool base_only = std::getenv("SLIM_WALK_BASE_ONLY") != nullptr;                 // E0a2
for (double f : base_fEdd) {
    if (base_only && f != base_fEdd[0]) break;                                        // E0a2
```
Executor: verify each insertion against surrounding context before applying (line
numbers drift); `#include <cstdlib>` if not already present in the probe.

### 1c. Gates + verification

```bash
./build.sh
./build/slim-omp-gate-probe          # MUST report bit-identical (0.000e+00 deltas)
./build/test-slim-coupled-jacobian   # MUST pass
./build/test-slim-disk-radial        # MUST pass
./build/test-slim-jacobian           # MUST pass
SLIM_DIAG=1 SLIM_WALK_BASE_ONLY=1 SLIM_WALK_WALL_S=600 SLIM_WALK_N=12 SLIM_WALK_NZ=96 \
  OMP_NUM_THREADS=8 timeout 900 ./build/slim-coupled-walk-probe 2>&1 | tee /tmp/e0a_smoke.log
# smoke: runs, prints ls_rej lines, respects base-only + wall budget
```
Acceptance: gates green; smoke shows `ls_rej` diagnostics; omp-gate bit-identity proves
observability-only. Commit message handed to Seth (do NOT commit).

---

## Task 2 — E0b: `slim_fadv_ok` grid-origin fix (`src/slim_disk_radial.cpp:2523–2531`)

All three callers pass the FULL state (4N+2 entries, r_s at U[4N+1]) — verified:
`:1077` (coupled line search, Utry), `:2665` (validity gate, reads `U[4*N+1]` itself),
`:3608` (arclength collector, block reads `Utry[4*N+1]` two lines above). One-site fix.

BEFORE (exact):
```cpp
    const int N = std::max(in.n_nodes, 4);
    std::vector<double> rgrid(N);
    {
        const double lr0 = std::log(in.r_in), lr1 = std::log(in.r_out);
        for (int i = 0; i < N; ++i) {
            const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
            rgrid[i] = std::exp(lr0 + (lr1 - lr0) * t);
        }
    }
```

AFTER:
```cpp
    const int N = std::max(in.n_nodes, 4);
    std::vector<double> rgrid(N);
    {
        // Grid origin = the STATE's sonic radius r_s (U[4N+1]) — the [r_s, r_out] grid
        // every consumer of U lives on.  in.r_in (0.5·r_ISCO) is the DOMAIN floor, ~2×
        // too small at node 0; Qadv ∝ 1/r² ⇒ ~4× f_adv error there, mis-firing this
        // gate's line-search rejections.  (Audit checklist #7; design 2026-09-28.)
        const double r_s = (U.size() >= size_t(4 * N + 2)) ? U[4 * N + 1] : in.r_in;
        const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
        for (int i = 0; i < N; ++i) {
            const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
            rgrid[i] = std::exp(lr0 + (lr1 - lr0) * t);
        }
    }
```

### Gates + A/B (behavior change expected — this is a fix, not instrumentation)

```bash
./build.sh
./build/slim-omp-gate-probe && ./build/test-slim-coupled-jacobian \
  && ./build/test-slim-disk-radial && ./build/test-slim-jacobian && ./build/test-column-coupled
# A/B on the established fast rig (structure comparison only — NOT physics diagnosis):
SLIM_DIAG=1 SLIM_WALK_BASE_ONLY=1 SLIM_WALK_WALL_S=3600 SLIM_WALK_N=12 SLIM_WALK_NZ=96 \
  OMP_NUM_THREADS=8 ./build/slim-coupled-walk-probe 2>&1 | tee /tmp/e0b_fast.log
# compare vs /tmp/e0a_smoke.log (longer run if affordable): ls_rej fadv count, λ
# trajectory, merit floor.  Record deltas in the design doc.
```
Acceptance: all gates green; fadv-rejection behavior change documented (either count
drops — gate was mis-firing — or unchanged, which redirects suspicion off H2).
Commit message handed to Seth. NOTE: this fix also affects the one-zone arclength
collector (:3608) and validity gate (:2665) — intended; `test-slim-jacobian` +
`test-slim-disk-radial` cover those paths.

---

## Task 3 — Pod bring-up (RunPod campaign 1, ≤ $10)

Pricing verified 2026-09-28 via MCP `list_cpu_types`: **cpu3c Compute-Optimized,
$0.03/vCPU·h ⇒ 32 vCPU = $0.96/h, 64 GB RAM, max 32 vCPU, availability HIGH in
US-CA-2.** $10 cap ≈ 10.4 h all-in. (cpu3g general-purpose 4 GB/vCPU = $1.28/h — only
if RAM pressure appears; not expected.)

1. `tool_describe` + `tool_call` `mcp__runpod__create_pod`: **4 vCPU** cpu5c (2 vCPU = ONE
   physical EPYC core with 2 SMT threads sharing an FPU — verified via lscpu on pod 1;
   the solver is FPU-bound, so size vCPU = 2× physical cores needed), US-CA-2 or
   scheduler's choice, Ubuntu 24.04 template `runpod-ubuntu-2404`, ~20 GB disk,
   `startSsh: true` (Seth's Thinkpad ed25519 key already registered — verified).
   **Toolchain: GCC ≥ 14 required** — `src/api.cpp` uses `std::print`; Ubuntu 24.04's
   default GCC 13 fails the `grrt` lib target, which `test-slim-disk-radial`,
   `test-slim-jacobian`, and `test-column-coupled` link (verified on pod 1: those three
   exit 127 while slim-only targets build clean). Install `g++-14`, configure with
   `-DCMAKE_CXX_COMPILER=g++-14`.
   **FP equivalence with the Windows build (Seth 2026-09-28 — accuracy matters):**
   `grrt` lib is strict on both platforms (MSVC `/fp:precise`; GCC `-fno-fast-math
   -ffp-contract=off`, CMakeLists :50-62). BUT the probe/test targets get NO
   `target_compile_options` — bare build-type defaults: MSVC `/O2 /fp:precise` (strict)
   vs GCC `-O3` + default **`-ffp-contract=fast`** (FMA contraction ON ⇒ NOT equivalent;
   pod-1 ran this way). Pod builds must configure with
   `-DCMAKE_CXX_FLAGS_RELEASE="-O2 -DNDEBUG -ffp-contract=off"` and the setup script
   hard-audits every target's `flags.make` before building. Pod-1 numbers are
   reference-only; never compare merit values across flag regimes. Cross-OS
   bit-identity remains impossible (glibc vs MSVC libm last-ulp differences in
   exp/log/pow); equivalence here means identical IEEE semantics, so convergence
   STRUCTURE transfers. Permanent repo fix (separate commit candidate, Seth's call):
   global `add_compile_options($<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-ffp-contract=off>)`
   after `project()`. Also flagged: `grrt-cli` carries `-ffast-math` on non-MSVC
   (CMakeLists :115) while the MSVC cli stays precise — renderer-side, Seth's call.
2. SSH in (accept-new host key), sanity: `nproc` = 32, `g++ --version` ≥ 13,
   `cmake --version` (apt install cmake if missing).
3. Ship the working tree (branch is LOCAL-only; do not push without Seth):
   ```bash
   cd ~/Projects/GRRT && tar czf - --exclude=build --exclude=.git . | \
     ssh -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST> 'mkdir -p ~/GRRT && tar xzf - -C ~/GRRT'
   ```
4. Build + gates on the pod (same binaries as Task 1/2 acceptance). omp-gate
   bit-identity must ALSO hold at OMP_NUM_THREADS=32.
5. Scaling probe: one base-rung outer iteration at OMP_NUM_THREADS=16 vs 32
   (SLIM_WALK_BASE_ONLY=1, short wall budget, count completed iterations / time).
   Record s/iter ⇒ refine E1 duration + E2 scope.
6. **STOP the pod** (`mcp__runpod__pod_action` STOP) whenever not actively running
   experiments; terminate at campaign close.

## Task 4 — E1: instrumented base-rung run (campaign 1)

```bash
tmux new -d -s e1 'cd ~/GRRT && SLIM_DIAG=1 SLIM_CHECKPOINT=$HOME/ckpt/e1 \
  SLIM_WALK_BASE_ONLY=1 SLIM_WALK_WALL_S=28800 OMP_NUM_THREADS=32 \
  ./build/slim-coupled-walk-probe 2>&1 | tee $HOME/e1.log'
```
(Verify SLIM_CHECKPOINT semantics — prefix vs directory — against `:3b8b8d6` code
before launch; create the path.) Poll every ~30 min via ssh; watch for: per-group
table at the merit floor, ls_rej counts, μ/maxrel trajectory, r_s movement.
Stop criteria: merit < 5e-3, OR ≥ 8 outer iterations at the floor, OR wall budget.
Then: `tar` logs + checkpoints back to laptop, STOP pod.

**E1 answers:** which row group pins merit at the floor (expect ang ≈ 87% — verify),
where its max sits (node 16?), whether fadv rejections were mis-firing (post-E0b),
whether the stall triad (μ↑, maxrel↓, r_s frozen) persists after E0a/E0b.

## Task 5 — E2: mesh/domain sweep (campaign 2, scoped AFTER E1 analysis, ≤ $10)

Config-only via Task-1 knobs. Baseline point: N=18, r_out=50 (E1 itself). Sweep:
N ∈ {12, 18, 24} × r_out ∈ {50, 65, 100} at f_Edd=1e-3, seed + 1–2 outer iterations
each (spike LOCATION is readable early), SLIM_FD_ONESIDED=0 A/B on the N=18/50 point.
Budget from measured s/iter (Task 3.5); prune points E1 makes redundant.
Decision: spike tracks mesh (second-to-last node, amplitude ∝ N, moves with r_out)
⇒ H1 discretization/BC; spike pins to r≈41.7M ⇒ physical/BC-validity.

## Task 6 — Analysis + decision gate (laptop)

Append findings to the design doc (evidence tables, not narrative); propose ONE
targeted fix per the design doc's Step-3 decision table; flag any load-bearing
physics claim for Seth's oracles; hand commit messages; Seth decides.

## Risks

- Pod scaling sublinear (column-count 18 < 32 threads; load imbalance) ⇒ measured in
  Task 3.5 BEFORE committing E1 hours; fallback: accept ~1.5–2× and extend campaign.
- E1 doesn't reproduce the 1.113 floor on the pod (different OpenMP scheduling) ⇒
  omp-gate bit-identity is the guard; if state diverges, diagnose scheduling
  dependence first (that itself would be a finding).
- Ubuntu template lacks GCC 13 ⇒ apt toolchain or pick another template (minutes).
- $10 overrun ⇒ wall budgets + tmux + STOP discipline; checkpoints make runs resumable.

## Definition of done (this plan)

E1 evidence table recorded; E2 mesh-dependence verdict on the node-16 spike; fadv-gate
bug fixed and gated; ONE next fix proposed with evidence; total pod spend ≤ $20 across
two campaigns; nothing committed without Seth.

---

## Session log — 2026-09-28 (Hermes, RunPod campaign 1)

**Status: PAUSED at Seth's request after Task 1. E0b (fadv grid fix) NOT yet applied.
Branch `feat/slim-disk` pushed with E0a + docs. Pod stopped (not terminated).**

### What ran

- **Pod 1** `hxqk55ewplgbzb` (cpu5c 2 vCPU = 1 physical core w/ SMT — wrong sizing;
  GCC 13; default FP flags): TERMINATED. Results: omp-gate PASS (bit-identical),
  test-slim-coupled-jacobian PASS; test-slim-disk-radial / test-slim-jacobian /
  test-column-coupled never built (GCC 13 lacks `<print>` for `src/api.cpp`). Log:
  laptop `/tmp/pod1_gates.log`. **Its numbers are reference-only (contract=fast build).**
- **Pod 2** `zw83l8qicsvigy` (cpu5c 4 vCPU = 2 physical cores, GCC 14, strict FP
  `-O2 -DNDEBUG -ffp-contract=off`, FLAG_AUDIT_PASS on all six targets): STOPPED after
  full pipeline. Log: laptop `/tmp/pod2_gates.log`, `/tmp/pod2_smoke_full.log`.

### Gate results (pod 2, E0a patch, strict FP)

| Gate | Result |
|---|---|
| slim-omp-gate-probe (OMP=4) | **PASS** — bit-identical, 0 checks failed |
| test-slim-coupled-jacobian | **PASS** (exit 0) |
| test-slim-jacobian | **PASS** (exit 0) |
| test-column-coupled | **PASS** (exit 0) |
| test-slim-disk-radial | **FAIL** — "a=0.900 / a=0.998 did not converge" (35.5 min) |
| smoke (walk-probe knobs) | PASS — knobs verified; clean budget abort; DONE(BASE-INFEASIBLE) |

### test-slim-disk-radial failure — attribution

- Cannot be E0a: the test links the `grrt` lib only; `slim_disk_coupled.cpp` is not a
  lib source (include-the-.cpp probes are the only consumers), and E0a touched only the
  coupled file + walk probe.
- Pristine control (unpatched `fix/volumetric-ring`, identical strict flags, GCC 14)
  cloned + built on pod 2; Seth confirmed 2026-09-28: **it does not converge** —
  pre-existing failure, not a Linux-only artifact, not caused by this session's work.
  (Control test was still mid-run when the pod was stopped at Seth's request; partial
  logs pulled to laptop `/tmp/pod2_control*.log` if available.)

### First E0a physics readouts (N=12/n_z=96 fast rig — structural only)

- Line-search rejections are **infeas-dominated, zero fadv**:
  `it=0 ls_rej phys/fadv/infeas=0/0/3 lam_acc=6.4e-2`, `it=1 ... 0/0/5 lam_acc=1.7e-2`.
  ⇒ λ truncation on the rig comes from column-solver infeasibility at trial points,
  not the fadv gate. H2's practical impact on the stall may be smaller than suspected
  (E0b still correct to make — the gate evaluates f_adv at wrong radii).
- Boundary signature reproduces at N=12: ang max at outermost node (row 23 = node 11),
  rad max at node 0 (sonic) — same structure as the N=18 stall story.

### Cost

Pod 1 ≈ $0.05 (40 min), Pod 2 ≈ $0.35 (≈2.5 h incl. control build) — well under the
$10/campaign cap. Pod 2 left STOPPED (container disk preserved; restart via MCP
`pod_action start`, direct SSH port may change on restart — re-read via `get_pod`).

### Resume checklist (next session)

1. Restart pod 2 (`mcp__runpod__pod_action` start `zw83l8qicsvigy`; `get_pod` for the
   new direct-SSH port) or terminate + recreate if stale.
2. Apply **E0b** locally (`slim_fadv_ok` grid origin `in.r_in` → `U[4N+1]`, plan Task 2),
   ship as a second patch, rebuild on pod, gates: omp bit-identity + cj + sj + cc green;
   sdr expected to still fail with the SAME two spins (any NEW failure = E0b signal).
3. Then **E1** (base rung, N=18, n_z=256, strict FP, tmux, checkpoints) — the run that
   answers which group pins merit at the 1.113 floor and whether the stall triad holds.
4. Open question for Seth's Windows box: does test-slim-disk-radial converge there at
   e7198f4? (His "it does not converge" is recorded above; if that was the Windows
   answer, the failure is platform-independent and belongs in the handoff as a known
   broken gate at this HEAD.)
