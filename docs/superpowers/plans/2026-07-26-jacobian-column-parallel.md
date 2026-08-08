# Per-column Parallelization of the Coupled Reduced Jacobian — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize the full-FD ℓ/ℓ_in/r_s column loop in `slim_coupled_reduced_jacobian` over columns (each with its own `ColumnCache` warm-started from a base-cache snapshot), cutting one coupled-relax outer-iteration cost by ~8–12× while keeping the reduced Jacobian bit-identical between serial and parallel builds.

**Architecture:** One localized rewrite of the full-FD block (`src/slim_disk_coupled.cpp:742–771`) plus a one-line snapshot after the base solve. Two steps: (1) make the serial full-FD loop base-seeded (snapshot the converged base cache; every column warm-starts from it) and measure the microscopic numeric shift; (2) parallelize that now-order-independent loop with `#pragma omp parallel for` and per-task cache/scratch. Correctness is guarded by the existing `slim-omp-gate-probe` (bit-identical serial==parallel + determinism + speedup) and `test-slim-coupled-jacobian` (reduced-J vs FD oracle).

**Tech Stack:** C++23, OpenMP, CMake / Visual Studio 2022 (Release). No new files.

**Spec:** `docs/superpowers/specs/2026-07-26-jacobian-column-parallel-design.md`

**Standing workflow constraints (OVERRIDE the sub-skill's defaults):**
- **NEVER run `git commit`.** Where a step below shows a commit, the implementer must instead STOP and hand the exact commit message to the human, who commits. Do not proceed to the next task until the human confirms the commit.
- **One change at a time.** Do not combine Task 1 and Task 2.
- This is a parallelization refactor, not a physics edit — the bit-identical gate is the primary correctness oracle. No fable/Wolfram pass required.

---

## Baseline facts (read before starting)

Current full-FD block, `src/slim_disk_coupled.cpp:742–771` (verbatim):

```cpp
    // ---------- (2) FULL-FD for the ℓ_i, ℓ_in, r_s columns (option (b): re-solve cols). ----------
    std::vector<double> R0; bool f0 = false;
    slim_coupled_residual(U, in, op, copt, cache, R0, f0);
    auto full_fd_col = [&](int col) {
        const double h = step_for(col);
        Up = U; Um = U; Up[col] += h; Um[col] -= h;
        bool fp = false, fm = false;
        slim_coupled_residual(Up, in, op, copt, cache, Rp, fp);
        slim_coupled_residual(Um, in, op, copt, cache, Rm, fm);
        for (int row = 0; row < n; ++row) {
            double d;
            if (fp && fm)      d = (Rp[row] - Rm[row]) / (2.0 * h);
            else if (fp && f0) d = (Rp[row] - R0[row]) / h;   // backward side infeasible
            else if (fm && f0) d = (R0[row] - Rm[row]) / h;   // forward side infeasible
            else               d = 0.0;                        // both sides infeasible
            if (!std::isfinite(d)) d = 0.0;
            J[(size_t)row * n + col] = d;
        }
    };
    for (int i = 0; i < N; ++i) full_fd_col(4*i+2);   // ℓ_i
    full_fd_col(4*N+0);                               // ℓ_in
    full_fd_col(4*N+1);                               // r_s
```

- `Up, Um, Rp, Rm` are function-level shared vectors declared earlier (`~L725`) and reused by the serial `frozen_fd_col` block above. The full-FD lambda captures them by reference — that is exactly what must become thread-local when parallelized.
- `n = 4*N + 2`. `J` is row-major `J[row*n + col]`; each column writes a distinct `col` → no write races across columns.
- The base solve at `~L694` fills `cache` with the converged base columns and sets `base_infeasible`; the function `return false`s at `~L707` on an infeasible base. After that point `cache` holds the converged base columns — the snapshot source.
- `ColumnCache` (`src/slim_disk_coupled.cpp:69`) is copyable (vectors of doubles + `std::vector<char> valid`); copying it is cheap relative to a column solve.
- OpenMP nesting is OFF by default, so the inner per-node `#pragma omp parallel for` inside `slim_coupled_residual` runs serially when reached from inside an outer parallel region. Do not enable nesting.

Build/run commands (Git Bash; the repo also builds under VS 2022):

```bash
cmake --build build --config Release --target slim-omp-gate-probe test-slim-coupled-jacobian
# gate probe writes its verdict to STDERR (stdout is noisy column diagnostics):
./build/Release/slim-omp-gate-probe.exe 2>gate.txt 1>/dev/null; cat gate.txt
./build/Release/test-slim-coupled-jacobian.exe
```

---

## Task 1: Base-seed the serial full-FD loop (snapshot the base cache)

Make every full-FD column warm-start from a snapshot of the converged base cache instead of from the previous column's mutated cache. This removes the accidental serial dependency **without** parallelizing yet, and lets us measure the resulting numeric shift in isolation.

**Files:**
- Modify: `src/slim_disk_coupled.cpp:742–771` (the full-FD block) and add a snapshot line after the base-solve `return false` (`~L707`).
- Verify with: `tools/slim_omp_gate_probe.cpp` (unchanged), `tests/test_slim_coupled_jacobian.cpp` (unchanged).

- [ ] **Step 1: Capture the pre-change reference J (baseline snapshot).**

Before editing any source, build and run the gate probe on the CURRENT code and save its output as the numeric baseline:

```bash
cmake --build build --config Release --target slim-omp-gate-probe
./build/Release/slim-omp-gate-probe.exe 2>gate_before.txt 1>/dev/null; cat gate_before.txt
```

Expected: test (3) reports `J = 50x50 ... max rel = 0.000e+00 => PASS` (serial==parallel already holds; the full-FD loop is serial today) and test (5) prints a speedup near ~1.8×. Keep `gate_before.txt`.

- [ ] **Step 2: Add the base-cache snapshot after the base solve.**

In `slim_coupled_reduced_jacobian`, immediately AFTER the base-solve infeasibility guard (the line `if (base_infeasible) return false;`, `~L707`), add:

```cpp
    // Snapshot the converged base columns. Every full-FD column below warm-starts from
    // THIS pristine snapshot (not from the previous column's mutated cache), so each
    // column's result is independent of evaluation order — the precondition for a
    // bit-identical parallelization (Task 2). See spec 2026-07-26-jacobian-column-parallel.
    const ColumnCache base_snap = cache;
```

- [ ] **Step 3: Compute R0 from a throwaway copy so the snapshot stays pristine.**

Replace the R0 computation (`~L746–747`):

```cpp
    std::vector<double> R0; bool f0 = false;
    slim_coupled_residual(U, in, op, copt, cache, R0, f0);
```

with a version that uses a disposable copy of the snapshot (so `cache`/`base_snap` are not mutated by the anchor solve):

```cpp
    std::vector<double> R0; bool f0 = false;
    { ColumnCache anchor_cache = base_snap;
      slim_coupled_residual(U, in, op, copt, anchor_cache, R0, f0); }
```

- [ ] **Step 4: Base-seed each full-FD column (still serial).**

Replace the `full_fd_col` lambda body so it uses a per-call cache seeded from the snapshot and local scratch buffers (do NOT touch the shared `Up/Um/Rp/Rm` here — leave them for the frozen block above):

```cpp
    auto full_fd_col = [&](int col) {
        const double h = step_for(col);
        ColumnCache col_cache = base_snap;                 // warm-start from the pristine base
        std::vector<double> Upc = U, Umc = U, Rpc, Rmc;
        Upc[col] += h; Umc[col] -= h;
        bool fp = false, fm = false;
        slim_coupled_residual(Upc, in, op, copt, col_cache, Rpc, fp);
        slim_coupled_residual(Umc, in, op, copt, col_cache, Rmc, fm);
        for (int row = 0; row < n; ++row) {
            double d;
            if (fp && fm)      d = (Rpc[row] - Rmc[row]) / (2.0 * h);
            else if (fp && f0) d = (Rpc[row] - R0[row]) / h;   // backward side infeasible
            else if (fm && f0) d = (R0[row] - Rmc[row]) / h;   // forward side infeasible
            else               d = 0.0;                        // both sides infeasible
            if (!std::isfinite(d)) d = 0.0;
            J[(size_t)row * n + col] = d;
        }
    };
    for (int i = 0; i < N; ++i) full_fd_col(4*i+2);   // ℓ_i
    full_fd_col(4*N+0);                               // ℓ_in
    full_fd_col(4*N+1);                               // r_s
```

(Within a single column, `col_cache` is reused across the Up and Um solves — sequential within the call, deterministic. Each column starts fresh from `base_snap`.)

- [ ] **Step 5: Build and run both gates; measure the shift.**

```bash
cmake --build build --config Release --target slim-omp-gate-probe test-slim-coupled-jacobian
./build/Release/slim-omp-gate-probe.exe 2>gate_after1.txt 1>/dev/null; cat gate_after1.txt
./build/Release/test-slim-coupled-jacobian.exe
```

Expected:
- `slim-omp-gate-probe` test (3) still PASS (`max rel = 0.000e+00`) and test (4) determinism PASS — the loop is still serial, so serial==parallel is unaffected.
- `test-slim-coupled-jacobian` PASS (all gated columns `< 1e-3`) — this is the guard that base-seeding did not move the physics / break the reduced-J assembly. (The probe reports only *within-build* serial-vs-parallel deltas, so it cannot itself show the before/after base-seeding shift; the FD-oracle staying green is the evidence that shift is physics-neutral.)
- If you want the shift quantified directly (spec claims ≲ 1e-6): optional — dump `J` to a file before and after this task from a throwaway `main` or a scratch probe and diff; not required for the gate, which is `test-slim-coupled-jacobian < 1e-3`.

- [ ] **Step 6: Hand off the commit (DO NOT run git commit).**

Stage nothing yourself. Give the human this message and WAIT for confirmation:

```
refactor(slim-coupled): base-seed the reduced-Jacobian full-FD columns from a cache snapshot

Snapshot the converged base ColumnCache after the base solve; every full-FD
ℓ/ℓ_in/r_s column now warm-starts from that pristine snapshot instead of from the
previous column's mutated cache, removing an accidental order dependency. Serial
behavior only (no parallelization yet). Reduced-J unchanged to column-solver
tolerance: slim-omp-gate-probe test(3)=PASS (0.0), test-slim-coupled-jacobian PASS.
Prereq for the bit-identical per-column parallelization.
```

---

## Task 2: Parallelize the full-FD loop over columns

Now that each column is order-independent, run the columns concurrently with OpenMP, each on its own thread-local cache and scratch. Result must stay bit-identical (the columns are independent) and the speedup must jump.

**Files:**
- Modify: `src/slim_disk_coupled.cpp` — the `for (int i...) full_fd_col(...)` driver at the end of the full-FD block (`~L769–771`).
- Verify with: `tools/slim_omp_gate_probe.cpp` (unchanged).

- [ ] **Step 1: Confirm the pre-parallel gate is green (regression anchor).**

```bash
./build/Release/slim-omp-gate-probe.exe 2>gate_pre2.txt 1>/dev/null; cat gate_pre2.txt
```

Expected: test (3) PASS (`0.000e+00`), test (5) speedup still ~1.8× (full-FD loop still serial). This is the "before" for the speedup improvement.

- [ ] **Step 2: Replace the serial column driver with a parallel loop over an explicit column list.**

Replace the three driver lines (`~L769–771`):

```cpp
    for (int i = 0; i < N; ++i) full_fd_col(4*i+2);   // ℓ_i
    full_fd_col(4*N+0);                               // ℓ_in
    full_fd_col(4*N+1);                               // r_s
```

with:

```cpp
    // Parallelize over the ℓ_i / ℓ_in / r_s columns. Each column is INDEPENDENT: full_fd_col
    // seeds its own col_cache from base_snap and writes only its own J column, so there is
    // no shared mutable state and the result is identical to the serial order (Task 1
    // base-seeding is the precondition). schedule(dynamic): every column task solves ALL
    // nodes incl. the one dominant high-Σ node, so tasks are near-equal-cost and dynamic
    // scheduling keeps threads full. The inner per-node parallel-for inside slim_coupled_
    // residual runs serially here (OpenMP nesting is off by default) — intended.
    std::vector<int> fd_cols;
    fd_cols.reserve(N + 2);
    for (int i = 0; i < N; ++i) fd_cols.push_back(4*i + 2);  // ℓ_i
    fd_cols.push_back(4*N + 0);                              // ℓ_in
    fd_cols.push_back(4*N + 1);                              // r_s
    #pragma omp parallel for schedule(dynamic)
    for (int k = 0; k < (int)fd_cols.size(); ++k) full_fd_col(fd_cols[k]);
```

No other change is needed: `full_fd_col` already allocates its own `col_cache`, `Upc/Umc/Rpc/Rmc`, and `fp/fm` per call (Task 1), and reads only the shared read-only `U`, `base_snap`, `R0`, `f0`, `n`. `#pragma omp` on a non-OpenMP build is ignored, so the serial fallback is automatic.

- [ ] **Step 3: Build and run the bit-identical + determinism + speedup gate.**

```bash
cmake --build build --config Release --target slim-omp-gate-probe
./build/Release/slim-omp-gate-probe.exe 2>gate_after2.txt 1>/dev/null; cat gate_after2.txt
```

Expected (ACCEPTANCE):
- test (2) Residual PASS (`max rel <= 1e-12`) — unchanged path.
- test (3) Jacobian serial(1)==parallel(max) PASS, `max rel = 0.000e+00` (bit-identical). This is the primary correctness gate for the parallelization.
- test (4) determinism (parallel twice) PASS, `0.000e+00`.
- test (5) speedup vs `gate_pre2.txt`: materially higher than the ~1.8× baseline — expect ≥ ~4× at the probe grid (N=12, n_z=96, ~14 columns / available threads). If `omp_get_max_threads()` is small, judge relative to it; the correctness gates are the hard pass/fail.

- [ ] **Step 4: Run the reduced-J regression gate.**

```bash
cmake --build build --config Release --target test-slim-coupled-jacobian
./build/Release/test-slim-coupled-jacobian.exe
```

Expected: PASS (all gated columns `< 1e-3`) — parallelization is numerically transparent to the reduced-J assembly.

- [ ] **Step 5: Hand off the commit (DO NOT run git commit).**

Give the human this message and WAIT for confirmation:

```
perf(slim-coupled): parallelize the reduced-Jacobian full-FD columns over threads

Run the ℓ/ℓ_in/r_s full-FD columns concurrently (#pragma omp parallel for,
schedule(dynamic)), each on its own base_snap-seeded ColumnCache + local scratch.
Every column task solves all nodes, so the dominant high-Σ node is amortized across
tasks → near-ideal load balance (unlike the 1.85×-capped per-node parallelism).
Bit-identical: slim-omp-gate-probe test(3)/test(4)=0.0; speedup <old>→<new>×.
test-slim-coupled-jacobian PASS.
```

Fill `<old>`/`<new>` from `gate_pre2.txt` / `gate_after2.txt`.

---

## Task 3: (Optional, informational) n_z=256 timing line

Add a production-grid timing readout so we can see the real per-iteration win, without making machine-dependent timing a hard gate.

**Files:**
- Modify: `tools/slim_omp_gate_probe.cpp` — add an OPTIONAL, guarded second timing pass at n_z=256.

- [ ] **Step 1: Add an env-gated n_z=256 timing block.**

At the end of `main()` in `tools/slim_omp_gate_probe.cpp`, before the final return, add a block that only runs when `SLIM_GATE_BIG=1`, rebuilding `copt.n_z = 256` and timing one `build_J(1,...)` vs `build_J(max,...)`, printing both wall times and the ratio to stderr via the existing `G(...)` helper. Reuse the existing `build_J` lambda and `U`/`in`/`op` already in scope; only override `copt.n_z` for the block. Do not assert on the ratio — print only.

```cpp
    if (const char* big = std::getenv("SLIM_GATE_BIG"); big && big[0] == '1') {
        G("### (6) n_z=256 timing (informational; SLIM_GATE_BIG=1) ###\n");
        copt.n_z = 256;                       // override for this block only
        std::vector<double> Jb1, Jb2; double tb1 = 0.0, tb2 = 0.0;
        const bool ok1 = build_J(1, Jb1, tb1);
        const bool ok2 = build_J(max_threads, Jb2, tb2);
        if (ok1 && ok2 && tb2 > 0.0)
            G("  n_z=256: serial=%.1fs  parallel(%d)=%.1fs  speedup=%.2fx\n",
              tb1, max_threads, tb2, tb1 / tb2);
        else
            G("  n_z=256 base infeasible at probe state (ok1=%d ok2=%d) — timing skipped.\n",
              ok1, ok2);
    }
```

Add `#include <cstdlib>` at the top of the file if not already present (for `std::getenv`).

- [ ] **Step 2: Build and run the big timing pass.**

```bash
cmake --build build --config Release --target slim-omp-gate-probe
SLIM_GATE_BIG=1 ./build/Release/slim-omp-gate-probe.exe 2>gate_big.txt 1>/dev/null; cat gate_big.txt
```

Expected: the default gates (2)–(5) still PASS, and section (6) prints a serial/parallel/ speedup line at n_z=256 (expect ≈ 8–12×; if the probe state is infeasible at n_z=256 it prints the skip line — acceptable, the property is already gated at n_z=96).

- [ ] **Step 3: Hand off the commit (DO NOT run git commit).**

Give the human this message and WAIT for confirmation:

```
test(slim-coupled): optional n_z=256 timing line in slim-omp-gate-probe (SLIM_GATE_BIG=1)

Informational-only production-grid timing of the reduced-Jacobian build, serial vs
parallel. Prints; does not gate (machine-dependent). Default gates unchanged.
```

---

## After all tasks

- The base rung's per-iteration cost should drop from ~2.7 h toward ~15–20 min at n_z=256. Re-launch the merit-trajectory run (`SLIM_DIAG=1 OMP_NUM_THREADS=12 slim-coupled-walk-probe`) as a background job to finally read `merit=` across several iterations — that is the downstream datum this speedup unblocks (do this as a separate step after the human has committed Tasks 1–2, not inside this plan).
- Dispatch a final code-review subagent over the combined diff before finishing the branch work.
