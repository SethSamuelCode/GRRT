# Stretched Vertical Grid — Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. NEVER `git commit` — hand the message over. ONE change, ONE commit.

**Goal:** surface-clustered vertical q-grid in the column BVP (default-on), via per-interval `dq_i` in the ODE residual + Jacobian rows, so high-Σ columns resolve their photosphere at low n_z. Spec: `docs/superpowers/specs/2026-07-24-stretched-vertical-grid-design.md`.

**Gates:** (1) uniform-recovery bit-identical @ stretch=0, (2) FD-oracle clean on a stretched grid, (3) existing column suite green (re-baseline shifted tolerances vs high-n_z, don't loosen blindly), (4) render regression [controller], (5) full256 n_z=96 stretched → 18/18 [controller].

---

### Task 1 (only task)

**Files:** `include/grrt/scene/disk_column_bvp.h` (+ `disk_column_coupled.h` if `ColumnCoupledInputs` lives there) for the `stretch` field + `kDefaultStretch`; `src/disk_column_bvp.cpp` (grid gen + `column_residual` + `analytic_jacobian` + `solve_column_bvp`); `src/disk_column_coupled.cpp` (`coupled_column_residual` + `coupled_column_jacobian` + `solve_column_coupled` + `base_inputs_from` propagation); `tests/test_column_coupled.cpp` (+ `tests/test_column_bvp.cpp`) for gates 1-2.

**Preflight (READ — the code map, verify against current source):**
- Grid `q[i]=i/(N-1)`: `disk_column_bvp.cpp:~172,883`. `dz_dq=Σ0/(2ρ)` per-node: `~94` (NO change).
- Constant `dq=1/(N-1)`: `disk_column_bvp.cpp:122,243`; `disk_column_coupled.cpp:60,112`. Residual rows use `0.5*dq`: bvp `~135-138`; coupled `~71-74`. Jacobian `half_dq=0.5*dq` scattered: bvp `~333,350-389`; coupled `~196,208-248`.
- Adaptive quadratures (Σ,τ,η3,η4) at `disk_column_bvp.cpp:891-898`, `disk_column_coupled.cpp:976-985,1304-1320` — **NO change** (already use `z[i+1]-z[i]`). BCs `disk_column_bvp.cpp:143-149`, `disk_column_coupled.cpp:79-88` — **NO change**.
- `ColumnInputs` / `ColumnCoupledInputs` struct definitions (add `double stretch = kDefaultStretch;`). `base_inputs_from(in,Te,fa)` (`disk_column_coupled.cpp:~385`) — propagate `b.stretch = in.stretch`.

- [ ] **Step 1 — grid generator + fields.**
  - Add `inline constexpr double kDefaultStretch = 2.5;` (near the column headers).
  - Add `double stretch = kDefaultStretch;` to `ColumnInputs` and `ColumnCoupledInputs`.
  - Add, in `disk_column_bvp.cpp` (or a shared header), surface-clustered grid:
```cpp
// Vertical q-grid on [0,1], clustered toward the SURFACE (q=1, the photosphere).
// stretch<=0 => uniform (q[i]=i/(N-1)), bit-identical to the legacy grid.
static std::vector<double> column_q_grid(int N, double stretch) {
    std::vector<double> q(N);
    if (N <= 1) { if (N==1) q[0]=0.0; return q; }
    if (!(stretch > 0.0)) { for (int i=0;i<N;++i) q[i]=double(i)/double(N-1); return q; }
    const double th = std::tanh(stretch);
    for (int i=0;i<N;++i) {
        const double u = double(i)/double(N-1);          // 0..1
        q[i] = 1.0 - std::tanh(stretch*(1.0-u))/th;      // dense near q=1
    }
    q[0]=0.0; q[N-1]=1.0;                                  // exact ends
    return q;
}
```

- [ ] **Step 2 — thread per-interval `dq_i` through the 4 ODE functions.** In each of `column_residual`, `analytic_jacobian`, `coupled_column_residual`, `coupled_column_jacobian`: accept the `q` array (add a `const std::vector<double>& q` parameter), and inside the trapezoidal loop over interval `(i,i+1)` replace the constant `dq`/`half_dq` with `const double dq_i = q[i+1]-q[i];` (and `0.5*dq_i`). Every `0.5*dq` scatter in the Jacobian rows for that interval uses `0.5*dq_i`. Update the callers in `solve_column_bvp` / `solve_column_coupled` to compute `q = column_q_grid(N, in.stretch)` once and pass it. Keep `base_inputs_from` propagating `stretch`. Do NOT touch the quadratures or BCs.

- [ ] **Step 3 — gate 1 (uniform-recovery, WRITE-FIRST intent but implement alongside).** Add to `tests/test_column_bvp.cpp` a test that solves a reference column with `stretch=0` and asserts the converged `(Sigma0, z0, T.front(), F)` match hardcoded reference values captured from the CURRENT (pre-change) build to **machine precision** (rel < 1e-12). This proves the per-interval threading equals the legacy constant-`dq` at uniform spacing. (Capture the reference by building the pre-change target once and recording the printed values, OR assert equality between a `stretch=0` solve and the legacy formula on a tiny hand-checked N.) Run: must PASS.

- [ ] **Step 4 — gate 2 (FD-oracle on a STRETCHED grid).** Extend the existing analytic-vs-FD Jacobian oracle tests (bvp `test_analytic_vs_numerical_jacobian`; coupled `test_coupled_jacobian_convective_state` / `test_coupled_repose_roundtrip`'s FD gate) to run with `stretch=kDefaultStretch` (>0). Assert mismatch `< 4e-4` for BOTH `analytic_jacobian` and `coupled_column_jacobian`. This is the correctness anchor: it fails loudly if the residual's `dq_i` and the Jacobian's `0.5*dq_i` disagree.

- [ ] **Step 5 — gate 3 (existing suite green, default-on).** Run `test-column-bvp` + `test-column-coupled` with the default (stretched) grid. `F=σT_eff⁴` is BC-pinned → unchanged. `z0`/`η` may shift at the quadrature level. If any assertion fails: build a high-n_z (n_z=512, uniform) reference for that quantity and confirm the stretched value is **closer to it** than the old uniform-n_z value before updating the test's reference/tolerance. Document each re-baseline. Do NOT loosen a tolerance without the closer-to-continuum check.

- [ ] **Step 6 — build the library** (`cmake --build build --config Release`) to confirm the render path (`volumetric_disk` → `solve_column_bvp`) still COMPILES with the new signatures (it passes default `ColumnInputs`, so it now uses stretch=2.5 — intended). Do NOT run a render (controller does the render regression).

- [ ] **Step 7 — hand the commit message over (NO commit).** Stage the touched src/include/test files.
```
feat(disk-column): surface-clustered (stretched) vertical grid, default-on

High-Σ optically-thick columns need their photosphere resolved; a uniform n_z=96
grid can't (base-rung feasibility 15/18 @ n_z=96 vs 18/18 @ 256). Add a per-solve
`stretch` (default 2.5) selecting a tanh grid clustered toward the surface (q=1);
replace the trapezoidal ODE rows' constant dq=1/(N-1) with per-interval
dq_i=q[i+1]-q[i] in column_residual/analytic_jacobian and
coupled_column_residual/coupled_column_jacobian. Quadratures (Σ,τ,η3,η4) and BCs
already use actual node spacing — untouched. stretch=0 recovers the uniform grid
bit-identically (gate 1). FD-oracle clean on a stretched grid for both Jacobians
(gate 2). Default-on also sharpens the volumetric-disk render path (same solver,
interface unchanged — no renderer change); render output improves, re-baselined.

full256 n_z=96 (stretched): 15/18 -> <N>/18.
```
Report DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED, listing every re-baselined test + its closer-to-continuum justification. Do NOT touch quadratures/BCs, do NOT run the relax, do NOT set stretch inside volumetric_disk.
