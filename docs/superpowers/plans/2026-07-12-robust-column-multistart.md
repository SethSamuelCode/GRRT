# Robust Multi-Start for the Coupled Column Solve — Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. **NEVER `git commit`** — hand the message over. ONE change, ONE commit.

**Goal:** Add a multi-start retry of the cold-seed 2-D f_adv-freeing bring-up inside `solve_column_coupled`, so scattered basin-miss nodes converge. Transparent to all callers.

**Spec:** `docs/superpowers/specs/2026-07-12-robust-column-multistart-design.md`.

**Isolated gate:** `slim-full256-probe` 16/18 → 18/18 and `slim-full256-probe 32 256` 29/32 → 32/32, with NOTHING else changed.

---

### Task 1 (only task): multi-start fallback in `solve_column_coupled`

**Files:** Modify `src/disk_column_coupled.cpp` (inside `solve_column_coupled`). Test: `tests/test_column_coupled.cpp`.

**Preflight (read + confirm before coding):**
- `solve_column_coupled` cold-seed path (~lines 944-1002): how it calls `build_coupled_seed_2d(in, op, U2d)` (~986) and `build_coupled_seed` (~995), and picks the lower-merit seed. Confirm `build_coupled_seed_2d`'s exact signature and whether it accepts an initial `f_adv` / uses `in.Teff_guess` / `in.rho_mid_guess`.
- The fallback hierarchy (~1019-1095): primary `affine_invariant_newton` (~1019), warm-start 2-D bring-up (~1049-1064), Σ-continuation "fresh-anchor" (~1079-1095). The new multi-start goes **after the primary Newton fails, before Σ-continuation**.
- `ColumnCoupledInputs` fields `Teff_guess`, `rho_mid_guess` (are they consumed by `build_coupled_seed_2d`?). If `build_coupled_seed_2d` does not read an initial f_adv, vary only the knobs it does read (`Teff_guess`, `rho_mid_guess`) — do NOT invent a parameter; report what knobs are actually available.

**Design of the retry (adapt to the confirmed knobs):**
On primary-solve non-convergence, loop over a bounded spread of a **mutable copy** of `in` (do not mutate the const `in`; `ColumnCoupledInputs in2 = in;`), perturbing only the *starting-guess* knobs `build_coupled_seed_2d` actually reads:
- `Teff_guess ∈ { default, ×0.5, ×2, ×0.25, ×4 }` (the default = the value the primary used / `estimate_Teff_guess`),
- `rho_mid_guess ∈ { default, ×0.3, ×3 }`,
- and, IF `build_coupled_seed_2d` accepts an initial f_adv, `f_adv0 ∈ {0, 0.5, 2.0}`.
For each combo: rebuild the 2-D seed with `in2`, run `affine_invariant_newton`, and **return on the first converged** result (or keep the lowest-merit converged). **Cap total combos at ≤12**; on exhaustion, fall through to the existing Σ-continuation unchanged. Σ and T_c pins are never altered (only `in2`'s guess fields change).

- [ ] **Step 1 — write the failing unit test FIRST.**
Identify a previously-failing basin-miss node from the full256 set and pin its exact inputs. Use the thin seed to get node-3's geometry (r≈3.92) and demanded Σ, and a T_c at which a root exists (use the manifold T_c from a *converged neighbor* if node 3's own manifold T_c is stale). Concretely:
```cpp
// LC/MS gate: a node that fails from the default 2-D guess must converge via multi-start,
// at the SAME pinned (Σ, T_c). (If the chosen node turns out to be pinned-T_c-unreachable
// rather than a basin miss — multi-start can't help it — pick a different node that the
// N=32 grid-move showed is feasible, and note it. Do NOT weaken the assertion.)
static void test_multistart_converges_basin_miss() {
    std::printf("\n=== MS1: multi-start converges a basin-miss column ===\n");
    auto op = build_opacity_luts(1e-14,1e6,3000.0,1e8);
    SlimDiskInputs in{}; in.mass=1.0; in.spin=0.9; in.alpha=0.1; in.r_g=1.48e6; in.r_out=50.0;
    in.n_nodes=18; in.tol=1e-8; in.r_in=0.5*grrt::slim_detail::isco_prograde(in.mass,in.spin);
    in.mdot=1.6399e16;
    std::vector<double> U = build_thin_disk_seed(in, op);
    const int N=std::max(in.n_nodes,4); const int i=3;                 // r≈3.92, a full256 failure
    const double r_s=U[4*N+1], lr0=std::log(r_s), lr1=std::log(in.r_out);
    const double t=double(i)/double(N-1); const double ri=std::exp(lr0+(lr1-lr0)*t);
    const int j=i+1; const double tj=double(j)/double(N-1); const double rj=std::exp(lr0+(lr1-lr0)*tj);
    const double Omi=grrt::slim_detail::omega_from_ell(in.mass,in.spin,ri,U[4*i+2]);
    const double Omj=grrt::slim_detail::omega_from_ell(in.mass,in.spin,rj,U[4*j+2]);
    // T_c: borrow the manifold T_c the neighbor node 4 accepts (it converges at higher Σ),
    // so the pin is REACHABLE and the only obstacle is basin entry.
    grrt::slim_coupled_detail::ColumnCoupledInputs ci{};
    ci.Sigma_target=std::max(U[4*i+0],1e2); ci.Tc=std::max(U[4*i+3],1.0);
    ci.shear=std::max(grrt::slim_coupled_detail::shear_cgs(in,ri,Omi,rj,Omj),1e-300);
    ci.omega_z=std::max(grrt::slim_coupled_detail::omega_perp_cgs(in,ri),1e-300);
    ci.alpha=in.alpha;
    ci.rho_mid_guess=std::max(grrt::slim_detail::one_zone_closure(ci.Sigma_target,ci.Tc,ri,in,op).rho_mid,1e-30);
    ci.n_nodes=96; ci.max_iters=300; ci.tol=1e-8; ci.Teff_guess=0.0;
    const bool ok = grrt::slim_coupled_detail::solve_column_coupled(ci, op, nullptr).converged;
    std::printf("  node %d r=%.3f Σ=%.3e Tc=%.3e -> converged=%d\n", i, ri, ci.Sigma_target, ci.Tc, ok);
    if (!ok) { std::printf("  FAIL: multi-start did not converge the basin-miss node\n"); failures++; }
}
```
Register in `main()`. Build + run: confirm it **FAILS** today (converged=0) — the red state. If node 3 turns out pinned-T_c-unreachable (multi-start structurally can't fix it), the implementer selects a different demonstrably-feasible-but-basin-missed node (from the full256/N=32 data) and documents the swap; do not weaken the gate.

- [ ] **Step 2 — implement the multi-start** (per the design above; adapt knobs to `build_coupled_seed_2d`'s real signature). Insert after the primary Newton failure, before Σ-continuation. Only entered on failure → healthy path untouched.

- [ ] **Step 3 — run to GREEN + regression.** `test-column-coupled` full suite `## 0 failure(s) ##` (MS1 green, all pre-existing green — healthy path bit-identical).

- [ ] **Step 4 — isolated integration gate.** Rebuild `slim-full256-probe`; run both `./slim-full256-probe.exe` and `./slim-full256-probe.exe 32 256`. Report the counts. Target 18/18 and 32/32. **If a node remains infeasible, report which and diagnose** (basin-miss the retry couldn't crack vs pinned-T_c-unreachable) — that honest per-node result is the deliverable; do NOT loosen anything or bundle the T_c fix.

- [ ] **Step 5 — hand the commit message over (NO commit).** Stage only `src/disk_column_coupled.cpp` + `tests/test_column_coupled.cpp`.
```
feat(disk-column): multi-start the 2-D bring-up in solve_column_coupled

Scattered inner nodes fail the column solve as a solver-basin miss (proven
non-physical: failing radii MOVE when the radial grid is re-gridded). High-Σ
inner nodes have no f_adv=0 root (grey Σ0 saturates ~few×10³ ≪ demand ~1.5e4),
so they need the 2-D f_adv-freeing bring-up — which misses its basin from a
single starting guess. On primary-solve failure, retry the cold 2-D bring-up
from a bounded spread of (Teff_guess, rho_mid_guess[, f_adv0]); first-converged
wins, ≤12 tries, then fall through to the existing Σ-continuation. Pinned Σ/T_c
untouched; entered only on failure so the healthy path is bit-identical.

slim-full256-probe: <N>/18 and <M>/32 feasible (was 16/18, 29/32). Gate MS1
converges a previously basin-missed node at its pinned (Σ,T_c).
```

## Self-review
- Spec coverage: multi-start location (Task 1 ✓), starting-guess-only spread (✓), always-on/cost-on-failure (✓), transparent to relax (✓, no call-site change), isolated full256 gate (Step 4 ✓), honest pinned-T_c caveat (Step 4 ✓). 
- No placeholders: knobs are "confirm against `build_coupled_seed_2d`" with an explicit fallback (vary only what it reads) — not a vague TODO.
- Types: `ColumnCoupledInputs in2 = in;` copy; `solve_column_coupled` signature unchanged.
