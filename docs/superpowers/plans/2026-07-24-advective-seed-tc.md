# Advective Seed-T_c — Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. NEVER `git commit` — hand the message over. ONE change, ONE commit.

**Goal:** `build_coupled_seed_advective` (f_adv-laddered seed builder) + wire it as the fallback where `build_coupled_seed` (f_adv=0) fails, so high-Σ nodes get a reachable seed T_c. Spec: `docs/superpowers/specs/2026-07-24-advective-seed-tc-design.md`.

**Isolated gate:** `slim-full256-probe` 17/18 → 18/18 and `32 256` → 32/32, nothing else changed.

---

### Task 1 (only task)

**Files:** `src/disk_column_coupled.cpp` (new helper), `tools/slim_full256_probe.cpp` + `tools/slim_coupled_walk_probe.cpp` (wiring), `tests/test_column_coupled.cpp` (unit test).

**Preflight:** Read `build_coupled_seed` (`src/disk_column_coupled.cpp:430-528`): its T_eff-secant, the Σ0-match gate (`SIGMA_SEED_BAND=0.30`), and how it packs `U` (per-node `P_gas,Q,T,z`; tail `U[4N]=z0, U[4N+1]=Σ_target, U[4N+2]=T_eff, U[4N+3]=f_adv`). Read `base_inputs_from(in, Te, fa)` (used by both seed builders). Confirm `build_coupled_seed`/`build_coupled_seed_advective` are visible to the test TU (it `#include`s the .cpp chain).

- [ ] **Step 1 — failing unit test FIRST** (`tests/test_column_coupled.cpp`, register in `main()`):
```cpp
// AS1: advective seed builds a reachable T_c where f_adv=0 has none (node 10 geometry).
static void test_advective_seed_high_sigma() {
    std::printf("\n=== AS1: advective seed-T_c for a high-Σ node ===\n");
    auto op = build_opacity_luts(1e-14,1e6,3000.0,1e8);
    SlimDiskInputs in{}; in.mass=1.0; in.spin=0.9; in.alpha=0.1; in.r_g=1.48e6; in.r_out=50.0;
    in.n_nodes=18; in.tol=1e-8; in.r_in=0.5*grrt::slim_detail::isco_prograde(in.mass,in.spin);
    in.mdot=1.6399e16;
    std::vector<double> U = build_thin_disk_seed(in, op);
    const int N=std::max(in.n_nodes,4); const int i=10;                 // r≈14.0, the full256 holdout
    const double r_s=U[4*N+1], lr0=std::log(r_s), lr1=std::log(in.r_out);
    const double t=double(i)/double(N-1); const double ri=std::exp(lr0+(lr1-lr0)*t);
    const int j=i+1; const double tj=double(j)/double(N-1); const double rj=std::exp(lr0+(lr1-lr0)*tj);
    const double Omi=grrt::slim_detail::omega_from_ell(in.mass,in.spin,ri,U[4*i+2]);
    const double Omj=grrt::slim_detail::omega_from_ell(in.mass,in.spin,rj,U[4*j+2]);
    ColumnCoupledInputs ci{};
    ci.Sigma_target=std::max(U[4*i+0],1e2); ci.Tc=std::max(U[4*i+3],1.0);   // thin Tc (stale)
    ci.shear=std::max(grrt::slim_coupled_detail::shear_cgs(in,ri,Omi,rj,Omj),1e-300);
    ci.omega_z=std::max(grrt::slim_coupled_detail::omega_perp_cgs(in,ri),1e-300);
    ci.alpha=in.alpha;
    ci.rho_mid_guess=std::max(grrt::slim_detail::one_zone_closure(ci.Sigma_target,ci.Tc,ri,in,op).rho_mid,1e-30);
    ci.n_nodes=96; ci.max_iters=300; ci.tol=1e-8; ci.Teff_guess=0.0;
    // f_adv=0 seed has no root here (Σ0 ceiling); advective seed must find one.
    std::vector<double> U0, Ua;
    const bool ok0 = grrt::slim_coupled_detail::build_coupled_seed(ci, op, U0);
    const bool oka = grrt::slim_coupled_detail::build_coupled_seed_advective(ci, op, Ua);
    std::printf("  build_coupled_seed(f_adv=0)=%d  build_coupled_seed_advective=%d\n", ok0, oka);
    if (!oka) { std::printf("  FAIL: advective seed did not build\n"); failures++; return; }
    const int Na=ci.n_nodes; const double Tc_adv=Ua[4*Na+2 - 4*Na + 2]; // = Ua[2] midplane T_c
    const double fadv=Ua[4*Na+3];
    std::printf("  advective seed: T_c=%.3e  f_adv=%.3f\n", Ua[2], fadv); (void)Tc_adv;
    // Pin the reachable T_c and confirm the coupled column converges.
    ColumnCoupledInputs cj=ci; cj.Tc=std::max(Ua[2],1.0);
    const bool okc = grrt::slim_coupled_detail::solve_column_coupled(cj, op, nullptr).converged;
    std::printf("  solve at (Σ,T_c_adv) converged=%d\n", okc);
    if (!(fadv>0.0 && Ua[2]>3.0e6 && okc)) { std::printf("  FAIL: advective seed not usable\n"); failures++; }
}
```
Build `test-column-coupled`, confirm COMPILE failure (`build_coupled_seed_advective` undefined) → red. (Adapt namespace qualifiers to what compiles — the existing tests use `grrt::slim_coupled_detail::` for these; match them. Drop the dead `Tc_adv` line if it offends the compiler — the real read is `Ua[2]`.)

- [ ] **Step 2 — implement `build_coupled_seed_advective`** in `src/disk_column_coupled.cpp`, placed right after `build_coupled_seed`. Reuse `build_coupled_seed`'s secant + Σ0-match + packing, parameterized by `fa`:
```cpp
// f_adv-laddered sibling of build_coupled_seed: for advective (high-Σ) nodes the f_adv=0
// grey Σ0 saturates below Σ_target, so build_coupled_seed returns false. Freeing f_adv
// raises the Σ0 ceiling; find the SMALLEST f_adv>0 whose T_eff-secant Σ0-matches Σ_target,
// and read the column's OWN midplane T_c (never pinned). Same 0.30 Σ0-match band and packing.
static bool build_coupled_seed_advective(const ColumnCoupledInputs& in, const OpacityLUTs& op,
                                         std::vector<double>& U) {
    for (double fa : {0.5, 1.0, 2.0, 4.0}) {
        // ... same secant on T_eff as build_coupled_seed but with base_inputs_from(in, Te, fa),
        //     same SIGMA_SEED_BAND=0.30 gate; on match pack U (P_gas,Q,T,z per node; z0;
        //     U[4N+1]=Σ_target; U[4N+2]=T_eff; U[4N+3]=fa) and return true ...
        // (Factor the shared secant into secant_Teff_at_fadv(in,op,fa,U) if clean; else duplicate.
        //  Do NOT change build_coupled_seed's fa=0 behavior.)
    }
    return false;
}
```
Implement the secant fully (mirror lines 434-524 of `build_coupled_seed`, substituting `fa` for the hard `0.0` in `base_inputs_from`).

- [ ] **Step 3 — wire the fallback** (two call sites; replace the lone `build_coupled_seed` with `build_coupled_seed(...) || build_coupled_seed_advective(...)`, reading `Uc[2]` as T_c either way):
  - `tools/slim_full256_probe.cpp`, the `Tc_manif` step (`if (build_coupled_seed(cm, op, Uc)) Tc_manif=...`).
  - `tools/slim_coupled_walk_probe.cpp`, `calibrate_seed_to_manifold` (`if (build_coupled_seed(ci, op, Uc)) { U[4*i+3]=Uc[2]; ++n_ok; }`).

- [ ] **Step 4 — GREEN + regression.** Run `test-column-coupled`: AS1 green (build_coupled_seed=0, advective=1, converged=1) AND all pre-existing green (`build_coupled_seed` byte-identical at f_adv=0 nodes). Confirm a pre-existing convergence test's output is unchanged.

- [ ] **Step 5 — hand the commit message over (NO commit).** Stage `src/disk_column_coupled.cpp`, `tests/test_column_coupled.cpp`, `tools/slim_full256_probe.cpp`, `tools/slim_coupled_walk_probe.cpp`. (The controller runs the full256 18/18 integration gate and fills the count.)
```
feat(disk-column): advective seed-T_c fallback for high-Σ nodes

build_coupled_seed pins f_adv=0, whose grey Σ0 saturates ~few×10³ ≪ inner-node
demand ~1.5e4, so it honestly returns false there and the seed-T_c derivation
fell back to a stale cold thin T_c — an unreachable pin (full256 node 10 stuck
at T_c=3.2e6). Add build_coupled_seed_advective: the same T_eff-secant Σ0-match,
run at a small f_adv>0 ladder, reading the column's OWN midplane T_c (never
pinned). Wire it as the fallback in the two seed-T_c callers. build_coupled_seed
and its hot-path use are untouched; the advective builder is consulted only
where f_adv=0 already failed.

slim-full256-probe: 17/18 -> <N>/18. Gate AS1: f_adv=0 seed fails at node 10 but
the advective seed builds a reachable T_c and the coupled column converges.
```
Do not loosen gates, do not touch `build_coupled_seed` or its Σ0-match gate, do not run the relax. Report DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED.
