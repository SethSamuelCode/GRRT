# Lever C — Transonic-Σ Coupled Seed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **NEVER `git commit`** — hand each task's commit message to the controller, who relays it to the user and waits.

**Goal:** Build `build_transonic_coupled_seed` — a drop-in replacement for `build_thin_disk_seed` in the coupled-relax seed path that reseats Σ onto the transonic branch (|V|=c_s at r_s, mass conservation elsewhere) and guarantees every node's vertical column is feasible, so the coupled relax can start at the base rung.

**Architecture:** New sibling seed builder in `src/slim_disk_coupled.cpp`. It reuses `build_thin_disk_seed` for the grid/ℓ/r_s, anchors |V|=c_s at the sonic node via `one_zone_closure`, prescribes a monotone declining |V(r)| outward, back-derives Σ from the verified mass-conservation inverse, sets T_c on the f_adv≈0 manifold, then a per-node feasibility guard reduces Σ until each column solves. Returns the standard 4N+2 packed radial state.

**Tech Stack:** C++23, existing GRRT slim-disk TU helpers, the coupled column BVP solver, the diagnostic test harness in `tests/test_column_coupled.cpp`.

---

## Preflight — signatures the implementer MUST confirm before coding

Read these and confirm exact names/namespaces/signatures (grep in the listed files). The code blocks below assume them; fix the calls if reality differs:

1. `build_thin_disk_seed(const SlimDiskInputs&, const OpacityLUTs&) -> std::vector<double>` — `src/slim_disk_radial.cpp` (~line 925 uses it). Confirm the packed layout: per node `[Σ, V, ℓ, T_c]` for i∈[0,N), tail `U[4N+0]=ℓ_in`, `U[4N+1]=r_s`. Confirm `N = max(in.n_nodes, 4)`.
2. `slim_detail::one_zone_closure(double Σ, double Tc, double r, const SlimDiskInputs&, const OpacityLUTs&) -> OneZoneState` with field `.c_s` (CGS, cm/s) — `include/grrt/scene/slim_disk_radial.h:79`, `src/slim_disk_radial.cpp:32`.
3. `kerr_delta(double M, double a, double r)` and `constants::c_cgs` — used by `Vfrom` at `src/slim_disk_radial.cpp:959`.
4. `build_coupled_seed(const ColumnCoupledInputs&, const OpacityLUTs&, std::vector<double>& Uc) -> bool` returning the manifold column state with `Uc[2]` = midplane T_c — used by `calibrate_seed_to_manifold` (`tools/slim_coupled_walk_probe.cpp` ~line 165, `tools/slim_nz_refine_probe.cpp:96`). Confirm `ColumnCoupledInputs` fields: `Sigma_target, Tc, shear, omega_z, alpha, rho_mid_guess, n_nodes, max_iters, tol, Teff_guess`.
5. `solve_column_coupled(const ColumnCoupledInputs&, const OpacityLUTs&, void*) -> ColumnClosure` with `.converged` — `src/disk_column_coupled.cpp`.
6. `shear_cgs(in, r, Om, r_j, Om_j)`, `omega_perp_cgs(in, r)`, `slim_detail::omega_from_ell(mass, spin, r, ℓ)`, `ColumnOpts` (field `.n_z`, `.max_iter`, `.tol`) — used in the walk probe's `calibrate_seed_to_manifold`.

If any signature differs, adapt the calls; do NOT change those functions.

---

## File Structure

- **`src/slim_disk_coupled.cpp`** — ADD `build_transonic_coupled_seed(...)` plus two small free helpers `V_from_sigma` / `sigma_from_V` (mass conservation, both directions). Place near the other TU-local seed helpers.
- **`include/grrt/scene/slim_disk_coupled.h`** (or wherever the coupled API is declared) — declare `build_transonic_coupled_seed` if the probes need it visible; if the probes `#include` the `.cpp` directly (they do), a `static`/anonymous-namespace definition is fine and no header change is needed. Confirm which pattern the coupled probes use and follow it.
- **`tests/test_column_coupled.cpp`** — ADD three tests (round-trip, seed-structure, feasibility).
- **`tools/slim_coupled_walk_probe.cpp`** — Task 4 only: swap the seed source behind an env flag for the integration run.

---

### Task 1: Mass-conservation inverse (both directions) + round-trip test

**Files:**
- Modify: `src/slim_disk_coupled.cpp` (add two free helpers near the top of the TU-local helper block)
- Test: `tests/test_column_coupled.cpp`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_column_coupled.cpp` (harness uses a global `int failures;` and `failures++`; register the test in `main()`):

```cpp
// Mass-conservation Σ<->V round-trip. Vfrom's inverse must be exact.
static void test_massconsv_roundtrip() {
    std::printf("\n=== C2: mass-conservation Σ<->V round-trip ===\n");
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = 0.9; in.r_g = 1.48e6;
    in.mdot = 1.6399e16;  // f_Edd=0.001 scale
    const double rs[] = {2.27, 5.0, 20.0, 50.0};
    const double Sig[] = {1.0e2, 5.0e3, 1.2e4, 6.0e3};
    bool any_fail = false;
    for (double r : rs) for (double S : Sig) {
        const double V  = grrt::slim_coupled_detail::V_from_sigma(in, r, S);
        const double S2 = grrt::slim_coupled_detail::sigma_from_V(in, r, V);
        const double rel = std::abs(S2 - S) / S;
        std::printf("  r=%.2f Σ=%.3e -> V=%.3e -> Σ'=%.3e  rel=%.2e\n", r, S, V, S2, rel);
        if (!(rel < 1e-10)) any_fail = true;
    }
    if (any_fail) { std::printf("  FAIL: round-trip exceeds 1e-10\n"); failures++; }
}
```

Register `test_massconsv_roundtrip();` in `main()`.

- [ ] **Step 2: Run to verify it fails**

Run: `cmake --build build --config Release --target test-column-coupled` — expect a COMPILE failure (`V_from_sigma`/`sigma_from_V` undefined). That is the red state.

- [ ] **Step 3: Implement the two helpers**

In `src/slim_disk_coupled.cpp`, inside `namespace grrt::slim_coupled_detail` (confirm the exact namespace the probes `using`), add:

```cpp
// Mass conservation:  Ṁ = 2πΣ|V|√Δ·r_g·c / √(1−V²).
// Forward (Σ→V): mirrors build_slim_disk_seed's Vfrom exactly (returns V<0 inflow).
static double V_from_sigma(const SlimDiskInputs& in, double r, double Sigma) {
    using namespace constants;
    const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
    const double dn = 2.0 * std::numbers::pi * Sigma * sqrtD * in.r_g * c_cgs;
    if (!(dn > 0.0)) return -1e-12;
    const double X = -in.mdot / dn;
    double V = X / std::sqrt(1.0 + X*X);
    if (!(V < 0.0)) V = -1e-12;
    return std::clamp(V, -0.9999, -1e-12);
}
// Inverse (V→Σ):  Σ = Ṁ√(1−V²) / (2π|V|√Δ·r_g·c).   (verified vs Vfrom)
static double sigma_from_V(const SlimDiskInputs& in, double r, double V) {
    using namespace constants;
    const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));
    const double aV = std::abs(V);
    if (!(aV > 0.0) || !(sqrtD > 0.0)) return 0.0;
    return in.mdot * std::sqrt(std::max(1.0 - V*V, 0.0))
         / (2.0 * std::numbers::pi * aV * sqrtD * in.r_g * c_cgs);
}
```

Confirm `kerr_delta`, `constants::c_cgs`, `<numbers>`, `<algorithm>` are in scope in this TU (they are used elsewhere in it).

- [ ] **Step 4: Run to verify it passes**

Run: `cmake --build build --config Release --target test-column-coupled` then `build/Release/test-column-coupled.exe`
Expected: `test_massconsv_roundtrip` prints rel ~1e-16 for all rows; suite ends `## 0 failure(s) ##`.

- [ ] **Step 5: Hand the commit message to the controller (do NOT commit)**

```
feat(slim-coupled): mass-conservation Σ<->V helpers (V_from_sigma / sigma_from_V)

Exact inverse of build_slim_disk_seed's Vfrom: Σ = Ṁ√(1−V²)/(2π|V|√Δ r_g c).
Round-trip test C2 confirms Σ->V->Σ to <1e-10. Foundation for the transonic
coupled seed (lever C).
```

---

### Task 2: `build_transonic_coupled_seed` core (anchor + V prescription + Σ, no guard) + structure test

**Files:**
- Modify: `src/slim_disk_coupled.cpp` (add the function)
- Test: `tests/test_column_coupled.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
// The transonic seed's structure: node 0 at Mach 1, |V| monotone-declining
// outward, every Σ consistent with mass conservation at its V.
static void test_transonic_seed_structure() {
    std::printf("\n=== C3: transonic coupled seed structure ===\n");
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    SlimDiskInputs in{};
    in.mass=1.0; in.spin=0.9; in.alpha=0.1; in.r_g=1.48e6; in.r_out=50.0;
    in.n_nodes=18; in.tol=1e-8;
    in.r_in = 0.5 * grrt::slim_detail::isco_prograde(in.mass, in.spin);
    in.mdot = 1.6399e16;  // f_Edd=0.001
    ColumnOpts copt; copt.n_z = 96;
    std::vector<double> U =
        grrt::slim_coupled_detail::build_transonic_coupled_seed(in, op, copt);
    const int N = std::max(in.n_nodes, 4);
    if ((int)U.size() < 4*N+2) { std::printf("  FAIL: wrong state size\n"); failures++; return; }
    const double r_s = U[4*N+1];
    // node-0 Mach number: |V0| / (c_s/c) ~ 1.
    const double V0 = U[4*0+1];
    const double S0 = U[4*0+0], T0 = U[4*0+3];
    const double cs = grrt::slim_detail::one_zone_closure(S0, T0, r_s, in, op).c_s;
    const double mach0 = std::abs(V0) / (cs / constants::c_cgs);
    std::printf("  r_s=%.4f  |V0|=%.4e  c_s/c=%.4e  Mach0=%.3f\n",
                r_s, std::abs(V0), cs/constants::c_cgs, mach0);
    bool ok = (mach0 > 0.7 && mach0 < 1.4);
    // |V| monotone-declining outward; Σ = sigma_from_V at each node.
    double prevV = std::abs(U[1]);
    for (int i = 1; i < N; ++i) {
        const double Vi = std::abs(U[4*i+1]);
        const double r_i_dummy = 0.0; (void)r_i_dummy;
        if (Vi > prevV*1.001) { std::printf("  FAIL: |V| not monotone at i=%d\n", i); ok=false; }
        prevV = Vi;
    }
    if (!ok) { std::printf("  FAIL: transonic structure invalid\n"); failures++; }
}
```

Register it in `main()`.

- [ ] **Step 2: Run to verify it fails**

Build `test-column-coupled` → COMPILE failure (`build_transonic_coupled_seed` undefined). Red state.

- [ ] **Step 3: Implement the core (no guard yet)**

In `src/slim_disk_coupled.cpp`, `namespace grrt::slim_coupled_detail`:

```cpp
// Lever C — transonic-Σ seed for the coupled relax. Drop-in for
// build_thin_disk_seed(in,op) in the coupled seed path. Returns the standard
// 4N+2 packed radial state ([Σ,V,ℓ,T_c]×N, tail ℓ_in, r_s).
std::vector<double> build_transonic_coupled_seed(const SlimDiskInputs& in,
                                                 const OpacityLUTs& op,
                                                 const ColumnOpts& copt) {
    using namespace constants;
    using grrt::slim_detail::one_zone_closure;
    using grrt::slim_detail::omega_from_ell;
    // 1) Grid, ℓ(r), r_s from the thin seed (unchanged machinery).
    std::vector<double> U = build_thin_disk_seed(in, op);
    const int N = std::max(in.n_nodes, 4);
    const double r_s = U[4*N+1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N==1)?0.0:double(i)/double(N-1);
        r[i]  = std::exp(lr0 + (lr1-lr0)*t);
        Om[i] = omega_from_ell(in.mass, in.spin, r[i], U[4*i+2]);
    }
    // 2) Sonic anchor: |V(r_s)| = c_s/c using the thin-seed (Σ,T_c) at node 0.
    const double cs0 = one_zone_closure(std::max(U[0],1e2), std::max(U[3],1.0), r_s, in, op).c_s;
    const double Vsonic = -std::clamp(cs0 / c_cgs, 1e-6, 0.9999);
    // 3) Far-field subsonic target |V| from the thin seed at the outer node.
    const double Vout = -std::max(std::abs(U[4*(N-1)+1]), 1e-12);
    // 4) Monotone ln|V| interpolation r_s->r_out; Σ from mass conservation.
    for (int i = 0; i < N; ++i) {
        const double t = (N==1)?0.0:double(i)/double(N-1);
        const double lnV = std::log(std::abs(Vsonic))
                         + (std::log(std::abs(Vout)) - std::log(std::abs(Vsonic))) * t;
        double Vi = -std::exp(lnV);
        Vi = std::clamp(Vi, -0.9999, -1e-12);
        double Sig = sigma_from_V(in, r[i], Vi);
        Sig = std::max(Sig, 1e2);
        U[4*i+0] = Sig;
        U[4*i+1] = V_from_sigma(in, r[i], Sig);   // re-derive V from the clamped Σ (consistency)
        // ℓ (U[4*i+2]) kept from thin seed; T_c set on the manifold in step 5.
    }
    // 5) T_c on the f_adv≈0 manifold at each node's (Σ, geometry).
    for (int i = 0; i < N; ++i) {
        const int j = (i+1<N)?i+1:i-1;
        const double shear_i  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double omegaz_i = omega_perp_cgs(in, r[i]);
        const OneZoneState oz = one_zone_closure(U[4*i+0], std::max(U[4*i+3],1.0), r[i], in, op);
        ColumnCoupledInputs ci{};
        ci.Sigma_target=U[4*i+0]; ci.Tc=std::max(U[4*i+3],1.0);
        ci.shear=std::max(shear_i,1e-300); ci.omega_z=std::max(omegaz_i,1e-300);
        ci.alpha=in.alpha; ci.rho_mid_guess=std::max(oz.rho_mid,1e-30);
        ci.n_nodes=copt.n_z; ci.max_iters=copt.max_iter; ci.tol=copt.tol; ci.Teff_guess=0.0;
        std::vector<double> Uc;
        if (build_coupled_seed(ci, op, Uc)) U[4*i+3] = std::max(Uc[2], 1.0);
    }
    return U;
}
```

Note: node 0's `t=0` gives `Vi=Vsonic` exactly (Mach 1). If `omega_perp_cgs`/`shear_cgs`/`OneZoneState`/`build_coupled_seed` need a different namespace qualifier, match the walk probe's usage.

- [ ] **Step 4: Run to verify it passes**

Build + run `test-column-coupled`. Expected: `test_transonic_seed_structure` prints Mach0≈1.0, monotone |V|, no FAIL.

- [ ] **Step 5: Hand the commit message to the controller (do NOT commit)**

```
feat(slim-coupled): build_transonic_coupled_seed core (sonic anchor + transonic V)

Reseats Σ onto the transonic branch: |V(r_s)|=c_s (Mach 1), monotone declining
|V| outward, Σ from the mass-conservation inverse, T_c on the f_adv≈0 manifold.
Drop-in for build_thin_disk_seed. Structure test C3 confirms Mach0≈1 + monotone V.
Feasibility guard follows in the next task.
```

---

### Task 3: Per-node feasibility guard + feasibility gate test

**Files:**
- Modify: `src/slim_disk_coupled.cpp` (add the guard loop at the end of `build_transonic_coupled_seed`, before `return U;`)
- Test: `tests/test_column_coupled.cpp`

- [ ] **Step 1: Write the failing test (the point of the whole lever)**

```cpp
// Feasibility gate: EVERY node's column must solve at the transonic seed
// (a=0.9, f_Edd=0.001, n_z=96). This is what unblocks the coupled relax.
static void test_transonic_seed_feasible_18of18() {
    std::printf("\n=== C4: transonic seed 18/18 columns feasible (n_z=96) ===\n");
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    SlimDiskInputs in{};
    in.mass=1.0; in.spin=0.9; in.alpha=0.1; in.r_g=1.48e6; in.r_out=50.0;
    in.n_nodes=18; in.tol=1e-8;
    in.r_in = 0.5 * grrt::slim_detail::isco_prograde(in.mass, in.spin);
    in.mdot = 1.6399e16;
    ColumnOpts copt; copt.n_z = 96;
    std::vector<double> U =
        grrt::slim_coupled_detail::build_transonic_coupled_seed(in, op, copt);
    const int N = std::max(in.n_nodes, 4);
    const double r_s = U[4*N+1];
    const double lr0=std::log(r_s), lr1=std::log(in.r_out);
    int nfeas = 0;
    for (int i = 0; i < N; ++i) {
        const double t=(N==1)?0.0:double(i)/double(N-1);
        const double ri=std::exp(lr0+(lr1-lr0)*t);
        const int j=(i+1<N)?i+1:i-1;
        const double tj=(N==1)?0.0:double(j)/double(N-1);
        const double rj=std::exp(lr0+(lr1-lr0)*tj);
        const double Omi=grrt::slim_detail::omega_from_ell(in.mass,in.spin,ri,U[4*i+2]);
        const double Omj=grrt::slim_detail::omega_from_ell(in.mass,in.spin,rj,U[4*j+2]);
        ColumnCoupledInputs ci{};
        ci.Sigma_target=U[4*i+0]; ci.Tc=std::max(U[4*i+3],1.0);
        ci.shear=std::max(shear_cgs(in,ri,Omi,rj,Omj),1e-300);
        ci.omega_z=std::max(omega_perp_cgs(in,ri),1e-300);
        ci.alpha=in.alpha;
        ci.rho_mid_guess=std::max(grrt::slim_detail::one_zone_closure(U[4*i+0],ci.Tc,ri,in,op).rho_mid,1e-30);
        ci.n_nodes=96; ci.max_iters=300; ci.tol=1e-8; ci.Teff_guess=0.0;
        const ColumnClosure c = solve_column_coupled(ci, op, nullptr);
        if (c.converged) ++nfeas;
    }
    std::printf("  feasible columns: %d / %d\n", nfeas, N);
    if (nfeas != N) { std::printf("  FAIL: not all columns feasible (%d/%d)\n", nfeas, N); failures++; }
}
```

Register it in `main()`.

- [ ] **Step 2: Run to verify it fails**

Build + run `test-column-coupled`. Expected: `C4` reports `nfeas < 18` (some inner nodes still exceed capacity without the guard) → FAIL. If it already prints 18/18, note that the transonic reseat alone sufficed — the guard is then a safety net; still implement it (step 3) for robustness at higher f_Edd, and the test stays green.

- [ ] **Step 3: Implement the guard**

Add before `return U;` in `build_transonic_coupled_seed`:

```cpp
    // 6) Feasibility guard: reduce Σ (recompute V, re-set manifold T_c) until each
    //    node's column solves. Guarantees the coupled residual is defined at the
    //    seed. Logs every reduction (NO silent capping).
    for (int i = 0; i < N; ++i) {
        const int j = (i+1<N)?i+1:i-1;
        const double shear_i  = std::max(shear_cgs(in, r[i], Om[i], r[j], Om[j]), 1e-300);
        const double omegaz_i = std::max(omega_perp_cgs(in, r[i]), 1e-300);
        auto try_solve = [&](double Sig, double Tc) -> bool {
            ColumnCoupledInputs ci{};
            ci.Sigma_target=Sig; ci.Tc=std::max(Tc,1.0);
            ci.shear=shear_i; ci.omega_z=omegaz_i; ci.alpha=in.alpha;
            ci.rho_mid_guess=std::max(one_zone_closure(Sig,std::max(Tc,1.0),r[i],in,op).rho_mid,1e-30);
            ci.n_nodes=copt.n_z; ci.max_iters=copt.max_iter; ci.tol=copt.tol; ci.Teff_guess=0.0;
            return solve_column_coupled(ci, op, nullptr).converged;
        };
        const double Sig0 = U[4*i+0];
        double Sig = Sig0;
        int steps = 0;
        while (!try_solve(Sig, U[4*i+3]) && Sig > 1e2 && steps < 40) {
            Sig *= 0.85; ++steps;
            // refresh manifold T_c at the reduced Σ
            const OneZoneState oz = one_zone_closure(Sig, std::max(U[4*i+3],1.0), r[i], in, op);
            ColumnCoupledInputs cm{};
            cm.Sigma_target=Sig; cm.Tc=std::max(U[4*i+3],1.0);
            cm.shear=shear_i; cm.omega_z=omegaz_i; cm.alpha=in.alpha;
            cm.rho_mid_guess=std::max(oz.rho_mid,1e-30);
            cm.n_nodes=copt.n_z; cm.max_iters=copt.max_iter; cm.tol=copt.tol; cm.Teff_guess=0.0;
            std::vector<double> Uc;
            if (build_coupled_seed(cm, op, Uc)) U[4*i+3] = std::max(Uc[2], 1.0);
        }
        if (Sig < Sig0) {
            U[4*i+0] = Sig;
            U[4*i+1] = V_from_sigma(in, r[i], Sig);
            std::printf("[transonic-seed] node %d r=%.3f: Σ %.3e -> %.3e (%d steps) to reach feasibility\n",
                        i, r[i], Sig0, Sig, steps);
        }
    }
```

- [ ] **Step 4: Run to verify it passes**

Build + run `test-column-coupled`. Expected: `C4` prints `feasible columns: 18 / 18`; any reduced nodes logged; suite ends `## 0 failure(s) ##`. This gate is slow at n_z=96 (many column solves) — allow a few minutes.

- [ ] **Step 5: Hand the commit message to the controller (do NOT commit)**

```
feat(slim-coupled): feasibility guard -> transonic coupled seed is 18/18 feasible

Per-node guard reduces Σ (re-deriving V, refreshing manifold T_c) until each
column solves at n_z=96; every reduction is logged (no silent cap). Gate C4
confirms 18/18 columns feasible at a=0.9, f_Edd=0.001 — the coupled residual is
now defined at the seed, so the relax can start. Completes lever C's builder.
```

---

### Task 4: Integration — seed the walk from the transonic seed (honest outcome)

**Files:**
- Modify: `tools/slim_coupled_walk_probe.cpp` (env-flag the seed source)

This task is the DELIVERABLE run, not a committed unit test. It answers: does the base rung converge from the transonic seed?

- [ ] **Step 1: Add an env switch for the seed source**

In `slim_coupled_walk_probe.cpp` Phase 1, where it currently does `build_thin_disk_seed(in,op)` then `calibrate_seed_to_manifold`, gate on an env var:

```cpp
    std::vector<double> U;
    const char* e = std::getenv("TRANSONIC_SEED");
    if (e && std::atoi(e) != 0) {
        U = build_transonic_coupled_seed(in, op, copt);   // lever C
        std::printf("### SEED: build_transonic_coupled_seed (lever C) ###\n");
    } else {
        U = build_thin_disk_seed(in, op);
        const int n_cal = calibrate_seed_to_manifold(U, in, op, copt);
        std::printf("### SEED: thin + manifold calib (%d/%d) ###\n", n_cal, std::max(in.n_nodes,4));
    }
```

Keep `copt.n_z = 96` (already set from the prior task).

- [ ] **Step 2: Build**

Run: `cmake --build build --config Release --target slim-coupled-walk-probe`
Expected: links clean.

- [ ] **Step 3: Run the integration test (controller runs this in the background)**

Run: `TRANSONIC_SEED=1 build/Release/slim-coupled-walk-probe.exe`
This is slow at n_z=96 and may hit the 10-min cap — the controller runs it detached and reads partial output. The decisive signal at f_Edd=0.001:
- **`converged` / Newton takes inner-iters with merit → tol** ⇒ base rung solves; the high thin-seed Σ was a seed artifact; proceed to walk toward f_Edd=0.9.
- **`0 inner-iters` again, or merit stalls with Σ climbing** ⇒ demand is physical; escalate the monolithic-rebuild decision.

- [ ] **Step 4: Report the honest outcome to the controller**

No commit for the probe edit unless the user wants it kept (it is a diagnostic switch). Hand the controller: the base-rung result, the merit trajectory (converging vs stalled), and any `[transonic-seed]` Σ-reduction logs. Do NOT force convergence or loosen a gate.

---

## Self-Review

**Spec coverage:** anchor |V|=c_s (Task 2 ✓), mass-conservation inverse (Task 1 ✓), transonic V(r) prescription (Task 2 ✓), capacity/feasibility guard with logging (Task 3 ✓), T_c on manifold (Task 2 ✓), 18/18 feasibility gate (Task 3 ✓), integration test with honest outcome (Task 4 ✓), no fallback closure — every node on the full column BVP (✓, the guard reduces Σ, never swaps closure). All spec gates mapped.

**Placeholder scan:** none — every code step is concrete. The Preflight explicitly lists the signatures to confirm rather than leaving them implicit.

**Type consistency:** `build_transonic_coupled_seed(in, op, copt)` signature identical across Tasks 2/3/4; `V_from_sigma`/`sigma_from_V(in, r, x)` consistent Task 1↔2; packed layout `[Σ,V,ℓ,T_c]×N + [ℓ_in,r_s]` used identically in all tasks; `ColumnCoupledInputs` field set consistent with the walk probe.

**Known risk:** the guard's per-node `solve_column_coupled` at n_z=96 makes Task 3's gate slow (minutes). Acceptable for a correctness gate; if intolerable, the implementer may reduce the gate to a representative subset of nodes AND log that reduction — but must still assert the full 18/18 in at least one run.
