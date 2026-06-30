# Convective Vertical Column (Component B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Sądowski-2011 mixing-length convection to the grey vertical-column BVP so the column's Σ0-capacity rises (~2×) and its NT factor f_F lifts from ~0.42 toward ~0.9 — the prerequisite for a physical f_Edd≈0.9 disk.

**Architecture:** Convection modifies exactly ONE thing — the temperature-gradient (`dT/dz`) term in `node_deriv` (`src/disk_column_bvp.cpp`). A new pure helper `convective_gradient()` applies the Schwarzschild criterion: where `∇_rad ≤ ∇_ad` it returns the radiative gradient (today's column, bit-identical); where `∇_rad > ∇_ad` it solves the MLT cubic and returns the shallower convective gradient. The analytic Jacobian gains the convective `dT` partials by local finite-difference of that helper (the rest of the Jacobian stays analytic). Reduces exactly to the current pure-radiative column in the stable regime → NT gate preserved by construction.

**Tech Stack:** C++23, the existing `disk_column_bvp.cpp` Newton column solver + its analytic/numerical Jacobian cross-check harness (`tests/test_column_bvp.cpp`, target `test-column-bvp`). All formulas: `docs/superpowers/references/disk-physics-formulas.md` §24 (opus+Wolfram-verified). EOS: fully-ionized (`mu_fully_ionized` constant), gas+radiation, NO partial ionization.

**Workflow constraints (non-negotiable):** Never `git commit` — hand the commit message to the user and WAIT. Present every reviewer recommendation with a take and WAIT. Gates green; convergence ≠ physical.

---

## File Structure

- **`src/disk_column_bvp.cpp`** (modify): add `nabla_ad()`, `c_p_gas_rad()`, `convective_gradient()` helpers in the anonymous namespace (near `node_deriv`, ~line 83); modify the `d.dT` computation in `node_deriv` (line 104); add convective `dT` partials in `analytic_jacobian` (~lines 285–290). One file, one responsibility (the column BVP) — no new files.
- **`tests/test_column_bvp.cpp`** (modify): add unit tests for the helpers + the pure-radiative-reduction gate + extend the analytic-vs-numerical Jacobian test to a convective state.
- **`tools/slim_convection_probe.cpp`** (create): the Σ0-capacity-lift + f_F-lift validation probe at the f_Edd=0.9 inner geometry.
- **`docs/superpowers/references/disk-physics-formulas.md` §24** (already written, doc-first).

Key constants (already in `grrt::constants`): `sigma_SB`, `a_rad`, `k_B`, `m_p`, `mu_fully_ionized`. The specific gas constant is `R_g = k_B/(mu_fully_ionized*m_p)` (constant — fully ionized).

---

## Task 1: Closed-form thermodynamic helpers (∇_ad, C_p)

**Files:**
- Modify: `src/disk_column_bvp.cpp` (anonymous namespace, just above `struct Deriv` ~line 82)
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Write the failing test** (add to `tests/test_column_bvp.cpp`, and call it from `main`)

```cpp
// Convection helpers: gas+radiation closed forms (disk-physics-formulas.md §24).
static void test_convection_thermo_helpers() {
    using namespace grrt::detail_bvp;   // namespace exposed in Step 3
    std::printf("\n=== convection: nabla_ad / c_p closed forms ===\n");
    // nabla_ad limits: beta=1 (gas) -> 0.4 ; beta=0 (radiation) -> 0.25
    if (std::abs(nabla_ad(1.0) - 0.40) > 1e-12) { std::printf("  FAIL nabla_ad(1)=%.6f\n", nabla_ad(1.0)); failures++; }
    if (std::abs(nabla_ad(0.0) - 0.25) > 1e-12) { std::printf("  FAIL nabla_ad(0)=%.6f\n", nabla_ad(0.0)); failures++; }
    // mid-beta is finite and inside (0.25, 0.40)
    const double na = nabla_ad(0.5);
    if (!(na > 0.25 && na < 0.40)) { std::printf("  FAIL nabla_ad(0.5)=%.6f out of band\n", na); failures++; }
    // c_p: beta=1 -> (5/2) R_g
    const double Rg = grrt::constants::k_B / (grrt::constants::mu_fully_ionized * grrt::constants::m_p);
    if (std::abs(c_p_gas_rad(1.0) - 2.5 * Rg) > 1e-6 * 2.5 * Rg) { std::printf("  FAIL c_p(1)\n"); failures++; }
    // c_p grows as beta -> 0 (radiation)
    if (!(c_p_gas_rad(0.3) > c_p_gas_rad(1.0))) { std::printf("  FAIL c_p monotonic\n"); failures++; }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --config Release --target test-column-bvp`
Expected: COMPILE FAIL — `nabla_ad` / `c_p_gas_rad` / `grrt::detail_bvp` undefined.

- [ ] **Step 3: Write minimal implementation** (in `src/disk_column_bvp.cpp`, anonymous namespace ~line 82; expose via a named detail namespace so tests can reach it)

```cpp
} // close anonymous namespace temporarily? NO — keep these in a NAMED detail ns:
// Put the convection helpers in grrt::detail_bvp so tests can call them, but keep
// node_deriv et al. where they are. Add near the top of the anonymous-namespace block:
} // end anonymous namespace  (if not already)
namespace grrt { namespace detail_bvp {
using namespace grrt::constants;
// Adiabatic gradient, gas+radiation mixture (gamma=5/3), beta = p_gas/p_total.
// disk-physics-formulas.md §24 (Wolfram-verified). Limits: 0.40 (gas) / 0.25 (rad).
inline double nabla_ad(double beta) {
    const double b = std::clamp(beta, 0.0, 1.0);
    return (4.0 - 3.0*b) / (16.0 - 12.0*b - 1.5*b*b);
}
// Specific heat at constant pressure, gas+radiation. §24: C_p = R_g(16/b^2 - 12/b - 3/2).
inline double c_p_gas_rad(double beta) {
    const double b = std::clamp(beta, 1e-12, 1.0);   // b->0 diverges (radiation); floor it
    const double Rg = k_B / (mu_fully_ionized * m_p);
    return Rg * (16.0/(b*b) - 12.0/b - 1.5);
}
}} // namespace grrt::detail_bvp
namespace { // reopen anonymous namespace for node_deriv etc.
using namespace grrt::constants;
using grrt::detail_bvp::nabla_ad;
using grrt::detail_bvp::c_p_gas_rad;
```
(Adjust the namespace open/close to the file's actual structure — the helpers must be visible to `node_deriv` AND to the test. Simplest: define them in `grrt::detail_bvp` BEFORE the anonymous namespace block, then `using` them inside.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --config Release --target test-column-bvp && ./build/Release/test-column-bvp`
Expected: the `=== convection: nabla_ad / c_p closed forms ===` block prints no FAIL.

- [ ] **Step 5: Commit** (hand message to user — do NOT run git commit)

```
feat(disk-column): gas+radiation nabla_ad + C_p closed forms (convection #13, §24)
```

---

## Task 2: The `convective_gradient()` helper (criterion + MLT cubic)

**Files:**
- Modify: `src/disk_column_bvp.cpp` (`grrt::detail_bvp`, after `c_p_gas_rad`)
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Write the failing test**

```cpp
static void test_convective_gradient() {
    using namespace grrt::detail_bvp;
    using namespace grrt::constants;
    std::printf("\n=== convection: convective_gradient ===\n");
    // (a) STABLE: tiny flux Q -> nabla_rad << nabla_ad -> returns radiative gradient,
    //     and grad == the bare radiative value (criterion off).
    {
        const double rho=1e-2, T=1e7, Ptot=1e15, Q=1e3, kR=0.34, z=1e4, omega_z=1e-3;
        double nab; bool convective;
        const double g = convective_gradient(rho, T, Ptot, Q, kR, z, omega_z, nab, convective);
        const double dPdz = -rho*omega_z*omega_z*z;
        const double dTdz_rad = -3.0*kR*rho*Q/(16.0*sigma_SB*T*T*T);
        if (convective) { std::printf("  FAIL: stable node flagged convective\n"); failures++; }
        if (std::abs(g - dTdz_rad) > 1e-12*std::abs(dTdz_rad)) { std::printf("  FAIL: stable grad != radiative\n"); failures++; }
        (void)dPdz; (void)nab;
    }
    // (b) UNSTABLE: large flux Q -> nabla_rad > nabla_ad -> convective; the returned
    //     dT/dz is SHALLOWER (smaller |dT/dz|) than radiative, and nabla in [nab_ad, nab_rad].
    {
        const double rho=1.0, T=3e7, Ptot=3e16, Q=1e17, kR=0.34, z=1e3, omega_z=3e-3;
        double nab; bool convective;
        const double g = convective_gradient(rho, T, Ptot, Q, kR, z, omega_z, nab, convective);
        const double dTdz_rad = -3.0*kR*rho*Q/(16.0*sigma_SB*T*T*T);
        const double beta = Ptot>0 ? (Ptot - (a_rad/3.0)*T*T*T*T)/Ptot : 1.0;
        const double na = nabla_ad(beta);
        const double dPdz = -rho*omega_z*omega_z*z;
        const double nr = (Ptot/T)*(dTdz_rad/dPdz);
        if (!convective) { std::printf("  FAIL: unstable node not flagged convective\n"); failures++; }
        if (!(std::abs(g) < std::abs(dTdz_rad))) { std::printf("  FAIL: convective grad not shallower (%.3e vs %.3e)\n", g, dTdz_rad); failures++; }
        if (!(nab >= na - 1e-9 && nab <= nr + 1e-9)) { std::printf("  FAIL: nabla %.4f outside [%.4f,%.4f]\n", nab, na, nr); failures++; }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --config Release --target test-column-bvp`
Expected: COMPILE FAIL — `convective_gradient` undefined.

- [ ] **Step 3: Write minimal implementation** (in `grrt::detail_bvp`)

```cpp
// Solve the Sądowski-2011 MLT efficiency cubic  A y^3 + w y^2 + w^2 y - w = 0  for the
// unique real root y>0 (§24 Eq 20). Guarded Newton from y0=min(1,1/w); the cubic is
// monotone increasing in y for y>0 (A,w>0) with F(0)=-w<0, so Newton converges.
inline double mlt_solve_y(double A, double w) {
    double y = (w > 1.0) ? 1.0/w : 1.0;   // good for both efficient (w small) & inefficient (w large)
    for (int it = 0; it < 40; ++it) {
        const double F  = A*y*y*y + w*y*y + w*w*y - w;
        const double dF = 3.0*A*y*y + 2.0*w*y + w*w;
        const double step = F/dF;
        y -= step;
        if (y <= 0.0) y = 1e-12;
        if (std::abs(step) <= 1e-12*(std::abs(y)+1e-12)) break;
    }
    return std::max(y, 0.0);
}

// Returns dT/dz (the SAME quantity node_deriv multiplies by dz_dq). When the column is
// convectively stable (nabla_rad <= nabla_ad) it returns the bare radiative gradient
// (BIT-IDENTICAL to the old code). When unstable it returns nabla_conv*(T/Ptot)*dP/dz.
// Outputs: nabla_out = the actual dlnT/dlnP used; convective = whether MLT was applied.
inline double convective_gradient(double rho, double T, double Ptot, double Q, double kR,
                                  double z, double omega_z,
                                  double& nabla_out, bool& convective) {
    const double dTdz_rad = -3.0*kR*rho*Q/(16.0*sigma_SB*T*T*T);   // §23 radiative law
    const double dPdz     = -rho*omega_z*omega_z*z;                // hydrostatic dP/dz
    convective = false;
    // Midplane guard: z->0 makes nabla_rad 0/0 (Q,dPdz both ~z). No flux to convect there.
    if (!(z > 0.0) || !(dPdz < 0.0) || !(Q > 0.0)) { nabla_out = 0.0; return dTdz_rad; }
    const double Pg   = Ptot - (a_rad/3.0)*T*T*T*T;
    const double beta = (Ptot > 0.0) ? std::clamp(Pg/Ptot, 0.0, 1.0) : 1.0;
    const double nab_rad = (Ptot/T) * (dTdz_rad/dPdz);            // §24 Eq 16 (ratio form)
    const double nab_ad  = nabla_ad(beta);
    nabla_out = nab_rad;
    if (nab_rad <= nab_ad) return dTdz_rad;                       // STABLE: radiative, bit-identical
    // --- convectively unstable: MLT (§24 Eqs 17-21) ---
    const double Hp  = Ptot / (rho*omega_z*omega_z*z + std::sqrt(Ptot*rho)*omega_z);  // Eq 18
    const double Hml = Hp;                                        // alpha_ML = 1
    const double tau = rho*kR*Hml;                               // tau_ml
    const double Cp  = c_p_gas_rad(beta);
    const double delta = (4.0 - 3.0*beta)/std::max(beta,1e-12);  // = -(dlnrho/dlnT)_p > 0 (SIGN-RESOLVED)
    const double T6 = T*T*T*T*T*T;
    const double pref = (3.0+tau*tau)/(3.0*tau);
    const double inv_w2 = pref*pref
        * (omega_z*omega_z * z * Hml*Hml * rho*rho * Cp*Cp) / (512.0*sigma_SB*sigma_SB*T6*Hp)
        * delta * (nab_rad - nab_ad);                            // §24 Eq 21 (delta>0 => inv_w2>0)
    const double w = 1.0/std::sqrt(std::max(inv_w2, 1e-300));
    const double A = (9.0/4.0) * (tau*tau)/(3.0 + tau*tau);      // Eq 20 cubic coeff
    const double y = mlt_solve_y(A, w);
    double frac = y*(y + w);                                     // (nab_conv-nab_ad)/(nab_rad-nab_ad) in [0,1]
    frac = std::clamp(frac, 0.0, 1.0);
    const double nab_conv = nab_ad + (nab_rad - nab_ad)*frac;    // Eq 19
    nabla_out = nab_conv;
    convective = true;
    return nab_conv * (T/Ptot) * dPdz;                          // dT/dz at the convective gradient
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --config Release --target test-column-bvp && ./build/Release/test-column-bvp`
Expected: `=== convection: convective_gradient ===` prints no FAIL.

- [ ] **Step 5: Commit** (hand message to user)

```
feat(disk-column): convective_gradient() — Schwarzschild + MLT cubic (§24 Eqs 16-21)
```

---

## Task 3: Wire convection into `node_deriv` — pure-radiative reduction gate FIRST

**Files:**
- Modify: `src/disk_column_bvp.cpp:104` (`node_deriv`, the `d.dT` line)
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Write the failing test** (the SAFETY NET — gate 1 of the spec)

```cpp
static void test_pure_radiative_reduction() {
    using namespace grrt::constants;
    std::printf("\n=== convection: pure-radiative reduction (bit-identical) ===\n");
    // A cool, low-flux column is convectively STABLE everywhere -> the solved column must
    // be BIT-IDENTICAL to the pre-convection solver. We assert the converged Sigma0/z0 and
    // the residual at the seed are unchanged vs a captured baseline.
    grrt::ColumnInputs in{};
    in.n_nodes = 32; in.T_eff = 5e5; in.shear = 1e-3; in.omega_z = 1e-3;
    in.alpha = 0.1; in.f_adv = 0.0; in.rho_mid_guess = 1e-3;
    auto lut = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    grrt::ColumnBVPSolution sol = grrt::solve_column_bvp(in, lut, nullptr);
    if (!sol.converged) { std::printf("  FAIL: stable column did not converge\n"); failures++; return; }
    // Baseline numbers captured from the PRE-convection build (fill in at Step 2):
    const double Sigma0_baseline = /* CAPTURE */ 0.0;
    if (Sigma0_baseline > 0.0 && std::abs(sol.Sigma0 - Sigma0_baseline) > 1e-10*Sigma0_baseline) {
        std::printf("  FAIL: Sigma0 drifted %.12e vs baseline %.12e\n", sol.Sigma0, Sigma0_baseline); failures++;
    }
    std::printf("  stable Sigma0 = %.12e (record as baseline)\n", sol.Sigma0);
}
```

- [ ] **Step 2: Capture the baseline, run to verify reduction**

Run (BEFORE editing `node_deriv`): `cmake --build build --config Release --target test-column-bvp && ./build/Release/test-column-bvp`
Record the printed `stable Sigma0 = …`; paste it into `Sigma0_baseline`. This is the pre-convection number the convective build must reproduce bit-for-bit.

- [ ] **Step 3: Modify `node_deriv`** (`src/disk_column_bvp.cpp`, replace line 104)

```cpp
    // dT/dz: grey radiative diffusion, OR the shallower MLT convective gradient where
    // the column is convectively unstable (Schwarzschild). Stable nodes are bit-identical
    // to the old radiative form (convective_gradient returns dTdz_rad there). §24.
    double nabla_unused; bool convective_unused;
    const double dTdz = convective_gradient(rho, T, Ptot, Q, kR, z, omega_z,
                                            nabla_unused, convective_unused);
    d.dT = dTdz * dz_dq;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --config Release --target test-column-bvp && ./build/Release/test-column-bvp`
Expected: `Sigma0` matches the baseline to 1e-10 (stable column unchanged). Also run the full suite: `test-column-coupled`, `slim-coupled-nt-probe` must stay green (NT reduction at low Ṁ is a stable column).

- [ ] **Step 5: Commit** (hand message to user)

```
feat(disk-column): apply MLT convective gradient in node_deriv; stable reduction bit-identical
```

---

## Task 4: Analytic Jacobian — convective `dT` partials (local FD), gated by the oracle

**Files:**
- Modify: `src/disk_column_bvp.cpp` `analytic_jacobian` (~lines 285–290, the `dT_*` partials)
- Test: `tests/test_column_bvp.cpp` (`test_analytic_vs_numerical_jacobian`, add a convective state)

- [ ] **Step 1: Write the failing test** (extend the existing FD cross-check to a convective column)

```cpp
static void test_jacobian_convective_state() {
    std::printf("\n=== convection: analytic vs numerical Jacobian (convective) ===\n");
    grrt::ColumnInputs in{};
    in.n_nodes = 24; in.T_eff = 3e7; in.shear = 6e3; in.omega_z = 3e-3;   // hot, high-flux => convective
    in.alpha = 0.1; in.f_adv = 0.0; in.rho_mid_guess = 1.0;
    auto lut = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    double maxrel = grrt::column_jacobian_maxrel_test(in, lut);   // analytic vs numerical, exposed hook
    std::printf("  max rel mismatch = %.3e\n", maxrel);
    if (!(maxrel < 1e-5)) { std::printf("  FAIL: convective Jacobian mismatch\n"); failures++; }
}
```
(If `column_jacobian_maxrel_test` does not exist, add it next to `column_numerical_jacobian_test` in the test-hook section of `disk_column_bvp.cpp` — it builds both Jacobians at the seed and returns the max relative entry mismatch.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cmake --build build --config Release --target test-column-bvp && ./build/Release/test-column-bvp`
Expected: FAIL — the analytic `dT` partials still use the radiative-only formulas (lines 285–290), so they disagree with the FD Jacobian at convective nodes.

- [ ] **Step 3: Implement convective `dT` partials by local FD of `node_deriv`'s dT** (in `analytic_jacobian`, where the per-node `J.dT_*` are set ~lines 285–290)

```cpp
        // dT/dq partials. Radiative form is analytic (rho cancels; no z-dependence).
        // Where the node is CONVECTIVE, dT/dq depends on z and on (Pg,Q,T) through the
        // MLT cubic in a way that is messy to differentiate analytically; we central-FD
        // the dT term of node_deriv for THIS node only (a smooth scalar of 4 locals,
        // far less noisy than FD-ing the whole residual, and gated by the FD oracle).
        // Everything else in the Jacobian stays analytic.
        double nabla_chk; bool is_conv;
        convective_gradient(rho, T, Ptot, Q, kappa, z, in.omega_z, nabla_chk, is_conv);
        if (!is_conv) {
            // STABLE: keep the existing analytic radiative partials (unchanged).
            J.dT_dQ = -3.0 * kappa * Sigma0 / (32.0 * sigma_SB * T3);
            J.dT_dS = -3.0 * kappa * Q / (32.0 * sigma_SB * T3);
            J.dT_dP = -3.0 * Q * Sigma0 / (32.0 * sigma_SB * T3) * dk_dP;
            J.dT_dT = -3.0 * Q * Sigma0 / (32.0 * sigma_SB) * (dk_dT / T3 - 3.0 * kappa / (T3 * T));
            J.dT_dz = 0.0;
        } else {
            // CONVECTIVE: dT/dq = node_deriv(...).dT ; central-FD w.r.t. Pg,Q,T,z and Sigma0.
            auto dTdq = [&](double Pg_, double Q_, double T_, double z_, double S_) {
                return node_deriv(Pg_, Q_, T_, z_, S_, in.alpha, in.shear, in.omega_z, in.f_adv, op).dT;
            };
            const double hP=1e-6*std::max(std::abs(Pg),1e-300), hQ=1e-6*std::max(std::abs(Q),1e-300);
            const double hT=1e-6*std::max(T,1.0), hz=1e-6*std::max(std::abs(z),1e-300), hS=1e-6*std::max(Sigma0,1e-300);
            J.dT_dP = (dTdq(Pg+hP,Q,T,z,Sigma0) - dTdq(Pg-hP,Q,T,z,Sigma0))/(2*hP);
            J.dT_dQ = (dTdq(Pg,Q+hQ,T,z,Sigma0) - dTdq(Pg,Q-hQ,T,z,Sigma0))/(2*hQ);
            J.dT_dT = (dTdq(Pg,Q,T+hT,z,Sigma0) - dTdq(Pg,Q,T-hT,z,Sigma0))/(2*hT);
            J.dT_dz = (dTdq(Pg,Q,T,z+hz,Sigma0) - dTdq(Pg,Q,T,z-hz,Sigma0))/(2*hz);
            J.dT_dS = (dTdq(Pg,Q,T,z,Sigma0+hS) - dTdq(Pg,Q,T,z,Sigma0-hS))/(2*hS);
        }
```
Add a `double dT_dz;` field to the per-node partial struct `J`, and scatter it into the `R_T` row alongside the others: in the `R_T` assembly block (~line 344) add `at(r, ci+3) += -half_dq * ji.dT_dz;` and `at(r, cj+3) += -half_dq * jj.dT_dz;` (the `+3` column is `z`). Confirm the variable names `Pg`, `kappa`, `dk_dP`, `dk_dT`, `T3`, `Sigma0`, `op` match the surrounding scope (they are defined in the loop body ~lines 256–290).

- [ ] **Step 4: Run test to verify it passes**

Run: `cmake --build build --config Release --target test-column-bvp && ./build/Release/test-column-bvp`
Expected: `max rel mismatch < 1e-5` on the convective state AND the existing radiative `test_analytic_vs_numerical_jacobian` still passes (stable node path unchanged).

- [ ] **Step 5: Commit** (hand message to user)

```
feat(disk-column): convective dT Jacobian partials (local-FD of node_deriv); FD-oracle clean
```

---

## Task 5: Validation — Σ0-capacity ~2× lift + f_F lift at the f_Edd=0.9 inner geometry

**Files:**
- Create: `tools/slim_convection_probe.cpp`
- Modify: `CMakeLists.txt` (add `slim-convection-probe` target — mirror `slim-sonic-sigma-probe` registration)

- [ ] **Step 1: Write the probe** (the success metric for the whole component)

```cpp
// SLIM CONVECTION VALIDATION  (DIAGNOSTIC — DELETABLE)
// At the f_Edd=0.9 inner-disk geometry (the slim-sonic-sigma-probe node, r~1.94-2.84),
// compare the column Σ0 capacity WITH convection (this build) vs the pure-radiative
// ceiling (~1e4 from slim-sonic-sigma-probe). GATE: capacity rises ~2x and crosses the
// sonic-Σ demand (col/req >= 1 at r_s). Also report f_F = 2F/(64 sigma Tc^4/3 kappa Sigma).
#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1
#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include <cstdio>
#include <cmath>
#include <algorithm>
using namespace grrt;
int main() {
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    // inner-node geometry from slim-sonic-sigma-probe (a=0.9, f_Edd=0.9, r_s~1.94):
    const double shear = 6.059e3, omega_z = 3.148e3;
    double cap = -1.0;
    for (int it = 0; it <= 18; ++it) {
        const double Te = 3e5 * std::pow(1e8/3e5, it/18.0);
        ColumnInputs in{}; in.n_nodes = 96; in.T_eff = Te; in.shear = shear;
        in.omega_z = omega_z; in.alpha = 0.1; in.f_adv = 0.0; in.rho_mid_guess = 1.0;
        ColumnBVPSolution s = solve_column_bvp(in, op, nullptr);
        if (s.converged && s.Sigma0 > cap) cap = s.Sigma0;
    }
    const double pure_radiative_ceiling = 9.6e3;   // from slim-sonic-sigma-probe (n_z=96)
    std::printf("convective Sigma0 capacity = %.4e ; pure-radiative = %.4e ; lift = %.2fx\n",
                cap, pure_radiative_ceiling, cap/pure_radiative_ceiling);
    std::printf("%s\n", (cap > 1.5*pure_radiative_ceiling)
        ? "PASS: convection lifts the Sigma0 capacity (>=1.5x)"
        : "INVESTIGATE: capacity lift < 1.5x");
    return 0;
}
```

- [ ] **Step 2: Register + build**

Add to `CMakeLists.txt` (after the `slim-sonic-sigma-probe` block):
```cmake
add_executable(slim-convection-probe tools/slim_convection_probe.cpp)
target_include_directories(slim-convection-probe PRIVATE include/ ${CMAKE_BINARY_DIR}/include third_party/)
```
Run: `cmake --build build --config Release --target slim-convection-probe`
Expected: builds clean.

- [ ] **Step 3: Run + judge the lift**

Run: `./build/Release/slim-convection-probe`
Expected: `lift >= ~2x` and capacity crosses ~1.5e4 (the sonic-Σ demand at r_s). If lift < 1.5×, STOP and investigate (the convection isn't doing enough — re-check Eq 21 sign/units before proceeding to component A). **convergence ≠ physical: also eyeball that the convective column's T_c is COOLER than the radiative one at the same Σ (the physical signature).**

- [ ] **Step 4: Commit** (hand message to user)

```
test(slim-disk): convection Σ0-capacity-lift probe (component B success gate)
```

---

## Component B done-criteria

All green: Task-1 helper limits; Task-2 gradient (reduction + bracket + shallower); Task-3 pure-radiative reduction bit-identical + NT/coupled gates still green; Task-4 convective Jacobian FD-clean <1e-5; Task-5 Σ0 capacity lift ~2× crossing the sonic-Σ demand + cooler T_c. Then proceed to **component A** (transonic-V seed) → couple at f_Edd=0.9.

**Deferred contingency:** the ⚡ C¹ boundary-blend lever (spec §hazards) — only if Task-4's FD oracle or the column Newton shows chatter at the convective boundary (a node flipping radiative↔convective between iterations). Not built up front.
