# Convective vertical column (component B) — design spec (2026-06-30)

**Goal:** add mixing-length convection to the grey vertical-column BVP (`disk_column_bvp.cpp`) so the column's surface-density capacity rises (~2×) and its NT factor `f_F` lifts from the pure-radiative ~0.42 toward the literature ~0.9 — the prerequisite for a physical `f_Edd≈0.9` disk. Convection is the **Sądowski 2011 §2.2(iii)** MLT model; **all formulas are verified in `references/disk-physics-formulas.md` §24** (opus+Wolfram).

**Why (grounded):** the rigorous sonic-Σ test showed the pure-radiative column saturates at `Σ0≈1e4–1.3e4`, but the f_Edd=0.9 disk needs ≥ that at its thinnest point — the convective onset. This is **component B** of the seed-pivot plan (B first, then A=transonic-V seed, then couple); see `slim-disk-handoff-2026-06-30-AUDIT-and-seed-pivot.md`.

**Scope (LOCKED):** closed-form **gas+radiation** EOS for `∇_ad`/`C_p` (NO partial ionization — we render only `T_c>10⁷ K`, fully ionized); `α_ML=1` (Sądowski's value, no free parameter); steady-state (not time-stepped). Out of scope: partial ionization (possible later #14), the transonic seed (component A), the coupling wiring.

---

## Architecture

One new internal module + one modified ODE row; everything else in the column solver is untouched.

**Data flow per column node (depth z):** existing state → closure gives `ρ, P, T, κ_R, β` → compute `∇_rad` (§24 Eq16) and `∇_ad` (§24, gas+rad) → **Schwarzschild test**: if `∇_rad ≤ ∇_ad`, `∇ = ∇_rad` (today's path, bit-identical); else compute `C_p, δ, H_p, H_ml, τ_ml, w`, solve the cubic (§24 Eq20) for `y>0`, set `∇ = ∇_conv` (§24 Eq19) → the `dT/dz` row uses `∇`: `dT/dz = ∇ · (T/P) · dP/dz`.

**Files:**
- **Modify `src/disk_column_bvp.cpp`:** the `node_deriv` temperature row (today implicitly `∇=∇_rad`); add a `convective_gradient(...)` helper (criterion + cubic + ∇_conv) returning `∇` and its partials; extend the analytic Jacobian (`column_*jacobian`) with `∂∇/∂{P_gas,T,Q,z}` via implicit differentiation of the cubic.
- **Modify `references/disk-physics-formulas.md`:** §24 (DONE, doc-first).
- **Test `tests/test_disk_column_bvp.cpp`** (+ a focused `tools/slim_convection_probe.cpp`).

**Key boundary:** `convective_gradient` is a pure function of the local closure outputs — testable in isolation, reduces to `∇_rad` continuously at the `∇_rad=∇_ad` boundary.

---

## The physics (all from §24 — verified)

Schwarzschild: `∇ = ∇_rad` if `∇_rad≤∇_ad` else `∇_conv`.
`∇_rad = 3κ_R Q P/(16σT⁴Ω_⊥²z)` · `∇_ad = (4−3β)/(16−12β−1.5β²)` · `C_p = R_g(16/β²−12/β−3/2)` · `δ = (4−3β)/β` (SIGN: positive, NOT the signed `∂lnρ/∂lnT`).
`H_p = p/(ρΩ_⊥²z+√(pρ)Ω_⊥)`, `H_ml=H_p`, `τ_ml=ρκ_R H_ml`.
`∇_conv = ∇_ad+(∇_rad−∇_ad)y(y+w)`; cubic `(9/4)(τ_ml²/(3+τ_ml²))y³+wy²+w²y−w=0`; `1/w²=[(3+τ_ml²)/(3τ_ml)]²·[Ω_⊥²zH_ml²ρ²C_p²]/(512σ²T⁶H_p)·δ·(∇_rad−∇_ad)`.

---

## Error handling / numerical hazards

- **The δ sign trap (§24):** use `δ=(4−3β)/β>0`. A unit test asserts `1/w²>0` whenever `∇_rad>∇_ad`.
- **Cubic root selection:** take the unique positive real root `y>0` (Cardano or a guarded Newton from `y₀≈min(1, w⁻¹)`); assert `y(y+w)∈[0,1]` (i.e. `∇_ad ≤ ∇_conv ≤ ∇_rad`). Clamp to that interval on roundoff.
- **Midplane z→0:** `∇_rad` is `0/0` (`Q∝z`, `dP/dz∝z`); evaluate via the `Q/z→dQ/dz|₀` limit already used by the radiative row — convection is inactive at the very midplane only if `∇_rad≤∇_ad` there (usually it's most active just off-midplane).
- **Boundary stiffness — ⚡ CONTINGENCY LEVER (evidence-driven, do NOT build up front):**
  The Schwarzschild switch is continuous in value (`∇_conv→∇_rad` as `∇_rad↓∇_ad`), and the onset is *naturally soft* — working the cubic in the inefficient (near-boundary) limit, `∇_rad − ∇_conv ∝ (∇_rad − ∇_ad)³`, i.e. convection turns on with continuous value AND slope (≈C¹/C²). So a damaging Jacobian kink most likely **never forms**, and we ship the plain switch.
  **IF** the analytic-vs-FD Jacobian cross-check (gate 4) or the column Newton shows **chatter at the convective boundary** (the solver oscillating across `∇_rad=∇_ad` without converging — symptom: a node whose `∇` flips radiative↔convective between iterations, merit stalling), **THEN pull this lever:** add a narrow C¹ blend that interpolates `∇` smoothly across `∇_rad ∈ [∇_ad, ∇_ad·(1+ε)]` (ε~1e-3) so value and slope are continuous. This is a small, localized add; it is deferred only because the natural smoothness likely makes it unnecessary. Decision = measured chatter, not preemptive worry.
- **Analytic Jacobian:** `∂∇_conv/∂(state)` via implicit diff of the cubic (`dy = −(∂F/∂params)/(∂F/∂y)`). Gate against the FD oracle; if a term is too messy, FD *that one partial* only as a stopgap — but the column/Schur Jacobian must stay overwhelmingly analytic (FD noise is what we cured).

---

## Testing / validation gates (TDD)

1. **Pure-radiative reduction (the safety net):** where `∇_rad≤∇_ad`, the column output is **bit-for-bit** today's result → `test-slim-coupled-nt-probe` / NT reduction at low Ṁ stays green. *Write this test first.*
2. **∇_ad / C_p unit tests:** match §24 closed forms; limits 0.40/0.25 (∇_ad) and 5/2·R_g (C_p) at β=1.
3. **Cubic correctness:** `∇_ad ≤ ∇_conv ≤ ∇_rad`; efficient limit (τ_ml→∞ ⇒ ∇_conv→∇_ad), inefficient (τ_ml→0 ⇒ ∇_conv→∇_rad).
4. **Analytic-vs-FD Jacobian** cross-check on a convective state (the permanent oracle) — clean to ~1e-6.
5. **Σ0-capacity lift:** at the f_Edd=0.9 inner geometry (`slim-sonic-sigma-probe` node), the convective column holds **~2× more Σ0** than pure-radiative → `Σ0_cap` crosses the sonic-Σ demand (col/req ≥ 1 at r_s).
6. **f_F lift:** the NT-reduction factor rises from ~0.42 toward ~0.9 (NOT required to hit 0.94 exactly — no partial ionization).

**Done = component B:** gates 1–6 green, committed (message handed over). Then component A (transonic-V seed), then couple at f_Edd=0.9.

---

## Workflow
Doc-first ✓ (§24 written + verified before this spec). Never `git commit` — hand the message over. Present every reviewer rec & WAIT. Verify load-bearing claims opus+Wolfram. Gates green; convergence ≠ physical.
