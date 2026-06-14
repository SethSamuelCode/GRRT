# DRAFT SCOPE — Slim-disk vertical-BVP coupling (cure the one-zone closure inadequacy)

**Status:** DRAFT scope/design (not a task-by-task plan). Written 2026-06-14.
**Owner context:** resume from `docs/superpowers/slim-disk-handoff-2026-06-12.md`; refinements `disk-approach-a-refinements.md`; verified formulas `references/disk-physics-formulas.md` §22–§23.

---

## 0. Problem statement (the verdict this scope responds to)

The `f_Edd≈0.9`, `a=0.9` slim disk is **unreachable in our one-zone vertical closure by construction** — not a numerical artifact, not (only) an instability, but a **closure inadequacy**: the real Sądowski disk exists but is **not a root of our equations**.

**Mechanism (cross-checked, see §1):** in the radiation-pressure-dominated (`β→0`) inner disk our one-zone closure ties the radiated flux to the thickness,
```
Q_rad = 64 σ T_c⁴/(3 κ Σ)  ──(b-dominated limit)──►  Q_rad = 8 c H Ω_⊥²/κ   ∝  H/r,
```
independent of (T,Σ). So radiating the required `Q_vis ≈ F_NT ≈ 1.0e26 erg/cm²/s` at `r≈r_s` forces `H/r ≈ 2–5` (a torus). A physical `H/r ≲ 0.5` can shed only ≈10–25% of `F_NT`, leaving an O(0.8·F_NT) surplus the model cannot dispose of — so the slim root does not exist in the one-zone model.

**Fix:** replace the one-zone `Q_rad` + `H = c_s/Ω_⊥` closure with a **vertical-structure-informed** flux (flux from `∫(dℱ/dz)dz` via radiative diffusion), decoupled from `H/r`, à la Sądowski 2011 (S11). The grey vertical BVP already exists in-tree: `src/disk_column_bvp.cpp`.

---

## 1. Cross-check of the verdict (evidence summary)

### 1a. Magnitude — CONFIRMED (code's own constants)
At `f_Edd=0.9, a=0.9, r≈r_s≈2.2 r_g`, `r_g=1.48e6 cm`, `κ≥κ_es=0.34`:
- `L_Edd = 4πGMc/κ = 1.48e39 erg/s`; `Ṁ = f_Edd·10·L_Edd/c² = 1.48e19 g/s` (M≈1 M_⊙ scale).
- **Energy that must be radiated:** `Ṁc²/(4πr_cm²) = 1.0e26 erg/cm²/s` at `r≈r_s` ⇒ `F_NT ≈ 1.0e26` (matches the diagnostic's `9.9e25`; the relativistic `f(r)` factor is O(1) at the inner edge).
- **Radiated by one-zone (b-limit):** `Q_rad = 8cΩ²H/κ` with `H=(H/r)·r_cm`:
  - using `Ω_orb²≈2.4e7` ⇒ coefficient `≈5.4e25·(H/r)`;
  - using the smaller relativistic `Ω_⊥²` (the ℋ-suppressed vertical epicyclic) ⇒ coefficient `≈2e25·(H/r)` (the diagnostic's number).
- **Balance:** `H/r ≈ F_NT / coeff = 2–5`. A physical `H/r≲0.5` sheds `≈0.5·(2–5)e25/1e26 ≈ 10–25%`, surplus `≈ 0.75–0.9·F_NT`. **CONFIRMED.**

### 1b. Papers — CONFIRMED (with one nuance)
- **S11 uses a FULL vertical integration**, NOT a one-zone `H=c_s/Ω`: hydrostatic (Eq 12), energy **generation** `dℱ/dz = (3𝒟/2𝒞)(αp/(1+f_adv))(M/r³)^½` (Eq 13), **radiative diffusion** `ℱ(z) = −(16σT³/3κρ)dT/dz` (Eq 14/16), surface BC `ℱ(h)≡σT_eff⁴ = 2σT⁴(h)` (Eddington, τ(h)=0). CONFIRMED.
- **Our `64σT_c⁴/(3κΣ)` IS S11 Eq 42** — but Eq 42 is explicitly the **one-zone/polytropic** flux (S11 §5 comparison), and Eq 45 carries a correction factor **`f_F = 0.94·𝒮_F`** (quoted for `α=0.01`; α-dependence to be confirmed for our `α=0.1`). CONFIRMED; our code omits `f_F`.
- **S11 says the polytropic/one-zone closure overestimates the photosphere height by >20% (at 0.1 Ṁ_Edd) and ~30% (at 1.0 Ṁ_Edd)** — Fig 17, §5. CONFIRMED. (Note: 20–30% is the *thin/efficient* regime overestimate; it is not the same as the O(1) `H/r≈4–5` torus the one-zone *radial* solve is driven to when forced to shed `F_NT` at `β→0` — that is a stronger, model-breaking failure, consistent with but larger than the polytropic 20–30%.)
- **S11 `𝒩₁` (Eqs 29, 32–33) keeps `(P/Σ)dlnη₃/dlnr` and `Ω_⊥²(η₄/η₃)dlnη₄/dlnr`**; the full `Q_adv` (Eq 29) also keeps `+η₃(P/Σ)dlnη₃/dlnr + Ω_⊥²η₄ dlnη₄/dlnr`. **Our `calN1` drops both** (`slim_disk_radial.cpp:578` documents "η-grad & Ω_⊥² drop"; the `Q_adv` bracket at `:1234` is only `[η₃dlnP−(1+η₃)dlnΣ]`). CONFIRMED.

### 1c. Empirical — CONFIRMED
`slim-sadowski-residual-probe.exe 48` (the production Sądowski-shape seed fed to `slim_radial_residual`), group magnitudes at `a=0.9`:

| f_Edd | ang  | rad  | merit |
|-------|------|------|-------|
| 0.05  | 28.2 | 22.1 | 17.8  |
| 0.10  | 31.2 | 31.2 | 21.9  |
| 0.20  | 50.8 | 49.0 | 34.9  |
| 0.40  | 85.9 | 104  | 66.7  |
| 0.60  | 119  | 169  | 102   |
| 0.90  | **214** | **335** | **196** |

The `rad` and `ang` groups grow **monotonically with radiation dominance**, reaching O(200–300) at `f_Edd=0.9` — **~5 orders above the 1e-3 merit floor**. A bounded relaxation does NOT reduce them (budget tripped, merit unmoved at 196). The Sądowski structure is **not near a root** of our one-zone residual. **CONFIRMED.**

### 1d. VERDICT
**The (b) closure-inadequacy verdict HOLDS UP.** All three independent lines (analytic magnitude, primary-source closure comparison, empirical residual) confirm it. Caveats, not refutations:
- The diagnostic's `Q_rad≈2e25·(H/r)` uses `Ω_⊥²`; with `Ω_orb²` it is `≈5e25·(H/r)` (still ⇒ `H/r≈2`, still a torus). The qualitative verdict is robust to this factor-of-2.5.
- "Closure inadequacy" and "instability" are **not mutually exclusive**: even with the correct vertical closure, a *stable* `f_Edd≈0.9` slim root must still exist and be reachable. The fix below is **necessary** to make the root exist; the stability-atlas (refinements #10) remains the separate check that it is *reachable/stable*. The scope should not claim the BVP coupling alone guarantees `f_Edd=0.9` converges — only that it removes the structural obstruction.

---

## 2. `disk_column_bvp` interface (what it gives, what it lacks)

Signatures (`include/grrt/scene/disk_column_bvp.h`):
- `ColumnInputs` (`:11`): `T_eff` [K], `shear` [1/s], `omega_z` (Ω_z) [1/s], `alpha`, `rho_mid_guess`, `n_nodes` (~150), `max_iters`, `tol`.
- `ColumnBVPSolution` (`:24`): profiles `q,z,P,P_gas,Q,T,rho`; scalars `z0` (half-thickness = z_max), `Sigma0` (Σ = 2∫ρdz), `tau_mid`, `converged`, `iters`, `final_residual`.
- `solve_column_bvp(in, opacity, warm_start=nullptr)` (`:54`) — Newton relaxation; **accepts a warm-start `U` (length 4N+2) from a converged neighbour** (continuation). Returns empty profiles on non-convergence (no fabricated fallback).
- Test hooks (`:60,65,70`): `column_residual_test`, `column_numerical_jacobian_test`, `column_jacobians_test` (analytic-vs-FD `∂R/∂U` cross-check).

**Critical interface facts for the coupling:**
1. **The column is driven by `T_eff`, not by a heating rate.** The surface flux is *prescribed* `Q(N-1)=σT_eff⁴` (`disk_column_bvp.cpp:133,138`); the internal `dℱ/dz` generation (`α·shear·…`) is present but the surface BC pins the emergent flux. **So `T_eff` is an INPUT and the emergent flux is an OUTPUT only trivially (= σT_eff⁴).** The radial solve needs the *opposite* causality (given Σ,heating → emergent flux). This is the central interface impedance mismatch (see §3.1).
2. **Returns `z0` (→H) and `Sigma0` and `tau_mid`** — but **does NOT return vertical moments η₃/η₄** (`∫E dz / ∫P dz`, `∫P/ρ`-type). Those must be computed by post-integrating the returned `P,T,rho,z` profile (straightforward; the profile is in hand).
3. **NO Jacobian of OUTPUTS wrt INPUTS is exposed.** The analytic Jacobian (`disk_column_bvp.cpp:229`) is the *internal* Newton `∂R/∂U`. There is **no** exported `∂{z0,Sigma0,F_emergent,η₃,η₄}/∂{Σ,T_eff,shear,Ω_z}` sensitivity. This is the crux gap for the radial Newton (§3.2).

---

## 3. Coupling design

### 3.0 What the radial residual consumes from the BVP, per node
Replace, at each radial node:
- `Q_rad = 64σT_c⁴/(3κΣ)`  →  **emergent flux** `F = ∫₀^{z₀}(dℱ/dz)dz = σT_eff⁴` from the column (decoupled from H/r).
- `H = one_zone_closure(...).H`  →  **`z₀`** from the column.
- one-zone `P/Σ`, `c_s²`, `β`  →  vertical-integral analogues from the column profile (Σ-weighted).
- `η₃, η₄` (currently `η₃(β)=3−1.5β`, `η₄` absent)  →  **vertical moments** post-integrated from the column `P,E,ρ,z`; enables restoring the dropped `𝒩₁` gradient terms (§1b, §3.5).

### 3.1 The causality fix (prerequisite)
The radial unknowns are `(Σ, V, ℓ, T_c)` per node. The column wants `(T_eff, shear, Ω_z, α, Σ)`. Two options:
- **(A) Invert the column to be Σ+heating-driven.** Add a sibling entry that, given `Σ` and the **viscous heating rate** `Q⁺ = (3/2)αPΩ`-equivalent (radial `Q_vis`), solves for the emergent `T_eff` such that `∫dℱ/dz = Q⁺` (replace the `T_eff` surface BC by a `∫generation = prescribed Q⁺` integral BC; `T_eff` becomes an unknown). This is the physically correct direction and is the S11 coupling (energy generation `dℱ/dz` set by `αp`, flux floats). Moderate solver change; reuses the analytic `∂R/∂U`.
- **(B) Keep `T_eff`-driven, close the loop in the radial Newton.** Treat `T_c` (radial) ↔ `T_eff` (column) as the matching variable: the radial energy row becomes `Q_vis − F_column(T_eff(T_c)) − Q_adv = 0`, with `F_column = σT_eff⁴` and a separate relation `T_eff = g(T_c, Σ, …)` from the column's midplane-to-surface T drop. Simpler to bolt on but leaves the flux still `∝σT_eff⁴`; the decoupling from H/r comes through `z₀` being set by the *vertical* hydrostatic balance, not `c_s/Ω`. **Recommend (A)** — it is the genuine S11 closure and avoids re-introducing a `σT⁴`-flux that is again structurally tied to a single temperature.

### 3.2 The analytic-Jacobian challenge (the crux — feasibility-determining)
The radial Newton needs `∂(BVP outputs)/∂(radial unknowns)`, i.e. `∂{F, z₀, η₃, η₄}/∂{Σ, T_c}` (V, ℓ enter the column only through `shear`/`Ω_z`, which are geometric, not unknowns — so the live sensitivities are wrt `Σ` and `T_c`/heating). Options:
- **(i) Chain-rule through the column's own analytic `∂R/∂U`.** The column converged at `R(U;p)=0` for parameters `p=(Σ,T_eff,…)`. By the implicit function theorem, `dU/dp = −(∂R/∂U)⁻¹ (∂R/∂p)`. The column **already factorizes `∂R/∂U`** for its Newton step, so one extra back-substitution per parameter column gives `dU/dp`; the outputs `{F,z₀,η₃,η₄}` are explicit functions of `U`, so `∂outputs/∂p = (∂outputs/∂U)(dU/dp)`. **This is the right answer: cheap (reuse the existing LU factor), exact, and consistent with the FD-oracle discipline.** It requires exporting `∂R/∂p` (the parameter-sensitivity columns) from the column solver — a *new* but mechanical addition (the residual's `Σ`/`T_eff` dependence is already in `node_deriv`).
- **(ii) FD per column** (perturb `Σ`, `T_c`, re-solve the column, difference the outputs). Robust, trivial to write, but **2–3× the column cost per radial node per radial Newton iteration** and FD-noisy (the very ceiling the slim solver fought). Acceptable as a *bring-up* oracle / fallback, not the production engine.
- **Verdict:** (i) is feasible and is the only option that keeps the radial Newton's quadratic convergence and the FD-cross-check discipline. **Effort is concentrated here.** Risk: the column's `(∂R/∂U)⁻¹` is only valid at a *converged* column; if a column fails to converge mid-radial-Newton the sensitivity is undefined — needs a graceful degrade (fall back to one-zone `Q_rad` for that node + flag, or FD).

### 3.3 Cost / blow-up
- A vertical BVP per column, `N_col ≈ 48–150` (radial nodes) × `n_z ≈ 150` (vertical nodes) × per radial Newton iteration (`~20–60`).
- Naive: `O(N_col · n_z³ · radial_iters)` if each column is a fresh cold solve — prohibitive (dense column Jacobian is `O(n_z³)`).
- **Mitigations (most already in-tree):**
  - **Warm-start each column** from (a) its converged radial-neighbour, (b) its own previous radial-Newton-iterate. `solve_column_bvp` already takes a `warm_start U` — a converged column moves only slightly between radial iterations, so this drops column cost to a few Newton steps. (refinements 1b: `bootstrap_column` continuation + neighbour warm-start already exist.)
  - **Band the column solve** (refinements #2: block-tridiagonal Thomas, `O(n_z)` not `O(n_z³)`) — *prerequisite* for affordability at `N_col·radial_iters` column solves.
  - **Coarse-then-fine in `n_z`** (refinements #1: Richardson) — run `n_z≈48` during early radial iterations, refine near convergence.
  - **Reuse the column LU factor** for the §3.2(i) sensitivity back-substitution (free once factored).
- Estimate with mitigations: warm-started + banded column ≈ `O(n_z)` × few steps ≈ cheap; total `≈ N_col · radial_iters · (few · n_z)` — comparable to today's one-zone radial solve × a modest constant (single-digit ×), not the naive 10³×.

### 3.4 Convergence strategy — RECOMMEND staged 2D fixed-point (with a fully-coupled fallback)
- **Fully-coupled radial+vertical Newton** (one giant Jacobian over radial unknowns *and* all column states): most robust quadratic convergence, but the Jacobian is enormous (`(4N_col+2 + N_col·(4n_z+2))²`) and couples everything; high implementation risk.
- **Staged 2D fixed-point (Sądowski's own scheme, §22 "self-consistent 2D iteration"):** (1) freeze radial `(Σ,V,ℓ,T_c)`; (2) solve all columns (warm-started) → get `{F,z₀,η₃,η₄}` per node; (3) feed those into the radial residual coefficients and take radial Newton steps (using §3.2(i) sensitivities for the column-derived terms); (4) repeat to outer-loop convergence. **RECOMMEND this** — it matches S11's structure, isolates failures (a bad column is visible), reuses the existing radial Newton and column solver almost unchanged, and the §3.2(i) sensitivities make the radial step properly Newton (not a slow Picard). Risk: outer-loop fixed-point can be slow/oscillatory near the fold — damp with under-relaxation on `{F,z₀}` and/or Anderson acceleration if needed.

### 3.5 Interim / lighter option (assess honestly)
Two cheap, in-the-radial-model-only changes (no BVP coupling):
- **(a) Carry S11's `f_F≈0.94` flux factor** on `Q_rad` (Eq 45). **Effect: ~6% on the flux. Does NOT relieve the H/r tie** — `Q_rad` is still `∝H/r` in the b-limit; a 6% coefficient bump cannot close an O(0.8·F_NT) gap. Cosmetic for the `f_Edd=0.9` problem (though correct physics worth carrying; α-dependence of `f_F` must be checked for `α=0.1`).
- **(b) Restore the dropped `𝒩₁` terms** `(P/Σ)dlnη₃/dlnr` and `Ω_⊥²(η₄/η₃)dlnη₄/dlnr` (S11 Eqs 29/32–33), re-derived in GRRT's `Ω_⊥²=Ω_K²ℋ` convention (§22 note). **Effect: corrects the transonic eigenvalue/`f_adv` shape; it makes the *radial* structure more S11-faithful but does NOT change `Q_rad`'s `∝H/r` dependence.** So (b) improves correctness of the advective/transonic physics but, like (a), **does not by itself create the missing slim root.**
- **Honest assessment:** **Only the full BVP coupling (§3.1–3.4) is sufficient** to make the `f_Edd≈0.9` root exist, because only it breaks `Q_rad ∝ H/r`. (a)+(b) are worthwhile correctness improvements (and (b) is a prerequisite for *using* the η₄ moment the BVP will provide), but they are **not** a shortcut to the render target. They should land as independent, gated correctness commits — not be oversold as a fix.

---

## 4. Validation & gates
- **NT-reduction gate (must stay green):** as `Ṁ→0` the coupled solver must still match `VolumetricDisk::compute_radial_structure` (`slim-nt-term-probe`, `Q_vis/F_NT≈1` flat at `a=0.9,f_Edd=0.02`). The BVP coupling must reduce to the thin-disk flux in the gas-dominated limit.
- **FD-Jacobian cross-check (must stay green):** the new §3.2(i) chain-rule sensitivities must match an FD oracle (perturb radial `Σ,T_c`, re-solve the column, difference) — extend `test-slim-jacobian` to cover the column-derived rows.
- **Column-internal cross-check (existing):** `column_jacobians_test` analytic ≡ FD for `∂R/∂U`.
- **The target gate:** a converged, physical `f_Edd≈0.9`, `a=0.9` disk with `H/r ≲ 0.5` passing all of the above, with the `rad`/`ang` residual groups at the 1e-3 floor (vs the O(200–300) one-zone values in §1c). Cross-check `H/r`, `T_eff(r)`, spectrum ballpark vs S11 figures.
- **Re-run `slim-sadowski-residual-probe`** under the coupled model: the seed residual groups should collapse toward the floor (the structure becomes a root).

## 5. Risks / unknowns
- **Column non-convergence inside the radial Newton** (undefined sensitivity) — needs graceful degrade + a robust column warm-start chain.
- **f_F α-dependence** (quoted for α=0.01; we run α=0.1) — verify before relying on Eq 45.
- **Cost** could still bite if banding (#2) isn't done first — sequence #2 before the coupling.
- **Outer-loop oscillation near the fold** — may need damping/Anderson; the staged scheme's failure modes are gentler than a monolithic Newton's.
- **η₄ definition** must be transcribed from S11 (Eqs 8/11) and gated by its own β/vertical-moment probe (mirror refinement #11's η₃ gate).

## 6. Effort sizing (rough)
- Banded column solve (#2): prerequisite, S.
- Column causality inversion (§3.1(A)) + parameter-sensitivity export `∂R/∂p` (§3.2(i)): M–L (the crux).
- Vertical-moment post-integration (η₃/η₄) + restoring `𝒩₁` terms (§3.5(b)): M.
- Staged 2D outer loop wiring (§3.4) + warm-start chain: M.
- Gates/probes extension: S–M.
- Overall: a multi-week research-grade subsystem, but every piece sits on existing in-tree machinery (column solver + analytic Jacobian, warm-start, arclength). Highest-risk item is the §3.2(i) sensitivity export + keeping the FD cross-check exact.

---
*Sources: S11 [arXiv:1006.4309] Eqs 12–16, 29, 32–33, 42, 45, §5/Fig 17; S09 [arXiv:0906.0355] Eqs 1, 4, 5, 6; Abramowicz et al. 1997 (vertical equilibrium). Code: `src/slim_disk_radial.cpp` (one_zone_closure :32, calN1 :578/:580, Gbalance/Q_rad :1298–1338, Q_adv :1228–1244), `src/disk_column_bvp.cpp` / `include/grrt/scene/disk_column_bvp.h`. Empirical: `tools/slim_sadowski_residual_probe.cpp`.*
