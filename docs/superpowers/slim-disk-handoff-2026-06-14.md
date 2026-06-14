# Slim-disk handoff — resume here (2026-06-14)

**Read this first after compaction.** Supersedes the 2026-06-12/13 handoffs. The headline changed: the disk **physics is now correct and committed**, and a rigorous diagnostic (A) **proved why f_Edd≈0.9 is unreachable** — it's a **one-zone-closure inadequacy**, not a bug or a seed problem. The path forward is the **vertical-BVP closure coupling** (Phase-3). Scope: `docs/superpowers/plans/2026-06-14-slim-disk-vertical-bvp-coupling.md`.

---

## 0. THE NEXT TASK (the resume action)

**Couple the existing grey vertical-structure BVP (`src/disk_column_bvp.cpp`) per-column into the radial slim solve**, so the radiative flux comes from the vertical integration `∫dF/dz` (decoupled from H/r) instead of the one-zone `Q_rad = 64σT_c⁴/(3κΣ)`. This is the "self-consistent 2D iteration" §22 always described as Phase-3, and it is the **derived, verified** fix for the f_Edd≈0.9 wall (see §2). The column solver is already in-tree with an analytic Jacobian (Approach-A refinement #3, FD-cross-checked). See the scope doc for the design, the analytic-Jacobian challenge, cost, and the interim option. Gates: NT-reduction stays green; success = a **physical** f_Edd≈0.9 disk (H/r≲0.5, gas-dominated outward, passes all validity gates) — NOT the torus.

(Interim/lighter option, assess first: carry S11's `f_F≈0.94` flux factor + restore the dropped `dlnη₃/dlnr`, `Ω_⊥²η₄` vertical-coupling terms in `calN1` — but the diagnostic suggests only the full BVP coupling truly relieves the H/r tie.)

---

## 1. THE GOVERNING THEME — incorrect-model vs unstable vs impossible

Non-convergence (or odd-but-converged results) is ONE of three things; telling them apart is the whole game:
1. **Incorrect model / numerical artifact** — a bug, or numerics can't reach a solution that exists, **OR the model is too simplified to admit the real solution.**
2. **Physically unstable** — a real but unstable steady state (S-curve middle branch); exists, nature limit-cycles.
3. **Genuinely impossible** — no steady solution.

**The f_Edd≈0.9 wall is the definitive category-1 example of the *third sub-kind*: model inadequacy.** A real, stable f_Edd≈0.9 slim disk EXISTS (Sądowski's full-vertical-structure model), but it is **not a root of our one-zone equations** — our closure is too simplified to represent it. This is distinct from (a) "a reachable root we can't seed to" and (c) "genuinely unstable/absent." **Convergence ≠ physical** was hammered home twice: the f_Edd=0.9 "thick-seed success" was a **TORUS artifact** that passed every in-house gate, and the diagnostic then proved the torus is the one-zone closure's *only* energy-balanced solution there. Judge converged states with INDEPENDENT checks (the in-house NT-reduction theorem; literature-shape comparison; the f_adv>−1 physical bound) — never gate-pass alone.

---

## 2. THE CLOSURE-INADEQUACY VERDICT (the headline — diagnostic A, 2026-06-14)

**f_Edd≈0.9 is unreachable in the one-zone model BY CONSTRUCTION.** The mechanism (Wolfram-confirmed):
- In the radiation-dominated inner disk, the one-zone closure makes the radiative cooling **`Q_rad = 8·c·H·Ω²/κ` — independent of T and Σ, linear in H** (the closure ties flux to thickness). Equivalently `Q_rad ∝ H/r` at fixed r.
- The disk must radiate `Q_vis ≈ F_NT ≈ 9.9e25 erg/cm²/s`; the one-zone `Q_rad ≈ 2.0e25·(H/r)`, and the advective ceiling is only ~0.04–0.35·F_NT. **So energy balance at the rad-dominated sonic point *demands* H/r ≈ 4–5 — the torus.** A physical H/r≲0.5 sheds ≲18% of the flux, leaving an ~+0.8·F_NT surplus no physical (Σ,T) can zero at Mach-1.
- **The torus was never a bug** — H/r≈4 is the one-zone closure's only energy-balanced root in the rad-dominated regime. The right-*shaped* seed (verified anti-torus) gives residuals O(100–300) at f_Edd=0.9 because there is no nearby root.

**Why Sądowski can and we can't:** S11 does a **full vertical-structure integration** — the emergent flux solves the vertical BVP (`dℱ/dz` generation + `ℱ=−16σT³/3κρ·dT/dz` diffusion + photosphere BC), so flux is **decoupled from H/r** (a moderate-H/r disk radiates a large flux). Our `Q_rad=64σT_c⁴/(3κΣ)` is S11's Eq. 42 *with* an `f_F≈0.94` correction; S11 states the one-zone/polytropic closure overestimates thickness ~20–30%. Our `calN1` also drops the `dlnη₃/dlnr` and `Ω_⊥²(η₄/η₃)dlnη₄/dlnr` vertical-coupling terms S11 keeps. (Cross-check of the paper claims + the empirical residual probe + the magnitude is in task #154 / the scope doc.)

**Implication:** no seed / continuation / arclength / spin trick reaches f_Edd≈0.9 — any "converged" 0.9 in the one-zone model is the torus. The session's correct-physics work (§3) was **necessary but not sufficient**; the **closure** is the ceiling. Fix = §0.

---

## 3. WHAT'S COMMITTED — the physics is now correct

`fable`'s equation audit (fable since pulled — §7) + opus + Wolfram found/fixed and implemented, all COMMITTED:
- **Three §23 transcription bugs:** Q_vis length divisor `/r_cm` not `/r_g`; Q_vis metric factor `A^½Γ/(Δ^½r²)` (S09 Eq6×Eq4); the Q_adv η₃-inversion bracket `[η₃dlnP−(1+η₃)dlnΣ]`, η₃=1/(Γ₁−1) (S11 Eq29).
- **#11 — state-dependent moments** `η₃(β)=3−1.5β`, `Γ̃₁(β)=1+1/η₃` (residual + analytic Jacobian + β-clamp).
- **#12 — full azimuthal Lorentz factor** `Γ²=1/(1−V²)+ℓ²r²/A` in the torque law + Q_vis (mass law stays radial u^r); Jacobian: mass-row Γ³ decoupled, corrected `∂Γ/∂V=V/((1−V²)²Γ)`, new `∂Γ/∂ℓ=ℓr²/(AΓ)`. Wolfram-confirmed derivatives. NT band re-baselined 0.91–1.13 → **1.10–1.20** (the +15% is the intentional `2Fℓ` radiative-angular-momentum-flux S09 drops — documented, not a regression; this is the SAME family of "dropped term" as the closure issue).
- **Arclength hardening** (secant orientation, fold-aware ds, f_adv gate) + the **`1+f_adv>ε` physical gate** (rejects the torus's Q_rad singularity).

**Gates (keep green):** NT-reduction (`slim-nt-term-probe`, now band ~1.1–1.2), FD-Jacobian (`test-slim-jacobian`, 0 failures), entropy probe (`slim-qadv-entropy-probe`). Don't tune to pass.

---

## 4. ARCHITECTURE (`src/slim_disk_radial.{h,cpp}`)

Two-level hybrid transonic relaxation (= Sądowski 2009's method): outer `ℓ_in` bracket (`solve_outer_bracket`, the 𝒟-sign topology) + inner fixed-`ℓ_in` Newton relaxation (`relax_structure`), free sonic node `r_s=U[4N+1]`, state `U` length `4N+2`. Exact analytic Jacobian (`slim_analytic_jacobian`) validated by the FD oracle. Warm-start Ṁ-continuation driver (`solve_slim_disk_radial`). Hardened pseudo-arclength (`solve_slim_disk_arclength`) — rounds the fold to 0.143 on physical states. Seeds: `build_thin_disk_seed` (f_Edd-aware r_s/ℓ_in, "P1") + `build_slim_disk_seed` (the principled Sądowski anti-torus shape — correct SHAPE, but no root in the one-zone model → §2). One-zone closure `one_zone_closure` (the inadequacy). Validity gate adds `1+f_adv>ε`. **The column solver `src/disk_column_bvp.cpp` (Approach-A, analytic-Jacobian DONE) is the coupling target.**

---

## 5. CURRENT STATE / OPEN ISSUES

- **f_Edd≈0.9 (the render target):** GATED on the vertical-BVP closure coupling (§0/§2). The one-zone model cannot reach it. This is the #1 blocker and it's now precisely diagnosed + scoped.
- **Fold post-#12:** moved up (#12's inner heating extends the branch; cold seed converges 0.16/0.18 at a=0.9; deep sub-fold 0.02/0.05 clean). #12-related **cold-seed mid-range basin hole (0.06–0.14)** — a seed-tooling artifact (warm-start sidesteps; deep sub-fold + 0.16+ converge); retune or rely on warm-start.
- **Near-extremal spin (a→0.999):** parallel conditioning/continuation track (spin-homotopy seam bug); downstream of the closure fix. The geometry machinery (#12, metric factors) is in place.
- **Volumetric thick-disk rendering:** at f_Edd≈1 the disk is thick (H/r~0.3–0.5) → needs the volumetric render path (branch `fix/volumetric-ring`), not the thin-equatorial model. Downstream; state not re-examined this session.
- **β/f_adv sawtooth at f_Edd=0.05 outer disk** (#140) — mesh/convergence artifact; tracked.
- Deferred: the `2Fℓ` radiative angular-momentum flux (the accepted +15% NT offset; beyond S09); the stability atlas (refinements #10).

---

## 6. KEY FILES & SOURCES

- `src/slim_disk_radial.{h,cpp}` (solver), `src/disk_column_bvp.cpp` (**the coupling target**), `src/opacity.cpp`, `src/volumetric_disk.cpp::compute_radial_structure` (the NT reference).
- `docs/superpowers/references/disk-physics-formulas.md` §19–§23. `docs/superpowers/disk-approach-a-refinements.md` (#10 atlas, #11/#12 DONE). `docs/superpowers/plans/2026-06-14-slim-disk-vertical-bvp-coupling.md` (**the scope**). Spec `2026-06-08-relativistic-slim-disk-design.md`.
- Probes (deletable, `tools/`): `slim-nt-term-probe` (NT gate, has the #12 Qvis_az column), `slim-qadv-entropy-probe`, `slim-coldseed-sweep`, `slim-warmstart-sweep`, `slim-arclength-probe`, `slim-slimseed-probe`, `slim-sadowski-residual-probe` (the A empirical test).
- Sources: S09 [arXiv:0906.0355], S11 [arXiv:1006.4309] (full vertical structure: flux from `∫dF/dz`, `f_F≈0.94`, one-zone overestimates thickness ~20–30%), AF13 [arXiv:1104.5499], BPT72, Page-Thorne, NT73, SS73. ar5iv HTML renders for fetching; PDF doesn't convert.

---

## 7. WORKFLOW CONSTRAINTS (non-negotiable)

- **Never `git commit`** — hand the message to the user. **Present every reviewer rec with a take and WAIT.** **Doc-first** for formula edits.
- **Gates green** (NT-reduction + FD-Jacobian + entropy probe). Don't tune to pass.
- **Verification = opus + Wolfram, NOT sonnet** (user: sonnet "not smart enough" for physics/Jacobian, 2026-06-13). **fable is PULLED** (Anthropic, 2026-06-13) — don't dispatch it. Wolfram (symbolic) + the FD-Jacobian gate are the model-independent oracles. (memory `reference_fable_access_pulled`.)
- Subagents conserve the orchestrator's context. Safety budget on; honest fallback; EVIDENCE vs inference labeled; **convergence ≠ physical** (§1/§2).
