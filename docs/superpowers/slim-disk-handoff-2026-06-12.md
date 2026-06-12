# Slim-disk handoff — resume here (2026-06-12)

**Read this first after compaction.** It captures the state, the governing epistemic theme, the bugs fixed, what remains, and the next task. Then read the linked spec/plan/§22-§23 + the verified-formula doc.

---

## 0. THE NEXT TASK (the resume action)

**✅ AUDIT DONE (2026-06-12).** `fable` audited every §19–§23 equation vs the primary sources. Result: foundation sound (20+ equations CONFIRMED, incl. both prior Q_vis fixes), **one NEW transcription bug found + FIXED** (flag #1, Q_adv bracket — see §3), plus root-cause doc typo (#2), header comment (#3), doc gloss (#4). Both gates green post-fix (FD-Jacobian 0 failures; NT-reduction flat 0.91–1.13, unmoved). **NEXT: (a) commit the flag-#1 fix (message below / from chat); (b) implement refinements #11 (state-dependent η₃/Γ̃₁) then #12 (full azimuthal Γ) in `disk-approach-a-refinements.md` — the prioritized near-Eddington accuracy fixes; (c) re-measure the fold and attempt f_Edd≈0.9.** A NEW gate is still owed: an entropy-form probe asserting Q_adv = −(Ṁ/2πr²)·T·dS/dlnr at MODERATE Ṁ (both existing gates are blind to Q_adv).

**Commit message (flag #1):**
```
fix(slim-disk): correct §23 Q_adv entropy bracket [(Γ₁−1)dlnP−Γ₁dlnΣ] → [η₃dlnP−(1+η₃)dlnΣ], η₃=1/(Γ₁−1) (S11 Eq 29)

fable's equation audit found the one-zone advective-cooling bracket used the
INVERTED energy moment η₃=Γ₁−1 (=2/3) instead of η₃≡E/P=1/(Γ₁−1) (=3/2). Root
cause: a doc typo at §23 ("one-zone η₃→Γ₁−1") that propagated into the code.
Verified two independent ways: entropy identity TdS=d(E/Σ)+Pd(1/Σ) with E=η₃P, and
ideal-gas s=c_v ln(Pρ^−Γ), c_v=1/(Γ−1) — both give [η₃dlnP−(1+η₃)dlnΣ]=[1.5dlnP−2.5dlnΣ].
With the /η₃ normalization the 𝒩₁ advection term then collapses to exactly S11
Eq 32's −(P/Σ)[dlnP−Γ̃₁dlnΣ] (since (1+η₃)/η₃=Γ̃₁); the old bracket gave
−(P/Σ)[0.44dlnP−1.11dlnΣ], internally inconsistent with 𝒟₀=V²−Γ̃₁P/Σ.

Q_adv was ~2.4–5× too small (slope-dependent) and wrong-shaped — it corrupts the
energy rows, 𝒩₁, the ℓ_in/r_s eigenvalue, and f_adv. INVISIBLE to both gates
(NT-reduction runs at Ṁ→0 where Q_adv→0; FD-Jacobian was satisfied because the
analytic Jacobian implemented the same wrong bracket) — the derivation is the proof.
Doc-first (§23 bracket + η₃ definition + TdS gloss). Code: named kAdvP=η₃,
kAdvS=1+η₃; 13 bracket sites (residual 909/1004, extraction 2577, analytic Jacobian
1340/1353-1359/1538/1546-1549) + the nt-term-probe inline copy. Also: header f_adv
comment Q_adv/Q_vis→Q_adv/Q_rad (#3). Gates: test-slim-jacobian 0 failures;
slim-nt-term-probe Q_vis/F_NT flat 0.91–1.13 (unmoved, as expected). Refinements
#11 (state-dependent η₃→3 at β→0) and #12 (azimuthal Γ) documented as the next
near-Eddington fixes.
```

---

### (original audit brief, for reference)
**Have `fable` AUDIT EVERY EQUATION in the slim-disk solver against EXTERNAL PRIMARY SOURCES, and re-verify our first principles.** Opus originally verified the §22/§23 formulas and did a good job — but `fable` has since found **transcription errors opus missed** (see §3), so an independent line-by-line re-audit is warranted to catch any remaining slips before we trust high-Eddington results.

- **Scope:** every term in `slim_radial_residual` (`src/slim_disk_radial.cpp`) and every formula in `docs/superpowers/references/disk-physics-formulas.md` §19–§23 — mass conservation, angular momentum, radial-momentum transonic ODE (`𝒩₁/𝒟₀`, `𝒜`, regularity), energy (`Q_vis`, `Q_rad`, `Q_adv`, `f_adv`), the one-zone vertical closure, the Kerr factors (`A`, `Δ`, `Ω_K`, `𝒞`, `ℋ`, `Ω_⊥²`, `Ω(ℓ)`), the outer BCs, the seed, the opacity (Rosseland mean, H⁻, Saha), `L_Edd`/`Ṁ_Edd`.
- **Method (require for each equation):** (1) quote the source equation it derives from (S09 [arXiv:0906.0355], S11 [arXiv:1006.4309 / A&A 527 A17], Abramowicz & Fragile 2013 Living Rev., Bardeen-Press-Teukolsky 1972, Page-Thorne 1974, Novikov-Thorne 1973, Shakura-Sunyaev 1973); (2) a **dimensional check** (geometric units `G=c=1`, `M` the scale; CGS via `r_g`); (3) a **limit check** (reduce to Novikov-Thorne / Newtonian as `Ṁ→0`, `a→0`, large `r`); (4) confirm the CODE matches the verified formula at every call site. Output: for each equation, CONFIRMED or a flagged transcription/derivation error with the source evidence and the fix (doc-first).
- **Gates that must stay green during any fix:** the **NT-reduction** (`slim-nt-term-probe`: `Q_vis/F_NT≈1` flat at `a=0.9, f_Edd=0.02`) and the **FD Jacobian cross-check** (`test-slim-jacobian`: analytic == finite-difference oracle).
- Use `fable` (the user's explicit request; it has outperformed on these audit tasks). Doc-first for formula edits; never `git commit` (hand the message over).

---

## 1. THE GOVERNING THEME — incorrect model vs unstable vs impossible

When a solve fails to converge OR converges to something odd, it is ONE of three things, and **distinguishing them is the entire game** (we got it wrong repeatedly before learning to test, not assume):

1. **Incorrect model / numerical artifact** — the equations or implementation have a bug, OR the numerics (seed, Jacobian, discretization) can't reach a solution that *exists and is stable*. **Most of this session was category 1**: the two `Q_vis` transcription bugs; the FD-precision ceiling; the Σ-cliff; the wrong-branch (gas-seed) failures; the spin-walk re-projection seam; the opacity Rosseland-window bug.
2. **Physically unstable** — a *real* steady solution that is thermally/viscously **unstable** (the radiation-pressure / Lightman-Eardley middle branch of the disk S-curve). Exists mathematically; nature doesn't sit on it (it limit-cycles — e.g. GRS 1915+105); numerically a saddle, hard to converge (why the arclength corrector *dithers* on it).
3. **Genuinely impossible** — no steady solution exists (a real physical boundary). The **thin-disk fold**: above the fold `Ṁ` there is no steady *thin* solution; nature makes a *slim* disk instead (different structure, not a faster thin disk).

**The categorization IS the disk's thermal-equilibrium S-curve, classified by stability** (slope `dṀ/dΣ` sign, or the linearized growth-rate / Jacobian eigenvalue): rising branch = stable (realized); falling branch = unstable; gaps/folds = impossible. The **arclength continuation traces it**; the **analytic Jacobian gives the eigenvalues**. Automating this is the deferred "stability atlas / input pre-flight validator" (refinements doc item #10) — so the pipeline can tell you up front whether a requested `(M, a, f_Edd, r)` is stable-thin / stable-slim / unstable / impossible, instead of failing cryptically.

**Epistemic rule (hard-won):** *convergence ≠ physical.* A converged solution is a real root of the (possibly-wrong) model; judging it needs INDEPENDENT checks — the **slim→Novikov-Thorne reduction theorem** (our in-house, exact, no-external-trust gate: as `Ṁ→0` the slim solver MUST match our validated NT solver `VolumetricDisk::compute_radial_structure`), literature benchmarks (ballpark only — different model assumptions), and resolution/seed-independence. The NT theorem is what finally exposed the real `Q_vis` bugs after we'd misdiagnosed the wall as FD-precision, then as a BC, then as my own prior about disk shape.

---

## 2. ARCHITECTURE (what's built, all committed unless noted)

Two-level hybrid transonic solver in `src/slim_disk_radial.{h,cpp}` (test `test-slim-disk-radial`):
- **Outer:** 1-D bracket on the eigenvalue `ℓ_in` (`solve_outer_bracket`). **Inner:** fixed-`ℓ_in` Newton relaxation (`relax_structure`) over `[r_s, r_out]` with `r_s = U[4N+1]` the free **sonic node** (regularity `𝒟₀=𝒩₁=0` at node 0); state `U` length `4N+2` = `{Σ,V,ℓ,T_c}×N + {ℓ_in, r_s}`.
- **Exact analytic Jacobian** (`slim_analytic_jacobian`), validated permanently by the FD oracle `slim_numerical_jacobian` (`test-slim-jacobian`; `SLIM_FD_JAC=1` switches back to FD).
- **Gain-ratio Levenberg-Marquardt** (Nielsen) damping; row/col scaling; feasibility line search; **validity-gated** convergence at the FD-noise floor (`kMeritFloor=1e-3` + physical gate: mass conserved, V<0, Σ>0, `𝒟₀/𝒩₁≈0`, `r_s<r_isco`, smooth).
- **`SolveBudget` safety budget** (default 200k inner iters / 900 s wall) — no more runaways.
- **Σ-outlier de-glitch** (`deglitch_sigma_outliers`) keeps nodes on the warm branch.
- **Pseudo-arclength continuation** (`solve_slim_disk_arclength`) — traces the branch through the fold; uses the analytic Jacobian (bordered augmented `(4N+3)` system + Keller arclength + tangent). Locates the fold; **dithers on the unstable middle branch** (doesn't yet ride it onto the slim branch).
- **Ṁ-continuation + spin-homotopy** drivers in `solve_slim_disk_radial` (the spin-walk has a known warm-reprojection seam bug at small spin steps — DEFERRED; the cold seed reaches a=0.9 directly, so it's only needed near-extremal).
- Opacity (`src/opacity.cpp`): T-adaptive Rosseland window + H⁻ upper-T taper (both fixed); Saha is fine (the high-density "16% neutral" is a real LTE-validity edge, not a bug).

---

## 3. THE Q_vis BUGS — the headline of this session

The viscous-heating term `Q_vis` had **two transcription errors** (both in §23 of the formula doc AND the code), masking the real physics for the whole session (they made the disk artificially over-hot, radiation-pressure-dominated, and over-thick — worst at large r):

- **Bug 1 — length divisor (COMMITTED).** The assembly needed a `[1/cm]` and divided by the **constant `r_g`** instead of the **local `r_cm = r·r_g`** ⇒ `Q_vis` inflated by a factor of **r** (×3 at r=3, ×53 at r=50). Fix: `geomfac / in.r_g` → `geomfac / r_cm`. Doc `/r³`→`/r⁴`.
- **Bug 2 — metric factor (UNCOMMITTED — in the working tree, fable's latest; COMMIT IT).** `A^½Δ^½Γ/r⁴` should be `A^½Γ/(Δ^½r²)` (wrong by `Δ/r²`; Newtonian-equal so the r=50 NT gate barely saw it, but it **suppressed inner-disk heating ×0.29 at the a=0.9 ISCO**). Derived from **S09 Eq 6 × Eq 4** (=S11 Eq 23). Fixed doc-first + `Gbalance` + `Gbalance_jac` + both `build_thin_disk_seed` copies + probe inline copies. *(Commit message is in the chat; if compacted, re-derive from this paragraph.)*

**Both validated:** `Q_vis/F_NT` at `a=0.9, f_Edd=0.02` went from a radial *tilt* (0.39 inner → 1.06 outer, ×r) to **flat 0.91–1.13** — right magnitude AND right radial shape. FD cross-check stays green.

**The two flagged O(1) items were CORRECT as-is** (fable verified vs S09 Eq 6): `Γ` belongs in the numerator; the local `dΩ/dr` is right (the Keplerian `√(M/r³)` only appears in S11 Eq 13's *vertical* `𝒟/𝒞` convention, which we don't use).

**Deferred (flagged in §23, small):** S11's full `Γ² = 1/(1−V²) + ℒ²r²/A` (code uses radial-only Γ; missing azimuthal piece ≈ +10–25% at `r≲6`) — fixing needs torque law + `Q_vis` + analytic Jacobian together. And an *inferred* ~+10% residual `Q_vis/F_NT` plateau, attributed to the slim model's neglected radiative angular-momentum flux (the `2Fℒ` term Page-Thorne keeps, S09 drops).

---

## 4. CURRENT RESULTS (post both Q_vis fixes)

- **The solver now produces PHYSICAL disks.** Best converged: **f_Edd=0.045, a=0.9** — thin (H/r ≤ 0.015), **gas-dominated outward** (β: 0.29 inner → 0.71 at r=50), f_adv inner-positive→0, smooth T_c/Σ, **no mesh sawtooth** at N=48. The exact inverse of the old pathology.
- **The old "f_Edd=0.9 H/r=2 torus" is GONE** — confirmed an artifact of the inflated heating.
- **The fold (gas/thin-branch terminus at a=0.9) is now at f_Edd≈0.071** (cross-check: pre-fix 0.25 × Δ/r²≈0.29 = 0.072 — moved exactly as the inner-heating correction predicts).
- **f_Edd≈0.9 (the render target) is UNREACHABLE** — it's on the **slim upper branch beyond the fold**. The arclength crosses the fold (0.071) but dithers on the unstable middle branch; thick "slim-branch" seeds that "worked" before now fail (they were tuned to the buggy heating). This is now the *genuine* category-1/2/3 question: does a stable slim solution exist at f_Edd≈0.9, a=0.9, and can we reach it?

---

## 5. OPEN ISSUES / REGRESSIONS

1. **Production driver regression:** `solve_slim_disk_radial` at N=150 now FAILS its mid-Ṁ test (returns empty) — a driver budget/heuristics issue exposed by the harder, more radiation-dominated corrected physics (the *same* point converges via the probe at N=48 in ~52 s). No pre-change baseline was captured. Needs a driver follow-up (retune iters/budget/seed for the corrected physics; possibly re-base the test point).
2. **Reaching f_Edd≈0.9 (the slim upper branch):** options — (A) harden the arclength to ride the unstable middle branch; (B) build a *correct* slim/advective upper-branch seed for the fixed physics; (C) **first confirm via the stability atlas whether a stable slim solution even exists** at f_Edd≈0.9, a=0.9 (your "what if it's genuinely unstable?" — real near-Eddington disks limit-cycle).
3. **`slim_benchmark_probe`'s internal NT-flux reference is mis-derived** (shape invalid; needs a ×3134 normalization / Page-Thorne) — use `slim-nt-term-probe` as the authoritative NT gate.
4. The deferred §3 items (full `Γ²`, the +10% radiative-torque plateau).

---

## 6. KEY FILES & SOURCES

- `src/slim_disk_radial.{h,cpp}` — the solver. `src/opacity.cpp` — opacity. `src/volumetric_disk.cpp` `compute_radial_structure` — the validated NT solver (the NT-reduction reference).
- `docs/superpowers/references/disk-physics-formulas.md` — §19–§23 verified formulas (**the audit target**; §22/§23 = the slim disk; trap checklist at the end).
- `docs/superpowers/specs/2026-06-08-relativistic-slim-disk-design.md`; plans `2026-06-08-relativistic-slim-disk.md`, `2026-06-11-slim-disk-analytic-jacobian.md`, `2026-06-11-slim-disk-arclength-continuation.md`; refinements `disk-approach-a-refinements.md` (item #10 = stability atlas).
- Probes (deletable diagnostics, in `tools/`): `slim_nt_term_probe.cpp` (**the NT gate**), `slim_coldseed_sweep.cpp` (cold-seed f_Edd sweep at a=0.9), `slim_benchmark_probe.cpp` (NT ref buggy), `slim_slimseed_probe.cpp`/`_validate.cpp`, `slim_arclength_probe.cpp`, `slim_jacentry_scan.cpp`.
- Sources: S09 [arXiv:0906.0355], S11 [arXiv:1006.4309] / A&A 527 A17, Abramowicz & Fragile 2013 (Living Rev. Rel. 16,1), BPT72, Page-Thorne 1974, Novikov-Thorne 1973, SS73.

---

## 7. WORKFLOW CONSTRAINTS (carry forward, non-negotiable)

- **Never `git commit`** — hand the message to the user (memory `feedback_review_workflow`).
- **Present every reviewer recommendation with a take and WAIT** for the user's call.
- **Doc-first** for any formula edit (update `disk-physics-formulas.md` before the code).
- **Gates:** the NT-reduction theorem (in-house exact) + the FD Jacobian cross-check. Don't tune to pass them.
- Subagents **sonnet/opus/fable** (the build now exposes `fable`); **`fable` has found 2 transcription errors opus missed — use it for the equation audit.** Never haiku for physics.
- Safety budget on; small N for probes; honest fallback (no fabricated profiles); EVIDENCE vs inference labeled.
- **Uncommitted right now:** Bug-2 (metric factor) fix + the FD-oracle step tightening + the §23 doc update + new probes (`slim_jacentry_scan`, `slim_nt_term_probe`, etc.) + CMake. Commit before/after compaction.
