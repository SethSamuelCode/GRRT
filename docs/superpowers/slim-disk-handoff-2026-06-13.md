# Slim-disk handoff — resume here (2026-06-13)

**Read this first after compaction.** Supersedes `slim-disk-handoff-2026-06-12.md` (which predates flag #1, #11, the fold re-measurement, the arclength hardening, and the torus verdict). Captures state, the governing epistemic theme, what's committed, what's open, and the next task.

---

## 0. THE NEXT TASK (the resume action)

**Build the PROPER Sądowski-style global slim seed so the production driver reaches a PHYSICAL f_Edd=0.9 disk across f_Edd and spin.** The render target (f_Edd≈0.9, a=0.9→0.998) is NOT yet solved — see §4. The path (literature-faithful, per Sądowski 2009 §3) has three parts:

1. **Principled thick shape from target Ṁ (NOT hand-tuned σ_mult).** Construct the trial by the Sądowski recipe: **Novikov-Thorne outer BC** (thin, gas-dominated *outer* disk), thickening *inward*, with the sonic point placed by **𝒟-sign tracking** (integrate/evaluate `calD0(eval_node(...))` along the trial; the sign change of 𝒟 locates `r_s`). Bracket `ℓ_in` by the 𝒟-topology (too-high ℓ_in ⇒ 𝒟 vanishes prematurely; too-low ⇒ 𝒟 stays positive; the eigenvalue is the common limit). Make `r_s`/`ℓ_in` f_Edd-aware. The OLD probe seed (`tools/slim_slimseed_probe.cpp` `build_slim_branch_seed`) is a *uniform-thick* seed (β≪1 everywhere via a T_c floor) — that produces a TORUS, do NOT promote it as-is.
2. **Node-local Γ̃₁(β) (#11 consistency).** The probe seed uses the frozen `kGtilde1`; promote to `gtilde1_of_beta(beta_of(oz))` / `eta3_of_beta` everywhere the seed touches the moments.
3. **Tighten the f_adv validity gate to `1 + f_adv > ε` (reject the torus).** The torus signature is **f_adv → −1** — and `Q_rad = Q_vis/(1+f_adv)`, so `f_adv > −1` is a HARD physical requirement (else Q_rad → ∞ or negative). The current gate only checks `|f_adv| < 50`, which sails past the singularity. Tightening `slim_fadv_ok`/`slim_validity_gate` to require `1+f_adv > ε` (e.g. ε≈0.05) rejects this whole torus class and makes the gate trustworthy at high Ṁ. **Do this first — it is the JUDGE for the trial.**

**Goal/gate:** a converged f_Edd=0.9 disk that is PHYSICAL — inner-peaked H/r (~0.25–0.5, declining outward), **gas-dominated outward (β→1)**, f_adv ~ +0.3 (significant inner, vanishing outward), sonic point inside ISCO — i.e. matching the Sądowski/AF13 profiles (§4/§6), passing the *tightened* gate. If the proper trial STILL inflates to a torus, the problem is structural (outer-BC / radial-momentum doesn't pin H outward at high Ṁ) → that becomes the next investigation.

**Validation:** seed changes can't affect `test-slim-jacobian` (residual/Jacobian untouched). The f_adv-gate tightening touches `slim_validity_gate` (not the residual/Jacobian) so `test-slim-jacobian` stays green too. The REAL gate is convergence to a *physical* (literature-matching, tightened-gate-passing) profile. Verify any 0.9 "success" against the literature before trusting it (the prior one was a torus — §4).

---

## 1. THE GOVERNING THEME — incorrect model vs unstable vs impossible

Non-convergence OR odd-but-converged results split THREE ways; telling them apart is the whole game:
1. **Incorrect model / numerical artifact** — bug, or numerics (seed/Jacobian/discretization) can't reach a solution that exists. (The 3 transcription bugs; the seed-basin potholes; the **torus** — a converged-but-unphysical root.)
2. **Physically unstable** — a real but thermally/viscously unstable steady solution (the S-curve radiation-pressure middle branch; nature limit-cycles).
3. **Genuinely impossible** — no steady solution (above a fold).

**Epistemic rule (reinforced HARD on 2026-06-13): convergence ≠ physical.** The f_Edd=0.9 "thick-seed success" passed EVERY in-house validity gate (merit 8.6e-6, 𝒟₀≈0, 𝒩₁≈0, mass, V<0, smooth) and was a **torus artifact** — a true root of the (still-imperfect/under-gated) model. Judge converged states with INDEPENDENT checks: the in-house **NT-reduction theorem** (exact, low-Ṁ), **literature-benchmark comparison** (Sądowski/AF13 profiles — ballpark), the **f_adv→−1 physical bound**, and resolution/seed-independence. Do NOT trust a high-Ṁ "success" on gate-pass alone.

---

## 2. ARCHITECTURE (`src/slim_disk_radial.{h,cpp}`, test `test-slim-disk-radial`)

Two-level hybrid transonic relaxation solver (this IS Sądowski 2009's method):
- **Outer:** 1-D bracket on the eigenvalue `ℓ_in` (`solve_outer_bracket`, scans `[0.4–1.1]·ℓ_K(r_isco)`, uses the scaled 𝒩₁(r_s) residual, bisects — exactly Sądowski's 𝒟-topology bracket).
- **Inner:** fixed-`ℓ_in` Newton relaxation (`relax_structure`) over `[r_s, r_out]`, free sonic node `r_s = U[4N+1]` (regularity 𝒟₀=𝒩₁=0 at node 0). State `U` length **4N+2** = `{Σ,V,ℓ,T_c}×N + {ℓ_in, r_s}` (194-dim at N=48).
- **Exact analytic Jacobian** (`slim_analytic_jacobian`), validated permanently by the FD oracle (`test-slim-jacobian`, 0 failures).
- Gain-ratio LM, row/col scaling, feasibility line search, validity gate (`slim_validity_gate`: mass, V<0, Σ>0, 𝒟₀/𝒩₁≈0, r_s<r_isco, smooth, bounded f_adv), `SolveBudget` safety budget.
- **Warm-start Ṁ-continuation** driver (`solve_slim_disk_radial`, Mdot ladder, carries U across rungs) — = Sądowski's "sequential parameter-stepping, without explicit branch-following."
- **Pseudo-arclength continuation** (`solve_slim_disk_arclength`) — **hardened 2026-06-13** (committed): secant-based tangent orientation (the textbook fix), fold-aware `ds` shrink, f_adv validity gate in the corrector. Rounds the fold cleanly to 0.143 on physical states; does NOT ride the unstable middle to the slim branch (Sądowski doesn't either — use the global seed, not arclength).
- `build_thin_disk_seed` (~589–826): the trial. `r_s`/`ℓ_in` seeds were f_Edd-independent; **P1** (uncommitted, marginal) made them f_Edd-aware but that's not the lever — the SHAPE is.

---

## 3. PHYSICS CORRECTIONS (all COMMITTED) — the headline of this work

`fable`'s equation audit (fable now pulled — §7) + opus + Wolfram found/fixed **three transcription bugs** in the §23 energy terms, then implemented refinement #11:
- **Q_vis bug 1** — length divisor `/r_cm` not `/r_g` (heating r× too large). Committed.
- **Q_vis bug 2** — metric factor `A^½Γ/(Δ^½r²)` not `A^½Δ^½Γ/r⁴` (S09 Eq6×Eq4). Committed.
- **Q_adv bug (flag #1)** — entropy bracket `[η₃·dlnP − (1+η₃)·dlnΣ]`, η₃=1/(Γ₁−1)=3/2, NOT the inverted `[(Γ₁−1)dlnP − Γ₁dlnΣ]` (S11 Eq 29). Root cause: a doc typo. Proven 2 ways analytically + Wolfram. Q_adv was ~1.8× too small, invisible to both gates (NT runs at Q_adv≈0; FD-Jac was consistently wrong). Committed.
- **Refinement #11 (committed)** — state-dependent moments `η₃(β)=3−1.5β`, `Γ̃₁(β)=1+1/η₃` (β=p_gas/p_mid; β=1 gas⇒3/2,5/3; β=0 rad⇒3,4/3), replacing the frozen gas constants. The near-Eddington inner disk is radiation-dominated where the frozen value underestimated Q_adv ~1.8×. Residual + analytic Jacobian (∂η₃/∂{Σ,T} from closure `dp_gas`/`dp_mid`) + β-clamp. Validated: FD-Jacobian 0 failures, Wolfram-confirmed derivatives, NT unmoved, entropy probe Q_true2D/Q_code→1.
- **Arclength hardening (committed)** — see §2.

**Gates that must stay green:** NT-reduction (`slim-nt-term-probe`, Q_vis/F_NT flat 0.91–1.13), FD-Jacobian cross-check (`test-slim-jacobian`, 0 failures), entropy-form Q_adv probe (`slim-qadv-entropy-probe`, Q_true2D/Q_code→1). Don't tune to pass them.

---

## 4. CURRENT STATE (post all corrections)

- **#11 ~DOUBLED the reachable accretion rate.** The fold (gas/thin-branch terminus, a=0.9) moved from f_Edd≈0.071 (pre-#11) to **f_Edd≈0.14** (warm-start wall; mass-dominated breakdown + step-halving can't cross + step floor ⇒ genuine fold). Stronger radiation-regime advective cooling (correct #11 physics) stabilizes higher Ṁ — textbook slim behavior. Disks 0.10–0.14 are clean, progressively slimmer (H/r 0.03→0.04, β_inner→0.02, f_adv→−0.11).
- **The arclength rounds the fold to 0.143 on physical states** (no garbage, no thrash) but can't ride the unstable middle to the slim branch.
- **f_Edd≈0.9 is NOT solved.** The "direct thick-seed success" at f_Edd=0.9 is a **TORUS ARTIFACT** (verified 2026-06-13 vs Sądowski/AF13): H/r grows *outward* 0.45→**4.08** at r=50 (literature: ~0.25–0.5, peaks *inner*, thins outward); radiation dominance *increases* outward (p_rad/p_gas 4e3→5e5; real disks gas-dominated outward); f_adv→−1 over the outer disk (the Q_rad=Q_vis/(1+f_adv) singularity; literature f_adv~+0.3 inner). T_c was physically free (not floored) — it's a genuine wrong-shaped root. Same class as the eliminated H/r=2 torus. The uniform-thick seed biases to the radiation-dominated thick branch; the solver finds the nearest root = a torus.

---

## 5. OPEN ISSUES

1. **Reaching a PHYSICAL f_Edd=0.9 (the render target)** — the §0 task: proper Sądowski trial (NT-outer/inward-thickening/𝒟-sign) + node-local Γ̃₁ + the `1+f_adv>ε` gate. Open question: does a stable physical slim solution exist at 0.9/a=0.9, or does the outer-disk structure inherently allow the H-runaway (structural outer-BC issue)? The proper trial distinguishes these.
2. **f_adv gate gap** — `1+f_adv>ε` not yet enforced (part of §0). Physically mandatory (Q_rad>0).
3. **Refinement #12 (next physics fix after #11)** — full azimuthal Lorentz factor `Γ²=1/(1−V²)+ℓ²r²/A` (currently radial-only; +10–25% inner-disk heating at a=0.9). Touches torque + Q_vis + Jacobian together. Documented in `disk-approach-a-refinements.md` #12.
4. **β/f_adv odd-even sawtooth at f_Edd=0.05 outer disk** (~24/46 nodes; smooth at 0.02). Mesh/convergence artifact #11 did NOT smooth; passed gate_smooth (Σ-based). Tracked.
5. **P1 (f_Edd-aware scalar seeds) uncommitted + marginal** — converts a flaky 0.15 to a clean 0.15; not the lever (shape is). Keep or revert — user's call. The verification probe's profile dump (`tools/slim_slimseed_probe.cpp`) is also uncommitted (deletable diagnostic).
6. **N is not the limiter** — N=96 ruled out for the upper-rung failures (still fail, not budget-tripped). The wall is seed-SHAPE.
7. Deferred: the ~+10% Q_vis/F_NT plateau (inferred radiative-torque term); the stability atlas (refinements #10); spin-homotopy seam (near-extremal only).

---

## 6. KEY FILES & SOURCES

- `src/slim_disk_radial.{h,cpp}` — solver + seed + arclength. `src/opacity.cpp`. `src/volumetric_disk.cpp::compute_radial_structure` — the validated NT solver (NT-reduction reference).
- `docs/superpowers/references/disk-physics-formulas.md` §19–§23 (verified formulas; §22/§23 = slim disk; trap checklist). `docs/superpowers/disk-approach-a-refinements.md` (#10 stability atlas, #11 DONE, #12 azimuthal Γ).
- Probes (deletable, `tools/`): `slim-nt-term-probe` (NT gate), `slim-qadv-entropy-probe` (Q_adv gate), `slim-coldseed-sweep`, `slim-warmstart-sweep`, `slim-arclength-probe`, `slim-slimseed-probe` (the thick-seed prototype + 0.9 torus).
- **Literature method (Sądowski 2009 §3, fetched & confirmed):** relaxation on a 2-pt BVP, free sonic point, ℓ_in eigenvalue via 𝒟-sign topology bracket, trial = NT outer BC + integrate inward (dΩ/dr via diffusive-viscosity approx, Abramowicz+96 Eq 35 — exact form not yet recovered), ~100 log points, "without explicit branch-following." Sources: S09 [arXiv:0906.0355], S11 [arXiv:1006.4309], AF13 [arXiv:1104.5499] (slim H/r~0.25–0.5, h≲1 NOT >1, f_adv~O(1), β<1 inner gas-dominated outer), BPT72, Page-Thorne, NT73, SS73. ar5iv HTML renders for fetching; PDF doesn't convert.
- Tavleev/Lipunova/Malanchev 2023 [arXiv:2303.02184], code `github.com/AndreyTavleev/DiscVerSt` — vertical-structure only (no advection/sonic point); reference for vertical root-finding, not the global slim seed.

---

## 7. WORKFLOW CONSTRAINTS (carry forward, non-negotiable)

- **Never `git commit`** — hand the message to the user.
- **Present every reviewer recommendation with a take and WAIT** for the user's call.
- **Doc-first** for any formula edit (update `disk-physics-formulas.md` before code).
- **Gates green** (NT-reduction + FD-Jacobian + entropy probe). Don't tune to pass.
- **Verification = opus + Wolfram, NOT sonnet** (user: sonnet "not smart enough" for physics/Jacobian; 2026-06-13). **fable is PULLED** (Anthropic, 2026-06-13) — don't dispatch it. Wolfram (symbolic) + the FD-Jacobian gate are the model-independent oracles; opus reads code-vs-formula. (memory `reference_fable_access_pulled`).
- Subagents conserve the orchestrator's context (dispatch read-heavy investigations/implementations).
- Safety budget on; small N for probes; honest fallback (no fabricated profiles); EVIDENCE vs inference labeled; **convergence ≠ physical** (§1).
