# Slim-disk handoff — resume here (2026-06-20)

**Read this first after compaction.** Supersedes the 2026-06-14 handoff (which has the full closure-inadequacy *verdict*; this one carries it forward through the approved **design + implementation plan**). The disk physics is correct & committed; the f_Edd≈0.9 wall is a proven **one-zone closure inadequacy**; the **design and the task-by-task implementation plan are written**; the next action is to **execute that plan**.

---

## 0. THE NEXT TASK (the resume action)

**Execute the implementation plan for the vertical-BVP closure coupling (POC), subagent-driven.**
- **Plan:** `docs/superpowers/plans/2026-06-14-slim-disk-vertical-bvp-coupling-implementation-plan.md` (12 TDD tasks, C1–C5 + gates, grounded in the real interfaces, FD-cross-check gate per numerical component).
- **Spec (design, committed):** `docs/superpowers/specs/2026-06-14-slim-disk-vertical-bvp-coupling-design.md`.
- **Execution mode:** subagent-driven-development (fresh subagent per task + two-stage review). The user had not yet picked execute-mode at compaction — confirm, then proceed.
- **Goal (POC):** land a PHYSICAL f_Edd≈0.9, a=0.9 disk (H/r≲0.5, gas-dominated outward, all validity gates, `rad`/`ang` residuals at the merit floor — vs the one-zone O(200–300)). Defer performance (banding, coarse-fine).
- **START with C1 (column re-pose) via the BC ROW-SWAP, NOT the secant wrapper** — the robustness decision (§3).

---

## 1. WHY (the verdict, condensed — full detail in the 2026-06-14 handoff §2)

The disk **physics is now correct and committed** (3 §23 transcription bugs + refinement #11 state-dependent η₃/Γ̃₁ + #12 azimuthal Γ + arclength hardening + the `1+f_adv>ε` gate; all FD-Jacobian/Wolfram/NT-validated). But diagnostic **(A) proved f_Edd≈0.9 is unreachable in the ONE-ZONE closure BY CONSTRUCTION** — a **closure inadequacy**, not a bug or a seed problem. Mechanism (Wolfram-confirmed): in the radiation-dominated inner disk the one-zone closure makes `Q_rad = 8cHΩ²/κ` — independent of (T,Σ), **∝ H/r** — so radiating `F_NT` *forces* H/r≈4 (the torus). Sądowski avoids this with a **full vertical-structure integration** (flux from `∫dℱ/dz`, decoupled from H/r). The fix is to couple the in-tree grey vertical BVP (`src/disk_column_bvp.cpp`) per radial node. **Convergence ≠ physical** (the f_Edd=0.9 "thick-seed success" was a torus artifact that passed every gate).

---

## 2. THE DESIGN (approved & committed)

**Convergence engine — Nested Newton (nonlinear column elimination) via per-column Schur**, with the **analytic IFT sensitivity** as the reduced (radial-sized) Jacobian. Chosen for robustness ("settle it once"); it's the final, production-grade form (radial-sized ~194 solves, columns always physically converged, quadratic; no global matrix). Five components:
- **C1** — column closure-map re-pose: provide the column as `(Σ, T_c, f_adv) → (F, z₀, η₃, η₄)` (currently `T_eff`-driven). Energy row keeps its form `Q_vis − F − Q_adv = 0` (column F replaces `64σT_c⁴/3κΣ`); `H = z₀` from hydrostatic; `f_adv` feeds S11 Eq 13's `/(1+f_adv)` generation.
- **C2** — vertical moments: `η₃ = ∫E dz/∫P dz` (E=(3/2)p_gas+3p_rad), `η₄ = (1/Σ)∫ρz²dz`.
- **C3** — analytic sensitivity `dC/d{Σ,T_c}` via the implicit-function theorem through the column's `∂R_c/∂U`.
- **C4** — nested coupled Newton driver (the Schur reduction).
- **C5** — restore the dropped `𝒩₁` η-gradient terms `(P/Σ)dlnη₃/dlnr + Ω_⊥²(η₄/η₃)dlnη₄/dlnr` (in the *coupled* assembly, since the one-zone path's η are constant ⇒ gradients vanish).

POC scope; mass-independent (geometrized units — covers sub-stellar→ultramassive unchanged; mass is an opacity/thermodynamics axis handled by the mass-adaptive opacity table).

---

## 3. THE ROBUSTNESS DECISION (2026-06-20) — row-swap, not secant

C1's causality inversion has two implementations; **the BC ROW-SWAP is the primary path** (the plan was updated):
- The column is genuinely `T_eff`-driven (hard BC rows `Q(N-1)=σT_eff⁴`, `T(N-1)=T_eff`; `Σ0` a free output). The **row-swap** re-poses it: drop `T(N-1)=T_eff` (free `T_eff`), add midplane `T(0)=T_c`, pin `Σ0=Σ_target`. Result: a single, clean, **differentiable** Newton solve with `(Σ,T_c)` as inputs.
- **Why not the secant wrapper** (root-find on `T_eff` to hit Σ around the unmodified solver): it is NOT cleanly differentiable → forces FD-through-the-secant (the noisy ceiling) → undermines the analytic-sensitivity robustness we chose; and it nests a 3rd iteration level (cost). The row-swap is also required for C3/C6's `∂R_c/∂p` regardless.
- **Secant = bring-up fallback only**, if the row-swapped column is hard to converge near β→0 (its solution can seed the row-swapped Newton).

---

## 4. VERIFIED FORMULAS (user asked to check externally — all confirmed)
- IFT sensitivity `dU_c/dp = −(∂R_c/∂U_c)⁻¹(∂R_c/∂p)` — **Wolfram ✅**.
- Schur reduced Jacobian `J_red = ∂R_r/∂U_r + (∂R_r/∂C)(dC/dU_r)` = Schur complement — **Wolfram ✅**.
- `η₃ = E/P = 3 − 1.5β` — **Wolfram ✅**. `η₄ = (1/Σ)∫ρz²dz` (S11 density 2nd moment; `Ω_⊥²η₄`=2× vertical grav-PE) — **source + physical check ✅** (recorded in `references/disk-physics-formulas.md` §23).
- S11 Eq 13 `/(1+f_adv)` generation, the restored `𝒩₁` terms — **source-verified** (the (A) cross-check). NT/thin reduction (`F→64σT_c⁴/3κΣ·f_F`) — **gate-validated**.

---

## 5. INTERFACE DELTAS (from grounding the plan in the real code — watch these)
- **DELTA-A (minor):** no reusable LU object — `dense_solve` factors-and-solves in place. Plan extracts `column_lu_factor`/`column_lu_solve` (Task 2). Trivial.
- **DELTA-B (the stiff part):** the column is a forward map; the coupling needs the inverse → **the row-swap (§3).** This is where column cost + the sensitivity export concentrate; highest convergence-risk task. Watch it.
- **DELTA-C (resolved):** η₄ formula (§4).

---

## 6. STATE / OPEN ISSUES BEYOND THE COUPLING
- **Stability is a separate downstream check.** The coupling makes the f_Edd≈0.9 root *exist*; it does NOT guarantee it's *stable/reachable*. If, post-coupling and gate-clean, the solver still can't land it → the obstruction is **instability** (the S-curve middle branch / the stability atlas, refinements #10), a different well-defined question — not a regression. (An unstable-but-physical steady disk is still *renderable* via continuation — render it labeled "steady unstable.")
- Cold-seed mid-range basin retune (post-#12; warm-start sidesteps). Near-extremal spin (spin-walk seam bug) — parallel track, downstream. Volumetric thick-disk *rendering* integration (`fix/volumetric-ring`) — downstream. β/f_adv sawtooth at f_Edd=0.05 (#140).

---

## 7. WHAT'S COMMITTED vs UNCOMMITTED
- **Committed:** the 3 transcription bugs, #11, #12, arclength hardening, the `1+f_adv>ε` gate, the design spec.
- **Uncommitted at compaction (hand the user a message):** the implementation plan doc, the η₄ resolution in `§23` + the plan, this handover, and the `slim-sadowski-residual-probe` + its CMake (if not already committed — check `git status`). NEVER `git commit` yourself.

---

## 8. KEY FILES & SOURCES
- `src/slim_disk_radial.{h,cpp}` (radial solver), `src/disk_column_bvp.cpp` + header (**the coupling target**), `src/opacity.cpp`, `src/volumetric_disk.cpp::compute_radial_structure` (NT reference).
- Design `specs/2026-06-14-slim-disk-vertical-bvp-coupling-design.md`; plan `plans/2026-06-14-slim-disk-vertical-bvp-coupling-implementation-plan.md`; scope `plans/2026-06-14-slim-disk-vertical-bvp-coupling.md`; verdict `slim-disk-handoff-2026-06-14.md`; `references/disk-physics-formulas.md` §20–§23; `disk-approach-a-refinements.md` (#1 Richardson, #2 banding — both now coupling prerequisites; #10 stability atlas; #11/#12 DONE).
- Probes (`tools/`): `slim-nt-term-probe`, `slim-qadv-entropy-probe`, `slim-coldseed-sweep`, `slim-warmstart-sweep`, `slim-arclength-probe`, `slim-slimseed-probe`, `slim-sadowski-residual-probe`.
- Sources: S11 [arXiv:1006.4309] (full vertical structure: Eqs 6–9 the η-moments, Eq 13 generation, 29/32–33 `𝒩₁`, 42 one-zone flux, 45 `f_F`), S09 [arXiv:0906.0355], AF13 [arXiv:1104.5499], BPT72, Page-Thorne, NT73, SS73. ar5iv HTML renders for fetching; PDF doesn't convert.

---

## 9. WORKFLOW CONSTRAINTS (non-negotiable)
- **Never `git commit`** — hand the message to the user. **Present every reviewer rec with a take and WAIT.** **Doc-first** for formula edits.
- **Gates green** (NT-reduction + FD-Jacobian + entropy probe); extend the FD-Jacobian cross-check to the new column sensitivities. Don't tune to pass.
- **Verification = opus + Wolfram, NOT sonnet** (user: sonnet "not smart enough" for physics/Jacobian, 2026-06-13). **fable is PULLED** (Anthropic, 2026-06-13). Wolfram (symbolic) + the FD-Jacobian gate are the model-independent oracles. (memory `reference_fable_access_pulled`.)
- Subagents conserve orchestrator context. Safety budget on; honest fallback; EVIDENCE vs inference labeled; **convergence ≠ physical**.
