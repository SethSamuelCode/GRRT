# Slim-disk handoff — resume here (2026-06-30)

**Read this first after compaction.** Supersedes the 2026-06-20 handoff. The vertical-BVP coupling is **built and validated end-to-end** (C1–C5, the analytic Schur Jacobian, the NT gate, the pure-radiative finding, the column-robustness suite). The POC goal — *land a physical f_Edd≈0.9 disk* — is **not yet reached**, blocked at one **precisely-characterized frontier**: the coupled radial relax can't start because the inner columns can't hold the surface density the disk seed demands. The next action is one cheap, decisive check (§4).

---

## 0. THE NEXT TASK (the resume action)

**Run the Σ↔V check: is the demanded inner Σ≈5e4 g/cm² PHYSICAL or a thin-disk-seed artifact?** This single question picks the path forward (§4/§5). It's cheap (analytic + a tiny probe), no full relax.
- Mass conservation: `Σ = Ṁ / (2πr·|V|·Δ^½/√(1−V²))`. The thin-disk seed may set too-LOW an inner inflow `V` → too-HIGH Σ. The coupled disk's *self-consistent* inner V (transonic, near ISCO) could be higher → Σ lower → **within the column's ~1.3e4 g/cm² capacity**.
- **The check:** at the worst inner node (a=0.9, f_Edd=0.01, r≈2.73 just outside ISCO=2.32), take the column's converged max Σ0 ≈ **1.3e4** and Ṁ(f_Edd), and compute the `V` that mass conservation requires for Σ=1.3e4. Is that V physical (V<0, sub/transonic, |V|<1)?
  - **V physical** → Σ=5e4 is a SEED ARTIFACT. Fix = a transonic-aware seed (slim V profile → lower inner Σ within capacity). Tractable. → Option A.
  - **V unphysical** (needs |V|≳1 or V>0) → Σ=5e4 is FORCED; the pure-radiative column is genuinely inadequate at near-Eddington inner densities → **convection (#13) becomes the real answer** (mixing-length convection lets a column hold more mass). → Option C.

---

## 1. WHERE WE ARE — the coupling is BUILT & VALIDATED (all committed)

The entire vertical-BVP closure coupling is implemented, gated, and committed:
- **C1** — augmented `f_adv`-free column (`solve_column_coupled`, 4N+4, BC row-swap; `(Σ,T_c)→(F,z0,η3,η4,f_adv)`). `f_adv` is a determined OUTPUT (S11 §3.1-3.2; numerically + Wolfram verified).
- **C2** — vertical moments `η3=∫E/∫P` (→3−1.5β gated), `η4=(1/Σ)∫ρz²dz` (one-zone reductions gated).
- **C3** — analytic IFT sensitivity `dC/d{Σ,T_c}` (`column_sensitivity`), perturb-resolve oracle <1e-3.
- **C4** — coupled residual + driver (`slim_coupled_residual`, `solve_slim_disk_coupled`/`relax_coupled`); reroutes energy `Qrad→2F` (one-face F, both-face Qrad), closure `H→z0`, η→column η.
- **C5** — restored 𝒩₁ η-gradient terms.
- **Task 10** — the analytic **reduced/Schur Jacobian** (`slim_coupled_reduced_jacobian`): `J_red = ∂R_r/∂U_r|_C + Σ_i(∂R_r/∂C_i)(dC_i/dU_r)`; FD-validated <1e-3, clean throughout the walk (FD mismatch 0.000e+00).
- **Column robustness:** rad-pressure basin fix (T_eff continuation in `solve_column_bvp`), warm-start re-seed (tiny-Tc stall), Ruiz equilibration + affine-invariant Newton, and the **honest Σ0-match seed gate** (commit 461c394 — `build_coupled_seed` no longer reports false success with a garbage seed).
- **Task 11** — NT-reduction gate (`slim-coupled-nt-probe`) + the **pure-radiative finding** (§6).
- Build is **/fp:precise** (deterministic; restore /fp:fast after the coupling lands).

**Gates green:** `test-column-coupled` (0), `test-slim-coupled-jacobian` (0), `slim-coupled-nt-probe` (PASS), `slim-nt-term-probe` (Qvis/F_NT=1.10), `test-slim-jacobian` (0).

---

## 2. THE PRECISE FRONTIER (why f_Edd=0.9 isn't landed)

**The coupled radial relax cannot START.** `solve_slim_disk_coupled` from the thin-disk seed returns BASE-INFEASIBLE (0 inner-iters) at every f_Edd tried (1e-3…0.1). Root cause, proven by the de-risk probes:

**The inner columns can't hold the disk's demanded surface density.** At the worst inner node (r≈2.73, f_Edd=0.01): the column's Σ0 **capacity ceiling ≈ 1.3e4 g/cm²** (converged), but the thin-disk seed demands **Σ≈5e4** — 4× above capacity. **11 of 18 nodes** are above the ceiling → no column root exists there → the relax can't even evaluate its residual.

This is the column-level manifestation of the long-standing "the column can't hold the mass" theme. It is NOT: a Jacobian bug (Schur FD-clean), a solver-robustness gap (the augmented Newton converges fine where a root *exists*), a seeding-language bug (the plumbing audit fixed the calibration↔relax inconsistencies), or an n_z artifact (§3).

---

## 3. THE n_z FINDING (a real correctness fix to apply)

The relax runs columns at **n_z=24** (`ColumnOpts` default, "speed over accuracy for bring-up") — but `Σ0=2∫ρdz` is **3× under-resolved** there:

| n_z | column Σ0 ceiling (worst inner node) |
|---|---|
| 24 (relax) | 4.6e3 |
| 96 (unit tests) | 1.23e4 (~5% under) |
| 200 (Richardson-converged) | 1.32e4 |
| 400 | 1.33e4 |

**RECOMMENDED FIX:** bump `ColumnOpts::n_z` (in `src/slim_disk_coupled.cpp`) from 24 to ~**128–200**. Perf cost ~5–8× slower per column. **It does NOT unblock the POC** (the converged ceiling 1.3e4 is still 4× below the demanded 5e4) — but the relax was running the column at a resolution 3× too coarse vs what validated it, so it's a real correctness fix. Apply it deliberately (consider n_z=96 or 128 as a speed compromise once the §4 question is answered). Proven by `slim-coupled-nz-probe`.

---

## 4. THE LOAD-BEARING QUESTION → §0 (the resume action)

Is Σ=5e4 physical or a seed artifact? See §0. This is the one check that decides Option A vs C.

---

## 5. STRATEGIC OPTIONS (decide after §4)

- **A. Transonic-aware seed** (if §4 says seed artifact). Re-seed `relax_coupled` with the slim/transonic V profile (faster inner inflow → lower inner Σ within the column's ~1.3e4 capacity), instead of the thin-disk seed's V. Then the columns are feasible and the relax can start. Most tractable; the natural fix if Σ=5e4 is just a wrong starting guess.
- **B. Coupling-strength homotopy** from the converged one-zone f_Edd=0.9 disk (`solve_slim_disk_radial` reaches it). Ramp a coupling parameter λ: 0 (one-zone `Qrad`) → 1 (column `2F`), warm-starting. Walks in coupling space from a real converged disk. Caveat: still needs the columns feasible at the disk's operating points — so it depends on §4 too.
- **C. Convection (#13 — the Sądowski radiative+convective column)** (if §4 says physical limit). Mixing-length convection lets a column hold MORE surface density at given heating. If the pure-radiative column genuinely can't hold near-Eddington inner Σ, convection is the answer — and #13 flips from "optional structure refinement" to "the actual blocker." See `disk-approach-a-refinements.md` #13 + §6.

**My (orchestrator) lean:** run §4 first; if seed-artifact → A; if physical → C. Don't build B/C blind.

---

## 6. PURE-RADIATIVE vs CONVECTION (the f_F finding — context for #13)

opus+Wolfram verified (2026-06-29; `references/disk-physics-formulas.md` §23): GRRT's column is **pure grey radiative diffusion** (C_diff=3/4 textbook-exact). The NT-reduction factor `f_F≈0.42` (vs the literature's convective `0.94`) is CORRECT for pure radiative diffusion — the `64σT_c⁴/3κΣ` one-zone is the ~2×-larger radiative+**convective** closure (it needs flux-depth `g=1/4`, impossible for deep viscous heating; column measures g=0.595). **Emitted flux/T_eff/spectrum are correct by energy balance; only internal T_c (~20% hotter) and H/r differ from a convective disk.** Documented as refinement **#13** (mixing-length convection = the Sądowski model, a scoped upgrade). **The Σ-capacity frontier (§2) may be the same convection gap surfacing as "can't hold the mass" — that's the §4 question.**

---

## 7. KEY FILES & PROBES

- **Source:** `src/disk_column_coupled.cpp` (the augmented column + sensitivity + honest seed gate), `src/slim_disk_coupled.cpp` (coupled residual + Schur Jacobian + `relax_coupled` + `ColumnOpts::n_z=24` ← the §3 fix), `src/disk_column_bvp.cpp` (base column + T_eff continuation), `src/slim_disk_radial.cpp` (one-zone radial solver + seeds: `build_thin_disk_seed`, `build_slim_disk_seed`).
- **Gates:** `tests/test_column_coupled.cpp` (n_z=96), `tests/test_slim_coupled_jacobian.cpp` (Schur gate), `tools/slim_coupled_nt_probe.cpp` (NT/pure-radiative).
- **Task-12 de-risk probes (committed):** `tools/slim_coupled_walk_probe.cpp` (the f_Edd-walk harness — re-run after a fix; has `calibrate_seed_to_manifold` + the `[calib] feasible Y/18` line), `slim_coupled_hard_probe.cpp` (high-Σ Σ0-capacity diagnostic), `slim_coupled_nz_probe.cpp` (the resolution proof), `slim_plumbing_audit_probe.cpp` (calibration↔relax audit).
- **Docs:** `references/disk-physics-formulas.md` §23 (pure-radiative finding), `disk-approach-a-refinements.md` #13 (convection upgrade), `specs/2026-06-14-...-design.md`, `plans/2026-06-14-...-implementation-plan.md`.

---

## 8. WORKFLOW CONSTRAINTS (non-negotiable)
- **Never `git commit`** — hand the message to the user. **Present every reviewer rec with a take and WAIT.** **Doc-first** for formula edits.
- **Verification = opus + Wolfram, NOT sonnet** (`reference_fable_access_pulled`). The user's consistency-check instinct ("are they speaking the same language", "test vs tested path") has caught real bugs **three times** — honor it: verify load-bearing claims, don't accept convenient conclusions.
- Gates green; don't tune to pass; **convergence ≠ physical**; honest fallback (no fabricated profiles).
- Subagents conserve orchestrator context.

---

## 9. WHAT'S COMMITTED vs UNCOMMITTED
- **Committed:** all of C1–C5, the Schur Jacobian, the column-robustness suite, the NT gate + pure-radiative docs, the honest seed fix (461c394), the de-risk probes (this session's diagnostics commit).
- **Uncommitted at handoff:** this handoff doc + the memory update (hand the user a message). The §3 n_z bump is NOT yet applied (apply it deliberately next session). `git status` should otherwise be clean.
