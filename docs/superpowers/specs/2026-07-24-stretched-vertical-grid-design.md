# Stretched (Surface-Clustered) Vertical Grid for the Column BVP (design)

**Date:** 2026-07-24
**Status:** DESIGN — awaiting user review before writing-plans.
**Scope discipline:** ONE change. Introduce a surface-clustered q-grid + per-interval `dq_i` in the ODE residual & Jacobian rows. Nothing else (no relax run, no new physics, no seed changes).

---

## Problem (evidence-grounded)

With the advective seed-T_c fix, base-rung feasibility is **18/18 at n_z=256** but only **15/18 at n_z=96** — the 3 holdouts are nodes **3 (r=3.9), 4 (r=4.7), 9 (r=11.7)**, the highest-Σ inner columns (Σ≈1.46–1.59e4, T_c≈1.4–2e7, radiation-pressure-dominated). They *exist* (solve at 256) but the **uniform n_z=96 grid can't resolve their steep photosphere**. n_z=256 uniform is ~7× slower per pass (69s → ~500s), making a full relax impractical. A grid that clusters points at the photosphere gets n_z=256-quality feasibility at ~n_z=96 cost — the enabler for actually running the relax.

## Why it's contained (from the code map)

The column BVP solves on `q ∈ [0,1]` (midplane q=0 → surface q=1), `q[i]=i/(N−1)` uniform. Physical height `z` is a solved unknown (`U[4i+3]`); `dz/dq = Σ0/(2ρ)` is already per-node. **All output integrals (Σ, τ, η₃, η₄) already use actual `dz=z[i+1]−z[i]`** (adaptive) and **all BCs are grid-agnostic** — no changes there. The *only* uniform assumption is the trapezoidal ODE rows' constant `dq=1/(N−1)` (residual) and `0.5·dq` (Jacobian).

## The change

1. **Grid generator** — new `column_q_grid(N, stretch)` returning a monotone `q[0]=0 … q[N−1]=1` array clustered toward the **surface (q=1)**. Use a smooth one-sided map, e.g. tanh:
   `q[i] = 1 − tanh(stretch·(1 − i/(N−1))) / tanh(stretch)` (dense near q=1; `stretch→0` ⇒ uniform).
   **`stretch` is a per-solve parameter — a field on `ColumnInputs` and `ColumnCoupledInputs`, DEFAULT ON** (a single named constant, e.g. `kDefaultStretch=2.5`, tuned so full256 n_z=96 → 18/18; raise if a node resists). `stretch=0` recovers the uniform grid (used by the recovery gate + available as a fallback). `ColumnCoupledInputs.stretch` propagates into the base `ColumnInputs.stretch` via `base_inputs_from`. Default-on means the render path (`volumetric_disk`) also benefits from photosphere resolution — see Render-path note.
2. **Per-interval `dq_i` in the ODE rows** — replace the constant `dq`/`half_dq` with `dq_i = q[i+1]−q[i]` in the trapezoidal loops of the **four** functions:
   - `column_residual` (`disk_column_bvp.cpp:~135`), `analytic_jacobian` (`~333,350-389`),
   - `coupled_column_residual` (`disk_column_coupled.cpp:~71`), `coupled_column_jacobian` (`~196,208-248`).
   Thread the `q` array (or the precomputed `dq_i` vector) into these from the solver entry points (`solve_column_bvp`, `solve_column_coupled`), computed once per solve.
3. **No change** to: `node_deriv`/`dz_dq`, the Σ/τ/η₃/η₄ quadratures, or any boundary condition (all already adaptive/grid-agnostic — confirmed by the code map).

Cluster toward the **surface** because that's where ρ/T fall off steeply (the photosphere, τ→2/3); the midplane (pinned T_c, Σ0) is smooth and tolerates sparse points.

## Render-path note (NO redesign; output shifts, improves)

`volumetric_disk.cpp` calls `solve_column_bvp` (lines 1007/1016/1085) and **is in the render path** (`api.cpp` → `geodesic_tracer` → `disk_step_entry` → `volumetric_disk`). The **interface is unchanged** (`ColumnInputs`→`ColumnBVPSolution`), so **no renderer/architecture change is required** — the volumetric disk, tracer, C API, and Blender plugin are untouched. With default-on, the render path's columns are solved on the stretched grid too → more accurate (better photosphere), so the rendered image **shifts slightly (improves), not bit-identical**. This is intended. Consequence: any golden-image regression is **re-baselined** to the better-resolved image (never loosened blindly), and gate 4 confirms the render still yields a good image. If default-on were ever found to *degrade* the render (mistuned stretch), the parameter is already per-solve, so falling back to opt-in (`stretch=0` in `volumetric_disk` only) is a one-line change — but the default remains ON unless that happens.

## Testing / validation gates (TDD)

1. **Uniform-recovery (write FIRST):** with `stretch=0`, `column_q_grid` equals the uniform grid and `dq_i ≡ 1/(N−1)`, so a solve is **bit-identical** to today. Assert a converged column (C1-style) matches the pre-change result to machine precision. This proves the per-interval threading is correct before any clustering.
2. **FD-oracle on a STRETCHED grid (the correctness gate):** the analytic-vs-FD Jacobian mismatch stays `<4e-4` on a stretched grid, for BOTH `analytic_jacobian` and `coupled_column_jacobian` — proving the residual's per-interval `dq_i` and the Jacobian's per-interval `0.5·dq_i` stay consistent. (Extend the existing FD-oracle tests to run on the stretched grid.)
3. **Existing column suite green on default (stretched):** `test-column-coupled` + `test-column-bvp` pass with the default stretch on. Physical BC-pinned quantities (F=σT_eff⁴) are grid-independent; z0/η are quadrature-level and should move only within tolerance (a stretched grid is *more* accurate at the surface). If a tight-tolerance assertion shifts, verify against a high-n_z reference that the stretched answer is *closer* to continuum before touching the tolerance — do NOT loosen blindly.
4. **Render-path regression (default-on):** the volumetric-disk render path must still produce a *good* image on the default stretched grid. Run a CLI render (e.g. `grrt-cli --metric kerr --spin 0.9 ...`) or the volumetric-disk unit/smoke test if one exists; confirm it completes and the volumetric-disk column solves converge. The image will differ slightly from today (more accurate) — that is expected; **re-baseline** any golden image to the better-resolved result (do NOT loosen a tolerance to hide a real change; if the image looks WORSE, stop and report — that's the signal to make `volumetric_disk` opt-out with `stretch=0`). Gate 1 (uniform-recovery, a unit test) remains as the threading-correctness proof independent of this.
5. **Isolated integration (controller runs):** `slim-full256-probe 18 96` (stretched) → **18/18** (nodes 3/4/9 rescued), matching n_z=256; and confirm it's much faster than n_z=256 uniform. If a node resists at stretch=2.5, raise the clustering (single constant) — gated by this count. NOTE: the full256 probe must set `stretch>0` on its `ColumnCoupledInputs` for this (it's a relax-side caller, opting in).

## Non-goals
- No relax run yet (that's the NEXT one-thing, now affordable).
- No change to quadratures, BCs, `node_deriv`, seeds, or the multi-start/advective fallbacks.
- No per-node adaptive grid (fixed stretch profile only — YAGNI; revisit only if a fixed profile can't hit 18/18).

## Risks
- **Jacobian/residual `dq_i` mismatch** → the whole coupled Newton breaks. Mitigation: gate 2 (FD-oracle on a stretched grid) is the hard, objective catch; gate 1 (uniform bit-identical) isolates threading bugs from clustering effects.
- **Existing tests assert uniform-grid values** → gate 3 handles it with the "closer-to-continuum, don't loosen blindly" rule.

## Workflow
Never `git commit` — hand the message over. TDD, gate 1 (uniform-recovery) first, gate 2 (FD-oracle) is the correctness anchor. Present every reviewer rec & WAIT. One change, one commit.
