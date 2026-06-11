# Slim-disk pseudo-arclength continuation — implementation plan (2026-06-11)

> Implement task-by-task; commit per task (hand the message to the user, never `git commit`). Built on the exact analytic Jacobian (already in `relax_structure`).

**Goal:** Trace the slim-disk solution branch *through* the f_Edd≈0.11 turning point to reach the high-Ṁ slim branch (target f_Edd→0.9 at a=0.9, then a=0.998), using **Keller pseudo-arclength continuation**.

**Why:** The analytic-Jacobian build proved the f_Edd=0.11 wall is NOT precision — it's a genuine **turning point / fold** in the (Σ,V)/mass branch as β→0 (the slim-disk S-curve). Simple Ṁ-marching cannot cross a fold; arclength continuation parametrizes the branch by arclength and follows the tangent *around* the turning point (Ṁ may decrease through an unstable segment, then increase onto the slim branch). This is the literature-standard method for slim disks, and it directly uses the analytic Jacobian.

**Architecture:** Promote Ṁ to a continuation UNKNOWN. Augment the `4N+2` residual `R(U,Ṁ)=0` with the Keller arclength constraint; solve the bordered `(4N+3)` system by predictor (tangent step) + corrector (Newton). Keep the residual physics and the analytic Jacobian unchanged — this is a continuation-driver wrapper.

---

## Tasks

### Task 1 — `∂R/∂Ṁ` column
- Ṁ enters the residual via the mass row (`mdot_node − Ṁ` ⇒ `∂/∂Ṁ = −1` on those rows, in the scaled space) AND the `Q_vis/Q_adv/Q_rad` Ṁ-prefactors (energy + radial-momentum rows). Compute `R_Mdot = ∂R/∂Ṁ` — one column, so a central FD on Ṁ is cheap and adequate (or analytic; FD is fine here).
- Unit-test: `R_Mdot` finite, mass rows ≈ the expected `−1/scale`, energy rows nonzero. Cross-check vs FD of `slim_radial_residual` over Ṁ.
- Commit: `feat(slim-disk): dR/dMdot column for arclength continuation`.

### Task 2 — Bordered augmented solve + tangent
- `augmented_jacobian` = `[[ J (=slim_analytic_jacobian), R_Mdot ], [ tangent_Uᵀ, tangent_Mdot ]]`, a `(4N+3)²` dense system (bordered). Solve via the existing `dense_solve` (or a bordered solver).
- **Tangent** `(U̇, Ṁ̇)`: solve `J·U̇ + R_Mdot·Ṁ̇ = 0` with normalization `‖U̇‖² + Ṁ̇² = 1`; orient it to continue the previous direction (`prev_tangent · new_tangent > 0`). The tangent's `Ṁ̇` changes sign at the fold — that's the mechanism.
- Unit-test on a converged sub-fold point (a=0.9, f_Edd=0.10): the tangent is well-defined and `J·U̇ + R_Mdot·Ṁ̇ ≈ 0`.
- Commit: `feat(slim-disk): bordered augmented system + arclength tangent`.

### Task 3 — Predictor-corrector with arclength step control
- **Predictor:** `(U,Ṁ)_pred = (U₀,Ṁ₀) + Δs·(U̇₀,Ṁ̇₀)`.
- **Corrector:** Newton on `{R(U,Ṁ)=0, N≡(U−U₀)·U̇₀ + (Ṁ−Ṁ₀)·Ṁ̇₀ − Δs = 0}` using the augmented Jacobian. Re-use the row/col scaling, gain-ratio LM, line search, and the physical-validity gate from `relax_structure` (scaled augmented system).
- **Step control:** grow `Δs` (×~1.3) on easy corrector convergence, shrink (×0.5) on failure; floor/ceiling on `Δs`. Honest fallback if `Δs` underflows.
- NOTE: within the slim hybrid, `ℓ_in` is the inner eigenvalue from the outer bracket. For arclength, fold the bracket into the corrector OR (simpler first) carry `ℓ_in` as a state component pinned by the regularity row and let the augmented Newton solve it jointly. Pick the simpler that keeps the regularity satisfied; document the choice.
- Commit: `feat(slim-disk): pseudo-arclength predictor-corrector with step control`.

### Task 4 — Drive it + cross the fold (the payoff)
- New entry `solve_slim_disk_arclength(in, opacity)` (or a mode in `solve_slim_disk_radial`): converge a sub-fold anchor (a=0.9, f_Edd≈0.10) the existing way, compute the initial tangent, then arclength-step UP, recording each accepted point `(f_Edd, Σ-scale, r_sonic, H/r, β, f_adv, merit)`. Stop when Ṁ reaches the target (f_Edd≈0.9) on the upper branch (Ṁ̇ may go negative through the fold then positive again — track the branch, take the target-Ṁ point on the high-Σ/slim side).
- **Gate:** the trace **crosses f_Edd=0.11** (beats the FD/analytic ceiling) and reaches toward f_Edd≈0.9; report the `(Ṁ, Σ)` branch showing the fold, and the physics at the highest f_Edd (deep slim disk: H/r~0.1–0.3, β≪1, f_adv significant). Then test a=0.998.
- Re-run via `tools/slim_warmstart_sweep.cpp` analog (or a new `tools/slim_arclength_probe.cpp`), N=48, safety budget on.
- Commit: `feat(slim-disk): arclength continuation crosses the f_Edd=0.11 fold`.

---

## Notes
- The exact analytic Jacobian is essential here (the predictor-corrector and the tangent need an accurate `J`; FD would reintroduce noise at the fold). The FD cross-check test stays as the oracle.
- Do NOT change the §23 residual physics. If the trace reveals the branch genuinely has NO solution at f_Edd=0.9 (a hard Ṁ_max below 0.9), that's a physical finding to report — but slim disks exist at high Ṁ, so the branch should continue (possibly through an unstable segment).
- Safety budget (`SolveBudget`) stays on — no runaways.
