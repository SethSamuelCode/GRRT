# Slim-disk analytic Jacobian — implementation plan (2026-06-11)

> **For agentic workers:** implement task-by-task. Each analytic block is gated by an FD cross-check (assert it matches `slim_numerical_jacobian` to round-off) — no block ships until it matches. Follow TDD; commit per task (hand the message to the user, never `git commit`).

**Goal:** Replace the dense finite-difference Jacobian in `relax_structure` (`src/slim_disk_radial.cpp`) with an exact **analytic** Jacobian of `slim_radial_residual`, validated against the existing FD Jacobian, to push the transonic slim-disk solver past the measured FD ceiling.

**Architecture:** The residual `R(U)` (length `4N+2`) is already correct and FD-validated. This is a pure numerical-precision upgrade: derive `∂R/∂U` analytically, validate each block against `slim_numerical_jacobian`, then swap it into the Newton loop. The FD Jacobian is retained permanently as the cross-check test oracle.

**Tech stack:** C++23, `src/slim_disk_radial.cpp`, the existing `one_zone_closure`, Kerr `slim_detail::` factors, `OpacityLUTs::kappa_ross_with_grad` (already provides `∂κ_R/∂ln{ρ,T}`).

---

## Why (evidence-justified target + mechanism)

The slim-disk physics is **proven correct** — at a=0.9 the cold seed converges and crosses the radiation-pressure fold (H/r=0.33, β=3.6×10⁻⁴ at f_Edd=0.11), exactly what the thin disk could not do. The blocker is purely FD precision:

- **FD ceiling = f_Edd ≈ 0.11 at a=0.9** (genuine stall, NOT a budget cut: the failing rungs ran 168–182 s of 300 s and stalled at merit ~0.05 with `maxrel` stuck at ~0.08). A finer f_Edd step (0.005) also fails → conditioning, not perturbation size.
- **Two FD-precision weaknesses** to fix with exact derivatives:
  1. **The near-rank-deficient (Σ,V) / mass-conservation block** — "scale Σ up, V down at fixed Ṁ" — worsens as the disk deepens into radiation-pressure support (β→0); FD can't resolve it → the warm-start **mass** wall.
  2. **The `r_s` grid-stretch column** — moving the free sonic node shifts every node's radius; this dense column is the FD Jacobian's least-accurate → the cold-seed **regularity** wall (`r_sonic` frozen).
- An exact Jacobian gives quadratic convergence AND resolves both blocks → reach the render regime.

**Target (the bar to beat):** at a=0.9, warm-start continuation climbs **well past f_Edd=0.11 toward ~0.9**; the cold/continuation path reaches **a=0.998**. Re-run the existing warm-start climb (`tools/slim_warmstart_sweep.cpp`) and the cold-seed spin sweep — both must extend their ceilings.

---

## The discipline (non-negotiable)

`slim_numerical_jacobian` is KEPT as the oracle. Every analytic block is validated by a test that builds both Jacobians at representative operating points and asserts the max relative column mismatch `< 1e-6`. Operating points: (i) gas-dominated low f_Edd (a=0.9, f_Edd≈0.02); (ii) radiation-dominated near the ceiling (a=0.9, f_Edd≈0.11, β~3.6e-4); (iii) higher spin (a=0.998 at low f_Edd). Use a small N (e.g. 16–24) so the dense FD reference is cheap. A derivation slip cannot ship — it shows up as a column mismatch pinpointing the wrong term.

---

## Tasks

### Task 1 — Cross-check harness + analytic stub
- Add `void slim_analytic_jacobian(const std::vector<double>& U, const SlimDiskInputs&, const OpacityLUTs&, std::vector<double>& J)` (row-major `n×n`, `n=4N+2`). Stub: initially copy `slim_numerical_jacobian` output (so the harness passes trivially, then each task replaces a block with the analytic form and the gate stays green).
- Test `test_analytic_jacobian_matches_fd` (new `tests/test_slim_jacobian.cpp`, target `test-slim-jacobian`): build both Jacobians at the 3 operating points, assert max column relative mismatch `<1e-6`, and print the worst (row,col) per point. Wire the CMake target. **Run → passes (stub).**
- Commit: `feat(slim-disk): analytic-Jacobian scaffold + FD cross-check test`.

### Task 2 — Closure derivatives (the foundation everything depends on)
- Add `OneZoneJac one_zone_closure_jac(Σ, T_c, r, in, op)` returning `∂{H, ρ_mid, p_mid, p_gas, p_rad, P, c_s, S}/∂{Σ, T_c}`.
- Derive via implicit-function-theorem on the H-quadratic `Ω_⊥²H² − bH − c_s_gas² = 0` (`b=2 a_rad T_c⁴/(3Σ)`); for μ, either freeze it (document) or differentiate `op.lookup_mu` via finite slope. Use `kappa_ross_with_grad` for opacity partials where needed downstream.
- Unit test: cross-check `one_zone_closure_jac` against a central-difference of `one_zone_closure` to `<1e-6` at gas- and radiation-dominated points.
- Commit: `feat(slim-disk): analytic one-zone closure derivatives`.

### Task 3 — Algebraic rows: mass + angular momentum
- Analytic `∂R/∂U` for the N mass rows (`∂(mdot_node−Ṁ)/∂{Σ,V}`, incl. Γ(V)) and N angmom rows (`∂[(Ṁ/2π)(ℓ−ℓ_in) − (A^½Δ^½Γ/r)αP]/∂{Σ,V,ℓ,T_c,ℓ_in}`, P via Task 2).
- Replace those blocks in `slim_analytic_jacobian`. **Cross-check gate green** at all 3 points (these rows must match FD exactly).
- Commit: `feat(slim-disk): analytic mass + angular-momentum Jacobian rows`.

### Task 4 — Kerr mechanics derivatives
- `∂Ω/∂ℓ` (`omega_from_ell` via `1/(dℓ/dΩ)`), `∂𝒜/∂{ℓ,r}`, `∂𝒟₀/∂{V,Σ,T_c}`, `∂𝒩₁/∂{Σ,V,ℓ,T_c,…}` (using Task 2 for P/closure, Task’s FD-gradient terms for Q_adv).
- Unit-test these helper derivatives vs central differences; they feed Task 5.
- Commit: `feat(slim-disk): analytic Kerr mechanics + D0/N1 derivatives`.

### Task 5 — ODE rows: radial-momentum + energy (trapezoidal) incl. stencils & L'Hôpital
- The trapezoidal rows couple nodes `i, i±1`; the `dlnP/dlnr`, `dlnΣ/dlnr`, `dΩ/dr` stencils are LINEAR in node values, so their exact derivatives are the stencil coefficients × the closure/mechanics derivatives (Tasks 2,4). Differentiate the node-0 L'Hôpital rhs `(d𝒩₁/dr)/(d𝒟₀/dr)·(1−V₀²)`.
- Replace these blocks. **Cross-check gate green** at all 3 points.
- Commit: `feat(slim-disk): analytic radial-momentum + energy Jacobian rows`.

### Task 6 — BC + regularity rows + global columns (incl. the `r_s` grid-stretch)
- Outer BC rows (cubic ℓ extrapolation; §23 energy-balance BC), regularity rows (`𝒟₀`,`𝒩₁` at node 0), and the two global columns:
  - `ℓ_in` column (appears in angmom + energy).
  - **`r_s` grid-stretch column (the hard one):** `r_i = r_s^{1−t_i} r_out^{t_i}` ⇒ `∂r_i/∂r_s = (1−t_i) r_i / r_s`; the column is `Σ_i (∂R/∂r_i)·(1−t_i)r_i/r_s`, the radial derivative of every r-dependent factor (mechanics, `dlnr` spacings) chained through. Derive carefully.
- Replace these. **Cross-check gate green** at all 3 points (the `r_s` column FD reference is noisy — require a looser match there, e.g. `<1e-3`, and verify the analytic column is SMOOTH where FD is noisy — the whole point).
- Commit: `feat(slim-disk): analytic BC/regularity rows + ell_in/r_s columns`.

### Task 7 — Wire in + re-run the climb (the payoff)
- Swap `slim_analytic_jacobian` for `slim_numerical_jacobian` in `relax_structure`'s Newton loop (keep the reduced-index active-set handling, row/col scaling, gain-ratio LM, line search). Keep the FD function + the cross-check test permanently.
- Tighten `kMeritFloor` toward `1e-6` (the analytic Jacobian reaches it) — but keep the physical-validity gate.
- **Re-run** `tools/slim_warmstart_sweep.cpp` (a=0.9) — must climb **well past f_Edd=0.11** (target ~0.9); and the cold-seed spin sweep — target **a=0.998**. Report the new ceilings vs the FD baseline (f_Edd=0.11, a=0.9).
- Confirm the full `test-slim-disk-radial` suite passes (a=0.9 AND a=0.998 at f_Edd≈0.3 should now converge directly or via continuation).
- Commit: `feat(slim-disk): analytic Jacobian in the Newton loop; FD ceiling broken`.

---

## Notes
- Do NOT change the §23 residual physics — only add derivatives of it. If a cross-check mismatch traces to the residual (not the derivative), STOP and report (it would mean a residual bug, not a Jacobian bug).
- The spin-walk warm-start seam bug (`warm_reproject_spin`, a≈0.2) is a SEPARATE, deferred issue — the cold seed reaches a=0.9 directly, so the spin-walk is only needed for near-extremal a; revisit after the analytic Jacobian (which may make a cold/continuation path to a=0.998 work without it).
- Temp probes (`tools/slim_*_sweep.cpp`, `slim_spinwalk_probe.cpp`) are the re-run harnesses for Task 7 — keep until then, delete after.
