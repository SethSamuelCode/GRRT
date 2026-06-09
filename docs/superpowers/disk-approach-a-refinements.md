# Disk Approach-A — deferred refinements

A living list of refinements we consciously deferred while building the first-principles disk (Approach A). Each is a *known, bounded* improvement — not a bug — recorded with enough context to pick up later. Ordered roughly by value/ease.

---

### 1. Computed (Richardson) vertical resolution `n_nodes`
**Now:** the column BVP uses a fixed `n_nodes` (~150–300), a physics-informed default (the column spans ~4–9 pressure scale heights; that many uniform points resolve the photosphere for typical inner-disk columns).
**Refinement:** make it *computed* — solve at `N`, then `2N`; if `z₀`/`Σ₀`/the profile change by less than a tolerance, `N` was sufficient, else refine (Richardson convergence). This *guarantees* adequate resolution per column at ~2–3× solve cost. It's the clean replacement for the retired `nested_refine`/`compare_columns` machinery, should a fixed `N` ever under-resolve a column.
**Why deferred:** a fixed `N` is comfortably sufficient for the rendered inner disk; "simple now, principled if needed."

### 1b. Column convergence robustness (hot / extreme-flux columns) — ✅ largely resolved / superseded
**Done:** (a) the explicit **homotopy/continuation** refinement is implemented — `bootstrap_column` does parameter-space continuation from a known-good easy column, and the radial march warm-starts each column from its converged neighbour. (b) The **gas-pressure conditioning fix** (carry `P_gas`, not total `P`, as the Newton state variable — §21) removed a catastrophic EOS cancellation / ill-conditioning that stalled radiation-influenced columns, extending convergence from `β≈1e-6` to `β≈2.5e-3`; the analytic Jacobian cross-check is exact at hot points. (c) The radiation-aware total-sound-speed seed is in `build_seed`. The honest fallback was removed (non-convergence returns empty, never a fabricated profile).
**The remaining wall is PHYSICAL, not numerical:** below `β≈2.5e-3` the standard `α(total-P)` thin-disk vertical structure has **no stable steady solution** (Lightman-Eardley fold) — no seeding or continuation can converge to a structure that does not exist. This is being resolved by the **relativistic slim-disk subsystem** (radial advection removes the fold; spec `2026-06-08-relativistic-slim-disk-design.md`). So 1b's convergence machinery is done; the radiation-dominated regime is a physics redesign, not a convergence refinement.

### 2. Banded (block-tridiagonal) Jacobian solve
**Now:** the Newton step uses a *dense* numerical Jacobian + dense Gaussian-elimination solve, `O(n³)` per step (`n=4N+2`). Fine for the standalone solver tests at one column.
**Refinement:** the true Jacobian is block-tridiagonal (each node couples only to neighbors). A block-Thomas banded solve is `O(N)`, and the numerical Jacobian can be built banded (perturb by color). Needed before solving 4096 columns at construction time for the full disk (Plan 3).
**Why deferred:** correctness first; dense is fine for unit tests. Pairs with #3.

### 3. Analytic Jacobian as the production engine — ✅ DONE (Plan 2)
**Status: DONE.** The analytic block Jacobian IS the production engine in `solve_column_bvp` (using `kappa_total_with_grad`), validated by the numerical finite-difference Jacobian via the `column_jacobians_test` cross-check (exact 0.0 mismatch at both gas- and radiation-dominated operating points). Re-derived for the gas-pressure state variable in the conditioning fix (§21), cross-check still exact. The numerical Jacobian remains its permanent validation test.

### 4. `GRRT_TEST_EXPORT` macro for test hooks
**Now:** `column_residual_test`, `column_numerical_jacobian_test` (and similar testable internals) are `GRRT_EXPORT`, so they ship in the production DLL (standalone test exes link the DLL and need the symbols).
**Refinement:** a `GRRT_TEST_EXPORT` macro that expands to `GRRT_EXPORT` only under a test build, so internal test hooks don't pollute the public DLL surface. Codebase-wide cleanup (also affects `disk_step_entry`, etc.).

### 5. Audit the absorption Rosseland opacity bump (T ~ 5–10×10⁴ K)
**Observation:** during BVP Task 5, the converged column at `T_eff=5e4` was optically thick (`τ_mid≈43`) — physically reasonable, but driven by a large Kramers absorption opacity in `kappa_ross_lut`. Worth confirming the absorption Rosseland mean magnitude (free-free + bound-free) is right and not over-estimated in the `~10⁴–10⁵ K` opacity-bump region.
**Why deferred:** the disk being optically thick is correct physics regardless; the BVP now uses the *total* Rosseland opacity (`κ_abs + κ_es`), which is the important fix. A magnitude audit is a separate validation.

### 6. Molecular + dust opacity for cool (`< 3000 K`) outer disks
**Now:** the opacity model is atomic (Saha ionization, free-free, bound-free, H⁻); the LUT floors at `T_min = 3000 K` (the atomic-opacity validity edge). Cooler material is clamped to the table edge.
**Refinement:** add molecular-band (H₂, CO, TiO, H₂O) and dust opacity to model genuinely cool outer disks (e.g., extended AGN discs → the dusty "torus").
**Why deferred:** the rendered *inner* disk is hot (`≫3000 K`) for every BH mass, so the image is unaffected; cool outer-disk fidelity is a specialized, large addition. (Even today a cool annulus still *emits* the correct blackbody color — only its opacity structure is approximate.)

### 7. Non-grey / line opacity (frequency-resolved structure)
**Now:** the vertical structure is grey (Rosseland-mean). Continuum opacity only — no spectral lines.
**Refinement:** non-grey atmosphere (frequency carried through the structure) and bound-bound line opacity (atomic line lists, NLTE). Would enable e.g. the relativistically-broadened **Fe Kα** line used to measure spin. ~50–500× cost; its own opacity subsystem. (Spec §9/§16 follow-up.)

### 8. CUDA backend port
**Now:** suspended (CUDA is behind). The CPU path is the deliverable.
**Refinement:** carry the log-density LUT encoding + absolute `ρ_mid` into `cuda/cuda_vol_host_data.cpp` and the device interpolation; optionally GPU-parallelize the per-column BVP construction (embarrassingly parallel, plain-struct/no-virtual solver is already CUDA-shaped). Restores the `CUDA == CPU` invariant. (Spec §11/§16.)

### 9. Super-Eddington thick disk / radiation funnel (future model)
**Now:** the structure solver targets thin → near-Eddington disks (`f_Edd ≲ 1` accurate; graceful to ~1–2 via the slim-disk subsystem). Genuinely super-Eddington accretion (`f_Edd ≫ 1` — real and observed: ULX pulsars at 100–1000×, tidal disruption events, super-Eddington quasars) is **out of scope** of the slim disk.
**Refinement:** a separate **super-Eddington thick-disk / radiation-funnel model** ("Polish doughnut" — Abramowicz, Jaroszyński & Sikora 1978): a geometrically thick (`H/r ~ 1`), radiation-pressure-supported torus with a polar funnel, radiation-driven winds, and (for the most accurate version) radiation-MHD structure. The grey-diffusion approximation breaks down (`τ_eff < 1`) in the inner funnel, so this needs a different radiative-transport treatment. Enables rendering ULXs / TDEs / super-Eddington quasars. (Distinct from, and larger than, the slim-disk subsystem.)
**Why deferred:** the slim disk covers the `f_Edd ≈ 0.9` workhorse case and the whole sub-to-near-Eddington range (it reduces exactly to the thin disk at low `Ṁ`); the super-Eddington funnel is a distinct, larger, more uncertain subsystem (winds + thick geometry + non-grey transport).

---
*Source context: built during the Approach-A redesign (spec `2026-06-01-disk-first-principles-vertical-structure-design.md`, plans `2026-06-01-disk-first-principles-foundation.md` and `2026-06-04-disk-column-bvp-solver.md`). Verified formulas: `references/disk-physics-formulas.md`.*
