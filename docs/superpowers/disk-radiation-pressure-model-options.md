# Radiation-pressure-dominated inner disk — model options

**Date:** 2026-06-08 · **Branch:** fix/volumetric-ring
**Why:** The Approach-A column BVP solves the grey vertical structure with viscous heating `dQ/dz = α·P_total·|r dΩ/dr|` (stress ∝ **total** pressure — standard Shakura-Sunyaev). In the radiation-pressure-dominated inner disk (`β = P_gas/P ≲ 2.5e-3`) this prescription has **no stable steady solution** — the Lightman-Eardley (1974) viscous/thermal instability folds the solution branch, so the BVP can't converge. The numerics are already fixed (gas-pressure state variable, §21 of the formula ref); this is a **physics/closure** problem. Real radiation-dominated disks exist and are largely stable, so the instability is an artifact of the `α·P_total` closure. This doc surveys the alternatives (research 2026-06-08) and recommends a path.

---

## Executive summary — three strategies on an accuracy/complexity ladder

| Strategy | What changes | Removes the fold? | Keeps emergent Σ / per-column? | Implementation | Physical fidelity |
|---|---|---|---|---|---|
| **A. Stable stress closure** (Family 1) | only the stress law: `α·P_total` → `α·P_gas` (β-disk) or `α·P_total^μ P_gas^{1-μ}` with μ<3/7 | Yes (μ<3/7 always; μ=½ only for β≳0.08) | Yes — both | **1 line** + maybe 1 param | Low–moderate. Over-stabilises; misses magnetic support, advection, surface dissipation |
| **B. Prescribed dissipation profile** (Family 4, BHSPEC/TLUSTY — the production standard) | replace local `dQ/dz=α·P` with a **prescribed** surface-weighted heating profile `q(m)` (broken power-law fit to MHD sims), normalised to `∫q dm = σT_eff⁴` | Yes — heating no longer couples to P, so no runaway | Yes — both (Σ still pinned by the surface-pressure BC) | **Small–moderate**: swap the heating ODE for a prescribed flux profile | Moderate–high. This is *exactly* how production spectral codes handle it; captures the real (non-midplane) dissipation |
| **C. Add the missing physics** (Families 2,3) | radial **advection** (slim disk) and/or a **magnetic-pressure support** term `P_mag` | Yes — advection relieves local thermal balance; B-support raises effective β | Advection needs a **global radial transonic solve** (not per-column); B-support is per-column | **Large** (slim disk: new radial solver, ~weeks) / **small** (B-support term, but free parameter) | Highest. Slim disks are the accepted near-Eddington model; rad-MHD shows magnetic support + puffing are real |

**Key cross-cutting finding:** the mature production code **BHSPEC/TLUSTY does NOT evaluate `α·P_total` in the radiation-dominated vertical solve at all.** It uses the global α-model only to set the boundary conditions `(T_eff, Ω², Σ)`, and **prescribes the vertical dissipation profile** `Q(z)` (a broken power-law fit to Hirose-Krolik-Blaes 2009 MHD sims). The local solve never "sees" the instability. Our current formulation (local `α·P` heating) is precisely the thing the production codes avoid — which strongly reframes what the "right" fix is.

**Recommendation (see bottom for detail):** adopt **Strategy B** (prescribed surface-weighted dissipation profile) as the primary fix — it removes the fold cleanly, keeps our emergent-Σ per-column architecture, is production-validated, and is more physical than a bare stress-closure swap. Optionally add a **magnetic-pressure-support term** (Strategy C-lite, one parameter) for the puffing real disks show. Reserve full **slim-disk advection** for a later accuracy pass if we ever target near-Eddington disks (it needs a global radial solver).

---

## Family 1 — Alternative stress closures

The instability's root cause is stress ∝ **total** pressure. Every closure here softens that to a weaker dependence on `P_gas`. For the generalised law `τ_rφ = α P_total^μ P_gas^{1-μ}`, the thermal-stability boundary (Grzędzielski et al. 2017) is **instability only if `μ > 3/7 ≈ 0.43`** — so any `μ < 3/7` is globally stable.

- **β-prescription** `τ = α P_gas` (μ=0). Sakimoto & Coroniti 1981 (magnetic buoyancy caps stress at ~P_gas); Stella & Rosner 1984. **Stable at all β.** One-line change. *Cons:* over-stabilises — MRI sims show stress correlates with *total* pressure on long timescales, so pure β-disk under-heats/under-thickens the inner disk.
- **Geometric-mean** `τ = α √(P_gas·P_total)` (μ=½). Taam & Lin 1984; re-derived from MRI/corona coupling by Merloni & Nayakshin 2006. Stable for **β ≳ 0.08**; matches stable X-ray binary disks at 0.01–0.5 L_Edd. One line. *Cons:* μ=½ is just **above** the 3/7 threshold, so it can still fold in the deepest regime (β~1e-3) — needs a β-floor (~0.02) or μ→0 switch there.
- **Generalised power-law** `τ = α P_total^μ P_gas^{1-μ}`, choose `μ ≈ 0.4` (< 3/7). Szuszkiewicz 1990; Shadmehri et al. 2018 (μ≈0.4–0.5 gives realistic Σ(r)). **Stable everywhere**, one line + one parameter. Brackets the uncertainty.
- **Stress-lag closures** (Ciesielski et al. 2012): MRI stress *leads* pressure by ~one thermal time (Hirose+2009), which is what actually stabilises real disks — but this is a *dynamic* effect with no clean steady algebraic form. Use only as motivation for picking μ≈0.4–0.5, not directly.

**Family verdict:** cheapest possible fix. Best single choice = power-law `μ≈0.4` (or geometric-mean + β-floor). Guaranteed convergence; physically a coarse but defensible closure. Misses surface-weighted dissipation, magnetic support, advection.

**Refs:** Lightman & Eardley 1974 (ApJL 187 L1); Sakimoto & Coroniti 1981 (ApJ 247 19); Stella & Rosner 1984 (ApJ 277 312); Taam & Lin 1984 (ApJ 287 761); Merloni & Nayakshin 2006 ([arXiv:astro-ph/0603514], MNRAS 372 728); Watarai & Mineshige 2001 ([arXiv:astro-ph/0109019]); Ciesielski et al. 2012 ([arXiv:1106.2335]); Grzędzielski et al. 2017 ([arXiv:1609.09322], the μ_crit=3/7 stability map); Shadmehri et al. 2018 ([arXiv:1809.10671]).

---

## Family 2 — Advective / slim-disk models

**Mechanism.** Slim disks (Abramowicz, Czerny, Lasota & Szuszkiewicz 1988) replace local energy balance `Q_vis = Q_rad` with `Q_vis = Q_rad + Q_adv`, where `Q_adv` carries the un-radiated fraction *radially inward* (photon trapping). Because `Q_adv` grows steeply with Ṁ, it provides a powerful extra cooling channel that **stiffens the temperature response and damps the thermal runaway** — the unstable middle branch of the S-curve is replaced by a stable upper "slim-disk" branch with `dṀ/dΣ > 0`. The fold is gone. This is the **accepted standard model for Ṁ ≳ 0.3 Ṁ_Edd** (ULXs, high-state XRBs, super-Eddington AGN) — established, not speculative.

**Relativistic version.** Sądowski 2009 (Kerr radial solutions with sonic-point regularity); Sądowski et al. 2011 (adds vertical structure: hydrostatic + radiative diffusion + mixing-length convection on a grid over `(r, T_c, f_adv)`, with `f_adv` taken ≈ depth-independent). State of the art semi-analytic; used for XRB continuum fitting.

**Implementation cost — the catch.** `Q_adv ∝ dS/dr` is a **radial gradient** — it cannot be computed per-column. The slim-disk radial equations are **transonic** (radial velocity crosses the sound speed) and require a **regularity (eigenvalue) condition at an a-priori-unknown sonic radius** — a global free-boundary radial ODE solve. The documented two-stage path (Sądowski 2011): (1) solve the global radial slim disk once → tabulate `f_adv(r), Σ(r), Ω(r)`; (2) feed those into the per-column vertical BVP, where advection appears as a *depth-independent reduction of local radiative flux* (`f_adv` suppresses the flux that must be radiated). The vertical BVP we already have is reusable; the new component is the radial transonic solver (~2–3 weeks of focused work; well-posed, algorithms published).

**Effect on render:** inner disk **puffs up** (`H/r` → 0.25–0.3 near Ṁ_Edd), flatter `T(r)`, peak shifts inward, ~tens of % of energy advected inward and re-radiated near the plunge, softer inner spectrum. Emission from within the ISCO is physically included. *Caveat:* at Ṁ ≳ Ṁ_Edd the grey diffusion approximation breaks down (`τ_eff < 1`) in the innermost cells.

**Family verdict:** the most physically correct steady model for high accretion rates, and it removes the fold by real physics (advection), not a closure tweak. But it's the most invasive: a global transonic radial solve is unavoidable. Best reserved for a dedicated accuracy pass, especially if near-Eddington disks are a target.

**Refs:** Abramowicz et al. 1988 (ApJ 332 646); Sądowski 2009 ([arXiv:0906.0355]); Sądowski et al. 2011 ([arXiv:1006.4309], A&A 527 A17); Sądowski 2011 thesis ([arXiv:1108.0396]); Abramowicz & Fragile 2013 (Living Rev. Rel. 16 1); Poutanen et al. 2007 ([arXiv:astro-ph/0609274]).

---

## Family 3 — Radiation-MHD (MRI) turbulence results

These shearing-box and global simulations are the most first-principles handle — they let the MRI generate stress self-consistently instead of assuming `α·P`.

- **Hirose, Krolik & Blaes 2009** ([arXiv:0809.1708]): radiation-dominated shearing boxes are **thermally stable** over ~40 cooling times; magnetic stress *leads* radiation pressure by ~one thermal time (causality inverted vs the α assumption — this breaks the runaway loop). Photosphere rises to **~7H** (vs ~3H analytic) from magnetic support; dissipation is **surface-weighted, not midplane-concentrated**; significant radiative *advection* near the midplane. Effective α≈0.02.
- **Jiang, Stone & Davis 2013** ([arXiv:1309.5646]): with a better radiation solver (VET) and larger boxes, they instead see **thermal runaway** — stress scales only weakly (∝P^0.19) and peaks off-midplane. The Hirose-vs-Jiang stability question is **not fully resolved** (box size, transport scheme, net flux all matter).
- **Ross, Latter & Guilet 2017** ([arXiv:1703.00211]), **Hogg & Reynolds 2016**: in large no-net-flux boxes, stress scales ~ **gas pressure** — supporting an effective `α·P_gas` closure; instability is weakened to a biased random walk rather than exponential runaway.
- **Global rad-MHD** (Jiang, Stone & Davis 2014/2019; Huang, Davis & Jiang 2023): inner disks are **magnetic-pressure supported**, geometrically **thicker**, with **lower Σ** than thin-disk predictions and **radially-variable α** (~0.2 near ISCO → ~0.03 outward). Authors explicitly state the vertical density profile of magnetically-supported disks is **not yet understood analytically**. Stabilisation criterion (Jiang et al. 2022): `P_mag ≳ 0.5 P_total` at the midplane.
- **Blaes et al. 2006** ([arXiv:astro-ph/0601380]): published an **empirical broken-power-law vertical dissipation profile** from the sims — *the closest thing to a usable closure*, and exactly what BHSPEC adopts (Family 4).

**Family verdict:** first-principles but mostly **guidance, not drop-in formulas**. The directly-usable distillations are: (i) an effective `α·P_gas` or surface-weighted dissipation profile, and (ii) an optional **magnetic-pressure-support term** `P_mag ~ (0.1–0.5)P_total` in hydrostatic balance (free parameter, reproduces the observed puffing). Accuracy gain over `α·P_total`: substantial; residual uncertainty ~factor-of-a-few in Σ and H.

**Refs:** Hirose Krolik Blaes 2009 ([0809.1708]); Blaes Hirose Krolik 2009 ([0908.1117]), 2011 ([1103.5052]); Blaes & Krolik 2006 ([astro-ph/0601380]); Turner 2004 ([astro-ph/0402539]); Jiang Stone Davis 2013 ([1309.5646]), 2014 corona ([1402.2979]), 2014 global super-Edd ([1410.0678]), 2019 global sub-Edd ([1904.01674]); Ross Latter Guilet 2017 ([1703.00211]), 2022 stress lag ([2111.11226]); Huang Davis Jiang 2023 ([2301.12679]); Jiang et al. 2022 ([2209.03317]), 2025 ([2505.09671]).

---

## Family 4 — Disk-atmosphere codes & how production tools handle it

- **TLUSTY / BHSPEC** (Hubeny & Hubeny 1998; Davis, Blaes, Hubeny & Turner 2005; Davis & Hubeny 2006; Davis & Blaes 2013). The industry-standard XRB/AGN continuum-fitting code. Each annulus is a 1-D atmosphere given `(T_eff, Ω², Σ)` as **boundary conditions from the global disk model**, and crucially **the vertical dissipation profile `Q(z)` is PRESCRIBED** — a broken power-law in column fraction (`ε ∝ (Σ/Σ₀)^p`, break at 0.11, exponents ≈0.5/0.2) fit to Hirose-Krolik-Blaes 2009. Because heating is a prescribed function of column, **not** `α·P_total`, the Lightman-Eardley instability **never enters the per-annulus solve** — radiation-dominated annuli compute fine. This is the cleanest, most-validated resolution and the **most directly adoptable for us**: replace `dQ/dz = α·P·shear` with `dF/dz = ε(m/Σ)·σT_eff⁴` (a normalised broken power-law). Newton relaxation, grey opacity, emergent Σ (still pinned by the surface-pressure BC) all unchanged. *Cons:* the prescribed profile is calibrated from gas-/marginally-radiation-dominated sims; it omits radiative advection (which matters near the midplane in the deepest regime).
- **DiscVerSt** (Tavleev, Lipunova & Malanchev 2023, [arXiv:2303.02184], [github.com/AndreyTavleev/DiscVerSt]) — our §20 reference. Uses `α·P_total` and **explicitly refuses the radiation-dominated regime** ("the code does not calculate such discs"). Confirms the wall is real and standard. Useful only as a gas-dominated cross-check.
- **Sądowski 2011** — the slim-disk vertical code (see Family 2); resolves the regime via advection, needs the radial solve.
- **diskvert** (Gronkiewicz & Różańska 2020, [arXiv:1909.13858], [github.com/gronki/diskvert]) — adds an explicit **magnetic pressure term** `P_mag = η·P_gas` to vertical balance (raises effective β past the instability threshold) and has a fixed-density "post-corona" fallback for still-unstable annuli. Magnetic support is a one-line add to hydrostatic balance, but `η` is a free parameter.

**Family verdict:** production codes converge on **decoupling local heating from `α·P_total`** — either by prescribing `Q(z)` (BHSPEC, most adoptable) or by magnetic support (diskvert). BHSPEC's prescribed-dissipation profile is the single most directly adoptable, validated approach for our grey per-column BVP.

**Refs:** Hubeny & Hubeny 1998 ([astro-ph/9804288], ApJ 505 844); Davis Blaes Hubeny Turner 2005 ([astro-ph/0408590], ApJ 621 372); Davis & Hubeny 2006 (ApJS 164 530, BHSPEC); Davis & Blaes 2013 ([1305.3320]); Tavleev Lipunova Malanchev 2023 ([2303.02184]); Sądowski et al. 2011 ([1006.4309]); Gronkiewicz & Różańska 2020 (A&A 633 A35).

---

## Recommendation

For a renderer prioritising accuracy with no compute constraint, ordered by what I'd do:

1. **Primary — Strategy B (prescribed surface-weighted dissipation profile, BHSPEC-style).** Replace the local `dQ/dz = α·P_total·shear` heating with a prescribed dissipation profile `q(q_col)` (broken power-law in column-mass fraction, normalised so `∫q = σT_eff⁴`). This is what the production standard does, removes the fold by construction (heating no longer couples to pressure), keeps our per-column emergent-Σ Newton BVP essentially intact, and captures the real surface-weighted dissipation. Small–moderate change. Externally verify the exact profile (Hirose+2009 / Blaes+2006 fit) before coding.
2. **Pair with — magnetic-pressure support (Strategy C-lite).** Add `P_mag = η·P_total` (η ~ 0.1–0.3) to hydrostatic balance to reproduce the puffing global rad-MHD shows. One term, one parameter (tunable, documented as a modelling knob — *not* a hidden magic number).
3. **Fallback / simplest — Strategy A (power-law stress μ≈0.4).** If B proves fiddly, the one-line stress-closure swap guarantees convergence and is defensible, just less physical.
4. **Future accuracy pass — Strategy C-full (slim-disk advection).** Only if we target near-Eddington disks; needs a global transonic radial solver. Big, separate effort.

**Decision needed:** which strategy to implement now. My lead is **B (+ optional magnetic support)**; A is the quick guaranteed-convergence fallback; C-full is a future major effort.
