# Slim-disk literature review — 2026-08-09

Commissioned to answer: **is Sądowski et al. 2011 still the right architectural basis**, or has
something superseded it? Targets driving the question: Kerr spin 0→1, f_Edd up to ~1 and beyond
with a breakdown warning, feeding T_eff(r) and H(r) into a GR raytracer.

All claims below were verified by live fetch. Papers dated 2026 and late-2025 are at/after the
researching model's training cutoff and come from fetches, not memory.

---

## 1. VERDICT: Sądowski 2011 has NOT been superseded

**arXiv:1006.4309 / A&A 527, A17** remains the only stationary, general-relativistic, transonic
slim-disc model with self-consistently solved vertical structure.

- A sweep of ~40 post-2011 "slim disc" arXiv papers found **zero** new stationary relativistic
  structure models. All are applications (TDEs, ULXs, LRDs/JWST AGN, IMBHs, continuum fitting,
  cosmological sub-grid models).
- The two nearest structural papers are both **non-relativistic**:
  - Liu & Yuan 2025, arXiv:2505.03583 — height-averaged; unifies ADAF/SSD/slim branches. No GR,
    no vertical structure. Worth reading for its branch-structure treatment (see §3).
  - Wen et al. 2025, arXiv:2508.15150 — *pseudo*-Newtonian circumbinary slim disc.
- **Sądowski produced no successor.** He moved to radiation-GRMHD (KORAL): Sądowski 2016,
  arXiv:1601.06785 is a simulation paper.
- 2026 GRMHD papers still benchmark **against** it — e.g. Lančová/Wielgus et al. 2026,
  arXiv:2603.17922 uses Novikov–Thorne and the slim disc as its analytic baselines.
- `slimbh` (Sądowski 2011 radial + TLUSTY vertical; Straub et al. 2011, arXiv:1106.0009) is still
  the shipped XSPEC model for near-Eddington continuum fitting, 15 years on.

**Conclusion: building on Sądowski 2011 is not sunk cost. Nothing better exists in its class.**

---

## 2. Thermal instability of the radiation-pressure branch is GENUINELY CONTESTED

Do **not** over-engineer around it. Report the disagreement; don't pick a side.

The authoritative recent statement — Blaes, Jiang, Lasota & Lipunova 2025, "Non-Stationary Discs
and Instabilities", *Space Science Reviews*, arXiv:2505.04402:

> "it has never been clear whether these instabilities have a physical reality, or are merely an
> artifact of the assumptions behind the alpha-prescription."

**Established:** the S-curve topology is textbook. At fixed r the optically-thick thermal-equilibrium
curve in Σ–Ṁ has three branches — gas-pressure-dominated SSD (lower, stable), radiation-pressure
dominated (middle, dṀ/dΣ < 0), advective slim (upper, stable). The middle branch is *both* viscously
and thermally unstable **under the α·P_tot prescription** — a modelling assumption, not a derived
result.

**Camp A — instability is real (local shearing boxes):**
- Jiang, Stone & Davis 2013, arXiv:1309.5646 — radiation-MHD shearing boxes with a variable Eddington
  tensor: radiation-dominated equilibria "always eventually suffer runaway heating or cooling."
  Sensitive to box size and net magnetic flux (acknowledged by the authors).

**Camp B — suppressed in reality (global / magnetised):**
- Sądowski 2016, arXiv:1601.06785 — discs dominated by magnetic pressure are thermally stable;
  needs strong *net* vertical/radial flux. Radiative efficiency ~unchanged (5.5% vs 5.7%).
- Mishra et al. 2016 — "Thermal instability (or not?) in 3D global radiative GRMHD…"
- Jiang, Blaes, Stone & Davis 2019, arXiv:1904.01674 — global radiation-MHD, 7% and 20% Eddington:
  magnetically supported, Σ far below thin-disc prediction, H far above, and **"do not show any sign
  of thermal instability over many thermal time scales."**
- Lančová et al. 2019, ApJL 884, L37 — puffy discs stabilised by net vertical magnetic flux.

**Observationally:** BH XRB high/soft state is quiet over 0.01–0.5 L_Edd. GRS 1915+105 and
IGR J17091−3624 are the famous exceptions. Nature mostly does *not* show the instability where
α-theory predicts it.

**Implication for us:** select the gas branch because that is where the **global transonic solution**
lives — NOT because we have decided the other branch is unphysical. The leading explanation for
observed stability is magnetic support, which a hydrodynamic slim disc cannot represent at all.

---

## 3. BRANCH SELECTION IS STRUCTURAL, NOT LOCAL

This is the key methodological finding, and it matches our own diagnosis.

In the Sądowski/Abramowicz formulation you do **not** pick a branch at each (r, Σ):

- The ODE system has a critical (sonic) point where numerator and denominator both vanish. The
  **regularity condition there is satisfied for only one value of the angular momentum — the
  eigenvalue of the two-point BVP.** Solved by relaxation, with the critical-point location
  promoted to an extra unknown (free-boundary problem).
- Documented failure modes are diagnostic:
  - angular momentum too high → solution terminates before reaching the sonic point;
  - too low → passes the sonic radius but **"follows an improper branch."**
  That second one is branch mis-selection, and it is detected **globally**, not locally.
- So at fixed Ṁ the physical branch is fixed by **continuity from the outer boundary inward**,
  subject to the sonic-point eigenvalue.

**Therefore: local multiplicity at a node is EXPECTED and is not by itself an error.** The remedy is
continuation along r — seed each node from its neighbour — not an independent root-find per node.

**No paper offers a continuation/homotopy method for this problem.** The literature's answer is
uniformly "relaxation with a good trial solution", with explicit acknowledgement that "convergence
strictly depends on the quality of such initial guess." Our multi-start + advective-seed-T_c
machinery is at parity with published practice.

---

## 4. VALIDITY LIMITS — and the asymmetry that matters for rendering

- **Slim discs stay slim.** Lasota, Vieira, Sądowski, Narayan & Abramowicz 2016, arXiv:1510.09152:
  "the dominant effect of advection at high accretion rates precludes slim discs becoming thick" —
  H/R ≲ 1 even super-Eddington. The model does *not* self-invalidate by puffing up.
- **The real failure mode is MASS LOSS.** Ohsuga & Mineshige and successors: slim disc overestimates
  luminosity around L_Edd; photon trapping is appreciable already below L_Edd whereas the slim disc
  puts it above ~3 L_Edd. Above ~1.5× critical the net luminosity is super-Eddington; above ~5× the
  hole accretes super-Eddington; beyond that the wind goes quasi-spherical and the model breaks.
- **Strongest recent challenge:** Fragile, Middleton, Bollimpalli & Smith 2025, arXiv:2505.08859 —
  long-timescale super-critical GRMHD at Ṁ = 1–10 Ṁ_Edd. All runs settle to *net* accretion near
  Eddington at all radii (outflow cancels inflow); they find agreement with the **critical
  (wind-regulated)** disc model rather than the slim (advection-dominated) one, contradicting most
  previous numerical work.
- **Puffy discs (≥ ~0.3–0.5 L_Edd):** Lančová et al. 2019; Wielgus et al. 2022, arXiv:2202.08831;
  quantified in arXiv:2603.17922 — simulated photospheric heights are **much larger** than analytic
  models and nearly independent of Ṁ (h_τ/R ≈ 1); surface density **much lower**; magnetosonic point
  well inside the ISCO; α rises steeply inward. Wielgus et al. find XSPEC models fit puffy-disc
  synthetic spectra well but **recover the WRONG BLACK HOLE SPIN.**
- Abolmasov & Chashkina 2015, arXiv:1509.07261 — GR corrections raise the local Eddington limit ~2×;
  thickness raises it further; advection lowers it. "L = L_Edd" is not a sharp line.

### ⚠ THE ASYMMETRY WE CARE ABOUT
We export exactly two quantities, and they degrade very differently:

| quantity | behaviour above ~0.3 L_Edd |
|---|---|
| **H(r)** | **degrades badly** — factor-of-several underestimate of photospheric height |
| **T_eff(r)** | **comparatively robust** — "hardly changes" even when outflow is included (Ohsuga) |

**Recommended graduated warning** (replaces a single f_Edd=1 threshold):
- **≲ 0.3 L_Edd** — quantitatively trustworthy
- **0.3 → 1** — degraded; magnetic support and puffiness missing; H and Σ both wrong in the
  direction the puffy-disc papers quantify
- **> 1** — omits the physics (winds / mass loss) that the newest simulations say dominates

---

## 5. UPGRADE AVAILABLE: α is not constant

**Abramowicz, Brandenburg, Horák, Lančová, Miller, Szuszkiewicz & Wielgus 2026, arXiv:2603.10997** —
"Universal behaviour of α-viscosity in black hole accretion discs". From three independent GRMHD
simulations, α varies by ~an order of magnitude across the disc with a pronounced stress maximum
near the photon orbit. They give a fitting formula in terms of a "gyration radius", with the stated
intent "to improve analytic models by making them more realistic."

**Drop-in α(r) upgrade for a Sądowski-type solver.** Cheap fidelity win.

---

## 6. EXISTING IMPLEMENTATIONS (surveyed; we chose to stay self-contained)

- **`mbursa/disk-models`** — https://github.com/mbursa/disk-models. Public C library, unified
  interface, ships **both** Novikov–Thorne and the **Sądowski polytropic slim disc**; exposes radial
  profiles of T_c, Σ, P, h, v_r and emergent flux F(r). Compiles to `.so`, `dlopen()`-able, Python
  wrapper, integrates with the **SIM5** GR ray-tracing library. NOTE: it is the *polytropic
  height-averaged* variant, **not** the 2-D vertical-structure model — i.e. LESS detailed than what
  we are attempting. Expect the 20–30% offsets Sądowski 2011 documents.
- **`slimbh` XSPEC tables** — spectra on a 3-D grid of (spin, disc luminosity, inclination) with
  `vflag`/`lflag` switches. Spectra, not structure; an end-to-end benchmark, not a structure source.
- **Tavleev, Lipunova & Malanchev 2023, arXiv:2303.02184** — open **Python** code for vertical
  structure with radiative + convective transport, arbitrary EoS, tabular opacities, producing radial
  structure and stability **S-curves**. Non-relativistic, but a readable reference implementation of
  exactly the vertical-structure + S-curve machinery. Useful for cross-validating a column solver.
- **Shashank et al. 2024, arXiv:2407.12890** — slim-disk model with self-consistent thickening and
  self-shadowing, ray-traced; closest published thing to our pipeline. Unconfirmed whether they
  release tabulated H(r)/T_eff(r).

**EHT is NOT a comparison source for our regime** — M87*/Sgr A* are ~10⁻⁵ Eddington RIAFs modelled
with magnetised thick tori. The relevant comparison community is X-ray continuum fitting and
ULX/TDE modelling.

**Decision (2026-08-09): stay self-contained.** No external dependency; keep validating against
published figures plus our own FD-oracle discipline. `mbursa/disk-models` is recorded here as a
possible future cross-check.

---

## 7. ACTIONABLE SUMMARY

1. **Keep Sądowski 2011 as the basis.** Nothing better exists; it is still the 2026 benchmark.
2. **Precompute the (r, T_c, f_adv) vertical-structure table** — the reference architecture's own
   design, and the likely cure for our ~2 h/iteration cost (more so than additional OpenMP).
3. **Use continuation in r for branch tracking**, not per-node multi-start. Local multiplicity is
   expected; the global transonic solution + continuity is what selects the branch.
4. **Graduate the breakdown warning** to ~0.3 / ~1 L_Edd, and state the H(r)-vs-T_eff(r) asymmetry
   explicitly in user-facing docs.
5. **Consider α(r)** from arXiv:2603.10997 as a cheap fidelity upgrade.
6. Do **not** hard-code an assumption that the radiation-pressure branch is unphysical — that is
   contested. Select the gas branch on structural grounds instead.

## Caveat from the researcher
Section 5 of Blaes et al. 2025 (arXiv:2505.04402) could not be retrieved in full — PDF and HTML body
extraction both failed. The quotes above come from its introduction and partial HTML, so its final
verdict on the instability may be sharper than represented here.
