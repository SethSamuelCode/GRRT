#ifndef GRRT_SLIM_DISK_COUPLED_H
#define GRRT_SLIM_DISK_COUPLED_H
#include "grrt/scene/slim_disk_radial.h"
#include "grrt/scene/disk_column_coupled.h"
#include "grrt/color/opacity.h"
#include "grrt_export.h"

namespace grrt {

/// Nested coupled-column slim-disk radial solve (Tasks 8 "C5" + 9 "C4").
///
/// Mirrors solve_slim_disk_radial's relativistic transonic structure, but REPLACES
/// the one-zone vertical closure (per node) with the in-tree vertical BVP column
/// closure `solve_column_coupled` (the (Σ,T_c)→(F,z0,η3,η4,f_adv) map).  The radial
/// residual is rerouted so that, per node i:
///   • energy row:  Q_rad = 64σT_c⁴/(3κΣ)  →  2·F_i     (column emergent flux; F_i is
///                  ONE FACE, the radial Q_rad is BOTH FACES — see disk-physics §23),
///   • closure:     H_i  →  z0_i           (the column photosphere half-thickness),
///   • Q_adv / 𝒩₁:  η3(β) →  η3_i          (the column's vertical energy moment), and
///   • C5: the 𝒩₁ assembly restores the S11 η-gradient terms the one-zone path drops
///         ((P/Σ)·dlnη3/dlnr + Ω_⊥²·(η4/η3)·dlnη4/dlnr), using per-node η3_i, η4_i and
///         FD radial gradients (GRRT Ω_⊥² = Ω_K²·ℋ convention).
///
/// The driver mirrors the LM-damped Newton relax loop (group scaling, gain-ratio LM,
/// feasibility line search) but builds the Jacobian by NUMERICAL central differences
/// of the coupled residual (re-solving every column per perturbation — slow but
/// correct; the analytic Schur Jacobian is a later task).  The feasibility line
/// search ALSO rejects any step on which a node's column fails to converge.  On
/// non-convergence it returns SlimDiskRadial{converged=false} — never a fabricated
/// profile.
///
/// Reuses the existing SlimDiskInputs / SlimDiskRadial from slim_disk_radial.h.
GRRT_EXPORT SlimDiskRadial solve_slim_disk_coupled(const SlimDiskInputs& in,
                                                   const OpacityLUTs& op);

} // namespace grrt
#endif
