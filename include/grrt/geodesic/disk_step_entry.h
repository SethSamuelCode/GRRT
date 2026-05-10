#ifndef GRRT_GEODESIC_DISK_STEP_ENTRY_H
#define GRRT_GEODESIC_DISK_STEP_ENTRY_H

#include "grrt/geodesic/integrator.h"
#include "grrt_export.h"

namespace grrt {

// Forward declarations
class VolumetricDisk;
class Kerr;
class RK4;

/// Result of the three-tier disk-entry gate (see check_disk_step_entry).
struct DiskStepEntryResult {
    bool should_raymarch;
    GeodesicState refined_endpoint;   ///< valid only when should_raymarch == true
    int substep_invocations;          ///< Tier C subdivide() invocations consumed
};

/// Tunables for the helper's adaptive subdivision depth and curvature padding.
struct DiskStepEntryOptions {
    int    depth_limit_floor = 4;     ///< minimum subdivisions
    int    depth_limit_cap   = 10;    ///< hard ceiling (1024x refinement)
    double curvature_pad     = 0.5;   ///< chord-length multiplier (see spec §5.4)
};

/// Three-tier gate replacing the endpoint-only predicate at three sites in
/// geodesic_tracer.cpp. See docs/superpowers/specs/2026-05-10-disk-step-entry-design.md
/// for design rationale.
GRRT_EXPORT DiskStepEntryResult check_disk_step_entry(
    const GeodesicState& prev_state,
    const GeodesicState& new_state,
    double dlambda_full,
    const VolumetricDisk& disk,
    const Kerr& metric,
    const RK4& integrator,
    const DiskStepEntryOptions& opts = {});

} // namespace grrt

#endif
