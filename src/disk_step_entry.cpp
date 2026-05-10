#include "grrt/geodesic/disk_step_entry.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/geodesic/rk4.h"
#include <algorithm>
#include <cmath>

namespace grrt {

DiskStepEntryResult check_disk_step_entry(
    const GeodesicState& /*prev_state*/,
    const GeodesicState& /*new_state*/,
    double /*dlambda_full*/,
    const VolumetricDisk& /*disk*/,
    const Kerr& /*metric*/,
    const RK4& /*integrator*/,
    const DiskStepEntryOptions& /*opts*/)
{
    // Stub — Task 2 wires up Tier A; Task 5 wires the full orchestrator.
    return { false, {}, 0 };
}

} // namespace grrt
