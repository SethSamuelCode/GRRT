#include "grrt/scene/disk_column_bvp.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>

namespace grrt {

double eos_rho(double, double) { return 0.0; }   // implemented in Task 2

ColumnBVPSolution solve_column_bvp(const ColumnInputs& in, const OpacityLUTs&) {
    ColumnBVPSolution s;                          // implemented in later tasks
    s.q.assign(in.n_nodes, 0.0);
    return s;
}

} // namespace grrt
