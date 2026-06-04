#include "grrt/scene/disk_column_bvp.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>

namespace grrt {

double eos_rho(double P, double T) {
    using namespace constants;
    const double P_gas = P - (a_rad / 3.0) * T * T * T * T;   // P - P_rad
    if (P_gas <= 0.0 || T <= 0.0) return 0.0;                 // non-physical
    return P_gas * mu_fully_ionized * m_p / (k_B * T);
}

ColumnBVPSolution solve_column_bvp(const ColumnInputs& in, const OpacityLUTs&) {
    ColumnBVPSolution s;                          // implemented in later tasks
    s.q.assign(in.n_nodes, 0.0);
    return s;
}

} // namespace grrt
