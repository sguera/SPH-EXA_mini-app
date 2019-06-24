#pragma once

#include <vector>
#include <math.h>

#include "kernels.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeTimestep(const std::vector<int> &l, Dataset &d)
{
    const int n = l.size();
    const int *clist = l.data();

    const T *h = d.h.data();
    const T *c = d.c.data();
    const T *dt_m1 = d.dt_m1.data();
    const T Kcour = d.Kcour;
    const T maxDtIncrease = d.maxDtIncrease;

    T &ttot = d.ttot;
    T *dt = d.dt.data();

    T mini = INFINITY;

#pragma omp parallel for reduction(min : mini)
    for (int pi = 0; pi < n; pi++)
    {
        int i = clist[pi];
        // Time-scheme according to Press (2nd order)
        dt[i] = Kcour * (h[i] / c[i]);
        if (dt[i] < mini) mini = dt[i];
    }

    if (n > 0) mini = std::min(mini, maxDtIncrease * dt_m1[0]);

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &mini, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

#pragma omp parallel for
    for (int pi = 0; pi < n; pi++)
    {
        int i = clist[pi];
        dt[i] = mini;
    }

    ttot += mini;
}
} // namespace sph
} // namespace sphexa
