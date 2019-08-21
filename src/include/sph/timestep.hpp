#pragma once

#include <vector>
#include <math.h>
#include <algorithm>

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

    auto policy = hpx::parallel::execution::par;

    auto min_op = [h, c](T v1, size_t v2)
                  {
                    // v2 will assume value of clist[pi]
                    // Time-scheme according to Press (2nd order)
                    T quot = h[v2] / c[v2];
                    if (quot < v1) return quot;
                    else           return v1;
                  };

    mini = Kcour * hpx::parallel::reduce(policy, clist, clist+n, mini, min_op);

    //#pragma omp parallel for reduction(min : mini)
    //for (int pi = 0; pi < n; pi++)
    //{
    //    int i = clist[pi];
    //    // Time-scheme according to Press (2nd order)
    //    dt[i] = Kcour * (h[i] / c[i]);
    //    if (dt[i] < mini) mini = dt[i];
    //}

    if (n > 0) mini = std::min(mini, maxDtIncrease * dt_m1[0]);

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &mini, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

    hpx::parallel::for_loop(policy,
        0, n,
    [=](int pi)
    {
        int i = clist[pi];
        dt[i] = mini;
    });

    ttot += mini;
}
} // namespace sph
} // namespace sphexa
