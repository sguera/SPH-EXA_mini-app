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
void computeMinTimestepImpl(const Task &t, Dataset &d, T &minDt)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();

    const T *h = d.h.data();
    const T *c = d.c.data();
    const T *maxvsignal = d.maxvsignal.data();
    const T Kcour = d.Kcour;

    T minDtTmp = INFINITY;

#pragma omp parallel for reduction(min : minDtTmp)
    for (size_t pi = 0; pi < n; pi++)
    {
        int i = clist[pi];
        // Time-scheme according to Press (2nd order)
        if(maxvsignal[i] > 0.0)
        {
            T dt = Kcour*h[i]/maxvsignal[i];
            if (dt < minDtTmp) minDtTmp = dt;
        }
        else
        {
            T dt = Kcour * (h[i] / c[i]);
            if (dt < minDtTmp) minDtTmp = dt;
        }
    }

    minDt = minDtTmp;
}

template <typename T, class Dataset>
void setMinTimestepImpl(const Task &t, Dataset &d, const T minDt)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();

    T *dt = d.dt.data();

#pragma omp parallel for
    for (size_t pi = 0; pi < n; pi++)
    {
        int i = clist[pi];
        dt[i] = minDt;
    }
}

template <typename T, class Dataset>
void computeTimestep(const std::vector<Task> &taskList, Dataset &d)
{
    T minDtTmp = INFINITY;
    for (const auto &task : taskList)
    {
        T tminDt = 0.0;
        computeMinTimestepImpl<T>(task, d, tminDt);
        minDtTmp = std::min(tminDt, minDtTmp);
    }

    minDtTmp = std::min(minDtTmp, d.maxDtIncrease * d.minDt);

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &minDtTmp, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

    for (const auto &task : taskList)
    {
        setMinTimestepImpl(task, d, minDtTmp);
    }

    d.ttot += minDtTmp;
    d.minDt = minDtTmp;
}


} // namespace sph
} // namespace sphexa
