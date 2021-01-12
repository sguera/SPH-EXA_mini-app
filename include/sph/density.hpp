#pragma once

#include <vector>

#include "kernels.hpp"
#include "Task.hpp"
#include "lookupTables.hpp"
#include "cuda/sph.cuh"

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeDensityImpl(const Task &t, Dataset &d)
{
    const size_t n = t.clist.size();
    const size_t ngmax = t.ngmax;
    const int *clist = t.clist.data();
    const int *neighbors = t.neighbors.data();
    const int *neighborsCount = t.neighborsCount.data();

    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

    T *ro = d.ro.data();

    const BBox<T> bbox = d.bbox;

    const T K = d.K;
    const T sincIndex = d.sincIndex;

#if defined(USE_OMP_TARGET)
    // Apparently Cray with -O2 has a bug when calling target regions in a loop. (and computeDensityImpl can be called in a loop).
    // A workaround is to call some method or allocate memory to either prevent buggy optimization or other side effect.
    // with -O1 there is no problem
    // Tested with Cray 8.7.3 with NVIDIA Tesla P100 on PizDaint
    std::vector<T> imHereBecauseOfCrayCompilerO2Bug(4, 10);

    const size_t np = d.x.size();
    const size_t allNeighbors = n * ngmax;
// clang-format off
#pragma omp target map(to                                                                                                                  \
                       : clist [0:n], neighbors [:allNeighbors], neighborsCount [:n], m [0:np], h [0:np], x [0:np], y [0:np], z [0:np])    \
                   map(from                                                                                                                \
                       : ro [:n])
#pragma omp teams distribute parallel for
// clang-format on
#elif defined(USE_ACC)
    const size_t np = d.x.size();
    const size_t allNeighbors = n * ngmax;
#pragma acc parallel loop copyin(n, clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np], \
                                 z [0:np]) copyout(ro[:n])
#else
#pragma omp parallel for
#endif
    for (size_t pi = 0; pi < n; pi++)
    {
        const int i = clist[pi];
        const int nn = neighborsCount[pi];

        T roloc = 0.0;

        // int converstion to avoid a bug that prevents vectorization with some compilers
        for (int pj = 0; pj < nn; pj++)
        {
            const int j = neighbors[pi * ngmax + pj];

            // later can be stores into an array per particle
            T dist = distancePBC(bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); // store the distance from each neighbor

            // calculate the v as ratio between the distance and the smoothing length
            T vloc = dist / h[i];

#ifndef NDEBUG
            if (vloc > 2.0 + 1e-6 || vloc < 0.0)
                printf("ERROR:Density(%d,%d) vloc %f -- x %f %f %f -- %f %f %f -- dist %f -- hi %f\n", i, j, vloc, x[i], y[i], z[i], x[j],
                       y[j], z[j], dist, h[i]);
#endif

            const T w = K * math_namespace::pow(wharmonic(vloc), (int)sincIndex);
            const T value = w / (h[i] * h[i] * h[i]);
            roloc += value * m[j];
        }

        // ro[pi] = roloc + m[i] * K / (h[i] * h[i] * h[i]);
        ro[i] = roloc + m[i] * K / (h[i] * h[i] * h[i]);

#ifndef NDEBUG
        if (std::isnan(ro[i])) printf("ERROR::Density(%d) density %f, position: (%f %f %f), h: %f\n", i, ro[i], x[i], y[i], z[i], h[i]);
#endif
    }
}

template <typename T, class Dataset>
void computeDensity(const std::vector<Task> &taskList, Dataset &d)
{
#if defined(USE_CUDA)
    cuda::computeDensity<T>(taskList, d); // utils::partition(l, d.noOfGpuLoopSplits), d);
#else
    for (const auto &task : taskList)
    {
        computeDensityImpl<T>(task, d);
    }
#endif
}

template <typename T, class Dataset>
void initFluidDensityAtRestImpl(const Task &t, Dataset &d)
{
    const size_t n = t.clist.size();
    const int *clist = t.clist.data();

    const T *ro = d.ro.data();
    T *ro_0 = d.ro_0.data();

#pragma omp parallel for
    for (size_t pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];
        ro_0[i] = ro[i];
    }
}

template <typename T, class Dataset>
void initFluidDensityAtRest(const std::vector<Task> &taskList, Dataset &d)
{
    for (const auto &task : taskList)
    {
        initFluidDensityAtRestImpl<T>(task, d);
    }
}

} // namespace sph
} // namespace sphexa
