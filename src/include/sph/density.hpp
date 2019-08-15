#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeDensity(const std::vector<int> &l, Dataset &d)
{
    const size_t n = l.size();
    const size_t ngmax = d.ngmax;
    const int *clist = l.data();
    const int *neighborsCount = d.neighborsCount.data();

    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

    T *ro = d.ro.data();
    T *ro_0 = d.ro_0.data();

    const BBox<T> bbox = d.bbox;

    const T sincIndex = d.sincIndex;
    const T K = d.K;

#if defined(USE_OMP_TARGET)
    const size_t np = d.x.size();
    const size_t allNeighbors = n * ngmax;

    for (ushort s = 0; s < d.noOfGpuLoopSplits; ++s)
    {
        const size_t begin_n = s * n / d.noOfGpuLoopSplits;
        const size_t end_n = (s + 1) * n / d.noOfGpuLoopSplits;
        const size_t begin_neighbors = begin_n * ngmax;
        const size_t end_neighbors = end_n * ngmax;
        const size_t neighborsOffset = begin_neighbors;
        const size_t neighborsChunkSize = end_neighbors - begin_neighbors;

        const int *neighbors = d.neighbors.data() + neighborsOffset;

// clang-format off
#pragma omp target map(to                                               \
		       : clist [:n], neighbors [:neighborsChunkSize], neighborsCount [:n],  \
                         m [:np], h [:np], x [:np], y [:np], z [:np])                                                                      \
                   map(from                                                                                                                \
                       : ro [:n])
// clang-format on
#pragma omp teams distribute parallel for // dist_schedule(guided)
#elif defined(USE_ACC)
    const size_t neighborsOffset = 0;
    const int *neighbors = d.neighbors.data();
    const size_t np = d.x.size();
    const size_t allNeighbors = n * ngmax;
    const size_t begin_n = 0;
    const size_t end_n = n;
#pragma acc parallel loop copyin(n, clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np], \
                                 z [0:np]) copyout(ro [0:n])
#else
    const size_t neighborsOffset = 0;
    const int *neighbors = d.neighbors.data();
    const size_t begin_n = 0;
    const size_t end_n = n;
#pragma omp parallel for
#endif
        for (size_t pi = begin_n; pi < end_n; ++pi)
        {
            const int i = clist[pi];
            const int nn = neighborsCount[pi];

            T roloc = 0.0;
            ro[i] = 0.0;

            // int converstion to avoid a bug that prevents vectorization with some compilers
            for (int pj = 0; pj < nn; pj++)
            {
                const int j = neighbors[pi * ngmax + pj - neighborsOffset];
                // later can be stores into an array per particle
                const T dist = distancePBC(bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); // store the distance from each neighbor

                // calculate the v as ratio between the distance and the smoothing length
                const T vloc = dist / h[i];

#ifndef NDEBUG
                if (vloc > 2.0 + 1e-6 || vloc < 0.0)
                    printf("ERROR:Density(%d,%d) vloc %f -- x %f %f %f -- %f %f %f -- dist %f -- hi %f\n", i, j, vloc, x[i], y[i], z[i],
                           x[j], y[j], z[j], dist, h[i]);
#endif

                const T value = wharmonic(vloc, h[i], sincIndex, K);
                roloc += value * m[j];
            }

            ro[i] = roloc + m[i] * K / (h[i] * h[i] * h[i]);

#ifndef NDEBUG
            if (std::isnan(ro[i])) printf("ERROR::Density(%d) density %f, position: (%f %f %f), h: %f\n", i, ro[i], x[i], y[i], z[i], h[i]);
#endif
        }
#if defined(USE_OMP_TARGET)
    }
#endif

    // Initialization of fluid density at rest
    if (d.iteration == 0)
    {
#pragma omp parallel for
        for (size_t pi = 0; pi < n; ++pi)
        {
            const int i = clist[pi];
            ro_0[i] = ro[i];
        }
    }
}
} // namespace sph
} // namespace sphexa
