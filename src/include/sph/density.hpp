#pragma once

#include <cmath>
#include <vector>
#include <cassert>

#include "kernels.hpp"

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeDensity(const std::vector<int> &l, Dataset &d)
{
    const int64_t n = l.size();
    const int64_t ngmax = d.ngmax;
    const int *clist = l.data();
    const int *neighbors = d.neighbors.data();
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
    const int np = d.x.size();
    const int64_t allNeighbors = n * ngmax;
#pragma omp target map(to                                                                                                                  \
                       : clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np], z [0:np])  \
    map(from                                                                                                                               \
        : ro [0:n])
#pragma omp teams distribute parallel for// dist_schedule(guided) // parallel for
#elif defined(USE_ACC)
    const int np = d.x.size();
    const int64_t allNeighbors = n * ngmax;
#pragma acc parallel loop copyin(n, clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], m [0:np], h [0:np], x [0:np], y [0:np], \
                                 z [0:np]) copyout(ro [0:n])
#else
#pragma omp parallel for
#endif
    for (int pi = 0; pi < n; pi++)
    {
        const int i = clist[pi];
        const int nn = neighborsCount[pi];

        T roloc = 0.0;
        ro[i] = 0.0;

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

            T value = wharmonic(vloc, h[i], sincIndex, K);
            roloc += value * m[j];
        }

        ro[i] = roloc + m[i] * K / (h[i] * h[i] * h[i]);

#ifndef NDEBUG
        if (std::isnan(ro[i])) printf("ERROR::Density(%d) density %f, position: (%f %f %f), h: %f\n", i, ro[i], x[i], y[i], z[i], h[i]);
#endif
    }

    // Initialization of fluid density at rest
    if (d.iteration == 0)
    {
#pragma omp parallel for
        for (int pi = 0; pi < n; pi++)
        {
            const int i = clist[pi];
            ro_0[i] = ro[i];
        }
    }
}
} // namespace sph
} // namespace sphexa
