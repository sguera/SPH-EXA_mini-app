#pragma once

#include <vector>

#include <cmath>
#include "math.hpp"
#include "kernels.hpp"
#include "lookupTables.hpp"
#include "cuda/sph.cuh"

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
void computeIADImpl(const Task &t, Dataset &d)
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
    const T *ro = d.ro.data();

    T *c11 = d.c11.data();
    T *c12 = d.c12.data();
    T *c13 = d.c13.data();
    T *c22 = d.c22.data();
    T *c23 = d.c23.data();
    T *c33 = d.c33.data();

    const BBox<T> bbox = d.bbox;

    const T K = d.K;
    const T sincIndex = d.sincIndex;

    // std::vector<T> checkImem(n, 0), checkDeltaX(n, 0), checkDeltaY(n, 0), checkDeltaZ(n, 0);

#if defined(USE_OMP_TARGET)
    // Apparently Cray with -O2 has a bug when calling target regions in a loop. (and computeIADImpl can be called in a loop).
    // A workaround is to call some method or allocate memory to either prevent buggy optimization or other side effect.
    // with -O1 there is no problem
    // Tested with Cray 8.7.3 with NVIDIA Tesla P100 on PizDaint
    std::vector<T> imHereBecauseOfCrayCompilerO2Bug(4, 10);

    const int np = d.x.size();
    const size_t allNeighbors = n * ngmax;

// clang-format off
#pragma omp target map(to                                                                                                                  \
		       : clist [:n], neighbors[:allNeighbors], neighborsCount[:n],                                                         \
                       x [0:np], y [0:np], z [0:np], h [0:np], m [0:np], ro [0:np])                                                        \
                   map(from                                                                                                                \
                       : c11[:n], c12[:n], c13[:n], c22[:n], c23[:n], c33[:n])
// clang-format on
#pragma omp teams distribute parallel for // dist_schedule(guided)
#elif defined(USE_ACC)
    const int np = d.x.size();
    const size_t allNeighbors = n * ngmax;
// clang-format off
#pragma acc parallel loop copyin(clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n],                                            \
                                  x [0:np], y [0:np], z [0:np], h [0:np], m [0:np], ro [0:np])                                             \
                           copyout(c11 [:n], c12 [:n], c13 [:n], c22 [:n], c23 [:n],                                                       \
                                   c33 [:n])
// clang-format on
#else
#pragma omp parallel for schedule(guided)
#endif
    for (size_t pi = 0; pi < n; ++pi)
    {
        const int i = clist[pi];
        const int nn = neighborsCount[pi];

        T tau11 = 0.0, tau12 = 0.0, tau13 = 0.0, tau22 = 0.0, tau23 = 0.0, tau33 = 0.0;

        for (int pj = 0; pj < nn; ++pj)
        {
            const int j = neighbors[pi * ngmax + pj];

            // later can be stored into an array per particle
            const T dist = distancePBC(bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); // store the distance from each neighbor
            // calculate the v as ratio between the distance and the smoothing length
            const T vloc = dist / h[i];
            const T w = K * math_namespace::pow(wharmonic(vloc), (int)sincIndex);
            const T W = w / (h[i] * h[i] * h[i]);

            T r_ijx = (x[i] - x[j]);
            T r_ijy = (y[i] - y[j]);
            T r_ijz = (z[i] - z[j]);

            applyPBC(bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);

            tau11 += r_ijx * r_ijx * m[j] / ro[j] * W;
            tau12 += r_ijx * r_ijy * m[j] / ro[j] * W;
            tau13 += r_ijx * r_ijz * m[j] / ro[j] * W;
            tau22 += r_ijy * r_ijy * m[j] / ro[j] * W;
            tau23 += r_ijy * r_ijz * m[j] / ro[j] * W;
            tau33 += r_ijz * r_ijz * m[j] / ro[j] * W;

            /*
            checkImem[i] += m[j] / ro[j] * W;
            checkDeltaX[i] += m[j] / ro[j] * (r_ijx) * W;
            checkDeltaY[i] += m[j] / ro[j] * (r_ijy) * W;
            checkDeltaZ[i] += m[j] / ro[j] * (r_ijz) * W;
            */
        }

        const T det =
            tau11 * tau22 * tau33 + 2.0 * tau12 * tau23 * tau13 - tau11 * tau23 * tau23 - tau22 * tau13 * tau13 - tau33 * tau12 * tau12;

        // c11[pi] = (tau22 * tau33 - tau23 * tau23) / det;
        // c12[pi] = (tau13 * tau23 - tau33 * tau12) / det;
        // c13[pi] = (tau12 * tau23 - tau22 * tau13) / det;
        // c22[pi] = (tau11 * tau33 - tau13 * tau13) / det;
        // c23[pi] = (tau13 * tau12 - tau11 * tau23) / det;
        // c33[pi] = (tau11 * tau22 - tau12 * tau12) / det;

        c11[i] = (tau22 * tau33 - tau23 * tau23) / det;
        c12[i] = (tau13 * tau23 - tau33 * tau12) / det;
        c13[i] = (tau12 * tau23 - tau22 * tau13) / det;
        c22[i] = (tau11 * tau33 - tau13 * tau13) / det;
        c23[i] = (tau13 * tau12 - tau11 * tau23) / det;
        c33[i] = (tau11 * tau22 - tau12 * tau12) / det;
    }
}

template <typename T, class Dataset>
void computeIAD(const std::vector<Task> &taskList, Dataset &d)
{
#if defined(USE_CUDA)
    cuda::computeIAD<T>(taskList, d); // utils::partition(l, d.noOfGpuLoopSplits), d);
#else
    for (const auto &task : taskList)
    {
        computeIADImpl<T>(task, d);
    }
#endif
}

} // namespace sph

} // namespace sphexa
