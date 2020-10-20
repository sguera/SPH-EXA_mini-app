#pragma once

#include <vector>
#include "Task.hpp"
#include "ParticlesData.hpp"
#include "Octree.hpp"
#include "LinearOctree.hpp"
#include "cuda/sph.cuh"

namespace sphexa
{
namespace sph
{
namespace kernels
{

template <typename T>
T normalize(T d, T min, T max)
{
    return (d - min) / (max - min);
}

template <typename T>
void findNeighborsDispl(const int pi, const int *clist, const T *x, const T *y, const T *z, const T *h,
                        const T displx, const T disply, const T displz, const int ngmax, int *neighbors, int *neighborsCount,
                        // The linear tree
                   		const int *o_cells, const int *o_ncells, const int *o_localPadding, const int *o_localParticleCount, const T *o_xmin, const T *o_xmax, const T *o_ymin, const T *o_ymax, const T *o_zmin, const T *o_zmax)
{
    const int i = clist[pi];

    // // 64 is not enough... Depends on the bucket size and h...
    // // This can be created and stored on the GPU directly.
    // // For a fixed problem and size, if it works then it will always work
    int collisionsCount = 0;
    int collisionNodes[128];

    const T xi = x[i] + displx;
    const T yi = y[i] + disply;
    const T zi = z[i] + displz;
    const T ri = 2.0 * h[i];

    constexpr int nX = 2;
    constexpr int nY = 2;
    constexpr int nZ = 2;

    int stack[64];
    int stackptr = 0;
    stack[stackptr++] = -1;

    int node = 0;

    do
    {
        if (o_ncells[node] == 8)
        {
            int mix = std::max((int)(normalize(xi - ri, o_xmin[node], o_xmax[node]) * nX), 0);
            int miy = std::max((int)(normalize(yi - ri, o_ymin[node], o_ymax[node]) * nY), 0);
            int miz = std::max((int)(normalize(zi - ri, o_zmin[node], o_zmax[node]) * nZ), 0);
            int max = std::min((int)(normalize(xi + ri, o_xmin[node], o_xmax[node]) * nX), nX - 1);
            int may = std::min((int)(normalize(yi + ri, o_ymin[node], o_ymax[node]) * nY), nY - 1);
            int maz = std::min((int)(normalize(zi + ri, o_zmin[node], o_zmax[node]) * nZ), nZ - 1);

            // Maximize threads sync
            for (int hz = 0; hz < 2; hz++)
            {
                for (int hy = 0; hy < 2; hy++)
                {
                    for (int hx = 0; hx < 2; hx++)
                    {
                        // if overlap
                        if (hz >= miz && hz <= maz && hy >= miy && hy <= may && hx >= mix && hx <= max)
                        {
                            // int l = hz * nX * nY + hy * nX + hx;
                            // stack[stackptr++] = o_cells[node * 8 + l];
                            const int l = hz * nX * nY + hy * nX + hx;
                            const int child = o_cells[node * 8 + l];
                            if(o_localParticleCount[child] > 0)
                                stack[stackptr++] = child;
                        }
                    }
                }
            }
        }

        if (o_ncells[node] != 8) collisionNodes[collisionsCount++] = node;

        node = stack[--stackptr]; // Pop next
    } while (node > 0);

    //__syncthreads();

    int ngc = neighborsCount[pi];

    for (int ni = 0; ni < collisionsCount; ni++)
    {
        int node = collisionNodes[ni];
        T r2 = ri * ri;

        for (int pj = 0; pj < o_localParticleCount[node]; pj++)
        {
            int j = o_localPadding[node] + pj;

            T xj = x[j];
            T yj = y[j];
            T zj = z[j];

            T xx = xi - xj;
            T yy = yi - yj;
            T zz = zi - zj;

            T dist = xx * xx + yy * yy + zz * zz;

            if (dist < r2 && i != j && ngc < ngmax) neighbors[ngc++] = j;
        }
    }

    neighborsCount[pi] = ngc;

    //__syncthreads();
}

template <typename T>
void findNeighbors(const int pi, const int *clist, const T *x, const T *y, const T *z, const T *h, const T displx,
                   const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount,
                   // The linear tree
                   const int *o_cells, const int *o_ncells, const int *o_localPadding, const int *o_localParticleCount, const T *o_xmin, const T *o_xmax, const T *o_ymin, const T *o_ymax, const T *o_zmin, const T *o_zmax)
{
    T dispx[3], dispy[3], dispz[3];

    dispx[0] = 0;
    dispy[0] = 0;
    dispz[0] = 0;
    dispx[1] = -displx;
    dispy[1] = -disply;
    dispz[1] = -displz;
    dispx[2] = displx;
    dispy[2] = disply;
    dispz[2] = displz;

    neighborsCount[pi] = 0;

    for (int hz = 0; hz <= maz; hz++)
        for (int hy = 0; hy <= may; hy++)
            for (int hx = 0; hx <= max; hx++)
                findNeighborsDispl<T>(pi, clist, x, y, z, h, dispx[hx], dispy[hy], dispz[hz], ngmax, &neighbors[pi * ngmax], neighborsCount,
                                   // The linear tree
                                   o_cells, o_ncells, o_localPadding, o_localParticleCount, o_xmin, o_xmax, o_ymin, o_ymax, o_zmin, o_zmax);
}
} // namespace kernels

template <typename T, class Dataset>
void computeFindNeighbors(const LinearOctree<T> &o, std::vector<Task> &taskList, Dataset &d)
{
    const int maz = d.bbox.PBCz ? 2 : 0;
    const int may = d.bbox.PBCy ? 2 : 0;
    const int max = d.bbox.PBCx ? 2 : 0;

    const T displx = o.xmax[0] - o.xmin[0];
    const T disply = o.ymax[0] - o.ymin[0];
    const T displz = o.zmax[0] - o.zmin[0];

    const T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    // Device pointers
    const T *d_h = d.h.data();
    const T *d_x = d.x.data();
    const T *d_y = d.y.data();
    const T *d_z = d.z.data();

    // Map LinearTree to device pointers
    // Currently OpenMP implementations do not support very well the mapping of structures
    // So we convert everything to simple arrays and pass them to OpenMP
    const int *o_ncells = o.ncells.data();
    const int *o_cells = o.cells.data();
    const int *o_localPadding = o.localPadding.data();
    const int *o_localParticleCount = o.localParticleCount.data();
    const T *o_xmin = o.xmin.data();
    const T *o_xmax = o.xmax.data();
    const T *o_ymin = o.ymin.data();
    const T *o_ymax = o.ymax.data();
    const T *o_zmin = o.zmin.data();
    const T *o_zmax = o.zmax.data();

    for (auto &t : taskList)
    {
        // Device pointers
        const int *d_clist = t.clist.data();
        int *d_neighbors = t.neighbors.data();
        int *d_neighborsCount = t.neighborsCount.data();

        const size_t n = t.clist.size();
        
// clang-format off
#if defined(USE_OMP_TARGET)
    const size_t np = d.x.size();
    const size_t st = o.size;
    const size_t stt = o.size * 8;
    // Apparently Cray with -O2 has a bug when calling target regions in a loop. (and computeDensityImpl can be called in a loop).
    // A workaround is to call some method or allocate memory to either prevent buggy optimization or other side effect.
    // with -O1 there is no problem
    // Tested with Cray 8.7.3 with NVIDIA Tesla P100 on PizDaint
    std::vector<T> imHereBecauseOfCrayCompilerO2Bug(4, 10);
    
    const size_t nn = n * ngmax;

#pragma omp target map(to : n, ngmax, max, may, maz, displx, disply, displz, d_x[0:np], d_y[0:np], d_z[0:np], d_h[0:np], o_cells[0:stt], o_ncells[0:st], o_localPadding[0:st], o_localParticleCount[0:st], o_xmin[0:st], o_xmax[0:st], o_ymin[0:st], o_ymax[0:st], o_zmin[0:st], o_zmax[0:st], d_clist[0:n]) map(from: d_neighbors[0:nn], d_neighborsCount [0:n])
#pragma omp teams distribute parallel for
// clang-format on
#else
#pragma omp parallel for
#endif
        for (size_t pi = 0; pi < n; pi++)
            kernels::findNeighbors<T>(pi, d_clist, d_x, d_y, d_z, d_h, displx, disply, displz, max, may, maz, ngmax, d_neighbors, d_neighborsCount,
                                   // The linear tree
                                   o_cells, o_ncells, o_localPadding, o_localParticleCount, o_xmin, o_xmax, o_ymin, o_ymax, o_zmin, o_zmax);
    }
}

template <typename T, class Dataset>
void findNeighbors(const Octree<T> &o, std::vector<Task> &taskList, Dataset &d)
{

#if defined(USE_CUDA)
    //cuda::computeFindNeighbors2<T>(l, taskList, d);
#else
    LinearOctree<T> l;
    createLinearOctree(o, l);
    computeFindNeighbors<T>(l, taskList, d);
#endif
}

size_t neighborsSumImpl(const Task &t)
{
    size_t sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (unsigned int i = 0; i < t.clist.size(); i++)
        sum += t.neighborsCount[i];
    return sum;
}

size_t neighborsSum(const std::vector<Task> &taskList)
{
    size_t sum = 0;
    for (const auto &task : taskList)
    {
        sum += neighborsSumImpl(task);
    }

#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

    return sum;
}
} // namespace sph
} // namespace sphexa
