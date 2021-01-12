#pragma once

#include <cmath>
#include <memory>
#include <algorithm>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "Task.hpp"

#include <stdint.h>
#include <limits.h>

#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "what is happening here?"
#endif

namespace sphexa
{

template <typename T>
class Octree
{
public:
    Octree(const T xmin, const T xmax, const T ymin, const T ymax, const T zmin, const T zmax, const int comm_rank, const int comm_size)
        : xmin(xmin)
        , xmax(xmax)
        , ymin(ymin)
        , ymax(ymax)
        , zmin(zmin)
        , zmax(zmax)
        , comm_rank(comm_rank)
        , comm_size(comm_size)
    {
    }

    Octree() = default;

    virtual ~Octree() = default;

    std::vector<std::shared_ptr<Octree>> cells;

    T xmin = INFINITY, xmax = -INFINITY, ymin = INFINITY, ymax = -INFINITY, zmin = INFINITY, zmax = -INFINITY;

    int comm_rank = -1;
    int comm_size = -1;
    int assignee = -1;

    int localPadding = 0;

    int globalNodeCount = 0;

    int localParticleCount = 0;
    size_t globalParticleCount = 0;

    T localMaxH = 0.0;
    T globalMaxH = 0.0;

    bool halo = false;

    bool global = false;

    static const int nX = 2, nY = 2, nZ = 2;
    static const int ncells = 8;
    static const int noParticlesThatPreventParallelTaskCreation = 10000;

    static size_t bucketSize, minGlobalBucketSize, maxGlobalBucketSize;

    static inline T normalize(T d, T min, T max) { return (d - min) / (max - min); }

    static inline T distancesq(const T x1, const T y1, const T z1, const T x2, const T y2, const T z2)
    {
        T xx = x1 - x2;
        T yy = y1 - y2;
        T zz = z1 - z2;

        return xx * xx + yy * yy + zz * zz;
    }

    inline void check_add_start(const int id, const T *x, const T *y, const T *z, const T xi, const T yi, const T zi, const T r,
                                const int ngmax, int *neighbors, int &neighborsCount) const
    {
        T r2 = r * r;

        // int maxCount = std::min(localParticleCount, ngmax - neighborsCount);

        for (int i = 0; i < localParticleCount; i++)
        {
            int ordi = localPadding + i;

            T dist = distancesq(xi, yi, zi, x[ordi], y[ordi], z[ordi]);
            if (dist < r2 && ordi != id && neighborsCount < ngmax) neighbors[neighborsCount++] = ordi;
        }
    }

    inline void distributeParticles(const std::vector<int> &list, const std::vector<T> &ax, const std::vector<T> &ay,
                                    const std::vector<T> &az, std::vector<std::vector<int>> &cellList)
    {
        for (size_t i = 0; i < list.size(); i++)
        {
            T xx = ax[list[i]];
            T yy = ay[list[i]];
            T zz = az[list[i]];

            T hx = std::min(std::max((int)(normalize(xx, xmin, xmax) * nX), 0), nX - 1);
            T hy = std::min(std::max((int)(normalize(yy, ymin, ymax) * nY), 0), nY - 1);
            T hz = std::min(std::max((int)(normalize(zz, zmin, zmax) * nZ), 0), nZ - 1);

            size_t l = hz * nX * nY + hy * nX + hx;

            cellList[l].push_back(list[i]);
        }
    }

    void findNeighborsRec(const int id, const T *x, const T *y, const T *z, const T xi, const T yi, const T zi, const T ri, const int ngmax,
                          int *neighbors, int &neighborsCount) const
    {
        if ((int)cells.size() == ncells)
        {
            int mix = std::max((int)(normalize(xi - ri, xmin, xmax) * nX), 0);
            int miy = std::max((int)(normalize(yi - ri, ymin, ymax) * nY), 0);
            int miz = std::max((int)(normalize(zi - ri, zmin, zmax) * nZ), 0);
            int max = std::min((int)(normalize(xi + ri, xmin, xmax) * nX), nX - 1);
            int may = std::min((int)(normalize(yi + ri, ymin, ymax) * nY), nY - 1);
            int maz = std::min((int)(normalize(zi + ri, zmin, zmax) * nZ), nZ - 1);

            for (int hz = miz; hz <= maz; hz++)
            {
                for (int hy = miy; hy <= may; hy++)
                {
                    for (int hx = mix; hx <= max; hx++)
                    {
                        size_t l = hz * nX * nY + hy * nX + hx;

                        cells[l]->findNeighborsRec(id, x, y, z, xi, yi, zi, ri, ngmax, neighbors, neighborsCount);
                    }
                }
            }
        }
        else
            check_add_start(id, x, y, z, xi, yi, zi, ri, ngmax, neighbors, neighborsCount);
    }

    void findNeighbors(const int id, const T *x, const T *y, const T *z, const T xi, const T yi, const T zi, const T ri, const int ngmax,
                       int *neighbors, int &neighborsCount, const bool PBCx = false, const bool PBCy = false, const bool PBCz = false) const
    {
        if ((PBCx && (xi - ri < xmin || xi + ri > xmax)) || (PBCy && (yi - ri < ymin || yi + ri > ymax)) ||
            (PBCz && (zi - ri < zmin || zi + ri > zmax)))
        {
            int mix = (int)floor(normalize(xi - ri, xmin, xmax) * nX);
            int miy = (int)floor(normalize(yi - ri, ymin, ymax) * nY);
            int miz = (int)floor(normalize(zi - ri, zmin, zmax) * nZ);
            int max = (int)floor(normalize(xi + ri, xmin, xmax) * nX);
            int may = (int)floor(normalize(yi + ri, ymin, ymax) * nY);
            int maz = (int)floor(normalize(zi + ri, zmin, zmax) * nZ);

            if (!PBCx) mix = std::max(mix, 0);
            if (!PBCy) miy = std::max(miy, 0);
            if (!PBCz) miz = std::max(miz, 0);
            if (!PBCx) max = std::min(max, nX - 1);
            if (!PBCy) may = std::min(may, nY - 1);
            if (!PBCz) maz = std::min(maz, nZ - 1);

            for (int hz = miz; hz <= maz; hz++)
            {
                for (int hy = miy; hy <= may; hy++)
                {
                    for (int hx = mix; hx <= max; hx++)
                    {
                        T displz = PBCz ? ((hz < 0) - (hz >= nZ)) * (zmax - zmin) : 0;
                        T disply = PBCy ? ((hy < 0) - (hy >= nY)) * (ymax - ymin) : 0;
                        T displx = PBCx ? ((hx < 0) - (hx >= nX)) * (xmax - xmin) : 0;

                        int hzz = PBCz ? (hz % nZ) + (hz < 0) * nZ : hz;
                        int hyy = PBCy ? (hy % nY) + (hy < 0) * nY : hy;
                        int hxx = PBCx ? (hx % nX) + (hx < 0) * nX : hx;

                        size_t l = hzz * nY * nX + hyy * nX + hxx;

                        cells[l]->findNeighborsRec(id, x, y, z, xi + displx, yi + disply, zi + displz, ri, ngmax, neighbors,
                                                   neighborsCount);
                    }
                }
            }
        }
        else
            findNeighborsRec(id, x, y, z, xi, yi, zi, ri, ngmax, neighbors, neighborsCount);
    }

    void print(int l = 0)
    {
        if (global)
        {
            for (int i = 0; i < l; i++)
                printf("   ");
            printf("[%d] %d %d %d %d %d %d\n", assignee, localPadding, localParticleCount, globalParticleCount, globalNodeCount,
                   (int)cells.size(), halo);

            if ((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                    cells[i]->print(l + 1);
            }
        }
    }
    virtual void makeSubCells()
    {
        cells.resize(ncells);

        for (int hz = 0; hz < nZ; hz++)
        {
            for (int hy = 0; hy < nY; hy++)
            {
                for (int hx = 0; hx < nX; hx++)
                {
                    T ax = xmin + hx * (xmax - xmin) / nX;
                    T bx = xmin + (hx + 1) * (xmax - xmin) / nX;
                    T ay = ymin + hy * (ymax - ymin) / nY;
                    T by = ymin + (hy + 1) * (ymax - ymin) / nY;
                    T az = zmin + hz * (zmax - zmin) / nZ;
                    T bz = zmin + (hz + 1) * (zmax - zmin) / nZ;

                    size_t i = hz * nX * nY + hy * nX + hx;

                    if (cells[i] == nullptr) cells[i] = std::make_shared<Octree>(ax, bx, ay, by, az, bz, comm_rank, comm_size);
                }
            }
        }
    }

    int globalRebalance(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax, int lvl = 0)
    {
        this->xmin = xmin;
        this->xmax = xmax;
        this->ymin = ymin;
        this->ymax = ymax;
        this->zmin = zmin;
        this->zmax = zmax;

        this->assignee = -1;
        this->halo = false;

        this->localPadding = 0;
        this->globalNodeCount = 0;
        this->localParticleCount = 0;

        this->localMaxH = 0.0;
        this->globalMaxH = 0.0;

        int nsplits = 0;

        if (global)
        {
            this->globalNodeCount = 1;

            // Closing non global branches
            if ((int)cells.size() == ncells && cells[0]->global == false) cells.clear();

            if ((int)cells.size() == ncells)
            {
                if (globalParticleCount < minGlobalBucketSize)
                    cells.clear();
                else
                {
                    for (int hz = 0; hz < nZ; hz++)
                    {
                        for (int hy = 0; hy < nY; hy++)
                        {
                            for (int hx = 0; hx < nX; hx++)
                            {
                                T ax = xmin + hx * (xmax - xmin) / nX;
                                T bx = xmin + (hx + 1) * (xmax - xmin) / nX;
                                T ay = ymin + hy * (ymax - ymin) / nY;
                                T by = ymin + (hy + 1) * (ymax - ymin) / nY;
                                T az = zmin + hz * (zmax - zmin) / nZ;
                                T bz = zmin + (hz + 1) * (zmax - zmin) / nZ;

                                size_t i = hz * nX * nY + hy * nX + hx;

                                nsplits += cells[i]->globalRebalance(ax, bx, ay, by, az, bz, lvl + 1);
                                this->globalNodeCount += cells[i]->globalNodeCount;
                            }
                        }
                    }
                }
            }
            else if ((int)cells.size() == 0)
            {
                if (globalParticleCount > maxGlobalBucketSize)
                {
                    makeSubCells();
                    nsplits += ncells;
                    for (int i = 0; i < ncells; i++)
                    {
                        // cells[i]->approximateRec(cellList[i], x, y, z, h);
                        cells[i]->globalNodeCount = 1;
                        cells[i]->global = true;
                        this->globalNodeCount += cells[i]->globalNodeCount;
                    }
                }
                else
                    this->globalNodeCount = 1;
            }
        }

        this->globalParticleCount = 0;

        return nsplits;
    }
    void buildGlobalTreeAndGlobalCountAndGlobalMaxHRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y,
                                                       const std::vector<T> &z, const std::vector<T> &h, std::vector<int> &ordering,
                                                       std::vector<size_t> &globalParticleCount, std::vector<T> &globalMaxH,
                                                       int padding = 0, int ptri = 0)
    {
        this->localPadding = padding;
        this->localParticleCount = 0;
        this->globalParticleCount = 0;
        this->globalMaxH = 0.0;

        int it = ptri;

        ptri++;

        std::vector<std::vector<int>> cellList(ncells);
        distributeParticles(list, x, y, z, cellList);

        // if(comm_rank ==1)
        // {
        //     if(list.size() == 500000)
        //     {
        //         printf("%f %f %f %f %f %f\n", xmin, xmax, ymin, ymax, zmin, zmax);
        //         for(int i=0; i<10; i++)
        //             printf("%d %f %f %f\n", list[i], x[list[i]], y[list[i]], z[list[i]]);
        //     }

        //     printf("%d (%lu): ", it, list.size());
        //     for(int i=0; i<cellList.size(); i++)
        //         printf("%lu ", cellList[i].size());
        //     printf("\n\n");
        // }

        if ((int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
            {
                cells[i]->buildGlobalTreeAndGlobalCountAndGlobalMaxHRec(cellList[i], x, y, z, h, ordering, globalParticleCount, globalMaxH,
                                                                        padding, ptri);
                this->localParticleCount += cells[i]->localParticleCount;
                this->globalMaxH = std::max(this->globalMaxH, cells[i]->globalMaxH);
                padding += cells[i]->localParticleCount;
                ptri += cells[i]->globalNodeCount;
            }
        }
        else
        {
            for (int i = 0; i < (int)list.size(); i++)
            {
                ordering[padding + i] = list[i];
                if (h[list[i]] > this->globalMaxH) this->globalMaxH = h[list[i]];
            }

            this->localParticleCount = list.size();
        }

        // if(comm_rank == 1 && list.size() == 500000)
        //     printf("%d Wrong\n", it);

        globalMaxH[it] = this->globalMaxH;
        globalParticleCount[it] = this->localParticleCount;
    }

#ifdef USE_MPI
    void buildGlobalTreeAndGlobalCountAndGlobalMaxH(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y,
                                                    const std::vector<T> &z, const std::vector<T> &h, std::vector<int> &ordering)
    {
        std::vector<size_t> globalParticleCount(globalNodeCount, 0);
        std::vector<T> globalMaxH(globalNodeCount, 0.0);

        buildGlobalTreeAndGlobalCountAndGlobalMaxHRec(list, x, y, z, h, ordering, globalParticleCount, globalMaxH);

        MPI_Allreduce(MPI_IN_PLACE, &globalParticleCount[0], globalNodeCount, MPI_SIZE_T, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &globalMaxH[0], globalNodeCount, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        setParticleCountPerNode(globalParticleCount);
        setMaxHPerNode(globalMaxH);
    }
#endif
    void buildTreeWithHalosRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                               std::vector<int> &ordering, int padding = 0)
    {
        this->localPadding = padding;
        this->localParticleCount = 0; // this->globalParticleCount;

        if (assignee == -1 || assignee == comm_rank || halo == true)
        {
            std::vector<std::vector<int>> cellList(ncells);
            distributeParticles(list, x, y, z, cellList);

            if ((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->buildTreeWithHalosRec(cellList[i], x, y, z, ordering, padding);
                    this->localParticleCount += cells[i]->localParticleCount;
                    padding += cells[i]->localParticleCount;
                }
            }
            else if (assignee == comm_rank || halo == true)
            {
                // If this is a halo node then we may not have all the particles yet
                // But we know how much space to reserve!
                this->localParticleCount = this->globalParticleCount;
                for (int i = 0; i < (int)list.size(); i++)
                    ordering[padding + i] = list[i];
            }
        }
    }

    void buildTreeWithHalos(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                            std::vector<int> &ordering)
    {
        buildTreeWithHalosRec(list, x, y, z, ordering);
    }

    virtual void buildTreeRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                              const std::vector<T> &m, std::vector<int> &ordering, int padding = 0)
    {
        this->localPadding = padding;
        this->localParticleCount = list.size();

        std::vector<std::vector<int>> cellList(ncells);
        distributeParticles(list, x, y, z, cellList);
        if ((int)cells.size() == 0 && list.size() > bucketSize) makeSubCells();

        if (!global && assignee == -1) assignee = comm_rank;

        if ((int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
            {
                if (list.size() < noParticlesThatPreventParallelTaskCreation)
                {
                    cells[i]->buildTreeRec(cellList[i], x, y, z, m, ordering, padding);
                    padding += cellList[i].size();
                }
                else
                {
#pragma omp task shared(cellList, x, y, z, m, ordering) firstprivate(padding)
                    cells[i]->buildTreeRec(cellList[i], x, y, z, m, ordering, padding);
                    padding += cellList[i].size();
                }
            }
#pragma omp taskwait
        }
        else
        {
            for (int i = 0; i < (int)list.size(); i++)
                ordering[padding + i] = list[i];
        }
    }

    void buildTree(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                   const std::vector<T> &m, std::vector<int> &ordering)
    {
#pragma omp parallel
#pragma omp single
        buildTreeRec(list, x, y, z, m, ordering);
    }

    void assignProcessesRec(std::vector<size_t> &work, size_t &pi)
    {
        if (work[pi] <= 0 && pi + 1 < work.size()) pi++;

        // If the node fits on process pi, we assign it to this branch
        if (globalParticleCount <= work[pi] || pi + 1 == work.size()) assignee = pi;
        if ((int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
                cells[i]->assignProcessesRec(work, pi);
        }
        else
        {
            // Else if we are a leaf and it does not fit on process pi, we go to the next process
            if (globalParticleCount > work[pi] && pi + 1 < work.size())
            {
                pi++;
                // It may not fit
                // If that happens, it means the tree is not well balanced
                // Perhaps increase the sample size?
                if (globalParticleCount > work[pi])
                {
                    printf("Node has %lu particles > assignee %lu which has max %lu work\n", globalParticleCount, pi, work[pi]);
                    printf("Increase sample size?\n");
                }
            }

            assignee = pi;
            work[pi] -= globalParticleCount;
        }
    }

    void assignProcesses(const std::vector<size_t> &work, std::vector<size_t> &work_remaining)
    {
        size_t pi = 0;
        work_remaining = work;
        assignProcessesRec(work_remaining, pi);
    }

    void syncRec(std::unordered_map<int, std::vector<int>> &toSendCellsPadding, std::unordered_map<int, std::vector<int>> &toSendCellsCount,
                 int &needed)
    {
        if (global)
        {
            // The cell is ours
            if (assignee == comm_rank) { needed += globalParticleCount - localParticleCount; }
            // The cell is associated to a process but is not ours
            else if (assignee >= 0 && localParticleCount > 0)
            {
                toSendCellsPadding[assignee].push_back(localPadding);
                toSendCellsCount[assignee].push_back(localParticleCount);
            }
            // The cell is not associated. If it is not a leaf then
            else if ((int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->syncRec(toSendCellsPadding, toSendCellsCount, needed);
                }
            }
        }
    }

    inline bool overlap(Octree *a)
    {
        T radius = a->globalMaxH * 2.0;

        // Check if Box1's max is greater than Box2's min and Box1's min is less than Box2's max
        return (a->xmax + radius > xmin && a->xmin - radius < xmax && a->ymax + radius > ymin && a->ymin - radius < ymax &&
                a->zmax + radius > zmin && a->zmin - radius < zmax);
    }

    int findHalosList(Octree *a, std::unordered_map<int, std::unordered_map<int, Octree<T> *>> &toSendHalos, int ptri = 0)
    {
        int haloCount = 0;

        if (global)
        {
            ptri++;

            // If this node is a halo (i.e. overlap) node a
            if (globalParticleCount > 0)
            {
                if (assignee == -1 && (int)cells.size() == ncells && overlap(a))
                {
                    for (int i = 0; i < ncells; i++)
                    {
                        haloCount += cells[i]->findHalosList(a, toSendHalos, ptri);
                        ptri += cells[i]->globalNodeCount;
                    }
                }
                else if (assignee != a->assignee && (assignee == comm_rank || a->assignee == comm_rank) && overlap(a))
                {
                    if (a->assignee == comm_rank) halo = true;

                    if ((int)cells.size() == ncells)
                    {
                        for (int i = 0; i < ncells; i++)
                        {
                            haloCount += cells[i]->findHalosList(a, toSendHalos, ptri);
                            ptri += cells[i]->globalNodeCount;
                        }
                    }
                    else
                    {
                        if (toSendHalos[a->assignee].count(ptri) == 0)
                        {
                            if (a->assignee == comm_rank) { haloCount += globalParticleCount; }
                            toSendHalos[a->assignee][ptri] = this;
                        }
                    }
                }
            }
        }

        return haloCount;
    }

    int findHalosRec(Octree *root, std::unordered_map<int, std::unordered_map<int, Octree<T> *>> &toSendHalos, bool PBCx, bool PBCy,
                     bool PBCz)
    {
        int haloCount = 0;

        if (global)
        {
            if (assignee == -1 && (int)cells.size() == ncells)
            {
                for (int i = 0; i < ncells; i++)
                {
                    haloCount += cells[i]->findHalosRec(root, toSendHalos, PBCx, PBCy, PBCz);
                }
            }
            else if (assignee >= 0)
            {
                // Find halos from the root
                // haloCount += root->findHalosList(this, toSendHalos);
                T oldxmin = xmin, oldxmax = xmax;
                T oldymin = ymin, oldymax = ymax;
                T oldzmin = zmin, oldzmax = zmax;

                // Find halos from the root
                int mix = (int)floor(normalize(xmin - 2 * globalMaxH, root->xmin, root->xmax) * nX);
                int miy = (int)floor(normalize(ymin - 2 * globalMaxH, root->ymin, root->ymax) * nY);
                int miz = (int)floor(normalize(zmin - 2 * globalMaxH, root->zmin, root->zmax) * nZ);
                int max = (int)floor(normalize(xmax + 2 * globalMaxH, root->xmin, root->xmax) * nX);
                int may = (int)floor(normalize(ymax + 2 * globalMaxH, root->ymin, root->ymax) * nY);
                int maz = (int)floor(normalize(zmax + 2 * globalMaxH, root->zmin, root->zmax) * nZ);

                if (!PBCx) mix = std::max(mix, 0);
                if (!PBCy) miy = std::max(miy, 0);
                if (!PBCz) miz = std::max(miz, 0);
                if (!PBCx) max = std::min(max, nX - 1);
                if (!PBCy) may = std::min(may, nY - 1);
                if (!PBCz) maz = std::min(maz, nZ - 1);

                for (int hz = miz; hz <= maz; hz++)
                {
                    for (int hy = miy; hy <= may; hy++)
                    {
                        for (int hx = mix; hx <= max; hx++)
                        {
                            T displz = PBCz ? ((hz < 0) - (hz >= nZ)) * (root->zmax - root->zmin) : 0;
                            T disply = PBCy ? ((hy < 0) - (hy >= nY)) * (root->ymax - root->ymin) : 0;
                            T displx = PBCx ? ((hx < 0) - (hx >= nX)) * (root->xmax - root->xmin) : 0;

                            xmin = xmin + displx;
                            xmax = xmax + displx;
                            ymin = ymin + disply;
                            ymax = ymax + disply;
                            zmin = zmin + displz;
                            zmax = zmax + displz;

                            /*
                             * TEMP HACK TO MAKE EVRARD COLLAPSE WORK WITH MPI WITHOUT REMOTE GRAVITY CALCULATIONS
                             * uncomment lines below and comment out remote gravity calculations in evrard.cpp
                             * It's slow like hell but it's working.
                             */
                            // const auto tmpH = this->globalMaxH;
                            // this->globalMaxH = 1000;
                            haloCount += root->findHalosList(this, toSendHalos);
                            // this->globalMaxH = tmpH;

                            xmin = oldxmin;
                            xmax = oldxmax;
                            ymin = oldymin;
                            ymax = oldymax;
                            zmin = oldzmin;
                            zmax = oldzmax;
                        }
                    }
                }
            }
        }

        return haloCount;
    }

    int findHalos(std::unordered_map<int, std::unordered_map<int, Octree<T> *>> &toSendHalos, bool PBCx, bool PBCy, bool PBCz)
    {
        toSendHalos.clear();
        return findHalosRec(this, toSendHalos, PBCx, PBCy, PBCz);
    }

    void writeTree(FILE *fout)
    {
        fprintf(fout, "%f %f %f %f %f %f\n", xmin, xmax, ymin, ymax, zmin, zmax);

        if ((global && (assignee == comm_rank || assignee == -1)) && (int)cells.size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
            {
                cells[i]->writeTree(fout);
            }
        }
    }

    void getParticleCountPerNode(std::vector<size_t> &particleCount, int ptri = 0)
    {
        if (global)
        {
            particleCount[ptri] = this->localParticleCount;

            if ((int)cells.size() == ncells)
            {
                ptri++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->getParticleCountPerNode(particleCount, ptri);
                    ptri += cells[i]->globalNodeCount;
                }
            }
        }
    }

    void setParticleCountPerNode(const std::vector<size_t> &particleCount, int ptri = 0)
    {
        if (global)
        {
            this->globalParticleCount = particleCount[ptri];

            if ((int)cells.size() == ncells)
            {
                ptri++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->setParticleCountPerNode(particleCount, ptri);
                    ptri += cells[i]->globalNodeCount;
                }
            }
        }
    }

    void setMaxHPerNode(const std::vector<double> &hmax, int ptri = 0)
    {
        if (global)
        {
            this->globalMaxH = hmax[ptri];

            if ((int)cells.size() == ncells)
            {
                ptri++;
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->setMaxHPerNode(hmax, ptri);
                    ptri += cells[i]->globalNodeCount;
                }
            }
        }
    }

    void mapListRec(std::vector<int> &clist, int &it)
    {
        if ((int)cells.size() == ncells)
        {
            if (assignee == -1 || assignee == comm_rank)
            {
                for (int i = 0; i < ncells; i++)
                {
                    cells[i]->mapListRec(clist, it);
                }
            }
        }
        else
        {
            if (assignee == comm_rank && localParticleCount > 0)
            {
                for (int i = 0; i < localParticleCount; i++)
                    clist[it++] = localPadding + i;
            }
        }
    }

    void mapList(std::vector<int> &clist)
    {
        int it = 0;
        mapListRec(clist, it);
    }
};

template <typename T>
size_t Octree<T>::bucketSize;
template <typename T>
size_t Octree<T>::minGlobalBucketSize;
template <typename T>
size_t Octree<T>::maxGlobalBucketSize;

template <typename T>
struct GravityOctree : Octree<T>
{
    GravityOctree(const T xmin, const T xmax, const T ymin, const T ymax, const T zmin, const T zmax, const int comm_rank,
                  const int comm_size)
        : Octree<T>(xmin, xmax, ymin, ymax, zmin, zmax, comm_rank, comm_size)
    {
    }

    GravityOctree()
        : Octree<T>()
    {
    }

    ~GravityOctree() = default;

    void gravityBuildTree(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                          const std::vector<T> &m, bool withGravitySync = false)
    {
        particleIdxList = list;
        calcGeometricalCenter();
        dx = this->xmax - this->xmin;

        zeroGravValues();

        for (const auto i : list)
        {
            const T xx = x[i];
            const T yy = y[i];
            const T zz = z[i];

            const T m_i = m[i];

            xcm += xx * m_i;
            ycm += yy * m_i;
            zcm += zz * m_i;

            mTot += m_i;

            const T rx = xx - xce;
            const T ry = yy - yce;
            const T rz = zz - zce;

            qxxa += rx * rx * m_i;
            qxya += rx * ry * m_i;
            qxza += rx * rz * m_i;
            qyya += ry * ry * m_i;
            qyza += ry * rz * m_i;
            qzza += rz * rz * m_i;
        }

        if (withGravitySync) gatherGravValues();

        const size_t noParticles = list.size();
        if (noParticles > 1 || this->global)
        {
            xcm /= mTot;
            ycm /= mTot;
            zcm /= mTot;

            const T rx = xce - xcm;
            const T ry = yce - ycm;
            const T rz = zce - zcm;
            qxx = qxxa - rx * rx * mTot;
            qxy = qxya - rx * ry * mTot;
            qxz = qxza - rx * rz * mTot;
            qyy = qyya - ry * ry * mTot;
            qyz = qyza - ry * rz * mTot;
            qzz = qzza - rz * rz * mTot;

            trq = qxx + qyy + qzz;
        }
        else if (noParticles == 1)
        {
            // save the future particleIdx after reordering
            particleIdx = this->localPadding;
            const auto idx = list.front();

            xcm = x[idx];
            ycm = y[idx];
            zcm = z[idx];

            xce = x[idx];
            yce = y[idx];
            zce = z[idx];

            qxx = 0;
            qxy = 0;
            qxz = 0;
            qyy = 0;
            qyz = 0;
            qzz = 0;

            trq = 0;
            dx = 0; // used to indicate that node is a leaf
        }
    }

    inline void calcGeometricalCenter()
    {
        xce = (this->xmin + this->xmax) / 2.0;
        yce = (this->ymin + this->ymax) / 2.0;
        zce = (this->zmin + this->zmax) / 2.0;
    }

    void buildTreeRec(const std::vector<int> &list, const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                      const std::vector<T> &m, std::vector<int> &ordering, int padding = 0) override
    {
        this->localPadding = padding;
        this->localParticleCount = list.size();

        gravityBuildTree(list, x, y, z, m);

        std::vector<std::vector<int>> cellList(Octree<T>::ncells);
        this->distributeParticles(list, x, y, z, cellList);

        if ((int)this->cells.size() == 0 && list.size() > this->bucketSize) this->makeSubCells();

        if (!this->global && this->assignee == -1) this->assignee = this->comm_rank;

        if ((int)this->cells.size() == this->ncells)
        {
            for (int i = 0; i < this->ncells; i++)
            {
                if (list.size() < this->noParticlesThatPreventParallelTaskCreation)
                {
                    this->cells[i]->buildTreeRec(cellList[i], x, y, z, m, ordering, padding);
                    padding += cellList[i].size();
                }
                else
                {
#pragma omp task shared(cellList, x, y, z, m, ordering) firstprivate(padding)
                    this->cells[i]->buildTreeRec(cellList[i], x, y, z, m, ordering, padding);
                    padding += cellList[i].size();
                }
            }
#pragma omp taskwait
        }
        else
        {
            for (int i = 0; i < (int)list.size(); i++)
                ordering[padding + i] = list[i];
        }
    }

    void makeSubCells() override
    {
        this->cells.resize(this->ncells);

        for (int hz = 0; hz < this->nZ; hz++)
        {
            for (int hy = 0; hy < this->nY; hy++)
            {
                for (int hx = 0; hx < this->nX; hx++)
                {
                    T ax = this->xmin + hx * (this->xmax - this->xmin) / this->nX;
                    T bx = this->xmin + (hx + 1) * (this->xmax - this->xmin) / this->nX;
                    T ay = this->ymin + hy * (this->ymax - this->ymin) / this->nY;
                    T by = this->ymin + (hy + 1) * (this->ymax - this->ymin) / this->nY;
                    T az = this->zmin + hz * (this->zmax - this->zmin) / this->nZ;
                    T bz = this->zmin + (hz + 1) * (this->zmax - this->zmin) / this->nZ;

                    unsigned int i = hz * this->nX * this->nY + hy * this->nX + hx;

                    if (this->cells[i] == nullptr)
                        this->cells[i] = std::make_shared<GravityOctree>(ax, bx, ay, by, az, bz, this->comm_rank, this->comm_size);
                }
            }
        }
    }
    using Octree<T>::print;

    void buildGlobalGravityTree(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m)
    {
        zeroGravValues();
        for (const auto &cell : this->cells)
        {
            asGravityOctree(cell)->zeroGravityValuesInHaloNodesRec();
        }

        buildGlobalParticleListRec();
        for (const auto &cell : this->cells)
        {
            asGravityOctree(cell)->buildGlobalParticleListRec();
        }

        for (const auto &cell : this->cells)
        {
            asGravityOctree(cell)->buildGlobalGravityTreeRec(x, y, z, m);
        }
    }

    void buildGlobalGravityTreeRec(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z, const std::vector<T> &m)
    {
        if (this->global)
        {
            const bool withGravitySync = true;
            gravityBuildTree(globalParticleIdxList, x, y, z, m, withGravitySync);

            for (int i = 0; i < this->ncells; i++)
            {
                if ((int)this->cells.size() == this->ncells) { asGravityOctree(this->cells[i])->buildGlobalGravityTreeRec(x, y, z, m); }
            }
        }
    }

    bool doAllChildCellsBelongToGlobalTree()
    {
        return (int)this->cells.size() == this->ncells &&
               std::all_of(this->cells.begin(), this->cells.end(), [](const auto &t) { return t->global; });
    }

    void zeroGravityValuesInHaloNodesRec()
    {

        if (doAllChildCellsBelongToGlobalTree())
        {
            for (const auto &cell : this->cells)
            {
                asGravityOctree(cell)->zeroGravityValuesInHaloNodesRec();
            }
        } // traverse bottom up

        pcount = particleIdxList.size();
        globalParticleIdxList = particleIdxList;

        if (this->halo)
        {
            zeroGravValues();
            pcount = 0;
            globalParticleIdxList.clear();
        }
    }

    void buildGlobalParticleListRec()
    {

        if (doAllChildCellsBelongToGlobalTree())
        {
            pcount = 0;
            globalParticleIdxList.clear();

            const size_t accSize =
                std::accumulate(this->cells.begin(), this->cells.end(), size_t(0), [this](const size_t acc, const auto &el) {
                    return asGravityOctree(el)->globalParticleIdxList.size() + acc;
                });
            globalParticleIdxList.reserve(accSize);
            pcount = accSize;

            for (const auto &cell : this->cells)
            {
                GravityOctree<T> *gptr = asGravityOctree(cell);
                std::copy(gptr->globalParticleIdxList.begin(), gptr->globalParticleIdxList.end(),
                          std::back_inserter(globalParticleIdxList));
            }
            for (const auto &cell : this->cells)
            {
                asGravityOctree(cell)->buildGlobalParticleListRec();
            }
        }
    }

    void zeroGravValues()
    {
        qxxa = 0.0, qxya = 0.0, qxza = 0.0;
        qyya = 0.0, qyza = 0.0;
        qzza = 0.0;
        xcm = 0.0;
        ycm = 0.0;
        zcm = 0.0;
        mTot = 0.0;

        qxx = 0.0, qxy = 0.0, qxz = 0.0;
        qyy = 0.0, qyz = 0.0;
        qzz = 0.0;
        trq = 0.0;
    }

    void print(int l = 0)
    {
        if (this->global)
        {
            for (int i = 0; i < l; i++)
                printf("   ");
            printf("[%d] lp:%d gp:%d gn:%d h:%d mTot:%.15f qxx:%.15f trq:%.15f ycm:%.15f pcount=%d, globalParticleIdxList.size=%lu\n",
                   this->assignee, this->localParticleCount, this->globalParticleCount, this->globalNodeCount, this->halo, mTot, qxx, trq,
                   ycm, pcount, globalParticleIdxList.size());

            if ((int)this->cells.size() == this->ncells)
            {
                for (int i = 0; i < this->ncells; i++)
                {
                    asGravityOctree(this->cells[i])->print(l + 1);
                }
            }
        }
    }

    inline GravityOctree<T> *asGravityOctree(const std::shared_ptr<Octree<T>> &octree)
    {
        return dynamic_cast<GravityOctree<T> *>(octree.get());
    }

    void gatherGravValues()
    {
#ifdef USE_MPI
        // if (this->global)
        if (this->global && this->assignee == -1)
        {
            //	  printf("[%d] MPI_Allreduce in tree node with %lu particles and mTot=%.15f\n", this->comm_rank, pcount, mTot);
            MPI_Allreduce(MPI_IN_PLACE, &mTot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, &xcm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &ycm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &zcm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            MPI_Allreduce(MPI_IN_PLACE, &qxxa, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qxya, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qxza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qyya, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qyza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &qzza, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
#endif
    }

    T mTot = 0.0;
    T xce, yce, zce;
    T xcm = 0.0, ycm = 0.0, zcm = 0.0;

    T qxx, qxy, qxz;
    T qyy, qyz;
    T qzz;

    T qxxa = 0.0, qxya = 0.0, qxza = 0.0;
    T qyya = 0.0, qyza = 0.0;
    T qzza = 0.0;

    T trq;
    int pcount = 0;

    std::vector<int> particleIdxList;
    std::vector<int> globalParticleIdxList;

    T dx;                // side of a cell;
    int particleIdx = 0; // filled only if node is a leaf
};

} // namespace sphexa
