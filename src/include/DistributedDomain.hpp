#pragma once

#ifdef USE_MPI
#include "mpi.h"
#endif

#include <vector>
#include <cmath>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>

#include <unistd.h>
#include <chrono>
#include <ctime>

#include "Octree.hpp"
#include "Task.hpp"

namespace sphexa
{

template <typename T>
class DistributedDomain
{
public:
    DistributedDomain()
    {
#ifdef USE_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Get_processor_name(processor_name, &name_len);
#endif
    }

    template <class Dataset>
    void findNeighborsImpl(Task &t, Dataset &d)
    {
        size_t  n = t.clist.size();

#pragma omp parallel for schedule(guided)
        for (size_t pi = 0; pi < n; pi++)
        {
            int i = t.clist[pi];

            t.neighborsCount[pi] = 0;
            octree.findNeighbors(i, &d.x[0], &d.y[0], &d.z[0], d.x[i], d.y[i], d.z[i], 2.0 * d.h[i], t.ngmax, &t.neighbors[pi * t.ngmax],
                                 t.neighborsCount[pi], d.bbox.PBCx, d.bbox.PBCy, d.bbox.PBCz);

#ifndef NDEBUG
            if (t.neighborsCount[pi] == 0)
                printf("ERROR::FindNeighbors(%d) x %f y %f z %f h = %f ngi %d\n", i, d.x[i], d.y[i], d.z[i], d.h[i], t.neighborsCount[pi]);
#endif
        }
    }

    template <class Dataset>
    void findNeighbors(std::vector<Task> &taskList, Dataset &d)
    {
        for (auto &task : taskList)
        {
            findNeighborsImpl(task, d);
        }
    }

    size_t  neighborsSumImpl(const Task &t)
    {
        size_t  sum = 0;
#pragma omp parallel for reduction(+ : sum)
        for (unsigned int i = 0; i < t.clist.size(); i++)
            sum += t.neighborsCount[i];

        return sum;
    }

    size_t  neighborsSum(const std::vector<Task> &taskList)
    {
        size_t  sum = 0;
        for (const auto &task : taskList)
        {
            sum += neighborsSumImpl(task);
        }

#ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
#endif

        return sum;
    }

    inline T normalize(T d, T min, T max) { return (d - min) / (max - min); }

    void reorderSwap(const std::vector<int> &ordering, std::vector<T> &arrayList)
    {
        std::vector<T> tmp(ordering.size());
        for (unsigned int i = 0; i < ordering.size(); i++)
            tmp[i] = arrayList[ordering[i]];
        tmp.swap(arrayList);
    }

    void reorder(const std::vector<int> &ordering, std::vector<std::vector<T> *> &arrayList)
    {
        for (unsigned int i = 0; i < arrayList.size(); i++)
            reorderSwap(ordering, *arrayList[i]);
    }

    template <class Dataset>
    void reorder(const std::vector<int> &ordering, Dataset &d)
    {
        reorder(ordering, d.data);
    }

    void makeDataArray(std::vector<std::vector<T> *> &data, std::vector<T> *d) { data.push_back(d); }

    template <typename... Args>
    void makeDataArray(std::vector<std::vector<T> *> &data, std::vector<T> *first, Args... args)
    {
        data.push_back(first);
        makeDataArray(data, args...);
    }

#ifdef USE_MPI
    /*  Send the particles from the old arrays either to the new arrays or to another process
        Basically collects a map[process][nodes] to send to other processes
        Then it knows how many particle it is missing (particleCount - localParticleCount)
        And just loop receive until we have received all the missing particles from other processes */
    void sync(const std::vector<std::vector<T> *> &arrayList)
    {
        std::map<int, std::vector<int>> toSendCellsPadding, toSendCellsCount;
        std::vector<std::vector<T>> buff;
        std::vector<MPI_Request> requests;

        int needed = 0;

        octree.syncRec(toSendCellsPadding, toSendCellsCount, needed);

        for (int rank = 0; rank < comm_size; rank++)
        {
            if (toSendCellsCount[rank].size() > 0)
            {
                int rcount = requests.size();
                int nParticlesToSend = std::accumulate(toSendCellsCount[rank].begin(), toSendCellsCount[rank].end(), 0);

                requests.resize(rcount + arrayList.size() + 2);

                MPI_Isend(&comm_rank, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &requests[rcount]);
                MPI_Isend(&nParticlesToSend, 1, MPI_INT, rank, 1, MPI_COMM_WORLD, &requests[rcount + 1]);

                for (unsigned int i = 0; i < arrayList.size(); i++)
                {
                    std::vector<T> localBuffer;
                    localBuffer.reserve(nParticlesToSend);

                    // We go over every tree nodes to send for this rank
                    for (unsigned int j = 0; j < toSendCellsCount[rank].size(); j++)
                    {
                        int padding = toSendCellsPadding[rank][j];
                        int count = toSendCellsCount[rank][j];

                        std::copy((*arrayList[i]).begin() + padding, (*arrayList[i]).begin() + padding + count, std::back_inserter(localBuffer));
                    }

                    MPI_Isend(localBuffer.data(), nParticlesToSend, MPI_DOUBLE, rank, 2 + i, MPI_COMM_WORLD, &requests[rcount + 2 + i]);
                    buff.emplace_back(std::move(localBuffer));
                }
            }
        }

        // allocate space for the incoming particles
        int end = arrayList[0]->size();
        for (unsigned int i = 0; i < arrayList.size(); i++)
            (*arrayList[i]).resize(end + needed);

        while (needed > 0)
        {
            // printf("Needed: %d\n", needed);
            MPI_Status status[arrayList.size() + 2];

            int rank, count;
            MPI_Recv(&rank, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status[0]);
            MPI_Recv(&count, 1, MPI_INT, rank, 1, MPI_COMM_WORLD, &status[1]);

            for (unsigned int i = 0; i < arrayList.size(); i++)
                MPI_Recv(&(*arrayList[i])[end], count, MPI_DOUBLE, rank, 2 + i, MPI_COMM_WORLD, &status[2 + i]);

            end += count;
            needed -= count;
        }

        if (requests.size() > 0)
        {
            MPI_Status status[requests.size()];
            MPI_Waitall(requests.size(), &requests[0], status);
        }
    }

    template <typename... Args>
    void synchronizeHalos(Args... args)
    {
        std::vector<std::vector<T> *> data;
        makeDataArray(data, args...);
        synchronizeHalos(data);
    }

    void synchronizeHalos(std::vector<std::vector<T> *> &data)
    {
        static unsigned short int tag = 0;

        std::map<int, Octree<T> *> cellMap;

        struct ToSend
        {
            std::vector<int> ptris;
            int ptriCount;

            std::vector<std::vector<T>> buff;
            int count;
        };

        std::unordered_map<int, ToSend> sendMap;
        int needed = 0;

        // Which tree node to send to which processor
        for (auto const &itProc : toSendHalos)
        {
            int to = itProc.first;

            ToSend &_toSend = sendMap[to];

            for (auto const &itNodes : itProc.second)
            {
                int ptri = itNodes.first;

                Octree<T> *cell = itNodes.second;

                int cellCount = cell->globalParticleCount;
                int from = cell->assignee;

                if (to == comm_rank && from != comm_rank)
                    needed += cellCount;
                else if (from == comm_rank && to != comm_rank)
                {
                    _toSend.ptris.push_back(ptri);
                    _toSend.count += cellCount;
                }
            }
        }


        std::map<int, std::vector<int>> reHalos, sendHalos;
        std::map<int, Octree<T>*> reCellMap, sendCellMap;
        std::map<int, int> recvCount, sendCount;

        for (auto const& proc : toSendHalos)
        {
            int to = proc.first;
            for (auto const& halo : proc.second)
            {
                int ptri = halo.first;
                Octree<T>* cell = halo.second;
                int from = cell->assignee;

                if (to == comm_rank)
                {
                    reHalos[from].push_back(ptri);
                    reCellMap[ptri] = cell;
                    recvCount[from] += cell->globalParticleCount;
                }
                else
                {
                    sendHalos[to].push_back(ptri);
                    sendCellMap[ptri] = cell;
                    sendCount[to] = cell->globalParticleCount;
                }
            }
        }


        // Fill buffer
        for (auto &it : sendMap)
        {
            ToSend &_toSend = it.second;
            _toSend.ptriCount = _toSend.ptris.size();
            _toSend.buff.resize(data.size(), std::vector<T>(_toSend.count));

            int current = 0;
            for (const int &ptri : _toSend.ptris)
            {
                Octree<T> *cell = sendCellMap[ptri];

                int cellCount = cell->globalParticleCount;
                int padding = cell->localPadding;

                for (unsigned int i = 0; i < data.size(); i++)
                {
                    T *_buff = &(*data[i])[padding];
                    T *_toSendBuff = &_toSend.buff[i][current];

                    for (int j = 0; j < cellCount; j++)
                        _toSendBuff[j] = _buff[j];
                }

                current += cellCount;
            }
        }

        std::vector<MPI_Request> requests;

        // Send!!
        for (auto &it : sendMap)
        {
            int to = it.first;
            ToSend &_toSend = it.second;

            int rcount = requests.size();
            requests.resize(rcount + data.size() + 1);

            // Send rank
            MPI_Isend(&comm_rank, 1, MPI_INT, to, tag + data.size() + 0, MPI_COMM_WORLD, &requests[rcount]);

            // printf("[%d] send %d to %d\n", comm_rank, toSendCount[to], to); fflush(stdout);
            for (unsigned int i = 0; i < data.size(); i++)
            {
                MPI_Isend(&_toSend.buff[i][0], _toSend.count, MPI_DOUBLE, to, tag + data.size() + 1 + i, MPI_COMM_WORLD,
                          &requests[rcount + 1 + i]);
            }
        }


        std::vector<std::vector<int>> ptriBuffs;
        std::vector<std::vector<std::vector<T>>> recvBuffs;
        while (needed > 0)
        {
            MPI_Status status[data.size() + 1];

            int from = -1;
            MPI_Recv(&from, 1, MPI_INT, MPI_ANY_SOURCE, tag + data.size() + 0, MPI_COMM_WORLD, &status[0]);

            ptriBuffs.push_back(reHalos[from]);
            int count = recvCount[from];

            int recvBuffCount = recvBuffs.size();
            recvBuffs.resize(recvBuffCount + 1);

            std::vector<std::vector<T>> &recvBuff = recvBuffs[recvBuffCount++];
            recvBuff.resize(data.size());

            // printf("[%d] recv %d from %d\n", comm_rank, count, from); fflush(stdout);
            for (unsigned int i = 0; i < data.size(); i++)
            {
                recvBuff[i].resize(count);
                MPI_Recv(&recvBuffs[recvBuffCount - 1][i][0], count, MPI_DOUBLE, from, tag + data.size() + 1 + i, MPI_COMM_WORLD,
                         &status[1 + i]);
            }

            needed -= count;
        }


        for (unsigned int bi = 0; bi < recvBuffs.size(); bi++)
        {
            std::vector<int> &ptriBuff = ptriBuffs[bi];
            std::vector<std::vector<T>> &recvBuff = recvBuffs[bi];

            int current = 0;
            for (const int &ptri : ptriBuff)
            {
                Octree<T> *cell = reCellMap[ptri];

                int cellCount = cell->globalParticleCount;
                int padding = cell->localPadding;

                for (unsigned int i = 0; i < data.size(); i++)
                {
                    T *buff = &(*data[i])[padding];
                    for (int j = 0; j < cellCount; j++)
                        buff[j] = recvBuff[i][current + j];
                }

                current += cellCount;
            }
        }

        tag += data.size() + 1;

        if (requests.size() > 0)
        {
            MPI_Status status[requests.size()];
            MPI_Waitall(requests.size(), &requests[0], &status[0]);
        }
    }

    template <class Dataset>
    void create(Dataset &d)
    {
        const std::vector<T> &x = d.x;
        const std::vector<T> &y = d.y;
        const std::vector<T> &z = d.z;
        const std::vector<T> &h = d.h;

        // Prepare processes
        const size_t n = d.count;
        const size_t ntot = d.n;
        const size_t split = ntot / comm_size;
        const size_t remaining = ntot - comm_size * split;

        clist.resize(n);
        for (size_t i = 0; i < n; i++)
            clist[i] = i;

        d.bbox.computeGlobal(clist, d.x, d.y, d.z);

        octree.cells.clear();
        octree = Octree<T>(d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax, comm_rank, comm_size);

        octree.global = true;
        octree.globalNodeCount = 9;
        octree.globalParticleCount = 0;

        std::vector<int> ordering(n);
        octree.buildGlobalTreeAndGlobalCountAndGlobalMaxH(clist, x, y, z, h, ordering);

        int nsplits = 0;
        do
        {
            nsplits = octree.globalRebalance(d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax);
            octree.buildGlobalTreeAndGlobalCountAndGlobalMaxH(clist, x, y, z, h, ordering);
        } while (nsplits > 0);

        reorder(ordering, d);

        // We then map the tree to processes
        std::vector<size_t> work(comm_size, split);
        work[0] += remaining;

        std::vector<size_t> work_remaining(comm_size);
        octree.assignProcesses(work, work_remaining);

        workAssigned = work[comm_rank] - work_remaining[comm_rank];

        sync(d.data);

        // adjust clist to consider only the particles that belong to us (using the tree)
        clist.resize(workAssigned);
        ordering.resize(workAssigned);
        buildTree(d);

        // build the global tree using only the particles that belong to us
        octree.globalRebalance(d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax);
        octree.buildGlobalTreeAndGlobalCountAndGlobalMaxH(clist, x, y, z, h, ordering);
        octree.assignProcesses(work, work_remaining);

        // Get rid of particles that do not belong to us
        reorder(ordering, d);
        for (size_t i = 0; i < workAssigned; i++)
            clist[i] = i;
        d.resize(workAssigned);

        workAssigned = work[comm_rank] - work_remaining[comm_rank];

        sync(d.data);

        d.count = workAssigned;
    }

    template <class Dataset>
    void distribute(Dataset &d)
    {
        const std::vector<T> &x = d.x;
        const std::vector<T> &y = d.y;
        const std::vector<T> &z = d.z;
        const std::vector<T> &h = d.h;

        // Prepare processes
        const size_t n = d.count;
        const size_t ntot = d.n;
        const size_t split = ntot / comm_size;
        const size_t remaining = ntot - comm_size * split;

        d.bbox.computeGlobal(clist, d.x, d.y, d.z);

        std::vector<size_t> work(comm_size, split);
        work[0] += remaining;

        // We map the nodes to a 1D array and retrieve the order of the particles in the tree
        std::vector<int> ordering(n);

        octree.globalRebalance(d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax);
        octree.buildGlobalTreeAndGlobalCountAndGlobalMaxH(clist, x, y, z, h, ordering);

        // Getting rid of old halos
        reorder(ordering, d);

        // printf("[%d] Global tree nodes: %d\n", comm_rank, octree.globalNodeCount); fflush(stdout);

        // We then map the tree to processes
        std::vector<size_t> work_remaining(comm_size);
        octree.assignProcesses(work, work_remaining);

        // Quick check
        int totalAvail = 0, totalAlloc = 0;
        for (int i = 0; i < (int)work.size(); i++)
        {
            totalAlloc += work[i] - work_remaining[i];
            totalAvail += work[i];
        }

        // printf("[%d] Total Avail: %d, Total Alloc: %d || Got: %d\n", comm_rank, totalAvail, totalAlloc, work[comm_rank] -
        //   work_remaining[comm_rank]); fflush(stdout);

        sync(d.data);

        // printf("[%d] Total number of particles %d (local) %d (global)\n", comm_rank, octree.localParticleCount,
        // octree.globalParticleCount); fflush(stdout);

        // octree.computeGlobalMaxH();
        haloCount = octree.findHalos(toSendHalos, d.bbox.PBCx, d.bbox.PBCy, d.bbox.PBCz);

        workAssigned = work[comm_rank] - work_remaining[comm_rank];

        // int check = workAssigned;
        // MPI_Allreduce(MPI_IN_PLACE, &check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        // if(comm_rank == 0)
        //     printf("CHECK: %d\n", check);

        // printf("[%d] haloCount: %d (%.2f%%)\n", comm_rank, haloCount,
        //         haloCount / (double)(workAssigned) * 100.0); fflush(stdout);

        // Finally remap everything
        ordering.resize(workAssigned + haloCount);
#pragma omp parallel for
        for (size_t i = 0; i < ordering.size(); i++)
            ordering[i] = 0;

        // We map ALL particles
        // Particles that do not belong to us will be ignored in the localMapParticleFunction
        std::vector<int> list(d.x.size());
#pragma omp parallel for
        for (size_t i = 0; i < d.x.size(); i++)
            list[i] = i;

        clist.resize(workAssigned);
#pragma omp parallel for
        for (size_t i = 0; i < workAssigned; i++)
            clist[i] = i;

        octree.buildTreeWithHalos(list, x, y, z, ordering);
        reorder(ordering, d);

        // MUST DO SYNCHALOS AND BUILDTREE AFTER

        d.count = workAssigned;
    }
#else
    template <class Dataset>
    void create(Dataset &d)
    {
        const std::vector<T> &x = d.x;
        const std::vector<T> &y = d.y;
        const std::vector<T> &z = d.z;

        const size_t n = d.count;

        clist.resize(n);
        for (size_t i = 0; i < n; i++)
            clist[i] = i;

        std::vector<int> ordering(n);

        d.bbox.computeGlobal(clist, d.x, d.y, d.z);

        // Each process creates a tree based on the gathered sample
        octree.cells.clear();
        octree = Octree<T>(d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax, 0, 1);
        octree.buildTree(clist, x, y, z, ordering);
        reorder(ordering, d);
    }

    template <class Dataset>
    void distribute(Dataset &d)
    {
        d.bbox.computeGlobal(clist, d.x, d.y, d.z);
        octree.cells.clear();
        octree.xmin = d.bbox.xmin;
        octree.xmax = d.bbox.xmax;
        octree.ymin = d.bbox.ymin;
        octree.ymax = d.bbox.ymax;
        octree.zmin = d.bbox.zmin;
        octree.zmax = d.bbox.zmax;
        return;
    }
    template <typename... Args>
    void synchronizeHalos(Args...)
    {
        return;
    }

    void synchronizeHalos(std::vector<std::vector<T> *> &) { return; }
#endif

    template <class Dataset>
    void buildTree(Dataset &d)
    {
        // Finally remap everything
        std::vector<int> ordering(d.x.size());

        std::vector<int> list(d.x.size());
#pragma omp parallel for
        for (int i = 0; i < (int)d.x.size(); i++)
            list[i] = i;

        // We need this to expand halo
        octree.buildTree(list, d.x, d.y, d.z, ordering);
        reorder(ordering, d);

        octree.mapList(clist);
    }

    template <class Dataset>
    void updateSmoothingLength(Dataset &d)
    {
        const T c0 = 7.0;
        const T exp = 1.0 / 3.0;

        const size_t n = clist.size();
        const int *neighborsCount = d.neighborsCount.data();
        const int ng0 = d.ng0;
        T *h = d.h.data();

#pragma omp parallel for
        for (int pi = 0; pi < n; pi++)
        {
            const int i = clist[pi];
            const int nn = neighborsCount[pi];

            h[i] = h[i] * 0.5 * pow((1.0 + c0 * ng0 / nn), exp);

#ifndef NDEBUG
            if (std::isinf(h[i]) || std::isnan(h[i])) printf("ERROR::h(%d) ngi %d h %f\n", i, nn, h[i]);
#endif
        }
    }

    void createTasks(std::vector<Task> &taskList, const size_t nTasks)
    {
        const int partitionSize = clist.size() / nTasks;
        const int lastPartitionOffset = clist.size() - nTasks * partitionSize;

        taskList.resize(nTasks);

#pragma omp parallel for
        for (size_t i = 0; i < nTasks; ++i)
        {
            const int begin = i * partitionSize;
            const int end = (i + 1) * partitionSize + (i == nTasks - 1 ? lastPartitionOffset : 0);
            const size_t size = end - begin;

            taskList[i].resize(size);
            for (size_t j = 0; j < size; j++)
                taskList[i].clist[j] = clist[j + begin];
        }
    }

    const int local_sample_size = 100;

    int comm_size, comm_rank, name_len;
    char processor_name[256];

    std::map<int, std::map<int, Octree<T> *>> toSendHalos;

    size_t haloCount = 0;
    size_t workAssigned = 0;

    Octree<T> octree;

    std::vector<int> clist;
};

} // namespace sphexa
