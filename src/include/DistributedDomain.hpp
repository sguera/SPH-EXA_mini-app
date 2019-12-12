#pragma once

#include <unordered_map>
#include <map>

#include "Domain.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif

namespace sphexa
{

template <typename T, class Dataset>
class DistributedDomain : public Domain<T, Dataset>
{
public:
    DistributedDomain()
    {
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        MPI_Get_processor_name(processor_name, &name_len);
    }

    ~DistributedDomain() override = default;

    /*  Send the particles from the old arrays either to the new arrays or to another process
        Basically collects a map[process][nodes] to send to other processes
        Then it knows how many particle it is missing (particleCount - localParticleCount)
        And just loop receive until we have received all the missing particles from other processes */
    void sync(const std::vector<std::vector<T> *> &arrayList)
    {
        std::unordered_map<int, std::vector<int>> toSendCellsPadding, toSendCellsCount;
        std::vector<std::vector<T>> buff;
        std::vector<MPI_Request> requests;
        std::vector<int> particleCounts(comm_size);

        int needed = 0;

        Domain<T, Dataset>::octree.syncRec(toSendCellsPadding, toSendCellsCount, needed);

        for (int rank = 0; rank < comm_size; rank++)
        {
            if (toSendCellsCount[rank].size() > 0)
            {
                int rcount = requests.size();
                particleCounts[rank] = std::accumulate(toSendCellsCount[rank].begin(), toSendCellsCount[rank].end(), 0);

                requests.resize(rcount + arrayList.size() + 2);

                MPI_Isend(&comm_rank, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &requests[rcount]);
                MPI_Isend(&particleCounts[rank], 1, MPI_INT, rank, 1, MPI_COMM_WORLD, &requests[rcount + 1]);

                for (unsigned int i = 0; i < arrayList.size(); i++)
                {
                    std::vector<T> localBuffer;
                    localBuffer.reserve(particleCounts[rank]);

                    // We go over every tree nodes to send for this rank
                    for (unsigned int j = 0; j < toSendCellsCount[rank].size(); j++)
                    {
                        int padding = toSendCellsPadding[rank][j];
                        int count = toSendCellsCount[rank][j];

                        std::copy((*arrayList[i]).begin() + padding, (*arrayList[i]).begin() + padding + count,
                                  std::back_inserter(localBuffer));
                    }

                    MPI_Isend(localBuffer.data(), particleCounts[rank], MPI_DOUBLE, rank, 2 + i, MPI_COMM_WORLD, &requests[rcount + 2 + i]);
                    buff.emplace_back(std::move(localBuffer)); // Note: need to move to keep the buffer valid
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

        int needed = 0;

        std::map<int, std::vector<int>> recvHalos, sendHalos;
        std::map<int, Octree<T>*> recvCellMap, sendCellMap;
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
                    recvHalos[from].push_back(ptri);
                    recvCellMap[ptri] = cell;
                    recvCount[from] += cell->globalParticleCount;
                    needed += cell->globalParticleCount;
                }
                else
                {
                    sendHalos[to].push_back(ptri);
                    sendCellMap[ptri] = cell;
                    sendCount[to] += cell->globalParticleCount;
                }
            }
        }

        std::vector<MPI_Request> requests;

        // Fill Buffers
        std::vector<std::vector<std::vector<T>>> sendBuffers;
        for (auto& procPtriVec : sendHalos)
        {
            int to = procPtriVec.first; 

            std::vector<std::vector<T>> procBuff(data.size());
            for (auto& vec : procBuff) vec.reserve(sendCount[to]);

            for (const int& ptri : procPtriVec.second)
            {
                Octree<T> *cell = sendCellMap[ptri];

                int cellCount = cell->globalParticleCount;
                int padding = cell->localPadding;

                for (size_t i = 0; i < data.size(); i++)
                {
                    auto src = (*data[i]).begin() + padding;
                    std::copy(src, src + cellCount, std::back_inserter(procBuff[i]));
                }
            }

            // send
            requests.emplace_back(MPI_Request());
            MPI_Isend(&comm_rank, 1, MPI_INT, to, tag + data.size() + 0, MPI_COMM_WORLD, &(*requests.rbegin()));

            for (size_t i = 0; i < data.size(); i++)
            {
                requests.emplace_back(MPI_Request());
                MPI_Isend(procBuff[i].data(), procBuff[i].size(), MPI_DOUBLE, to, tag + data.size() + 1 + i, MPI_COMM_WORLD,
                          &(*requests.rbegin()));
            }

            sendBuffers.emplace_back(std::move(procBuff)); // Note: need to move to keep buffer valid
        }


        std::vector<int> fromIdx;
        std::vector<std::vector<std::vector<T>>> recvBuffs;
        while (needed > 0)
        {
            MPI_Status status[data.size() + 1];

            int from = -1;
            MPI_Recv(&from, 1, MPI_INT, MPI_ANY_SOURCE, tag + data.size() + 0, MPI_COMM_WORLD, &status[0]);
            fromIdx.push_back(from);

            int count = recvCount[from];
            recvBuffs.emplace_back(std::vector<std::vector<T>>(data.size(), std::vector<T>(count)));

            for (unsigned int i = 0; i < data.size(); i++)
            {
                MPI_Recv((*recvBuffs.rbegin())[i].data(), count, MPI_DOUBLE, from, tag + data.size() + 1 + i, MPI_COMM_WORLD,
                         &status[1 + i]);
            }

            needed -= count;
        }


        for (unsigned int bi = 0; bi < recvBuffs.size(); bi++)
        {
            std::vector<int> &ptriBuff = recvHalos[fromIdx[bi]];
            std::vector<std::vector<T>> &recvBuff = recvBuffs[bi];

            int current = 0;
            for (const int &ptri : ptriBuff)
            {
                Octree<T> *cell = recvCellMap[ptri];

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

    void create(Dataset &d) override
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

        std::vector<int> &clist = Domain<T, Dataset>::clist;

        clist.resize(n);

        #pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            clist[i] = i;

        d.bbox.computeGlobal(clist, d.x, d.y, d.z);

        Octree<T> &octree = Domain<T, Dataset>::octree;

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
        Domain<T, Dataset>::buildTree(d);

        // build the global tree using only the particles that belong to us
        octree.globalRebalance(d.bbox.xmin, d.bbox.xmax, d.bbox.ymin, d.bbox.ymax, d.bbox.zmin, d.bbox.zmax);
        octree.buildGlobalTreeAndGlobalCountAndGlobalMaxH(clist, x, y, z, h, ordering);
        octree.assignProcesses(work, work_remaining);

        // Get rid of particles that do not belong to us
        reorder(ordering, d);

        #pragma omp parallel for
        for (size_t i = 0; i < workAssigned; i++)
            clist[i] = i;
        
        d.resize(workAssigned);

        workAssigned = work[comm_rank] - work_remaining[comm_rank];

        sync(d.data);

        d.count = workAssigned;
    }

    void update(Dataset &d) override
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

        std::vector<int> &clist = Domain<T, Dataset>::clist;
        Octree<T> &octree = Domain<T, Dataset>::octree;

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

    int comm_size, comm_rank, name_len;
    char processor_name[256];

    std::unordered_map<int, std::unordered_map<int, Octree<T> *>> toSendHalos;

    size_t haloCount = 0;
    size_t workAssigned = 0;
};

}
