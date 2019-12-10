#pragma once

#include <unordered_map>

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

        int needed = 0;

        Domain<T, Dataset>::octree.syncRec(toSendCellsPadding, toSendCellsCount, needed);

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

                        std::copy((*arrayList[i]).begin() + padding, (*arrayList[i]).begin() + padding + count,
                                  std::back_inserter(localBuffer));
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

        // typedef std::chrono::high_resolution_clock Clock;
        // typedef std::chrono::time_point<Clock> TimePoint;
        // typedef std::chrono::duration<float> Time;

        std::unordered_map<int, Octree<T> *> cellMap;

        struct ToSend
        {
            std::vector<int> ptris;
            int ptriCount;

            std::vector<std::vector<T>> buff;
            int count;
        };

        std::unordered_map<int, ToSend> sendMap;
        int needed = 0;

        // MPI_Barrier(MPI_COMM_WORLD);

        // TimePoint tstart = Clock::now();

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

                // store corresponding cell for when recving
                cellMap[ptri] = cell;

                if (to == comm_rank && from != comm_rank)
                    needed += cellCount;
                else if (from == comm_rank && to != comm_rank)
                {
                    _toSend.ptris.push_back(ptri);
                    _toSend.count += cellCount;
                }
            }
        }

        // MPI_Barrier(MPI_COMM_WORLD);

        // TimePoint tstop = Clock::now();
        // float elaspedTime = std::chrono::duration_cast<Time>(tstop-tstart).count();

        // if(comm_rank == 0)
        // {
        //     printf("synchronizeHalos::collect() %f\n", elaspedTime);
        //     fflush(stdout);
        // }

        // tstart = Clock::now();

        // Fill buffer
        for (auto &it : sendMap)
        {
            ToSend &_toSend = it.second;
            _toSend.ptriCount = _toSend.ptris.size();
            _toSend.buff.resize(data.size(), std::vector<T>(_toSend.count));

            int current = 0;
            for (const int &ptri : _toSend.ptris)
            {
                Octree<T> *cell = cellMap[ptri];

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

        // MPI_Barrier(MPI_COMM_WORLD);

        // tstop = Clock::now();
        // elaspedTime = std::chrono::duration_cast<Time>(tstop-tstart).count();

        // if(comm_rank == 0)
        // {
        //     printf("synchronizeHalos::fill() %f\n", elaspedTime);
        //     fflush(stdout);
        // }

        // tstart = Clock::now();

        // Send!!
        for (auto &it : sendMap)
        {
            int to = it.first;
            ToSend &_toSend = it.second;

            int rcount = requests.size();
            requests.resize(rcount + data.size() + 4);

            // Send rank
            MPI_Isend(&comm_rank, 1, MPI_INT, to, tag + data.size() + 0, MPI_COMM_WORLD, &requests[rcount]);

            // Send ptriBuff
            MPI_Isend(&_toSend.ptriCount, 1, MPI_INT, to, tag + data.size() + 1, MPI_COMM_WORLD, &requests[rcount + 1]);
            MPI_Isend(&_toSend.ptris[0], _toSend.ptriCount, MPI_INT, to, tag + data.size() + 2, MPI_COMM_WORLD, &requests[rcount + 2]);

            // Send bigBuffer
            MPI_Isend(&_toSend.count, 1, MPI_INT, to, tag + data.size() + 3, MPI_COMM_WORLD, &requests[rcount + 3]);

            // printf("[%d] send %d to %d\n", comm_rank, toSendCount[to], to); fflush(stdout);
            for (unsigned int i = 0; i < data.size(); i++)
            {
                MPI_Isend(&_toSend.buff[i][0], _toSend.count, MPI_DOUBLE, to, tag + data.size() + 4 + i, MPI_COMM_WORLD,
                          &requests[rcount + 4 + i]);
            }
        }

        // MPI_Barrier(MPI_COMM_WORLD);

        // tstop = Clock::now();
        // elaspedTime = std::chrono::duration_cast<Time>(tstop-tstart).count();

        // if(comm_rank == 0)
        // {
        //     printf("synchronizeHalos::send() %f\n", elaspedTime);
        //     fflush(stdout);
        // }

        // tstart = Clock::now();

        std::vector<std::vector<int>> ptriBuffs;
        std::vector<std::vector<std::vector<T>>> recvBuffs;
        while (needed > 0)
        {
            MPI_Status status[data.size() + 4];

            int from = -1;
            MPI_Recv(&from, 1, MPI_INT, MPI_ANY_SOURCE, tag + data.size() + 0, MPI_COMM_WORLD, &status[0]);

            // int rcount = requests.size();
            // requests.resize(rcount + data.size());

            int ptriCount = 0;

            // Recv ptriBuff
            MPI_Recv(&ptriCount, 1, MPI_INT, from, tag + data.size() + 1, MPI_COMM_WORLD, &status[1]);

            int ptriBuffCount = ptriBuffs.size();
            ptriBuffs.resize(ptriBuffCount + 1);

            std::vector<int> &ptriBuff = ptriBuffs[ptriBuffCount++];
            ptriBuff.resize(ptriCount);

            MPI_Recv(&ptriBuff[0], ptriCount, MPI_INT, from, tag + data.size() + 2, MPI_COMM_WORLD, &status[2]);

            int count = 0;

            // Recv bigBuffer
            MPI_Recv(&count, 1, MPI_INT, from, tag + data.size() + 3, MPI_COMM_WORLD, &status[3]);

            int recvBuffCount = recvBuffs.size();
            recvBuffs.resize(recvBuffCount + 1);

            std::vector<std::vector<T>> &recvBuff = recvBuffs[recvBuffCount++];
            recvBuff.resize(data.size());

            // printf("[%d] recv %d from %d\n", comm_rank, count, from); fflush(stdout);
            for (unsigned int i = 0; i < data.size(); i++)
            {
                recvBuff[i].resize(count);
                MPI_Recv(&recvBuffs[recvBuffCount - 1][i][0], count, MPI_DOUBLE, from, tag + data.size() + 4 + i, MPI_COMM_WORLD,
                         &status[4 + i]);
            }

            needed -= count;
        }

        // MPI_Barrier(MPI_COMM_WORLD);

        // tstop = Clock::now();
        // elaspedTime = std::chrono::duration_cast<Time>(tstop-tstart).count();

        // if(comm_rank == 0)
        // {
        //     printf("synchronizeHalos::recv() %f\n", elaspedTime);
        //     fflush(stdout);
        // }

        // tstart = Clock::now();

        for (unsigned int bi = 0; bi < recvBuffs.size(); bi++)
        {
            std::vector<int> &ptriBuff = ptriBuffs[bi];
            std::vector<std::vector<T>> &recvBuff = recvBuffs[bi];

            int current = 0;
            for (const int &ptri : ptriBuff)
            {
                Octree<T> *cell = cellMap[ptri];

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

        // MPI_Barrier(MPI_COMM_WORLD);

        // tstop = Clock::now();
        // elaspedTime = std::chrono::duration_cast<Time>(tstop-tstart).count();

        // if(comm_rank == 0)
        // {
        //     printf("synchronizeHalos::finalize() %f\n", elaspedTime);
        //     fflush(stdout);
        // }

        tag += data.size() + 4;

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
