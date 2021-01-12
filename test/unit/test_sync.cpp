#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "mpi.h"

#ifndef USE_MPI
#define USE_MPI
#endif

#include "sphexa.hpp"
#include "SqPatchDataGenerator.hpp"

#include "gtest/gtest.h"

using namespace std;
using namespace sphexa;



template <class T>
bool inRange(T val, T min, T max)
{
    if (val >= min && val <= max)
        return true;
    else
        return false;
}

//template <class T>
//void makeDataArray(std::vector<std::vector<T> *> &data, std::vector<T> *d) { data.push_back(d); }
//
//template <class T, typename... Args>
//void makeDataArray(std::vector<std::vector<T> *> &data, std::vector<T> *first, Args... args)
//{
//    data.push_back(first);
//    makeDataArray(data, args...);
//}



template <class T>
void synchronizeHalos(int comm_rank, std::unordered_map<int, std::unordered_map<int, Octree<T>*>> &toSendHalos,
                      std::vector<std::vector<T> *> &data)
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

        // sort ptri buffer on the sender side
        std::sort(_toSend.ptris.begin(), _toSend.ptris.end());
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

    // ***********************************************************
    // TEST CODE
    // reconstruct ptri vector from list of Halos recorded for this rank
    std::map<int, std::vector<int>> reHalos;
    std::map<int, Octree<T>*> reCellMap;
    std::map<int, int> recvCount;

    for (auto const& proc : toSendHalos)
    {
        int to = proc.first;
        if (to == comm_rank)
        {
            for (auto const& halo : proc.second)
            {
                int ptri = halo.first;
                Octree<T>* cell = halo.second;
                int from = cell->assignee;

                reHalos[from].push_back(ptri);
                reCellMap[ptri] = cell;
                recvCount[from] += cell->globalParticleCount;
            }
        }
    }

    // sort ptri buffer on the receiver side
    for (auto& pv : reHalos) std::sort(pv.second.begin(), pv.second.end());

    // ***********************************************************


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

        ASSERT_EQ(ptriCount, reHalos[from].size());

        MPI_Recv(&ptriBuff[0], ptriCount, MPI_INT, from, tag + data.size() + 2, MPI_COMM_WORLD, &status[2]);

        for (size_t i = 0; i < ptriBuff.size(); ++i)
        {
            EXPECT_EQ(ptriBuff[i], reHalos[from][i]);
        }

        int count = 0;

        // Recv bigBuffer
        MPI_Recv(&count, 1, MPI_INT, from, tag + data.size() + 3, MPI_COMM_WORLD, &status[3]);
        EXPECT_EQ(count, recvCount[from]);

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

    for (unsigned int bi = 0; bi < recvBuffs.size(); bi++)
    {
        std::vector<int> &ptriBuff = ptriBuffs[bi];
        std::vector<std::vector<T>> &recvBuff = recvBuffs[bi];

        int current = 0;
        for (const int &ptri : ptriBuff)
        {
            Octree<T> *cell = cellMap[ptri];
            EXPECT_EQ(cell, reCellMap[ptri]);

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

    tag += data.size() + 4;

    if (requests.size() > 0)
    {
        MPI_Status status[requests.size()];
        MPI_Waitall(requests.size(), &requests[0], &status[0]);
    }
}


TEST(Octree, syncHalos) {

    using Real = double;
    using Dataset = ParticlesData<Real>;

    const int cubeSide = 50;
    const int maxStep = 10;

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    DistributedDomain<Real, Dataset> distributedDomain;

    distributedDomain.create(d);
    distributedDomain.update(d);

    std::vector<std::vector<Real> *> data;
    makeDataArray(data, &d.x, &d.y, &d.z);

    synchronizeHalos(distributedDomain.comm_rank, distributedDomain.toSendHalos, data);
}

int main(int argc, char **argv) {

  MPI_Init(NULL, NULL);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  MPI_Finalize();
  return ret;
}
