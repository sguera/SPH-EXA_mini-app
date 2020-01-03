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

// This test checks some properties of the Octree after syncHalos when the full tree
// including local nodes is built
template <class T>
void buildTreeRec(Octree<T>* node, const std::vector<int> &list,
                  const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
                  std::vector<int> &ordering, int padding = 0)
{
    if (node->global && (node->assignee == -1 || node->assignee == node->comm_rank))
    {
        // we expect global nodes to have valid particle count
        EXPECT_EQ(node->localParticleCount, list.size());

        // global leaf node
        if ((int)node->cells.size() == 0)
        {
        }
        else
        {
            EXPECT_EQ(node->localParticleCount, list.size());
        }

    }
    else {
        // if the node is global and is assigned to a different process,
        // it is a halo node
        if (node->global) {
            if (list.size() > 0) {
                EXPECT_TRUE(node->halo);
            }
        }
    }

    if (node->global && !(node->assignee == -1 || node->assignee == node->comm_rank))
    {
        if (node->halo == true)
            EXPECT_EQ(node->localParticleCount, list.size());
        else
            EXPECT_EQ(0, list.size());
    }

    node->localPadding = padding;
    node->localParticleCount = list.size();

    std::vector<std::vector<int>> cellList(Octree<T>::ncells);
    node->distributeParticles(list, x, y, z, cellList);

    if ((int)node->cells.size() == 0 && list.size() > Octree<T>::bucketSize) node->makeSubCells();

    if (!node->global && node->assignee == -1) node->assignee = node->comm_rank;

    if ((int)node->cells.size() == Octree<T>::ncells)
    {
        for (int i = 0; i < Octree<T>::ncells; i++)
        {
            buildTreeRec(node->cells[i].get(), cellList[i], x, y, z, ordering, padding);
            padding += cellList[i].size();
        }
    }
    else
    {
        for (int i = 0; i < (int)list.size(); i++)
            ordering[padding + i] = list[i];
    }
}

template <class T>
void buildTree(Octree<T>& otree, const std::vector<int> &list,
               const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &z,
               std::vector<int> &ordering)
{
    buildTreeRec(&otree, list, x, y, z, ordering);
}


TEST(Octree, buildTree) {

    using Real = double;
    using Dataset = ParticlesData<Real>;

    const int cubeSide = 50;
    const int maxStep = 10;

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    DistributedDomain<Real, Dataset> distributedDomain;

    distributedDomain.create(d);
    distributedDomain.update(d);
    distributedDomain.synchronizeHalos(&d.x, &d.y, &d.z, &d.h);


    std::vector<int> ordering(d.x.size());
    std::vector<int> list(d.x.size());
    for (int i=0; i < list.size(); ++i)
        list[i] = i;

    buildTree(distributedDomain.octree, list, d.x, d.y, d.z, ordering);

    EXPECT_EQ(ordering.size(), list.size());
}

int main(int argc, char **argv) {

  MPI_Init(NULL, NULL);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  MPI_Finalize();
  return ret;
}
