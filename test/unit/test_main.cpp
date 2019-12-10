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

// Check whether the "localPadding" = Space curve offset
// correctly locates the particles in the data arrays x,y,z
template <class T>
void checkParticlesRec(Octree<T>* node,
                       std::vector<T> & x,
                       std::vector<T> & y,
                       std::vector<T> & z,
                       int zoffset = 0,
                       int nodeIdx = 0
                      )
{
    int ncells = Octree<T>::ncells;
    T xmin = node->xmin;
    T xmax = node->xmax;
    T ymin = node->ymin;
    T ymax = node->ymax;
    T zmin = node->zmin;
    T zmax = node->zmax;

    if (node->assignee == node->comm_rank || node->assignee == -1)
    {
        if ((int)(node->cells).size() == ncells)
        {
            for (int i = 0; i < ncells; i++)
            {
                checkParticlesRec((node->cells)[i].get(), x, y, z);
            }
        }
        else
        {
            int offset = node->localPadding;
            int npart   = node->localParticleCount;

            // is not true!
            //EXPECT_EQ(node->localParticleCount, node->globalParticleCount);

            //std::cout << "[" << node->comm_rank << "]: " << node
            //                 << ": " << xmin << "," << xmax
            //                 << " " << ymin << "," << ymax
            //                 << " " << zmin << "," << zmax << std::endl;
            for (int i = 0; i < npart; i++)
            {
                EXPECT_TRUE(inRange(x[offset + i], xmin, xmax));
                EXPECT_TRUE(inRange(y[offset + i], ymin, ymax));
                EXPECT_TRUE(inRange(z[offset + i], zmin, zmax));
            }
        }
    }
}

template <class T, class Dataset>
void checkParticles(Octree<T>& tree, Dataset& d)
{
    checkParticlesRec(&tree, d.x, d.y, d.z);
}

TEST(Octree, SpaceCurveIndexCorrect) {

    using Real = double;
    using Dataset = ParticlesData<Real>;

    const int cubeSide = 50;
    const int maxStep = 10;

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    DistributedDomain<Real> distributedDomain;

    distributedDomain.create(d);
    distributedDomain.distribute(d);

    checkParticles(distributedDomain.octree, d);
}


// Check whether each node is assigned to process p
// if all its children are assigned to proc p
template <class T>
void checkProcessAssignmentRec(Octree<T>* node)
{
    int ncells = Octree<T>::ncells;

    if (node->assignee == -1)
    {
        if ((int)(node->cells).size() == ncells)
        {
            int assignee0 = (node->cells)[0]->assignee;
            bool all_equal = true;
            for (int i = 0; i < ncells; i++)
            {
                all_equal = all_equal && (assignee0 == (node->cells)[i]->assignee);
            }
            EXPECT_FALSE(all_equal);

            for (int i = 0; i < ncells; i++)
            {
                checkProcessAssignmentRec((node->cells)[i].get());
            }
        }
    }
}

template <class T>
void checkProcessAssignment(Octree<T>& tree)
{
    checkProcessAssignmentRec(&tree);
}

TEST(Octree, processAssignment) {

    using Real = double;
    using Dataset = ParticlesData<Real>;

    const int cubeSide = 50;
    const int maxStep = 10;

    auto d = SqPatchDataGenerator<Real>::generate(cubeSide);
    DistributedDomain<Real> distributedDomain;

    distributedDomain.create(d);
    distributedDomain.distribute(d);

    checkProcessAssignment(distributedDomain.octree);
}

int main(int argc, char **argv) {

  MPI_Init(NULL, NULL);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  MPI_Finalize();
  return ret;
}
