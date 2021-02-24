/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! \file
 * \brief Collision CPU driver test
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/collisions_cpu.hpp"
#include "cstone/octree_util.hpp"

#include "collision_reference/collisions_a2a.hpp"


using namespace cstone;

/*! \brief compare tree-traversal collision detection with the naive all-to-all algorithm
 *
 * @tparam I           32- or 64-bit unsigned integer
 * @tparam T           float or double
 * @param tree         cornerstone octree leaves
 * @param haloRadii    floating point collision radius per octree leaf
 * @param box          bounding box used to construct the octree
 *
 * This test goes through all leaf nodes of the input octree and computes
 * a list of all other leaves that overlap with the first one.
 * The computation is done with both the tree-traversal algorithm and the
 * naive all-to-all algorithm and the results are compared.
 */
template<class I, class T>
void generalCollisionTest(const std::vector<I>& tree, const std::vector<T>& haloRadii,
                          const Box<T>& box)
{
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    // tree traversal collision detection
    std::vector<CollisionList> collisions    = findAllCollisions(internalTree, tree, haloRadii, box);
    // naive all-to-all algorithm
    std::vector<CollisionList> refCollisions = findCollisionsAll2all(tree, haloRadii, box);

    for (std::size_t nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        std::vector<int> c{collisions[nodeIndex].begin(), collisions[nodeIndex].end()};
        std::vector<int> ref{refCollisions[nodeIndex].begin(), refCollisions[nodeIndex].end()};

        std::sort(begin(c), end(c));
        std::sort(begin(ref), end(ref));

        EXPECT_EQ(c, ref);
    }
}

//! \brief an irregular tree with level-3 nodes next to level-1 ones
template<class I, class T, bool Pbc>
void irregularTreeTraversal()
{
    auto tree = OctreeMaker<I>{}.divide().divide(0).divide(0,7).makeTree();

    Box<T> box(0, 1, 0, 1, 0, 1, Pbc, Pbc, Pbc);
    std::vector<T> haloRadii(nNodes(tree), 0.1);
    generalCollisionTest(tree, haloRadii, box);
}

TEST(Collisions, irregularTreeTraversal)
{
    irregularTreeTraversal<unsigned, float , false>();
    irregularTreeTraversal<uint64_t, float , false>();
    irregularTreeTraversal<unsigned, double, false>();
    irregularTreeTraversal<uint64_t, double, false>();
}

TEST(Collisions, irregularTreeTraversalPbc)
{
    irregularTreeTraversal<unsigned, float , true>();
    irregularTreeTraversal<uint64_t, float , true>();
    irregularTreeTraversal<unsigned, double, true>();
    irregularTreeTraversal<uint64_t, double, true>();
}


//! \brief a regular tree with level-3 nodes, 8x8x8 grid
template<class I, class T, bool Pbc>
void regularTreeTraversal()
{
    auto tree = makeUniformNLevelTree<I>(512, 1);

    Box<T> box(0, 1, 0, 1, 0, 1, Pbc, Pbc, Pbc);
    // node edge length is 0.125
    std::vector<T> haloRadii(nNodes(tree), 0.124);
    generalCollisionTest(tree, haloRadii, box);
}

TEST(Collisions, regularTreeTraversal)
{
    regularTreeTraversal<unsigned, float, false>();
    regularTreeTraversal<uint64_t, float, false>();
    regularTreeTraversal<unsigned, double, false>();
    regularTreeTraversal<uint64_t, double, false>();
}

TEST(Collisions, regularTreeTraversalPbc)
{
    regularTreeTraversal<unsigned, float , true>();
    regularTreeTraversal<uint64_t, float , true>();
    regularTreeTraversal<unsigned, double, true>();
    regularTreeTraversal<uint64_t, double, true>();
}

/*! \brief test tree traversal with anisotropic boxes
 *
 * anisotropic boxes with a single halo radius per node
 * results in different x,y,z halo search lengths once
 * the coordinates are normalized to the cubic unit box.
 */
class AnisotropicBoxTraversal : public testing::TestWithParam<std::array<int,6>>
{
public:
    template <class I, class T>
    void check()
    {
        // 8x8x8 grid
        auto tree = makeUniformNLevelTree<I>(512, 1);

        Box<T> box(std::get<0>(GetParam()),
                   std::get<1>(GetParam()),
                   std::get<2>(GetParam()),
                   std::get<3>(GetParam()),
                   std::get<4>(GetParam()),
                   std::get<5>(GetParam()));

        // node edge length is 0.125 in the compressed dimension
        // and 0.250 in the other two dimensions
        std::vector<T> haloRadii(nNodes(tree), 0.175);
        generalCollisionTest(tree, haloRadii, box);
    }
};

TEST_P(AnisotropicBoxTraversal, compressedAxis32f)
{
    check<unsigned, float>();
}

TEST_P(AnisotropicBoxTraversal, compressedAxis64f)
{
    check<uint64_t, float>();
}

TEST_P(AnisotropicBoxTraversal, compressedAxis32d)
{
    check<unsigned, double>();
}

TEST_P(AnisotropicBoxTraversal, compressedAxis64d)
{
    check<uint64_t, double>();
}

std::vector<std::array<int, 6>> boxLimits{{0,1,0,2,0,2},
                                          {0,2,0,1,0,2},
                                          {0,2,0,2,0,1}};

INSTANTIATE_TEST_SUITE_P(AnisotropicBoxTraversal,
                         AnisotropicBoxTraversal,
                         testing::ValuesIn(boxLimits));

//! \brief this tree results from 2 particles at (0,0,0) and at (1,1,1) with a bucket size of 1
std::vector<unsigned> makeEdgeTree()
{
    std::vector<unsigned> tree{0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64, 128, 192, 256, 320, 384,
                               448, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 8192, 12288, 16384, 20480, 24576,
                               28672, 32768, 65536, 98304, 131072, 163840, 196608, 229376, 262144, 524288, 786432, 1048576,
                               1310720, 1572864, 1835008, 2097152, 4194304, 6291456, 8388608, 10485760, 12582912, 14680064,
                               16777216, 33554432, 50331648, 67108864, 83886080, 100663296, 117440512, 134217728, 268435456,
                               402653184, 536870912, 671088640, 805306368, 939524096, 956301312, 973078528, 989855744, 1006632960,
                               1023410176, 1040187392, 1056964608, 1059061760, 1061158912, 1063256064, 1065353216, 1067450368,
                               1069547520, 1071644672, 1071906816, 1072168960, 1072431104, 1072693248, 1072955392, 1073217536,
                               1073479680, 1073512448, 1073545216, 1073577984, 1073610752, 1073643520, 1073676288, 1073709056,
                               1073713152, 1073717248, 1073721344, 1073725440, 1073729536, 1073733632, 1073737728, 1073738240,
                               1073738752, 1073739264, 1073739776, 1073740288, 1073740800, 1073741312, 1073741376, 1073741440,
                               1073741504, 1073741568, 1073741632, 1073741696, 1073741760, 1073741768, 1073741776, 1073741784,
                               1073741792, 1073741800, 1073741808, 1073741816, 1073741817, 1073741818, 1073741819, 1073741820,
                               1073741821, 1073741822, 1073741823, 1073741824};

    return tree;
}

/*! \brief a simple collision test with the edge tree from above
 *
 * Since the halo radius for the first and last node is bigger than the box,
 * these two nodes collide with all nodes in the tree, while all other nodes have
 * radius 0 and only collide with themselves.
 */
TEST(Collisions, adjacentEdgeRegression)
{
    std::vector<unsigned> tree = makeEdgeTree();
    auto internalTree = createInternalTree(tree);

    Box<double> box(0.5, 0.6);

    std::vector<double> haloRadii(nNodes(tree), 0);
    haloRadii[0] = 0.2;
    *haloRadii.rbegin() = 0.2;

    std::vector<int> allNodes(nNodes(tree));
    std::iota(begin(allNodes), end(allNodes), 0);

    for (std::size_t i = 0; i < nNodes(tree); ++i)
    {
        IBox haloBox = makeHaloBox(tree[i], tree[i+1], haloRadii[i], box);
        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

        std::vector<int> cnodes{collisions.begin(), collisions.end()};
        std::sort(begin(cnodes), end(cnodes));

        if (i == 0 || i == nNodes(tree) - 1)
        {
            EXPECT_EQ(cnodes, allNodes);
        }
        else
        {
            EXPECT_EQ(cnodes, std::vector<int>(1,i));
        }
    }

    generalCollisionTest(tree, haloRadii, box);

    Box<double> boxPbc(0.5, 0.6, 0.5, 0.6, 0.5, 0.6, true, true, true);
    generalCollisionTest(tree, haloRadii, boxPbc);
}

/*! \brief collisions test with a very small radius
 *
 * This tests that a very small, but non-zero halo radius
 * does not get rounded down to zero.
 */
TEST(Collisions, adjacentEdgeSmallRadius)
{
    std::vector<unsigned> tree = makeEdgeTree();
    auto internalTree = createInternalTree((tree));

    Box<double> box(0,1);

    // nNodes is 134
    int secondLastNode = 132;
    double radius = 0.0001;
    IBox haloBox = makeHaloBox(tree[secondLastNode], tree[secondLastNode+1], radius, box);

    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

    std::vector<int> cnodes{collisions.begin(), collisions.end()};
    std::sort(begin(cnodes), end(cnodes));

    std::vector<int> refNodes{125,126,127,128,129,130,131,132,133};
    EXPECT_EQ(cnodes, refNodes);
}

TEST(Collisions, adjacentEdgeLastNode)
{
    std::vector<unsigned> tree = makeEdgeTree();
    auto internalTree = createInternalTree((tree));

    Box<double> box(0,1);

    // nNodes is 134
    int lastNode = 133;
    double radius = 0.0;
    IBox haloBox = makeHaloBox(tree[lastNode], tree[lastNode+1], radius, box);

    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

    std::vector<int> cnodes{collisions.begin(), collisions.end()};
    std::sort(begin(cnodes), end(cnodes));

    std::vector<int> refNodes{133};
    EXPECT_EQ(cnodes, refNodes);
}

TEST(Collisions, regularLastNode)
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<unsigned> tree = makeUniformNLevelTree<unsigned>(64, 1);
    auto internalTree = createInternalTree(tree);

    IBox haloBox{1022, 1023, 1022, 1023, 1022, 1023};
    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox);
    EXPECT_EQ(collisions.size(), 1);
    EXPECT_EQ(collisions[0], 63);
}