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
 * \brief Binary radix tree traversal tests
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#include <vector>

#include "gtest/gtest.h"

#include "cstone/btreetraversal.hpp"
#include "cstone/octree.hpp"

using namespace cstone;

//! \brief test add(), operator[] and begin()/end() of the CollisionList class
TEST(BinaryTreeTraversal, collisionList)
{
    CollisionList collisions;
    collisions.add(3);
    collisions.add(7);
    collisions.add(10);
    collisions.add(0);

    EXPECT_EQ(collisions.size(), 4);
    EXPECT_EQ(collisions[0], 3);
    EXPECT_EQ(collisions[1], 7);
    EXPECT_EQ(collisions[2], 10);
    EXPECT_EQ(collisions[3], 0);

    std::vector<int> refValues{3,7,10,0};
    std::vector<int> probe{collisions.begin(), collisions.end()};

    EXPECT_EQ(refValues, probe);
}

/*! \brief test findCollisions with pbc halo boxes
 *
 * The example constructs a 4x4x4 regular octree with 64 nodes. A haloBox
 * with coordinates [-1,1]^3 will collide with all 8 nodes in the corners of the tree.
 * The same happens for a haloBox at the opposite diagonal end with
 * coordinates [2^(10 or 21)-1, 2^(10 or 21)+1]^3.
 */
template<class I>
void pbcCollision()
{
    std::vector<I>             tree         = makeUniformNLevelTree<I>(64, 1);
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    {
        IBox haloBox{-1, 1, -1, 1, -1, 1};

        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

        std::vector<int> collided(collisions.begin(), collisions.end());
        std::sort(begin(collided), end(collided));

        constexpr int level2 = 2;
        // nLevel2Codes is the number of Morton codes that a level2 node can cover,
        // corresponding to the number of contained octree leaf nodes at the maximum division level
        I nLevel2Codes = 1ul<<(3*(maxTreeLevel<I>{} - level2)); // 8^(maxLevel - 2)
        // reference from level-2 ix,iy,iz coordinates in [0:4], illustrates that all 8 corners collide
        std::vector<int> refCollided{
                int(codeFromBox<I>(0,0,0,level2) / nLevel2Codes),
                int(codeFromBox<I>(0,0,3,level2) / nLevel2Codes),
                int(codeFromBox<I>(0,3,0,level2) / nLevel2Codes),
                int(codeFromBox<I>(0,3,3,level2) / nLevel2Codes),
                int(codeFromBox<I>(3,0,0,level2) / nLevel2Codes),
                int(codeFromBox<I>(3,0,3,level2) / nLevel2Codes),
                int(codeFromBox<I>(3,3,0,level2) / nLevel2Codes),
                int(codeFromBox<I>(3,3,3,level2) / nLevel2Codes),
        };
        // node indices of the 8 corners in a level-2 tree with 4x4x4=64 nodes
        // explicitly specified
        std::vector<int> refCollidedExplicit{0,9,18,27,36,45,54,63};

        EXPECT_EQ(collided, refCollided);
        EXPECT_EQ(refCollided, refCollidedExplicit);
    }
    {
        int maxCoord = 1u<<maxTreeLevel<I>{};
        IBox haloBox{maxCoord-1, maxCoord+1, maxCoord-1, maxCoord+1, maxCoord-1, maxCoord+1};

        CollisionList collisions;
        findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

        std::vector<int> collided(collisions.begin(), collisions.end());
        std::sort(begin(collided), end(collided));
        // 8 corners
        std::vector<int> refCollided{0,9,18,27,36,45,54,63};

        EXPECT_EQ(collided, refCollided);
    }
}

TEST(BinaryTreeTraversal, pbcCollision)
{
    pbcCollision<unsigned>();
    pbcCollision<uint64_t>();
}


/*! \brief test collision detection with anisotropic halo ranges
 *
 * If the bounding box of the floating point boundary box is not cubic,
 * an isotropic search range with one halo radius per node will correspond
 * to an anisotropic range in the Morton code SFC which always gets mapped
 * to an unit cube.
 */
template <class I>
void anisotropicHaloBox()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    // node range in each dimension is 2^(10 or 21 - 2)
    std::vector<I>             tree         = makeUniformNLevelTree<I>(64, 1);
    std::vector<BinaryNode<I>> internalTree = createInternalTree(tree);

    int r = 1u<<(maxTreeLevel<I>{}-2);

    int queryIdx = 7;

    // this will hit two nodes in +x direction, not just one neighbor node
    IBox haloBox = makeHaloBox(tree[queryIdx], tree[queryIdx+1], 2*r, 0, 0);

    CollisionList collisions;
    findCollisions(internalTree.data(), tree.data(), collisions, haloBox);

    std::vector<int> collisionsSorted(collisions.begin(), collisions.end());
    std::sort(begin(collisionsSorted), end(collisionsSorted));

    std::vector<int> collisionsReference{3,7,35,39};
    EXPECT_EQ(collisionsSorted, collisionsReference);
}


TEST(BinaryTreeTraversal, anisotropicHalo)
{
    anisotropicHaloBox<unsigned>();
    anisotropicHaloBox<uint64_t>();
}


