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
 * \brief Test functions used to determine the arrangement of halo and assigned particles
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#include "gtest/gtest.h"

#include "cstone/layout.hpp"

using namespace cstone;

template<class I>
void computeLocalNodeRanges()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    // two domains
    SpaceCurveAssignment<I> assignment(2);
    assignment.addRange(Rank(0), tree[0], tree[32], 64);
    assignment.addRange(Rank(1), tree[32], tree[64], 64);

    {
        int rank = 0;
        std::vector<int> nodeIndexRanges = computeLocalNodeRanges(tree, assignment, rank);
        std::vector<int> ref{0,32};
        EXPECT_EQ(nodeIndexRanges, ref);
    }
    {
        int rank = 1;
        std::vector<int> nodeIndexRanges = computeLocalNodeRanges(tree, assignment, rank);
        std::vector<int> ref{32,64};
        EXPECT_EQ(nodeIndexRanges, ref);
    }
}

TEST(Layout, LocalLayout)
{
    computeLocalNodeRanges<unsigned>();
    computeLocalNodeRanges<uint64_t>();
}


TEST(Layout, flattenNodeList)
{
    std::vector<std::vector<int>> grouped{{0,1,2}, {3,4,5}, {6}, {}};

    std::vector<int> flattened = flattenNodeList(grouped);

    std::vector<int> ref{0,1,2,3,4,5,6};
    EXPECT_EQ(flattened, ref);
}


TEST(Layout, computeLayoutOffsets)
{
    int nNodes = 32;
    std::vector<unsigned> nodeCounts(nNodes, 1);

    // nodes 4-9 and 23-27 are local nodes
    std::vector<int> localNodes{4,10,23,28};

    std::vector<int> halos{1, 3, 14, 15, 16, 21, 30};
    nodeCounts[1]  = 2;
    nodeCounts[3]  = 3;
    nodeCounts[4]  = 5;
    nodeCounts[16] = 6;
    nodeCounts[24] = 8;
    nodeCounts[30] = 9;

    std::vector<int> presentNodes, offsets;
    computeLayoutOffsets(localNodes, halos, nodeCounts, presentNodes, offsets);

    std::vector<int> refPresentNodes{1,3,4,5,6,7,8,9,14,15,16,21,23,24,25,26,27,30};
    EXPECT_EQ(presentNodes, refPresentNodes);

    std::vector<int> refOffsets{0,2,5,10,11,12,13,14,15,16,17,23,24,25,33,34,35,36,45};
    EXPECT_EQ(offsets, refOffsets);
}

TEST(Layout, createHaloExchangeList)
{
    int nRanks = 3;
    std::vector<std::vector<int>> outgoingHaloNodes(nRanks);
    outgoingHaloNodes[1].push_back(1);
    outgoingHaloNodes[1].push_back(10);
    outgoingHaloNodes[2].push_back(12);
    outgoingHaloNodes[2].push_back(22);

    std::vector<int> presentNodes{1,2,10,12,20,22};
    std::vector<int>      offsets{0,1,3, 6, 10,15,21};

    SendList sendList = createHaloExchangeList(outgoingHaloNodes, presentNodes, offsets);

    SendList refSendList(nRanks);
    refSendList[1].addRange(0,1);
    refSendList[1].addRange(3,6);
    refSendList[2].addRange(6,10);
    refSendList[2].addRange(15,21);

    EXPECT_EQ(sendList, refSendList);
}

