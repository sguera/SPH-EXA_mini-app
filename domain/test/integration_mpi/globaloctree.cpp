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
 * \brief Global octree build test
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <vector>
#include <mpi.h>

#include <gtest/gtest.h>

#include "cstone/mortonconversions.hpp"
#include "cstone/octree_mpi.hpp"

using namespace cstone;

template<class I>
std::vector<I> makeRegularGrid(int rank)
{
    std::vector<I> codes;

    constexpr unsigned n     = 4;
    constexpr unsigned level = 2;

    unsigned nRanks = 2;
    unsigned istart = rank * n/nRanks;
    unsigned iend   = (rank + 1) * n/nRanks;

    // a regular n x n x n grid
    for (unsigned i = istart; i < iend; ++i)
        for (unsigned j = 0; j < n; ++j)
            for (unsigned k = 0; k < n; ++k)
    {
        codes.push_back(codeFromBox<I>(i,j,k, level));
    }

    std::sort(begin(codes), end(codes));

    return codes;
}

template<class I>
void buildTree(int rank)
{
    auto codes = makeRegularGrid<I>(rank);

    int bucketSize = 8;
    auto [tree, counts] = computeOctreeGlobal(codes.data(), codes.data() + codes.size(), bucketSize);

    std::vector<I> refTree{
        codeFromIndices<I>({0}),
        codeFromIndices<I>({1}),
        codeFromIndices<I>({2}),
        codeFromIndices<I>({3}),
        codeFromIndices<I>({4}),
        codeFromIndices<I>({5}),
        codeFromIndices<I>({6}),
        codeFromIndices<I>({7}),
        nodeRange<I>(0)
    };

    std::vector<unsigned> refCounts(8,8);

    EXPECT_EQ(counts, refCounts);
    EXPECT_EQ(tree, refTree);
}

TEST(GlobalTree, basicRegularTree32)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks)
        throw std::runtime_error("this test needs 2 ranks\n");

    buildTree<unsigned>(rank);
    buildTree<uint64_t>(rank);
}

template<class CodeType>
void computeNodeMax(int rank)
{
    // an (incomplete) tree with 4 nodes
    std::vector<CodeType> tree{0,8,16,24,32};

    // node boundaries:                 |    |     |   |           |
    std::vector<CodeType> particleCodes{ 2,4, 8,14, 20, 24,25,26,31 };

    std::vector<std::vector<float>> smoothingLs
    {
    //  |    |    |  |       |
        {1,1, 2,3, 6, 2,9,1,3}, // rank 0 smoothing lengths
        {1,2, 4,3, 5, 2,8,1,3}, // rank 1 smoothing lengths
    };

    // expected maximum per node across both ranks searching all nodes
    std::vector<float>    hMaxPerNode{4, 8, 12, 18};

    // trivial ordering
    std::vector<int> ordering(particleCodes.size());
    std::iota(begin(ordering), end(ordering), 0);

    {
        std::vector<float> probe(hMaxPerNode.size());

        computeHaloRadiiGlobal(tree.data(), nNodes(tree), particleCodes.data(),
                               particleCodes.data() + particleCodes.size(), ordering.data(),
                               smoothingLs[rank].data(), probe.data());

        EXPECT_EQ(probe, hMaxPerNode);
    }
}

TEST(GlobalTree, computeNodeMax)
{
    int rank = 0, nRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    constexpr int thisExampleRanks = 2;

    if (nRanks != thisExampleRanks)
        throw std::runtime_error("this test needs 2 ranks\n");

    computeNodeMax<unsigned>(rank);
    computeNodeMax<uint64_t>(rank);
}
