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
 * \brief Space filling curve octree assignment to ranks tests
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <numeric>

#include "gtest/gtest.h"

#include "cstone/domaindecomp.hpp"
#include "cstone/octree.hpp"
#include "coord_samples/random.hpp"

using namespace cstone;

TEST(DomainDecomposition, singleRangeSfcSplit)
{
    using CodeType = unsigned;
    {
        int nSplits = 2;
        std::vector<unsigned> counts{5, 5, 5, 5, 5, 6};
        std::vector<CodeType> tree{0, 1, 2, 3, 4, 5, 6};

        auto splits = singleRangeSfcSplit(tree, counts, nSplits);

        SpaceCurveAssignment<CodeType> ref(nSplits);
        ref.addRange(Rank(0),0,3,15);
        ref.addRange(Rank(1),3,6,16);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 2;
        std::vector<unsigned> counts{5, 5, 5, 15, 1, 0};
        std::vector<CodeType> tree{0, 1, 2, 3, 4, 5, 6};

        auto splits = singleRangeSfcSplit(tree, counts, nSplits);

        SpaceCurveAssignment<CodeType> ref(nSplits);
        ref.addRange(Rank(0),0,3,15);
        ref.addRange(Rank(1),3,6,16);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 2;
        std::vector<unsigned> counts{15, 0, 1, 5, 5, 5};
        std::vector<CodeType> tree{0, 1, 2, 3, 4, 5, 6};

        auto splits = singleRangeSfcSplit(tree, counts, nSplits);

        SpaceCurveAssignment<CodeType> ref(nSplits);
        ref.addRange(Rank(0),0,3,16);
        ref.addRange(Rank(1),3,6,15);
        EXPECT_EQ(ref, splits);
    }
    {
        int nSplits = 7;
        std::vector<unsigned> counts{4, 3, 4, 3, 4, 3, 4, 3, 4, 3};
        // should be grouped |4|7|3|7|4|7|3|
        std::vector<CodeType> tree{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        auto splits = singleRangeSfcSplit(tree, counts, nSplits);

        SpaceCurveAssignment<CodeType> ref(nSplits);
        ref.addRange(Rank(0),0,1,4);
        ref.addRange(Rank(1),1,3,7);
        ref.addRange(Rank(2),3,4,3);
        ref.addRange(Rank(3),4,6,7);
        ref.addRange(Rank(4),6,7,4);
        ref.addRange(Rank(5),7,9,7);
        ref.addRange(Rank(6),9,10,3);
        EXPECT_EQ(ref, splits);
    }
}

//! \brief test that the SfcLookupKey can lookup the rank for a given code
TEST(DomainDecomposition, SfcLookupMinimal)
{
    using I = unsigned;

    int nRanks = 4;
    SpaceCurveAssignment<I> assignment(nRanks);
    assignment.addRange(Rank(3), 0, 1, 1);
    assignment.addRange(Rank(2), 3, 4, 1);
    assignment.addRange(Rank(0), 1, 3, 1);
    assignment.addRange(Rank(1), 4, 5, 1);
    assignment.addRange(Rank(3), 5, 7, 1);

    SfcLookupKey<I> key(assignment);

    EXPECT_EQ(3, key.findRank(0));
    EXPECT_EQ(2, key.findRank(3));
    EXPECT_EQ(0, key.findRank(1));
    EXPECT_EQ(0, key.findRank(2));
    EXPECT_EQ(1, key.findRank(4));
    EXPECT_EQ(3, key.findRank(5));
    EXPECT_EQ(3, key.findRank(6));
}

TEST(DomainDecomposition, SfcLookupGrid)
{
    using I = unsigned;

    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    int nRanks = 2;
    SpaceCurveAssignment<I> assignment(nRanks);
    assignment.addRange(Rank(0), tree[0], tree[32], 0);
    assignment.addRange(Rank(1), tree[32], tree[64], 0);

    SfcLookupKey<I> key(assignment);

    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(key.findRank(tree[i]), 0);
    }
    for (int i = 32; i < 64; ++i)
    {
        EXPECT_EQ(key.findRank(tree[i]), 1);
    }
}


/*! \brief test SendList creation from a SFC assignment
 *
 * This test creates an array with Morton codes and an
 * SFC assignment with Morton code ranges.
 * CreateSendList then translates the code ranges into indices
 * valid for the Morton code array.
 */
template<class I>
void createSendList()
{
    int nParticles = 10;
    std::vector<I> codes(nParticles);
    std::iota(begin(codes), end(codes), 10);

    int nRanks = 2;
    SpaceCurveAssignment<I> assignment(nRanks);
    assignment.addRange(Rank(0),9, 11, 1);   // range lower than lowest code
    assignment.addRange(Rank(0),13, 15, 2);
    assignment.addRange(Rank(1),17, 1000, 2); // range bigger than highest code

    // note: codes input needs to be sorted
    auto sendList = createSendList(assignment, codes.data(), codes.data() + nParticles);

    EXPECT_EQ(sendList[0].totalCount(), 3);
    EXPECT_EQ(sendList[1].totalCount(), 3);

    SendList refSendList(nRanks);
    refSendList[0].addRange(0, 1, 1);
    refSendList[0].addRange(3, 5, 2);
    refSendList[1].addRange(7, 10, 3);

    EXPECT_EQ(refSendList, sendList);
}

TEST(DomainDecomposition, createSendList)
{
    createSendList<unsigned>();
    createSendList<uint64_t>();
}

/*! \brief This test integrates octree generation, SFC assignment and SendList creation
 *
 * Test procedure:
 *
 * 1. create nParticles random gaussian distributed x,y,z coordinates in a box
 * 2. create (local) sfc-octree
 * 3. create sfc assignment based on octree from 2.
 * 4. create sendList from assignment
 *
 * Expected results:
 *
 * 1. assignment contains nSplit SFC ranges which all contain about nParticles/nSplit +- bucketSize particles
 * 2. each particle appears in the SendList, i.e. each particle did get assigned to some rank
 */
template<class I>
void assignSendRandomData()
{
    int nParticles = 1003;
    int bucketSize = 64;
    RandomGaussianCoordinates<double, I> coords(nParticles, {-1,1});

    auto [tree, counts] = computeOctree(coords.mortonCodes().data(),
                                                coords.mortonCodes().data() + nParticles,
                                                bucketSize);

    int  nSplits    = 4;
    auto assignment = singleRangeSfcSplit(tree, counts, nSplits);

    /// all splits except the last one should at least be assigned nParticles/nSplits
    for (int rank = 0; rank < nSplits; ++rank)
    {
        std::size_t rankCount = assignment.totalCount(rank);

        /// particles in each rank should be within avg per rank +- bucketCount
        EXPECT_LE(nParticles/nSplits - bucketSize, rankCount);
        EXPECT_LE(rankCount, nParticles/nSplits + bucketSize);
    }

    auto sendList = createSendList(assignment, coords.mortonCodes().data(),
                                           coords.mortonCodes().data() + nParticles);

    int particleRecount = 0;
    for (auto& manifest : sendList)
        for (std::size_t rangeIndex = 0; rangeIndex < manifest.nRanges(); ++rangeIndex)
            particleRecount += manifest.rangeEnd(rangeIndex) - manifest.rangeStart(rangeIndex);

    /// make sure that all particles present on the node got assigned to some rank
    EXPECT_EQ(nParticles, particleRecount);
}

TEST(DomainDecomposition, assignSendIntegration)
{
    assignSendRandomData<unsigned>();
    assignSendRandomData<uint64_t>();
}

/*! \brief Test that createSendBuffer can create the correct buffer from a source array
 *
 * @tparam I 32- or 64-bit unsigned integer
 *
 * Precondition: an array, corresponding ordering and SendManifest with valid index ranges into
 *               the array
 *
 * Expected Result: createSendBuffer extracts the elements that fall within the index ranges in the
 *                  sendManifest into the output buffer
 */
template<class I>
void createSendBuffer()
{
    int bufferSize = 64;
    // the source array from which to extract the buffer
    std::vector<double> x(bufferSize);
    std::iota(begin(x), end(x), 0);

    SendManifest manifest;
    manifest.addRange(0, 8, 8);
    manifest.addRange(40, 42, 2);
    manifest.addRange(50, 50, 0);

    std::vector<int> ordering(bufferSize);
    std::iota(begin(ordering), end(ordering), 0);

    // non-default ordering will make x appear sorted despite two elements being swapped
    std::swap(x[0], x[1]);
    std::swap(ordering[0], ordering[1]);

    auto buffer = createSendBuffer(manifest, x.data(), ordering.data());

    // note sorted reference
    std::vector<double> ref{0,1,2,3,4,5,6,7,40,41};
    EXPECT_EQ(buffer, ref);
}

TEST(DomainDecomposition, createSendBuffer)
{
    createSendBuffer<unsigned>();
    createSendBuffer<uint64_t>();
}
