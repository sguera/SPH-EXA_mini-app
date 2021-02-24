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
 * \brief Test morton code implementation
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/mortoncode.hpp"
#include "cstone/octree.hpp"
#include "cstone/octree_util.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

template<class I>
void printIndices(std::array<unsigned char, maxTreeLevel<I>{}> indices)
{
    for (int i = 0; i < maxTreeLevel<I>{}; ++i)
        std::cout << indices[i] << ",";
}


template<class T, std::size_t N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& input)
{
    unsigned lastNonZero = 0;
    for (int i = 0; i < N; ++i)
    {
        if (input[i] != 0)
        {
            lastNonZero = i;
        }
    }

    for (unsigned i = 0; i < lastNonZero; ++i)
        os << int(input[i]) << ",";
    os << int(input[lastNonZero]);

    return os;
}

template<class I>
void printTree(const I* tree, const int* counts, int nNodes)
{
    for (int i = 0; i < nNodes; ++i)
    {
        I thisNode     = tree[i];
        std::cout << indicesFromCode(thisNode) << " :" << counts[i] << std::endl;
    }
    std::cout << std::endl;
}

//! \brief test that computeNodeCounts correctly counts the number of codes for each node
template<class CodeType>
void checkCountTreeNodes()
{
    std::vector<CodeType> codes;

    constexpr unsigned n     = 4;
    constexpr unsigned level = 2;

    // a regular n x n x n grid
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j)
            for (unsigned k = 0; k < n; ++k)
    {
        codes.push_back(codeFromBox<CodeType>(i,j,k, level));
    }
    std::sort(begin(codes), end(codes));

    std::vector<CodeType>    tree = OctreeMaker<CodeType>{}.divide().divide(0).makeTree();
    std::vector<unsigned> counts(nNodes(tree));

    // doesn't affect the end result, but makes sure that
    // binary searching correctly finds the first tree node
    // with a _lower_ code than the first particle code
    codes[0]++;

    computeNodeCounts(tree.data(), counts.data(), nNodes(tree),
                      codes.data(), codes.data() + codes.size());

    // the level 2 nodes have 1/64 of the total volume/particle count
    for (std::size_t i = 0; i < 8; ++i)
        EXPECT_EQ(counts[i], 1);

    // the level 1 nodes have 1/8 of the total
    for (std::size_t i = 8; i < counts.size(); ++i)
        EXPECT_EQ(counts[i], 8);
}

TEST(CornerstoneOctree, countTreeNodes32)
{
    checkCountTreeNodes<unsigned>();
}

TEST(CornerstoneOctree, countTreeNodes64)
{
    checkCountTreeNodes<uint64_t>();
}

template<class CodeType, class LocalIndex>
void rebalanceDecision()
{
    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.divide().divide(0).makeTree();

    unsigned bucketSize = 4;
    std::vector<unsigned> counts{1,1,1,0,0,0,0,0, 2, 3, 4, 5, 6, 7, 8};

    std::vector<LocalIndex> nodeOps(nNodes(tree));
    rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

    std::vector<LocalIndex> reference{1,0,0,0,0,0,0,0, 1, 1, 1, 8, 8, 8, 8};
    EXPECT_EQ(nodeOps, reference);
}

TEST(CornerstoneOctree, rebalanceDecision)
{
    rebalanceDecision<unsigned, unsigned>();
    rebalanceDecision<uint64_t, unsigned>();
}

//! \brief check that nodes can be fused at the start of the tree
template<class CodeType>
void rebalanceShrinkStart()
{
    constexpr int bucketSize = 8;

    std::vector<CodeType> tree;
    tree.reserve(20);

    std::vector<unsigned> counts;
    counts.reserve(20);

    for (unsigned char i = 0; i < 8; ++i)
    {
        tree.push_back(codeFromIndices<CodeType>({0, i}));
        counts.push_back(1);
    }

    for (unsigned char i = 1; i < 8; ++i)
    {
        tree.push_back(codeFromIndices<CodeType>({i}));
        counts.push_back(1);
    }
    tree.push_back(nodeRange<CodeType>(0));

    std::vector<CodeType> balancedTree = rebalanceTree(tree.data(), counts.data(),
                                                       nNodes(tree), bucketSize);

    EXPECT_TRUE(checkOctreeInvariants(balancedTree.data(), nNodes(balancedTree)));
    EXPECT_TRUE(checkOctreeInvariants(tree.data(), nNodes(tree)));

    std::vector<CodeType> reference;
    reference.reserve(9);

    for (unsigned char i = 0; i < 8; ++i)
    {
        reference.push_back(codeFromIndices<CodeType>({i}));
    }
    reference.push_back(nodeRange<CodeType>(0));
    EXPECT_EQ(balancedTree, reference);
}

TEST(CornerstoneOctree, rebalanceShrinkStart32)
{
    rebalanceShrinkStart<unsigned>();
}

TEST(CornerstoneOctree, rebalanceShrinkStart64)
{
    rebalanceShrinkStart<uint64_t>();
}

//! \brief check that nodes can be fused in the middle of the tree
template<class CodeType>
void rebalanceShrinkMid()
{
    constexpr int bucketSize = 8;

    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.divide().divide(1).makeTree();

    std::vector<unsigned> counts(nNodes(tree), 1);
    std::vector<CodeType> balancedTree
        = rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_TRUE(checkOctreeInvariants(balancedTree.data(), nNodes(balancedTree)));
    EXPECT_TRUE(checkOctreeInvariants(tree.data(), nNodes(tree)));

    std::vector<CodeType> reference;
    reference.reserve(9);

    for (unsigned char i = 0; i < 8; ++i)
    {
        reference.push_back(codeFromIndices<CodeType>({i}));
    }
    reference.push_back(nodeRange<CodeType>(0));

    EXPECT_EQ(balancedTree, reference);
}

TEST(CornerstoneOctree, rebalanceShrinkMid32)
{
    rebalanceShrinkMid<unsigned>();
}

TEST(CornerstoneOctree, rebalanceShrinkMid64)
{
    rebalanceShrinkMid<uint64_t>();
}

//! \brief check that nodes can be fused at the end of the tree
template<class CodeType>
void rebalanceShrinkEnd()
{
    constexpr int bucketSize = 8;

    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.divide().divide(7).makeTree();

    std::vector<unsigned> counts(nNodes(tree), 1);

    std::vector<CodeType> balancedTree
        = rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_TRUE(checkOctreeInvariants(balancedTree.data(), nNodes(balancedTree)));
    EXPECT_TRUE(checkOctreeInvariants(tree.data(), nNodes(tree)));

    std::vector<CodeType> reference;
    reference.reserve(9);

    for (unsigned char i = 0; i < 8; ++i)
    {
        reference.push_back(codeFromIndices<CodeType>({i}));
    }
    reference.push_back(nodeRange<CodeType>(0));

    EXPECT_EQ(balancedTree, reference);
}

TEST(CornerstoneOctree, rebalanceShrinkEnd32)
{
    rebalanceShrinkEnd<unsigned>();
}

TEST(CornerstoneOctree, rebalanceShrinkEnd64)
{
    rebalanceShrinkEnd<uint64_t>();
}

//! \brief test invariance of a single root node under rebalancing if count < bucketsize
template<class I>
void rebalanceRootInvariant()
{
    using CodeType = I;
    constexpr int bucketSize = 8;

    // single root node
    std::vector<CodeType> tree{0, nodeRange<CodeType>(0)};
    std::vector<unsigned> counts{7};

    std::vector<CodeType> balancedTree
        = rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_EQ(balancedTree, tree);
}

TEST(CornerstoneOctree, rebalanceRootInvariant32)
{
    rebalanceRootInvariant<uint64_t>();
}

TEST(CornerstoneOctree, rebalanceRootInvariant64)
{
    rebalanceRootInvariant<uint64_t>();
}

//! \brief test splitting of a single root node
template<class I>
void rebalanceRootSplit()
{
    using CodeType = I;
    constexpr int bucketSize = 8;

    // single root node
    std::vector<CodeType> tree{0, nodeRange<CodeType>(0)};
    std::vector<unsigned> counts{9};

    std::vector<CodeType> balancedTree
        = rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    std::vector<CodeType> reference;
    reference.reserve(9);

    for (unsigned char i = 0; i < 8; ++i)
    {
        reference.push_back(codeFromIndices<CodeType>({i}));
    }
    reference.push_back(nodeRange<CodeType>(0));

    EXPECT_EQ(balancedTree, reference);
}

TEST(CornerstoneOctree, rebalanceRootSplit32)
{
    rebalanceRootSplit<unsigned>();
}

TEST(CornerstoneOctree, rebalanceRootSplit64)
{
    rebalanceRootSplit<uint64_t>();
}

//! \brief test node splitting and fusion simultaneously
template<class CodeType>
void rebalanceSplitShrink()
{
    constexpr int bucketSize = 8;

    std::vector<CodeType> tree = OctreeMaker<CodeType>{}.divide().divide(7).makeTree();

    // nodes {7,i} will need to be fused
    std::vector<unsigned> counts(nNodes(tree), 1);
    // node {1} will need to be split
    counts[1] = bucketSize+1;

    std::vector<CodeType> balancedTree
        = rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_TRUE(checkOctreeInvariants(balancedTree.data(), nNodes(balancedTree)));
    EXPECT_TRUE(checkOctreeInvariants(tree.data(), nNodes(tree)));

    std::vector<CodeType> reference;
    reference.reserve(9);

    reference.push_back(codeFromIndices<CodeType>({0}));
    for (unsigned char i = 0; i < 8; ++i)
        reference.push_back(codeFromIndices<CodeType>({1,i}));
    for (unsigned char i = 2; i < 8; ++i)
        reference.push_back(codeFromIndices<CodeType>({i}));

    reference.push_back(nodeRange<CodeType>(0));

    EXPECT_EQ(balancedTree, reference);
}

TEST(CornerstoneOctree, rebalanceSplitShrink32)
{
    rebalanceSplitShrink<unsigned>();
}

TEST(CornerstoneOctree, rebalanceSplitShrink64)
{
    rebalanceSplitShrink<uint64_t>();
}

/*! \brief test behavior of a maximum-depth tree under rebalancing
 *
 *  Node 0 is at the lowest octree level (10 or 21) and its particle
 *  count is bigger than the bucket size. This test verifies that
 *  this tree stays invariant under rebalancing because the capacity
 *  of the underlying 30-bit or 63 bit Morton code is exhausted.
 */
template<class CodeType>
void rebalanceInsufficentResolution()
{
    constexpr int bucketSize = 1;

    OctreeMaker<CodeType> octreeMaker;
    for (unsigned level = 0; level < maxTreeLevel<CodeType>{}; ++level)
        octreeMaker.divide({}, level);
    std::vector<CodeType> tree = octreeMaker.makeTree();

    std::vector<unsigned> counts(nNodes(tree), 1);
    counts[0] = bucketSize + 1;

    // the first node has two particles, one more than the bucketSize
    // since the first node is at the maximum subdivision layer, the tree
    // can't be further refined to satisfy the bucketSize
    std::vector<CodeType> balancedTree
        = rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize);

    EXPECT_EQ(balancedTree, tree);
}

TEST(CornerstoneOctree, rebalanceInsufficientResolution32)
{
    rebalanceInsufficentResolution<unsigned>();
}

TEST(CornerstoneOctree, rebalanceInsufficientResolution64)
{
    rebalanceInsufficentResolution<uint64_t>();
}


template<class I>
void checkOctreeWithCounts(const std::vector<I>& tree, const std::vector<unsigned>& counts, int bucketSize,
                           const std::vector<I>& mortonCodes)
{
    using CodeType = I;
    EXPECT_TRUE(checkOctreeInvariants(tree.data(), nNodes(tree)));

    int nParticles = mortonCodes.size();

    // check that referenced particles are within specified range
    for (std::size_t nodeIndex = 0; nodeIndex < nNodes(tree); ++nodeIndex)
    {
        int nodeStart = std::lower_bound(begin(mortonCodes), end(mortonCodes), tree[nodeIndex]) -
                        begin(mortonCodes);

        int nodeEnd = std::lower_bound(begin(mortonCodes), end(mortonCodes), tree[nodeIndex+1]) -
                      begin(mortonCodes);

        // check that counts are correct
        EXPECT_EQ(nodeEnd - nodeStart, counts[nodeIndex]);
        EXPECT_LE(counts[nodeIndex], bucketSize);

        if (counts[nodeIndex])
        {
            ASSERT_LT(nodeStart, nParticles);
        }

        for (std::size_t i = nodeStart; i < counts[nodeIndex]; ++i)
        {
            CodeType iCode = mortonCodes[i];
            EXPECT_TRUE(tree[nodeIndex] <= iCode);
            EXPECT_TRUE(iCode < tree[nodeIndex+1]);
        }
    }
}

class ComputeOctreeTester : public testing::TestWithParam<int>
{
public:
    template<class I, template <class...> class CoordinateType>
    void check(int bucketSize)
    {
        using CodeType = I;
        Box<double> box{-1, 1};

        int nParticles = 100000;

        CoordinateType<double, CodeType> randomBox(nParticles, box);

        // compute octree starting from default uniform octree
        auto [treeML, countsML] = computeOctree(randomBox.mortonCodes().data(),
                                                randomBox.mortonCodes().data() + nParticles,
                                                bucketSize);

        std::cout << "number of nodes: " << nNodes(treeML) << std::endl;

        checkOctreeWithCounts(treeML, countsML, bucketSize, randomBox.mortonCodes());

        // compute octree starting from just the root node
        auto [treeRN, countsRN] = computeOctree(randomBox.mortonCodes().data(),
                                                randomBox.mortonCodes().data() + nParticles,
                                                bucketSize, std::numeric_limits<unsigned>::max(),
                                                makeRootNodeTree<I>());

        checkOctreeWithCounts(treeML, countsRN, bucketSize, randomBox.mortonCodes());

        EXPECT_EQ(treeML, treeRN);
    }
};

TEST_P(ComputeOctreeTester, pingPongRandomNormal32)
{
    check<unsigned, RandomGaussianCoordinates>(GetParam());
}

TEST_P(ComputeOctreeTester, pingPongRandomNormal64)
{
    check<uint64_t, RandomGaussianCoordinates>(GetParam());
}

std::array<int, 3> bucketSizesPP{64, 1024, 10000};

INSTANTIATE_TEST_SUITE_P(RandomBoxPP, ComputeOctreeTester, testing::ValuesIn(bucketSizesPP));


TEST(CornerstoneOctree, computeHaloRadii)
{
    using CodeType = unsigned;

    std::vector<CodeType> tree{0,8,16,24,32};

    std::vector<CodeType> particleCodes{ 0,4, 8,14, 20, 24,25,26,31 };
    std::vector<float>    smoothingLs{   2,1, 4,3,   5, 8,2,1,3};
    std::vector<float>    hMaxPerNode{    4,   8,   10,  16};

    std::vector<int> ordering(particleCodes.size());
    std::iota(begin(ordering), end(ordering), 0);

    std::vector<float> probe(hMaxPerNode.size());

    computeHaloRadii(tree.data(), nNodes(tree), particleCodes.data(), particleCodes.data() + particleCodes.size(),
                     ordering.data(), smoothingLs.data(), probe.data());

    EXPECT_EQ(probe, hMaxPerNode);
}

TEST(CornerstoneOctree, nodeMaxRegression)
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

    EXPECT_TRUE(checkOctreeInvariants(tree.data(), nNodes(tree)));

    std::vector<unsigned> nodeCounts(nNodes(tree), 0);
    nodeCounts[0]        = 2;
    *nodeCounts.rbegin() = 2;

    std::vector<unsigned> codes{0, 0, 1073741823, 1073741823};

    {
        std::vector<unsigned> countsProbe(nNodes(tree));
        computeNodeCounts(tree.data(), countsProbe.data(), nNodes(tree), codes.data(), codes.data() + codes.size());
        EXPECT_EQ(nodeCounts, countsProbe);
    }

    std::vector<double> h{0.2, 0.2, 0.2, 0.2};
    std::vector<int> ordering{0,1,2,3};

    std::vector<double> hMaxPerNode(nNodes(tree), 0);
    computeHaloRadii(tree.data(), nNodes(tree), codes.data(), codes.data() + codes.size(), ordering.data(), h.data(),
                     hMaxPerNode.data());

    std::vector<double> refhMaxPerNode(nNodes(tree));
    refhMaxPerNode[0] = 0.4;
    refhMaxPerNode[nNodes(tree)-1] = 0.4;

    EXPECT_EQ(refhMaxPerNode, hMaxPerNode);
}