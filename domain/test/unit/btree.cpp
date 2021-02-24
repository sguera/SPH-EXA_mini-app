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
 * \brief Binary radix tree creation tests
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include "gtest/gtest.h"

#include "cstone/btree.hpp"
#include "cstone/mortonconversions.hpp"
#include "cstone/octree.hpp"

/*! \brief \file tests for binary tree generation
 */

using namespace cstone;

//! \brief check binary node prefixes
template <class I>
void internal4x4x4PrefixTest()
{
    // a tree with 4 subdivisions along each dimension, 64 nodes
    std::vector<I> tree = makeUniformNLevelTree<I>(64, 1);

    auto internalTree = createInternalTree(tree);
    EXPECT_EQ(internalTree[0].prefixLength, 0);
    EXPECT_EQ(internalTree[0].prefix, 0);

    EXPECT_EQ(internalTree[31].prefixLength, 1);
    EXPECT_EQ(internalTree[31].prefix, pad(I(0b0), 1));
    EXPECT_EQ(internalTree[32].prefixLength, 1);
    EXPECT_EQ(internalTree[32].prefix, pad(I(0b1), 1));

    EXPECT_EQ(internalTree[15].prefixLength, 2);
    EXPECT_EQ(internalTree[15].prefix, pad(I(0b00), 2));
    EXPECT_EQ(internalTree[16].prefixLength, 2);
    EXPECT_EQ(internalTree[16].prefix, pad(I(0b01), 2));

    EXPECT_EQ(internalTree[7].prefixLength, 3);
    EXPECT_EQ(internalTree[7].prefix, pad(I(0b000), 3));
    EXPECT_EQ(internalTree[8].prefixLength, 3);
    EXPECT_EQ(internalTree[8].prefix, pad(I(0b001), 3));

    // second (useless) root node
    EXPECT_EQ(internalTree[63].prefixLength, 0);
    EXPECT_EQ(internalTree[63].prefix, 0);
}

TEST(BinaryTree, internalTree4x4x4PrefixTest)
{
    internal4x4x4PrefixTest<unsigned>();
    internal4x4x4PrefixTest<uint64_t>();
}


/*! Create a set of irregular octree leaves which do not cover the whole space
 *
 * This example is illustrated in the original paper referenced in sfc/binarytree.hpp.
 * Please refer to the publication for a graphical illustration of the resulting
 * node connectivity.
 */

template<class I>
std::vector<I> makeExample()
{
    std::vector<I> ret
        {
            pad(I(0b00001), 5),
            pad(I(0b00010), 5),
            pad(I(0b00100), 5),
            pad(I(0b00101), 5),
            pad(I(0b10011), 5),
            pad(I(0b11000), 5),
            pad(I(0b11001), 5),
            pad(I(0b11110), 5),
        };
    return ret;
}

template<class I>
void findSplitTest()
{
    std::vector<I> example = makeExample<I>();

    {
        int split = findSplit(example.data(), 0, 7);
        EXPECT_EQ(split, 3);
    }
    {
        int split = findSplit(example.data(), 0, 3);
        EXPECT_EQ(split, 1);
    }
    {
        int split = findSplit(example.data(), 4, 7);
        EXPECT_EQ(split, 4);
    }
}

TEST(BinaryTree, findSplit)
{
    findSplitTest<unsigned>();
    findSplitTest<uint64_t>();
}

template<class I>
void paperExampleTest()
{
    using CodeType = I;

    std::vector<CodeType> example = makeExample<CodeType>();
    std::vector<BinaryNode<CodeType>> internalNodes(example.size() - 1);
    for (std::size_t i = 0; i < internalNodes.size(); ++i)
    {
        constructInternalNode(example.data(), example.size(), internalNodes.data(), i);
    }

    std::vector<BinaryNode<CodeType>*> refLeft
        {
            internalNodes.data() + 3,
            nullptr,
            nullptr,
            internalNodes.data() + 1,
            nullptr,
            internalNodes.data() + 6,
            nullptr
        };

    std::vector<BinaryNode<CodeType>*> refRight
        {
            internalNodes.data() + 4,
            nullptr,
            nullptr,
            internalNodes.data() + 2,
            internalNodes.data() + 5,
            nullptr,
            nullptr
        };

    std::vector<int> refLeftIndices {-1, 0, 2, -1, 4, -1, 5};
    std::vector<int> refRightIndices{-1, 1, 3, -1, -1, 7, 6};

    std::vector<int> refPrefixLengths{0, 3, 4, 2, 1, 2, 4};

    using Node = BinaryNode<I>;
    for (std::size_t idx = 0; idx < internalNodes.size(); ++idx)
    {
        EXPECT_EQ(internalNodes[idx].child[Node::left], refLeft[idx]);
        EXPECT_EQ(internalNodes[idx].leafIndex[Node::left],  refLeftIndices[idx]);
        EXPECT_EQ(internalNodes[idx].child[Node::right], refRight[idx]);
        EXPECT_EQ(internalNodes[idx].leafIndex[Node::right], refRightIndices[idx]);
        EXPECT_EQ(internalNodes[idx].prefixLength,   refPrefixLengths[idx]);
    }
}

TEST(BinaryTree, internalIrregular)
{
    paperExampleTest<unsigned>();
    paperExampleTest<uint64_t>();
}
