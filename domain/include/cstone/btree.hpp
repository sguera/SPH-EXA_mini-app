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

/*! \brief \file parallel binary radix tree construction implementation
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Algorithm published in https://dl.acm.org/doi/10.5555/2383795.2383801
 * and further illustrated at
 * https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
 *
 * Most SFC-based functionality for SPH like domain decomposition, global octree build
 * and neighbor search does not need tree-traversal and explicit construction
 * of tree-internal nodes.
 *
 * The halo search problem is an exception. Finding all nodes in the global octree
 * that overlap with a given node that is enlarged by the search radius in all
 * dimensions is essentially the same as collision detection between objects
 * in 3D graphics, where bounding volume hierarchies are constructed from binary radix trees.
 *
 * The reason why the halo search problem differs from the other SFC-based algorithms
 * for neighbor search and domain decomposition is that an octree node that is enlarged
 * by an arbitrary distance in each dimension cannot necessarily be composed from octree
 * nodes of similar size as the query node and therefore also not as a range of Morton codes,
 * or the sum of a small number of Morton code ranges. This is especially true at the boundaries
 * between octree nodes of different division level, i.e. when large nodes are next to
 * very small nodes. For this reason, nodes enlarged by a halo radius are best represented
 * by separate x,y,z coordinate ranges.
 *
 * The best way to implement collision detection for the 3D box defined x,y,z coordinate ranges
 * is by constructing the internal tree part over the
 * global octree leaf nodes as a binary radix tree. Each internal node of the binary
 * radix tree can be constructed independently and thus the algorithm is ideally suited
 * for GPUs. Subsequent tree traversal to detect collisions can also be done for all leaf
 * nodes in parallel. It is possible to convert the internal binary tree into an octree,
 * as 3 levels in the binary tree correspond to one level in the equivalent octree.
 * Doing so could potentially speed up traversal by a bit, but it is not clear whether it
 * would make up for the overhead of constructing the internal octree.
 */

#pragma once

#include "cstone/mortoncode.hpp"
#include "cstone/boxoverlap.hpp"

namespace cstone {

using TreeNodeIndex = int;

/*! \brief binary radix tree node
 *
 * @tparam I 32- or 64 bit unsigned integer
 */
template<class I>
struct BinaryNode
{
    enum ChildType{ left = 0, right = 1 };

    /*! \brief pointer to the left and right children nodes
     *
     * The left child adds a 0-bit to the prefix, while
     * the right child adds a 1-bit to the prefix.
     *
     * If the child is an octree node, the pointer is nullptr and left/rightLeafIndex
     * is set instead.
     */
    BinaryNode* child[2];

    /*! \brief the Morton code prefix
     *
     * Shared among all the node's children. Only the first prefixLength bits are relevant.
     */
    I   prefix;

    //! \brief number of bits in prefix to interpret
    int prefixLength;

    /*! \brief Indices of leaf codes
     *
     * If the left/right child is a leaf code of the tree used to construct the internal binary tree,
     * this integer stores the index (e.g. the global octree), otherwise it will be set to -1
     * if the children are also internal binary nodes.
     */
    TreeNodeIndex leafIndex[2];
};


/*! \brief find position of first differing bit
 *
 * @tparam I                 32 or 64 bit unsigned integer
 * @param sortedMortonCodes
 * @param first              first range index
 * @param last               last rang index
 * @return                   position of morton
 */
template<class I>
int findSplit(I*  sortedMortonCodes,
              int first,
              int last)
{
    // Identical Morton codes => split the range in the middle.
    I firstCode = sortedMortonCodes[first];
    I lastCode  = sortedMortonCodes[last];

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same for all objects
    int cpr = commonPrefix(firstCode, lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest Morton code that
    // shares more than commonPrefix bits with the first one.
    int split = first; // initial guess
    int step = last - first;

    do
    {
        step = (step + 1) / 2;      // exponential decrease
        int newSplit = split + step; // proposed new position

        if (newSplit < last)
        {
            I splitCode = sortedMortonCodes[newSplit];
            int splitPrefix = commonPrefix(firstCode, splitCode);
            if (splitPrefix > cpr)
                split = newSplit; // accept proposal
        }
    }
    while (step > 1);

    return split;
}

/*! \brief construct the internal binary tree node with index idx
 *
 * @tparam I                  32- or 64-bit unsigned integer
 * @param codes[in]           sorted Morton code sequence without duplicates
 * @param nCodes[in]          number of elements in \a codes
 * @param internalNodes[out]  output internal binary radix tree, size is nCodes - 1
 * @param firstIndex          element of \a internalNodes to construct,
 *                            permissible range is 0 <= firstIndex < nCodes -1
 */
template<class I>
void constructInternalNode(const I* codes, int nCodes, BinaryNode<I>* internalNodes, int firstIndex)
{
    BinaryNode<I>* outputNode = internalNodes + firstIndex;

    int d = 1;
    int minPrefixLength = -1;

    if (firstIndex > 0)
    {
        d = (commonPrefix(codes[firstIndex], codes[firstIndex + 1]) >
             commonPrefix(codes[firstIndex], codes[firstIndex - 1])) ? 1 : -1;
        minPrefixLength = commonPrefix(codes[firstIndex], codes[firstIndex -d]);
    }

    // determine searchRange, the maximum distance of secondIndex from firstIndex
    int searchRange = 2;
    int secondIndex = firstIndex + searchRange * d;
    while(0 <= secondIndex && secondIndex < nCodes
          && commonPrefix(codes[firstIndex], codes[secondIndex]) > minPrefixLength)
    {
        searchRange *= 2;
        secondIndex = firstIndex + searchRange * d;
    }

    // start binary search with known searchRange
    secondIndex = firstIndex;
    do
    {
        searchRange = (searchRange + 1) / 2;
        int newJdx = secondIndex + searchRange * d;
        if (0 <= newJdx && newJdx < nCodes
            && commonPrefix(codes[firstIndex], codes[newJdx]) > minPrefixLength)
        {
            secondIndex = newJdx;
        }
    } while (searchRange > 1);

    outputNode->prefixLength = commonPrefix(codes[firstIndex], codes[secondIndex]);
    outputNode->prefix       = zeroLowBits(codes[firstIndex], outputNode->prefixLength);

    // find position of highest differing bit between [firstIndex, secondIndex]
    int gamma = findSplit(codes, std::min(secondIndex, firstIndex), std::max(secondIndex, firstIndex));

    // establish child relationships
    if (std::min(secondIndex, firstIndex) == gamma)
    {
        // left child is a leaf
        outputNode->child[BinaryNode<I>::left]    = nullptr;
        outputNode->leafIndex[BinaryNode<I>::left] = gamma;
    }
    else
    {
        //left child is an internal binary node
        outputNode->child[BinaryNode<I>::left]    = internalNodes + gamma;
        outputNode->leafIndex[BinaryNode<I>::left] = -1;
    }

    if (std::max(secondIndex, firstIndex) == gamma + 1)
    {
        // right child is a leaf
        outputNode->child[BinaryNode<I>::right]    = nullptr;
        outputNode->leafIndex[BinaryNode<I>::right] = gamma + 1;
    }
    else
    {
        // right child is an internal binary node
        outputNode->child[BinaryNode<I>::right]    = internalNodes + gamma + 1;
        outputNode->leafIndex[BinaryNode<I>::right] = -1;
    }
}


/*! \brief create the internal part of an octree as internal nodes
 *
 * @tparam I    32- or 64-bit unsigned integer
 * @param tree  Sorted Morton codes representing the leaves of the (global) octree
 *              or the locations of objects in 3D.
 *              Cornerstone invariants are not a requirement for this function,
 *              only that the codes be sorted and not contain any duplicates.
 * @return      the internal part of the input tree constructed as binary nodes
 *
 * Note that if the input \a tree is a cornerstone octree, the root node with index
 * 0 in the returned binary tree only maps binary nodes 0 <= ... < tree.size() -1.
 * Due to the last element of tree being the maximum Morton code 2^(30 or 61),
 * the last node/element of the returned binary tree will be set up as a useless
 * second root node that is not reachable from the root node with index 0.
 * So if \a tree is a cornerstone octree of size N, there are / will be
 *      - N-1 octree leaf nodes
 *      - a binary tree of size N-1 with 0...N-2 as usable elements
 *
 * One could of course prevent the generation of the last binary node with index N-1,
 * but that would result in loss of generality for arbitrary sorted Morton code sequences
 * without duplicates.
 *
 * This is a CPU version that can be OpenMP parallelized.
 * In the GPU version, the for-loop body is designed such that one GPU-thread
 * can be launched for each for-loop element.
 */
template<class I>
std::vector<BinaryNode<I>> createInternalTree(const std::vector<I>& tree)
{
    std::vector<BinaryNode<I>> ret(tree.size() - 1);

    #pragma omp parallel for
    for (std::size_t idx = 0; idx < ret.size(); ++idx)
    {
        constructInternalNode(tree.data(), tree.size(), ret.data(), idx);
    }

    return ret;
}

} // namespace cstone
