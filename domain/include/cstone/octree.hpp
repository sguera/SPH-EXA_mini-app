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
 * \brief Generation of local and global octrees in cornerstone format
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * In the cornerstone format, the octree is stored as sequence of Morton codes
 * fulfilling three invariants. Each code in the sequence both signifies the
 * the start Morton code of an octree leaf node and serves as an upper Morton code bound
 * for the previous node.
 *
 * The invariants of the cornerstone format are:
 *      - code sequence contains code 0 and the maximum code 2^30 or 2^61
 *      - code sequence is sorted by ascending code value
 *      - difference between consecutive elements must be a power of 8
 *
 * The consequences of these invariants are:
 *      - the entire space is covered, i.e. there are no holes in the tree
 *      - only leaf nodes are stored
 *      - for each leaf node, all its siblings (nodes at the same subdivision level with
 *        the same parent) are present in the Morton code sequence
 *      - each node with index i is defined by its lowest possible Morton code at position
 *        i in the vector and the highest possible (excluding) Morton code at position i+1
 *        in the vector
 *      - a vector of length N represents N-1 leaf nodes
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <tuple>

#include "cstone/mortoncode.hpp"

#include "../../include/Timer.hpp"

namespace cstone
{

/*! \brief returns the number of nodes in a tree
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param tree         input tree
 * \return             the number of nodes
 *
 * This makes it explicit that a vector of n Morton codes
 * corresponds to a tree with n-1 nodes.
 */
template<class I>
std::size_t nNodes(const std::vector<I>& tree)
{
    assert(tree.size());
    return tree.size() - 1;
}

/*! \brief count number of particles in each octree node
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param tree         octree nodes given as Morton codes of length @a nNodes+1
 *                     needs to satisfy the octree invariants
 * \param counts       output particle counts per node, length = @a nNodes
 * \param nNodes       number of nodes in tree
 * \param codesStart   Morton code range start of particles to count
 * \param codesEnd     Morton code range end of particles to count
 */
template<class I>
void computeNodeCounts(const I* tree, std::size_t* counts, int nNodes, const I* codesStart, const I* codesEnd)
{
    int firstNode = 0;
    int lastNode  = nNodes;
    if (codesStart != codesEnd)
    {
        firstNode = std::upper_bound(tree, tree + nNodes, *codesStart) - tree - 1;
        lastNode  = std::upper_bound(tree, tree + nNodes, *(codesEnd-1)) - tree;
    }

    #pragma omp parallel for
    for (int i = 0; i < firstNode; ++i)
        counts[i] = 0;

    #pragma omp parallel for
    for (int i = lastNode; i < nNodes; ++i)
        counts[i] = 0;

    #pragma omp parallel for
    for (int i = firstNode; i < lastNode; ++i)
    {
        I nodeStart = tree[i];
        I nodeEnd   = tree[i+1];

        // count particles in range
        auto rangeStart = std::lower_bound(codesStart, codesEnd, nodeStart);
        auto rangeEnd   = std::lower_bound(codesStart, codesEnd, nodeEnd);
        counts[i]       = std::distance(rangeStart, rangeEnd);
    }
}

/*! \brief split or fuse octree nodes based on node counts relative to bucketSize
 *
 * \tparam I               32- or 64-bit unsigned integer type
 * \param[in] tree         octree nodes given as Morton codes of length @a nNodes
 *                         needs to satisfy the octree invariants
 * \param[in] counts       output particle counts per node, length = @a nNodes
 * \param[in] nNodes       number of nodes in tree
 * \param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * \param[out] converged   optional boolean flag to indicate convergence
 * \return                 the rebalanced Morton code octree
 */
template<class I>
std::vector<I> rebalanceTree(const I* tree, const std::size_t* counts, int nNodes,
                             unsigned bucketSize, bool* converged = nullptr)
{
    std::vector<I> balancedTree;
    balancedTree.reserve(nNodes + 1);

    int changes = 0;

    int i = 0;
    while(i < nNodes)
    {
        I thisNode     = tree[i];
        I range        = tree[i+1] - thisNode;
        unsigned level = treeLevel(range);

        if (counts[i] > bucketSize && level < maxTreeLevel<I>{})
        {
            // split
            for (std::size_t sibling = 0; sibling < 8; ++sibling)
            {
                balancedTree.push_back(thisNode + sibling * nodeRange<I>(level + 1));
            }
            changes++;
            i++;
        }
        else if (level > 0 && // level 0 cannot be fused
                 parentIndex(thisNode, level) == 0 &&  // current node is first of 8 siblings
                 tree[i+8] == thisNode + nodeRange<I>(level - 1) && // next 7 nodes are all siblings
                 std::accumulate(counts + i, counts + i + 8, std::size_t(0)) <= bucketSize) // parent count too small
        {
            // fuse, by omitting the 7 siblings
            balancedTree.push_back(thisNode);
            changes++;
            i += 8;
        }
        else
        {
            // do nothing
            balancedTree.push_back(thisNode);
            i++;
        }
    }
    balancedTree.push_back(nodeRange<I>(0));

    if (converged != nullptr)
    {
        *converged = (changes == 0);
    }

    return balancedTree;
}


//! \brief returns a uniform octree with 8^ceil(log8(nBuckets)) leaf nodes
template<class I>
std::vector<I> makeUniformNLevelTree(std::size_t nParticles, int bucketSize)
{
    // the minimum tree level needed is ceil(log8(nParticles/bucketSize))
    unsigned minTreeLevel = log8ceil(I(nParticles/std::size_t(bucketSize)));
    unsigned ticks        = 1u << minTreeLevel;

    std::vector<I> tree;
    tree.reserve(ticks*ticks*ticks + 1);

    // generate regular minTreeLevel tree
    for (unsigned x = 0; x < ticks; ++x)
        for (unsigned y = 0; y < ticks; ++y)
            for (unsigned z = 0; z < ticks; ++z)
            {
                tree.push_back(codeFromBox<I>(x,y,z, minTreeLevel));
            }

    tree.push_back(nodeRange<I>(0));

    sort(begin(tree), end(tree));

    return tree;
}


/*! \brief compute an octree from morton codes for a specified bucket size
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param codesStart   particle morton code sequence start
 * \param codesEnd     particle morton code sequence end
 * \param bucketSize   maximum number of particles/codes per octree leaf node
 * \param[inout] tree  initial tree for the first iteration
 * \return             the tree and the node counts
 */
template<class I, class Reduce = void>
std::tuple<std::vector<I>, std::vector<std::size_t>>
computeOctree(const I* codesStart, const I* codesEnd, unsigned bucketSize, std::vector<I>&& tree = std::vector<I>(0))
{
    if (!tree.size())
    {
        // tree containing just the root node
        tree.push_back(0);
        tree.push_back(nodeRange<I>(0));
    }

    int myRank = 0;
    if constexpr (!std::is_same_v<void, Reduce>)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    }
    sphexa::MasterProcessTimer timer(std::cout, myRank);

    std::vector<std::size_t> counts(nNodes(tree));

    timer.start();

    bool converged = false;
    while (!converged)
    {
        computeNodeCounts(tree.data(), counts.data(), nNodes(tree), codesStart, codesEnd);
        timer.step("            octree::count");
        if constexpr (!std::is_same_v<void, Reduce>) Reduce{}(counts);
        timer.step("            octree::reduce");
        std::vector<I> balancedTree =
            rebalanceTree(tree.data(), counts.data(), nNodes(tree), bucketSize, &converged);
        timer.step("            octree::rebalance");

        swap(tree, balancedTree);
        counts.resize(nNodes(tree));
    }

    return std::make_tuple(std::move(tree), std::move(counts));
}

/*! \brief Compute the halo radius of each node in the given octree
 *
 * This is the maximum distance beyond the node boundaries that a particle outside the
 * node could possibly interact with.
 *
 * TODO: Don't calculate the maximum smoothing length, calculate the maximum distance by
 *       which any of the particles plus radius protrude outside the node.
 *
 * \tparam I           32- or 64-bit unsigned integer type
 * \param tree         octree nodes given as Morton codes of length @a nNodes+1
 *                     This function does not rely on octree invariants, sortedness of the nodes
 *                     is the only requirement.
 * \param nNodes       number of nodes in tree
 * \param codesStart   sorted Morton code range start of particles to count
 * \param codesEnd     sorted Morton code range end of particles to count
 * \param ordering     Access input according to \a ordering
 *                     The sequence input[ordering[i]], i=0,...,N must list the elements of input
 *                     (i.e. the smoothing lengths) such that input[i] is a property of the particle
 *                     (x[i], y[i], z[i]), with x,y,z sorted according to Morton ordering.
 * \param input        Radii per particle, i.e. the smoothing lengths in SPH, length = codesEnd - codesStart
 * \param output       Radius per node, length = @a nNodes
 */
template<class I, class T>
void computeHaloRadii(const I* tree, int nNodes, const I* codesStart, const I* codesEnd,
                      const int* ordering, const T* input, T* output)
{
    int firstNode = 0;
    int lastNode  = nNodes;
    if (codesStart != codesEnd)
    {
        firstNode = std::upper_bound(tree, tree + nNodes, *codesStart) - tree - 1;
        lastNode  = std::upper_bound(tree, tree + nNodes, *(codesEnd-1)) - tree;
    }

    #pragma omp parallel for default(shared)
    for (int i = 0; i < firstNode; ++i)
        output[i] = 0;

    #pragma omp parallel for default(shared)
    for (int i = lastNode; i < nNodes; ++i)
        output[i] = 0;

    #pragma omp parallel for default(shared)
    for (int i = firstNode; i < lastNode; ++i)
    {
        I nodeStart = tree[i];
        I nodeEnd   = tree[i+1];

        // find elements belonging to particles in node i
        int startIndex = std::lower_bound(codesStart, codesEnd, nodeStart) - codesStart;
        int endIndex   = std::lower_bound(codesStart, codesEnd, nodeEnd)   - codesStart;

        T nodeMax = 0;
        for(int p = startIndex; p < endIndex; ++p)
        {
            T nodeElement = input[ordering[p]];
            nodeMax       = std::max(nodeMax, nodeElement);
        }

        // note factor of 2 due to SPH conventions
        output[i] = 2 * nodeMax;
    }
}

//////////////////////////////////////////////////////////
// Tested, but not currently used functionality
// the code below implements an alternative octree format
// where each node has an upper and lower bound.
// Therefore, this format allows to have holes in the space filling curve
// of morton codes - or, in octree language, empty nodes can be omitted,
// such that not all the siblings of a node in the tree are guaranteed to exist.
//////////////////////////////////////////////////////////

//! \brief Defines data to describe an octree node
template<class I>
struct SfcNode
{
    /*! The morton start and end codes define the scope of the node.
     *  They are equivalent to an integer (ix,iy,iz) index triple that describes the
     *  spatial location of the box plus the octree division level (or the size of the box)
     */
    I startCode;
    I endCode;

    /*! The particle content of the node.
     *
     *  coordinateIndex: Stores the index of the first morton code in the full array
     *                   of sorted morton codes that falls into the node.
     *  count:           Number of morton codes in this node. Since the morton code
     *                   array is assumed to be sorted, all particles in the range
     *                   [coordinateIndex, coordinateIndex + count] fall into the node.
     *
     *  Both indices are also valid for the x,y,z coordinate arrays, provided that they
     *  are sorted according to the ascending morton code ordering.
     */
    unsigned coordinateIndex;
    unsigned count;
};

template<class I>
inline bool operator<(const SfcNode<I>& lhs, const SfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode) < std::tie(rhs.startCode, rhs.endCode);
}

template<class I>
inline bool operator==(const SfcNode<I>& lhs, const SfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode, lhs.coordinateIndex, lhs.count)
           == std::tie(rhs.startCode, rhs.endCode, rhs.coordinateIndex, rhs.count);

}

//! \brief Defines data to describe an octree node, no coordinate reference
template<class I>
struct GlobalSfcNode
{
    GlobalSfcNode(I start, I end, [[maybe_unused]] unsigned ignore, std::size_t c)
        : startCode(start), endCode(end), count(c) { }

    //! start and end codes
    I startCode;
    I endCode;

    //! The particle content of the node.
    std::size_t count;
};

template<class I>
inline bool operator<(const GlobalSfcNode<I>& lhs, const GlobalSfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode) < std::tie(rhs.startCode, rhs.endCode);
}

template<class I>
inline bool operator==(const GlobalSfcNode<I>& lhs, const GlobalSfcNode<I>& rhs)
{
    return std::tie(lhs.startCode, lhs.endCode, lhs.count)
           == std::tie(rhs.startCode, rhs.endCode, rhs.count);

}

/*! \brief aggregate mortonCodes into octree leaf nodes of increased size
 *
 * \tparam NodeType       SfcNode, either with or without offset into coordinate arrays
 * \tparam I              32- or 64-bit unsigned integer
 * \param[in] mortonCodes input mortonCode array
 * \param[in] bucketSize  determine size of octree nodes such that
 *                        (leaf node).count <= bucketSize
 *                        and for their parents (<=> internal nodes)
 *                        (parent node).count > bucketSize
 *
 * \return vector with the sorted octree leaf nodes
 */
template<template<class> class NodeType, class I>
std::vector<NodeType<I>> trimZCurve(const std::vector<I>& mortonCodes, unsigned bucketSize)
{
    std::vector<SfcNode<I>> ret;

    unsigned n = mortonCodes.size();
    unsigned i = 0;

    I previousBoxEnd = 0;

    while (i < n)
    {
        I code = mortonCodes[i];

        // the smallest code more than bucketSize away
        // need to find a box that stays below it
        I codeLimit = (i + bucketSize < n) ? mortonCodes[i + bucketSize] : nodeRange<I>(0);

        // find smallest j in [i, i + bucketSize], such that codeLimit < get<1>(smallestCommonBox(mCodes[i], mCodes[j]))
        auto isInBox = [code](I c1_, I c2_){ return c1_ < smallestCommonBox(code, c2_)[1]; };
        auto jIt = std::upper_bound(cbegin(mortonCodes) + i, cbegin(mortonCodes) + std::min(n, i + bucketSize), codeLimit, isInBox);
        unsigned j = jIt - cbegin(mortonCodes);

        // find smallest k in [i, i + bucketSize], such that not(get<0>(smallestCommonBox(mCodes[i], mCodes[k])) < previousBoxEnd)
        auto isBelowBox = [code](I c1_, I c2_){ return !(smallestCommonBox(code, c1_)[0] < c2_); };
        auto kIt = std::lower_bound(cbegin(mortonCodes) + i, cbegin(mortonCodes) + std::min(n, i + bucketSize), previousBoxEnd, isBelowBox);
        unsigned k = kIt - cbegin(mortonCodes);

        // the smaller of the two indices is the one that produces a range of codes
        // with an enclosing octree box that both doesn't overlap with the previous one
        // and does not include more than bucketSize particles
        j = std::min(j, k);

        pair<I> box = smallestCommonBox(code, mortonCodes[j-1]);
        ret.push_back(NodeType<I>{box[0], box[1], i, j-i});
        i = j;
        previousBoxEnd = box[1];
    }

    return ret;
}

} // namespace cstone
