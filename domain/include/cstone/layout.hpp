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
 * \brief Utility functions for determining the layout of particle buffers on a given rank
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * Each rank will be assigned a part of the SFC, equating to one or multiple ranges of
 * node indices of the global cornerstone octree. In a addition to the assigned nodes,
 * each rank must also store particle data for those nodes in the global octree which are
 * halos of the assigned nodes. Both types of nodes present on the rank are stored in the same
 * particle array (x,y,z,h,...) according to increasing node index, which is the same
 * as increasing Morton code.
 *
 * Given
 *  - the global cornerstone tree
 *  - its assignment to ranks
 *  - lists of in/outgoing halo nodes (global indices) per rank,
 * the utility functions in this file determine the position and size of each node (halo or assigned node)
 * in the particle buffers. The resulting layout is valid for all particle buffers, such as x,y,z,h,d,p,...
 *
 * Note:
 * If a node of the global cornerstone octree has index i, this means its Morton code range is tree[i] - tree[i+1]
 */

#pragma once

#include <vector>

#include "cstone/domaindecomp.hpp"

namespace cstone
{

/*! \brief  Finds the ranges of node indices of the tree that are assigned to a given rank
 *
 * @tparam I           32- or 64-bit unsigned integer
 * @param tree         global cornerstone octree
 * @param assignment   assignment of Morton code ranges to ranks
 * @param rank         extract rank's part from assignment
 * @return             ranges of node indices in \a tree that belong to rank \a rank
 */
template<class I>
static std::vector<int> computeLocalNodeRanges(const std::vector<I>& tree,
                                               const SpaceCurveAssignment<I>& assignment,
                                               int rank)
{
    std::vector<int> ret;

    for (std::size_t rangeIndex = 0; rangeIndex < assignment.nRanges(rank); ++rangeIndex)
    {
        int firstNodeIndex  = std::lower_bound(begin(tree), end(tree),
                                               assignment.rangeStart(rank, rangeIndex)) - begin(tree);
        int secondNodeIndex = std::lower_bound(begin(tree), end(tree),
                                               assignment.rangeEnd(rank, rangeIndex)) - begin(tree);

        ret.push_back(firstNodeIndex);
        ret.push_back(secondNodeIndex);
    }

    return ret;
}

//! \brief create a sorted list of nodes from the hierarchical per rank node list
static std::vector<int> flattenNodeList(const std::vector<std::vector<int>>& groupedNodes)
{
    int nNodes = 0;
    for (auto& v : groupedNodes) nNodes += v.size();

    std::vector<int> nodeList;
    nodeList.reserve(nNodes);

    // add all halos to nodeList
    for (const auto& group : groupedNodes)
    {
        std::copy(begin(group), end(group), std::back_inserter(nodeList));
    }

    return nodeList;
}

/*! \brief computes the array layout for particle buffers of the executing rank
 *
 * @param localNodeRanges    Ranges of node indices, assigned to executing rank
 * @param haloNodes          List of halo node indices without duplicates.
 *                           From the perspective of the
 *                           executing rank, these are incoming halo nodes.
 * @param globalNodeCounts   Particle count per node in the global octree
 * @param presentNodes[out]  Upon return, will contain a sorted list of global node indices
 *                           present on the executing rank
 * @param offsets[out]       Will contain an offset index for each node in \a presentNodes,
 *                           indicating its position in the particle x,y,z,... buffers
 */
template<class IndexType>
static void computeLayoutOffsets(const std::vector<int>& localNodeRanges,
                                 const std::vector<int>& haloNodes,
                                 const std::vector<unsigned>& globalNodeCounts,
                                 std::vector<int>& presentNodes,
                                 std::vector<IndexType>& offsets)
{
    // add all halo nodes to present
    std::copy(begin(haloNodes), end(haloNodes), std::back_inserter(presentNodes));

    // add all local nodes to presentNodes
    for (std::size_t rangeIndex = 0; rangeIndex < localNodeRanges.size(); rangeIndex += 2)
    {
        int lower = localNodeRanges[rangeIndex];
        int upper = localNodeRanges[rangeIndex+1];
        for (int i = lower; i < upper; ++i)
            presentNodes.push_back(i);
    }

    std::sort(begin(presentNodes), end(presentNodes));

    // an extract of globalNodeCounts, containing only nodes listed in localNodes and incomingHalos
    std::vector<int> nodeCounts(presentNodes.size());

    // extract particle count information for all nodes in nodeList
    for (std::size_t i = 0; i < presentNodes.size(); ++i)
    {
        int globalNodeIndex = presentNodes[i];
        nodeCounts[i]       = globalNodeCounts[globalNodeIndex];
    }

    offsets.resize(presentNodes.size() + 1);
    {
        IndexType offset = 0;
        for (std::size_t i = 0; i < presentNodes.size(); ++i)
        {
            offsets[i] = offset;
            offset += nodeCounts[i];
        }
        // the last element stores the total size of the layout
        offsets[presentNodes.size()] = offset;
    }
}


/*! \brief translate global node indices involved in halo exchange to array ranges
 *
 * @param outgoingHaloNodes   For each rank, a list of global node indices to send or receive.
 *                            Each node referenced in these lists must be contained in
 *                            \a presentNodes.
 * @param presentNodes        Sorted unique list of global node indices present on the
 *                            executing rank
 * @param nodeOffsets         nodeOffset[i] stores the location of the node presentNodes[i]
 *                            in the particle arrays. nodeOffset[presentNodes.size()] stores
 *                            the total size of the particle arrays on the executing rank.
 *                            Size is presentNodes.size() + 1
 * @return                    For each rank, the returned sendList has one or multiple ranges of indices
 *                            of local particle arrays to send or receive.
 */
template<class IndexType>
static SendList createHaloExchangeList(const std::vector<std::vector<int>>& outgoingHaloNodes,
                                       const std::vector<int>& presentNodes,
                                       const std::vector<IndexType>& nodeOffsets)
{
    SendList sendList(outgoingHaloNodes.size());

    for (std::size_t rank = 0; rank < sendList.size(); ++rank)
    {
        for (int globalNodeIndex : outgoingHaloNodes[rank])
        {
            int localNodeIndex = std::lower_bound(begin(presentNodes), end(presentNodes), globalNodeIndex)
                                - begin(presentNodes);

            sendList[rank].addRange(nodeOffsets[localNodeIndex], nodeOffsets[localNodeIndex+1]);
        }
    }

    return sendList;
}


} // namespace cstone