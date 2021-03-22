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
 * \brief  Compute the internal part of a cornerstone octree
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * General algorithm:
 *      cornerstone octree (leaves) -> internal binary radix tree -> internal octree
 *
 * Like the cornerstone octree, the internal octree is stored in a linear memory layout
 * with tree nodes placed next to each other in a single buffer. Construction
 * is fully parallel and non-recursive and non-iterative. Traversal is possible non-recursively
 * in an iterative fashion with a local stack.
 */

#pragma once

#include <iterator>
#include <vector>

#include "cstone/cuda/annotation.hpp"
#include "cstone/primitives/gather.hpp"
#include "cstone/sfc/mortoncode.hpp"

#include "btree.hpp"
#include "octree.hpp"


namespace cstone
{

/*! \brief octree node for the internal part of cornerstone octrees
 *
 * @tparam I  32- or 64 bit unsigned integer
 */
template<class I>
struct OctreeNode
{
    //! \brief named bool to tag children of internal octree nodes
    enum ChildType : bool
    {
        internal = true,
        leaf     = false
    };

    /*! \brief the Morton code prefix
     *
     * Shared among all the node's children. Only the first prefixLength bits are relevant.
     */
    I   prefix;

    //! \brief octree division level, equals 1/3rd the number of bits in prefix to interpret
    int level;

    //! \brief internal node index of the parent node
    TreeNodeIndex parent;

    /*! \brief Child node indices
     *
     *  If childType[i] is ChildType::internal, child[i] is an internal node index,
     *  If childType[i] is ChildType::leaf, child[i] is the index of an octree leaf node.
     *  Note that the indices in these two cases refer to two different arrays!
     */
    TreeNodeIndex child[8];
    ChildType     childType[8];

    friend bool operator==(const OctreeNode<I>& lhs, const OctreeNode<I>& rhs)
    {
        bool eqChild = true;
        for (int i = 0; i < 8; ++i)
        {
            eqChild = eqChild &&
                      lhs.child[i]     == rhs.child[i] &&
                      lhs.childType[i] == rhs.childType[i];
        }

        return lhs.prefix == rhs.prefix &&
               lhs.level  == rhs.level &&
               lhs.parent == rhs.parent &&
               eqChild;
    }
};

/*! \brief atomically update a maximum value and return the previous maximum value
 *
 * @tparam T                     integer type
 * @param maximumValue[inout]    the maximum value to be atomically updated
 * @param newValue[in]           the value with which to compute the new maximum
 * @return                       the previous maximum value
 */
template<typename T>
T atomicMax(std::atomic<T>& maximumValue, const T& newValue) noexcept
{
    T previousValue = maximumValue;
    while(previousValue < newValue && !maximumValue.compare_exchange_weak(previousValue, newValue))
    {}
    return previousValue;
}

/*! \brief calculate distance to farthest leaf for each internal node in parallel
 *
 * @tparam I             32- or 64-bit integer type
 * @param octree[in]     an octree, length @a nNodes
 * @param nNodes[in]     number of (internal) nodes
 * @param depths[out]    array of length @a nNodes, contains
 *                       the distance to the farthest leaf for each node.
 *                       The distance is equal to 1 for each node whose children are only leaves.
 */
template<class I>
void nodeDepth(const OctreeNode<I>* octree, TreeNodeIndex nNodes, std::atomic<TreeNodeIndex>* depths)
{
    #pragma omp parallel for
    for (TreeNodeIndex i = 0; i < nNodes; ++i)
    {
        int nLeafChildren = 0;
        for (int octant = 0; octant < 8; ++octant)
        {
            if (octree[i].childType[octant] == OctreeNode<I>::leaf)
            {
                nLeafChildren++;
            }
        }

        if (nLeafChildren == 8) { depths[i] = 1; } // all children are leaves - maximum depth is 1
        else                    { continue; }

        TreeNodeIndex nodeIndex = i;
        TreeNodeIndex depth = 1;

        // race to the top
        do
        {   // ascend one level
            nodeIndex = octree[nodeIndex].parent;
            depth++;

            // set depths[nodeIndex] = max(depths[nodeIndex], depths) and store previous value
            // of depths[nodeIndex] in previousMax
            TreeNodeIndex previousMax = atomicMax(depths[nodeIndex], depth);
            if (previousMax >= depth)
            {
                // another thread already set a higher value for depths[nodeIndex], drop out of race
                break;
            }

        } while (nodeIndex != octree[nodeIndex].parent);
    }
}

/*! @brief calculates the tree node ordering for descending max leaf distance
 *
 * @tparam I                    32- or 64-bit integer
 * @param octree[in]            input array of OctreeNode<I>
 * @param nNodes[in]            number of input octree nodes
 * @param ordering[out]         output ordering, a permutation of [0:nNodes]
 * @param nNodesPerLevel[out]   number of nodes per value of farthest leaf distance
 *                              length is maxTreeLevel<I>{} (10 or 21)
 *
 *  nNodesPerLevel[0] --> number of leaf nodes (NOT set in this function)
 *  nNodesPerLevel[1] --> number of nodes with maxDepth = 1, children: only leaves
 *  nNodesPerLevel[2] --> number of internal nodes with maxDepth = 2, children: leaves and maxDepth = 1
 *
 * First calculates the distance of the farthest leaf for each node, then
 * determines an ordering that sorts the nodes according to decreasing distance.
 * The resulting ordering is valid from a scatter-perspective, i.e.
 * Nodes at <oldIndex> should be moved to @a ordering[<oldIndex>].
 */
template<class I>
void decreasingMaxDepthOrder(const OctreeNode<I>* octree,
                             TreeNodeIndex nNodes,
                             TreeNodeIndex* ordering,
                             TreeNodeIndex* nNodesPerLevel)
{
    std::vector<std::atomic<TreeNodeIndex>> depths(nNodes);

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nNodes; ++i)
    {
        depths[i] = 0;
    }

    nodeDepth(octree, nNodes, depths.data());

    // inverse ordering will be the inverse permutation of the
    // output ordering and also corresponds to the ordering
    // from a gather-perspective
    std::vector<TreeNodeIndex> inverseOrdering(nNodes);
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nNodes; ++i)
    {
        inverseOrdering[i] = i;
    }

    std::vector<TreeNodeIndex> depths_v(begin(depths), end(depths));
    // Todo: we need sort_by_key here
    sort_invert(begin(depths_v), end(depths_v), begin(inverseOrdering), std::greater<TreeNodeIndex>{});
    std::sort(begin(depths_v), end(depths_v), std::greater<TreeNodeIndex>{});

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nNodes; ++i)
    {
        ordering[i] = i;
    }

    sort_invert(begin(inverseOrdering), end(inverseOrdering), ordering);

    // count nodes per value of depth
    for (TreeNodeIndex depth = 1; depth < maxTreeLevel<I>{}; ++depth)
    {
        auto it1 = std::lower_bound(begin(depths_v), end(depths_v), depth, std::greater<TreeNodeIndex>{});
        auto it2 = std::upper_bound(begin(depths_v), end(depths_v), depth, std::greater<TreeNodeIndex>{});
        nNodesPerLevel[depth] = std::distance(it1, it2);
    }
}

/*! \brief reorder internal octree nodes according to a map
 *
 * @tparam I                   32- or 64-bit unsigned integer
 * @param oldNodes[in]         array of octree nodes, length @a nInternalNodes
 * @param rewireMap[in]        a permutation of [0:nInternalNodes]
 * @param nInternalNodes[in]   number of internal octree nodes
 * @param newNodes[out]        reordered array of octree nodes, length @a nInternalNodes
 */
template<class I>
void rewireInternal(const OctreeNode<I>* oldNodes,
                    const TreeNodeIndex* rewireMap,
                    TreeNodeIndex nInternalNodes,
                    OctreeNode<I>* newNodes)
{
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex oldIndex = 0; oldIndex < nInternalNodes; ++oldIndex)
    {
        // node at <oldIndex> moves to <newIndex>
        TreeNodeIndex newIndex = rewireMap[oldIndex];

        OctreeNode<I> newNode = oldNodes[oldIndex];
        newNode.parent = rewireMap[newNode.parent];
        for (int octant = 0; octant < 8; ++octant)
        {
            if (newNode.childType[octant] == OctreeNode<I>::internal)
            {
                TreeNodeIndex oldChild = newNode.child[octant];
                newNode.child[octant] = rewireMap[oldChild];
            }
        }

        newNodes[newIndex] = newNode;
    }
}

/*! \brief translate each input index according to a map
 *
 * @tparam Index         integer type
 * @param input[in]      input indices, length @a nElements
 * @param rewireMap[in]  index translation map
 * @param nElements[in]  number of Elements
 * @param output[out]    translated indices, length @a nElements
 */
template<class Index>
void rewireIndices(const Index* input,
                   const Index* rewireMap,
                   size_t nElements,
                   Index* output)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nElements; ++i)
    {
        output[i] = rewireMap[input[i]];
    }
}

/*! \brief construct the internal octree node with index \a nodeIndex
 *
 * @tparam I                       32- or 64-bit unsigned integer type
 * @param internalOctree[out]      linear array of OctreeNode<I>'s
 * @param binaryTree[in]           linear array of binary tree nodes
 * @param nodeIndex[in]            element of @a internalOctree to construct
 * @param octreeToBinaryIndex[in]  octreeToBinaryIndex[i] stores the index of the binary node in
 *                                 @a binaryTree with the identical prefix as the octree node with index i
 * @param binaryToOctreeIndex[in]  inverts @a octreeToBinaryIndex
 * @param leafParents[out]         linear array of indices to store the parent index of each octree leaf
 *                                 number of elements corresponds to the number of nodes in the cornerstone
 *                                 octree that was used to construct @a binaryTree
 *
 * This function sets all members of internalOctree[nodeIndex] except the parent member.
 * (Exception: the parent of the root node is set to 0)
 * In addition, it sets the parent member of the child nodes to \a nodeIndex.
 */
template<class I>
CUDA_HOST_DEVICE_FUN
inline void constructOctreeNode(OctreeNode<I>*       internalOctree,
                                const BinaryNode<I>* binaryTree,
                                TreeNodeIndex        nodeIndex,
                                const TreeNodeIndex* scatterMap,
                                const TreeNodeIndex* binaryToOctreeIndex,
                                TreeNodeIndex*       leafParents)
{
    OctreeNode<I>& octreeNode = internalOctree[nodeIndex];

    TreeNodeIndex bi = scatterMap[nodeIndex]; // binary tree index
    octreeNode.prefix = binaryTree[bi].prefix;
    octreeNode.level  = binaryTree[bi].prefixLength / 3;

    // the root node is its own parent
    if (octreeNode.level == 0)
    {
        octreeNode.parent = 0;
    }

    for (int hx = 0; hx < 2; ++hx)
    {
        for (int hy = 0; hy < 2; ++hy)
        {
            for (int hz = 0; hz < 2; ++hz)
            {
                int octant = 4*hx + 2*hy + hz;
                if (binaryTree[bi].child[hx]->child[hy]->child[hz] != nullptr)
                {
                    TreeNodeIndex childBinaryIndex = binaryTree[bi].child[hx]->child[hy]->child[hz]
                                                     - binaryTree;
                    TreeNodeIndex childOctreeIndex = binaryToOctreeIndex[childBinaryIndex];

                    octreeNode.child[octant]     = childOctreeIndex;
                    octreeNode.childType[octant] = OctreeNode<I>::internal;

                    internalOctree[childOctreeIndex].parent = nodeIndex;
                }
                else
                {
                    TreeNodeIndex octreeLeafIndex = binaryTree[bi].child[hx]->child[hy]->leafIndex[hz];
                    octreeNode.child[octant]      = octreeLeafIndex;
                    octreeNode.childType[octant]  = OctreeNode<I>::leaf;
                    leafParents[octreeLeafIndex]  = nodeIndex;
                }
            }
        }
    }
}

/*! \brief translate an internal binary radix tree into an internal octree
 *
 * @tparam I                   32- or 64-bit unsigned integer
 * @param binaryTree[in        binary tree nodes
 * @param nLeafNodes[in]       number of octree leaf nodes used to construct \a binaryTree
 * @param internalOctree[out]  output internal octree nodes, length = \a (nLeafNodes-1) / 7
 * @param leafParents[out]     node index of the parent node for each leaf, length = \a nLeafNodes
 *
 */
template<class I>
void createInternalOctreeCpu(const BinaryNode<I>* binaryTree, TreeNodeIndex nLeafNodes,
                             OctreeNode<I>* internalOctree, TreeNodeIndex* leafParents)
{
    // we ignore the last binary tree node which is a duplicate root node
    TreeNodeIndex nBinaryNodes = nLeafNodes - 1;

    // one extra element to store the total sum of the exclusive scan
    std::vector<TreeNodeIndex> prefixes(nBinaryNodes + 1);
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nBinaryNodes; ++i)
    {
        int  prefixLength = binaryTree[i].prefixLength;
        bool divisibleBy3 = prefixLength % 3 == 0;
        prefixes[i] = (divisibleBy3) ? 1 : 0;
    }

    // stream compaction: scan and scatter
    exclusiveScan(prefixes.data(), prefixes.size());

    // nInternalOctreeNodes is also equal to prefixes[nBinaryNodes]
    TreeNodeIndex nInternalOctreeNodes = (nLeafNodes-1)/7;
    std::vector<TreeNodeIndex> scatterMap(nInternalOctreeNodes);

    // compaction step, scatterMap -> compacted list of binary nodes that correspond to octree nodes
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nBinaryNodes; ++i)
    {
        bool isOctreeNode = (prefixes[i+1] - prefixes[i]) == 1;
        if (isOctreeNode)
        {
            int octreeNodeIndex = prefixes[i];
            scatterMap[octreeNodeIndex] = i;
        }
    }

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nInternalOctreeNodes; ++i)
    {
        constructOctreeNode(internalOctree, binaryTree, i, scatterMap.data(), prefixes.data(), leafParents);
    }
}


/*! \brief This class unifies a cornerstone octree with the internal part
 *
 * @tparam I          32- or 64-bit unsigned integer
 *
 * Leaves are stored separately from the internal nodes. For either type of node, just a single buffer is allocated.
 * All nodes are guaranteed to be stored ordered according to decreasing value of the distance of the farthest leaf.
 * This property is relied upon by the generic upsweep implementation.
 */
template<class I>
class Octree {
public:
    Octree() : nNodesPerLevel_(maxTreeLevel<I>{}) {}

    /*! \brief sets the leaves to the provided ones and updates the internal part based on them
     *
     * @param firstLeaf  first leaf
     * @param lastLeaf   last leaf
     *
     * internal state change:
     *      -full tree update
     */
    void update(const I* firstLeaf, const I* lastLeaf)
    {
        assert(lastLeaf > firstLeaf);

        // make space for leaf nodes
        TreeNodeIndex treeSize = lastLeaf - firstLeaf;
        nNodesPerLevel_[0]     = treeSize;
        cstoneTree_.resize(treeSize);
        std::copy(firstLeaf, lastLeaf, cstoneTree_.data());

        binaryTree_.resize(nNodes(cstoneTree_));
        createBinaryTree(cstoneTree_.data(), nNodes(cstoneTree_), binaryTree_.data());

        std::vector<OctreeNode<I>> preTree((nNodes(cstoneTree_) - 1) / 7);
        std::vector<TreeNodeIndex> preLeafParents(nNodes(cstoneTree_));

        createInternalOctreeCpu(binaryTree_.data(), nNodes(cstoneTree_), preTree.data(), preLeafParents.data());

        // re-sort internal nodes to establish a max-depth ordering
        std::vector<TreeNodeIndex> ordering(preTree.size());
        // determine ordering
        decreasingMaxDepthOrder(preTree.data(), preTree.size(), ordering.data(), nNodesPerLevel_.data());
        // apply the ordering to the internal tree;
        internalTree_.resize(preTree.size());
        rewireInternal(preTree.data(), ordering.data(), preTree.size(), internalTree_.data());

        // apply ordering to leaf parents
        leafParents_.resize(preLeafParents.size());
        rewireIndices(preLeafParents.data(), ordering.data(), preLeafParents.size(), leafParents_.data());
    }

    //! \brief total number of nodes in the tree
    [[nodiscard]] inline TreeNodeIndex nTreeNodes() const
    {
        return nNodes(cstoneTree_) + internalTree_.size();
    }

    /*! \brief number of nodes with given value of maxDepth
     *
     * @param maxDepth[in]  distance to farthest leaf
     * @return              number of nodes in the tree with given value for maxDepth
     *
     * Some relations with other node-count functions:
     *      nTreeNodes(0) == nLeafNodes()
     *      sum([nTreeNodes(i) for i in [0:maxTreeLevel<I>{}]]) == nTreeNodes()
     *      sum([nTreeNodes(i) for i in [1:maxTreeLevel<I>{}]]) == nInternalNodes()
     */
    [[nodiscard]] inline TreeNodeIndex nTreeNodes(int maxDepth) const
    {
        assert(maxDepth < maxTreeLevel<I>{});
        return nNodesPerLevel_[maxDepth];
    }

    //! \brief number of leaf nodes in the tree
    [[nodiscard]] inline TreeNodeIndex nLeafNodes() const
    {
        return nNodes(cstoneTree_);
    }

    //! \brief number of internal nodes in the tree, equal to (nLeafNodes()-1) / 7
    [[nodiscard]] inline TreeNodeIndex nInternalNodes() const
    {
        return internalTree_.size();
    }

    /*! \brief check whether node is a leaf
     *
     * @param node[in]    node index, range [0:nTreeNodes()]
     * @return            true or false
     */
    [[nodiscard]] inline bool isLeaf(TreeNodeIndex node) const
    {
        assert(node < nTreeNodes());
        return node >= internalTree_.size();
    }

    /*! \brief check whether child of node is a leaf
     *
     * @param node[in]    node index, range [0:nInternalNodes()]
     * @param octant[in]  octant index, range [0:8]
     * @return            true or false
     *
     * If \a node is not internal, behavior is undefined.
     */
    [[nodiscard]] inline bool isLeafChild(TreeNodeIndex node, int octant) const
    {
        assert(node < internalTree_.size());
        return internalTree_[node].childType[octant] == OctreeNode<I>::leaf;
    }

    //! \brief check if node is the root node
    [[nodiscard]] inline bool isRoot(TreeNodeIndex node) const
    {
        return node == 0;
    }

    /*! \brief return child node index
     *
     * @param node[in]    node index, range [0:nInternalNodes()]
     * @param octant[in]  octant index, range [0:8]
     * @return            child node index, range [0:nNodes()]
     *
     * If \a node is not internal, behavior is undefined.
     * Query with isLeaf(node) before calling this function.
     */
    [[nodiscard]] inline TreeNodeIndex child(TreeNodeIndex node, int octant) const
    {
        assert(node < internalTree_.size());

        TreeNodeIndex childIndex = internalTree_[node].child[octant];
        if (internalTree_[node].childType[octant] == OctreeNode<I>::leaf)
        {
            childIndex += nInternalNodes();
        }

        return childIndex;
    }

    /*! \brief return child node indices with leaf indices starting from 0
     *
     * @param node[in]    node index, range [0:nInternalNodes()]
     * @param octant[in]  octant index, range [0:8]
     * @return            child node index, range [0:nInternalNodes()] if child is internal
     *                    or [0:nLeafNodes()] if child is a leaf
     *
     * If @a node is not internal, behavior is undefined.
     * Note: the indices returned by this function refer to two different arrays, depending on
     * whether the child specified by @a node and @a octant is a leaf or an internal node.
     */
    [[nodiscard]] inline TreeNodeIndex childDirect(TreeNodeIndex node, int octant) const
    {
        return internalTree_[node].child[octant];
    }

    /*! \brief index of parent node
     *
     * Note: the root node is its own parent
     */
    [[nodiscard]] inline TreeNodeIndex parent(TreeNodeIndex node) const
    {
        if (node < nInternalNodes())
        {
            return internalTree_[node].parent;
        } else
        {
            return leafParents_[node - nInternalNodes()];
        }
    }

    //! \brief lowest SFC key contained int the geometrical box of \a node
    [[nodiscard]] inline I codeStart(TreeNodeIndex node) const
    {
        if (node < nInternalNodes())
        {
            return internalTree_[node].prefix;
        } else
        {
            return cstoneTree_[node - nInternalNodes()];
        }
    }

    //! \brief highest SFC key contained in the geometrical box of \a node
    [[nodiscard]] inline I codeEnd(TreeNodeIndex node) const
    {
        if (node < nInternalNodes())
        {
            return internalTree_[node].prefix + nodeRange<I>(internalTree_[node].level);
        } else
        {
            return cstoneTree_[node - nInternalNodes() + 1];
        }
    }

    /*! \brief octree subdivision level for \a node
     *
     * Returns 0 for the root node. Highest value is maxTreeLevel<I>{}.
     */
    [[nodiscard]] inline int level(TreeNodeIndex node) const
    {
        if (node < nInternalNodes())
        {
            return internalTree_[node].level;
        } else
        {
            return treeLevel(cstoneTree_[node - nInternalNodes() + 1] -
                             cstoneTree_[node - nInternalNodes()]);
        }
    }

    //! \brief returns (min,max) x-coordinate pair for \a node
    template<class T>
    [[nodiscard]] pair<T> x(TreeNodeIndex node, const Box<T>& box) const
    {
        constexpr int maxCoord = 1u<<maxTreeLevel<I>{};
        constexpr T uL = T(1.) / maxCoord;

        I prefix = codeStart(node);
        int ix = decodeMortonX(prefix);
        T xBox = box.xmin() + ix * uL * box.lx();
        int unitsPerBox = 1u<<(maxTreeLevel<I>{} - level(node));

        T uLx = uL * box.lx() * unitsPerBox;

        return {xBox, xBox + uLx};
    }

    //! \brief returns (min,max) y-coordinate pair for \a node
    template<class T>
    [[nodiscard]] pair<T> y(TreeNodeIndex node, const Box<T>& box) const
    {
        constexpr int maxCoord = 1u<<maxTreeLevel<I>{};
        constexpr T uL = T(1.) / maxCoord;

        I prefix = codeStart(node);
        int iy = decodeMortonY(prefix);
        T yBox = box.ymin() + iy * uL * box.ly();
        int unitsPerBox = 1u<<(maxTreeLevel<I>{} - level(node));

        T uLy = uL * box.ly() * unitsPerBox;

        return {yBox, yBox + uLy};
    }

    //! \brief returns (min,max) z-coordinate pair for \a node
    template<class T>
    [[nodiscard]] pair<T> z(TreeNodeIndex node, const Box<T>& box) const
    {
        constexpr int maxCoord = 1u<<maxTreeLevel<I>{};
        constexpr T uL = T(1.) / maxCoord;

        I prefix = codeStart(node);
        int iz = decodeMortonZ(prefix);
        T zBox = box.zmin() + iz * uL * box.lz();
        int unitsPerBox = 1u<<(maxTreeLevel<I>{} - level(node));

        T uLz = uL * box.lz() * unitsPerBox;

        return {zBox, zBox + uLz};
    }

private:

    //! \brief cornerstone octree, just the leaves
    std::vector<I>             cstoneTree_;

    //! \brief indices into internalTree_ to store the parent index of each leaf
    std::vector<TreeNodeIndex> leafParents_;

    //! \brief the internal tree
    std::vector<OctreeNode<I>> internalTree_;

    /*! \brief the internal part as binary radix nodes, precursor to internalTree_
     *
     * This is kept here because the binary format is faster for findHalos / collision detection
     */
    std::vector<BinaryNode<I>> binaryTree_;

    /*! \brief stores the number of nodes for each of the maxTreeLevel<I>{} possible values of
     *
     *  maxDepth is the distance of the farthest leaf, i.e.
     *  nNodesPerLevel[0] --> number of leaf nodes
     *  nNodesPerLevel[1] --> number of nodes with maxDepth = 1, children: only leaves
     *  nNodesPerLevel[2] --> number of internal nodes with maxDepth = 2, children: leaves and maxDepth = 1
     *  etc.
     */
     std::vector<TreeNodeIndex> nNodesPerLevel_;
};


} // namespace cstone
