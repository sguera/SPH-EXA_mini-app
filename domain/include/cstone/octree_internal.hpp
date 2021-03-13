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

#include "cstone/btree.hpp"
#include "cstone/octree.hpp"
#include "cstone/scan.hpp"

#include "cstone/cuda/annotation.hpp"

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
 * @tparam I           32- or 64-bit unsigned integer
 * @param binaryTree   binary tree nodes
 * @param nLeafNodes   number of octree leaf nodes used to construct \a binaryTree
 * @return             the internal octree nodes and octree leaf node parent indices
 */
template<class I>
std::tuple<std::vector<OctreeNode<I>>, std::vector<TreeNodeIndex>>
createInternalOctreeCpu(const std::vector<BinaryNode<I>>& binaryTree, TreeNodeIndex nLeafNodes)
{
    // we ignore the last binary tree node which is a duplicate root node
    TreeNodeIndex nBinaryNodes = binaryTree.size() - 1;

    // one extra element to store the total sum of the exclusive scan
    std::vector<TreeNodeIndex> prefixes(nBinaryNodes + 1);
    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nBinaryNodes; ++i)
    {
        int prefixLength = binaryTree[i].prefixLength;
        prefixes[i] = (prefixLength % 3) ? 0 : 1;
    }

    // stream compaction: scan and scatter
    exclusiveScan(prefixes.data(), prefixes.size());

    TreeNodeIndex nInternalOctreeNodes = *prefixes.rbegin();
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

    std::vector<OctreeNode<I>> internalOctree(nInternalOctreeNodes);
    std::vector<TreeNodeIndex> leafParents(nLeafNodes);

    #pragma omp parallel for schedule(static)
    for (TreeNodeIndex i = 0; i < nInternalOctreeNodes; ++i)
    {
        constructOctreeNode(internalOctree.data(), binaryTree.data(), i,
                            scatterMap.data(), prefixes.data(), leafParents.data());
    }

    return std::make_tuple(std::move(internalOctree), std::move(leafParents));
}


/*! \brief create the internal part for a given cornerstone octree with just leaves
 *
 * @tparam I           32- or 64-bit unsigned integer
 * @param tree         cornerstone octree with the octree leaves
 * @return             the internal octree and a vector of length nNodes(tree)
 *                     with an index into the internal octree to store the parent node
 *                     for each octree leaf of \a tree
 */
template<class I>
std::tuple<std::vector<OctreeNode<I>>, std::vector<TreeNodeIndex>>
createInternalOctree(const std::vector<I>& tree)
{
    std::vector<BinaryNode<I>> binaryTree = createInternalTree(tree);
    auto internalOctree = createInternalOctreeCpu(binaryTree, nNodes(tree));

    return internalOctree;
}



struct GlobalTree{};
struct LocalTree{};


/*! \brief This class unifies a cornerstone octree with the internal part
 *
 * @tparam I          32- or 64-bit unsigned integer
 * @tparam Locality   GlobalTree or LocalTree, selects between distributed and local trees
 *
 * Leaves are stored separately from the internal nodes.
 * For either type of node, just a single buffer is allocated.
 */
template<class I, class Locality = LocalTree>
class Octree {
public:
    Octree() = default;

    /*! \brief builds the tree from a range of particle Morton codes
     *
     * @param codesStart[in]    local particle Morton code buffer start
     * @param codesEnd[in]                                        end
     * @param bucketSize[in]    refine tree until each leaf has bucketSize or fewer particles
     *
     * internal state change:
     *      -full tree update
     */
    void compute(const I *codesStart, const I *codesEnd, unsigned bucketSize)
    {
        static_assert(std::is_same_v<Locality, LocalTree> || std::is_same_v<Locality, GlobalTree>);

        if constexpr (std::is_same_v<Locality, LocalTree>)
        {
            std::tie(tree_, nodeCounts_) = computeOctree(codesStart, codesEnd, bucketSize,
                                                         std::numeric_limits<unsigned>::max(), std::move(tree_));
        } else if constexpr (std::is_same_v<Locality, GlobalTree>)
        {
            std::tie(tree_, nodeCounts_) = computeOctreeGlobal(codesStart, codesEnd, bucketSize, std::move(tree_));
        }

        std::tie(internalTree_, leafParents_) = createInternalOctree(tree_);
        internalCounts_.resize(internalTree_.size());
    }

    //! \brief total number of nodes in the tree
    [[nodiscard]] inline TreeNodeIndex nTreeNodes() const
    { return nNodes(tree_) + internalTree_.size(); }

    //! \brief number of leaf nodes in the tree
    [[nodiscard]] inline TreeNodeIndex nLeaves() const
    { return nNodes(tree_); }

    //! \brief number of internal nodes in the tree, equal to (nLeaves()-1) / 7
    [[nodiscard]] inline TreeNodeIndex nInternalNodes() const
    { return internalTree_.size(); }

    //! \brief check if node is at the bottom of the tree
    [[nodiscard]] inline bool isLeaf(TreeNodeIndex node) const
    {
        assert(node < nTreeNodes());
        return node >= internalTree_.size();
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
    [[nodiscard]] TreeNodeIndex child(TreeNodeIndex node, TreeNodeIndex octant) const
    {
        assert(node < internalTree_.size());

        TreeNodeIndex childIndex = internalTree_[node].child[octant];
        if (internalTree_[node].childType[octant] == OctreeNode<I>::leaf)
        {
            childIndex += nInternalNodes();
        }

        return childIndex;
    }

    /*! \brief index of parent node
     *
     * Note: the root node is its own parent
     */
    [[nodiscard]] TreeNodeIndex parent(TreeNodeIndex node) const
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
    [[nodiscard]] I codeStart(TreeNodeIndex node) const
    {
        if (node < nInternalNodes())
        {
            return internalTree_[node].prefix;
        } else
        {
            return tree_[node - nInternalNodes()];
        }
    }

    //! \brief highest SFC key contained in the geometrical box of \a node
    [[nodiscard]] I codeEnd(TreeNodeIndex node) const
    {
        if (node < nInternalNodes())
        {
            return internalTree_[node].prefix + nodeRange<I>(internalTree_[node].level);
        } else
        {
            return tree_[node - nInternalNodes() + 1];
        }
    }

    /*! \brief octree subdivision level for \a node
     *
     * Returns 0 for the root node. Highest value is maxTreeLevel<I>{}.
     */
    [[nodiscard]] I level(TreeNodeIndex node) const
    {
        if (node < nInternalNodes())
        {
            return internalTree_[node].level;
        } else
        {
            return treeLevel(tree_[node - nInternalNodes() + 1] -
                             tree_[node - nInternalNodes()]);
        }
    }

    //! \brief returns (min,max) x-coordinate pair for \a node
    template<class T>
    [[nodiscard]] pair<T> x(TreeNodeIndex node, const Box<T>& box)
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
    [[nodiscard]] pair<T> y(TreeNodeIndex node, const Box<T>& box)
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
    [[nodiscard]] pair<T> z(TreeNodeIndex node, const Box<T>& box)
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

    //! \brief returns the internal octree
    const std::vector<OctreeNode<I>>& internalTree() const { return internalTree_; }

    //! \brief returns the cornerstone octree
    const std::vector<I>& tree() const { return tree_; }

    //! \brief returns node counts for the cornerstone octree
    const std::vector<I>& nodeCounts() const { return nodeCounts_; }

private:

    //! \brief cornerstone octree, just the leaves
    std::vector<I>             tree_;
    //! \brief leaf node particles counts
    std::vector<unsigned>      nodeCounts_;
    //! \brief indices into internalTree_ to store the parent index of each leaf
    std::vector<TreeNodeIndex> leafParents_;

    //! \brief the internal tree
    std::vector<OctreeNode<I>> internalTree_;
    //! \brief internal tree node particles counts
    std::vector<unsigned>      internalCounts_;
};


} // namespace cstone
