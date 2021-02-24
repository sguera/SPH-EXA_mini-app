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
 * \brief binary tree traversal implementation
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/btree.hpp"

namespace cstone
{

//! \brief stores indices of colliding octree leaf nodes
class CollisionList
{
public:
    //! \brief add an index to the list of colliding leaf tree nodes
    void add(int i)
    {
        list_[n_] = i;
        n_ = (n_ < collisionMax-1) ? n_+1 : n_;
    }

    //! \brief access collision list as a range
    [[nodiscard]] const int* begin() const { return list_; }
    [[nodiscard]] const int* end()   const { return list_ + n_; }

    //! \brief access collision list elements
    int operator[](int i) const
    {
        assert(i < collisionMax);
        return list_[i];
    }

    /*! \brief returns number of collisions
     *
     * Can (should) also be used to check whether the internal storage
     * was exhausted during collision detection.
     */
    [[nodiscard]] std::size_t size() const { return n_; };

    [[nodiscard]] bool exhausted() const { return n_ == collisionMax-1; }

private:
    static constexpr int collisionMax = 512;
    std::size_t n_{0};
    int list_[collisionMax]{0};
};

template<class I>
inline bool traverseNode(const BinaryNode<I>* node, const IBox& collisionBox)
{
    return (node != nullptr) && overlap(node->prefix, node->prefixLength, collisionBox);
}

template<class I>
inline bool leafOverlap(int leafIndex, const I* leafNodes,
                        const IBox& collisionBox)
{
    if (leafIndex < 0)
        return false;

    I leafCode = leafNodes[leafIndex];
    I leafUpperBound = leafNodes[leafIndex + 1];

    int prefixNBits = treeLevel(leafUpperBound - leafCode) * 3;

    return overlap(leafCode, prefixNBits, collisionBox);
}

/*! \brief find all collisions between a leaf node enlarged by (dx,dy,dz) and the rest of the tree
 *
 * @tparam I                  32- or 64-bit unsigned integer
 * @param[in]  internalRoot   root of the internal binary radix tree
 * @param[in]  leafNodes      octree leaf nodes
 * @param[out] collisionList  output list of indices of colliding nodes
 * @param[in]  collisionBox   query box to look for collisions
 *                            with leaf nodes
 *
 * At all traversal steps through the hierarchy of the internal binary radix tree,
 * all 3 x,y,z dimensions are checked to determine overlap with a binary node.
 * If the Morton codes used to construct the binary tree satisfy the requirements of
 * a cornerstone octree, it would be sufficient to only check
 *
 * x at nodes with prefixLength % 3 == 1
 * y at nodes with prefixLength % 3 == 2
 * z at nodes with prefixLength % 3 == 0
 *
 * because the cornerstone invariants guarantee that each internal binary node will
 * have one more bit in its prefix than its parent.
 *
 * However, the construction of the internal tree and the following traversal as
 * implemented also works if an arbitrary sorted sequence of Morton codes was used
 * for the construction of the internal tree, i.e. holes or omission of empty nodes
 * would be possible. Since this capability might be useful in the future, and the
 * cost to check all 3 dimensions at each step should not be very high, we keep
 * the implementation general.
 */
template <class I>
void findCollisions(const BinaryNode<I>* internalRoot, const I* leafNodes, CollisionList& collisionList,
                    const IBox& collisionBox)
{
    using Node    = BinaryNode<I>;
    using NodePtr = const Node*;

    NodePtr  stack[64];
    NodePtr* stackPtr = stack;

    *stackPtr++ = nullptr;

    const BinaryNode<I>* node = internalRoot;

    do
    {
        bool traverseL = traverseNode(node->child[Node::left], collisionBox);
        bool traverseR = traverseNode(node->child[Node::right], collisionBox);

        bool overlapLeafL = leafOverlap(node->leafIndex[Node::left], leafNodes, collisionBox);
        bool overlapLeafR = leafOverlap(node->leafIndex[Node::right], leafNodes, collisionBox);

        if (overlapLeafL) collisionList.add(node->leafIndex[Node::left]);
        if (overlapLeafR) collisionList.add(node->leafIndex[Node::right]);

        if (!traverseL and !traverseR)
        {
            node = *--stackPtr; // pop
        }
        else
        {
            if (traverseL && traverseR)
            {
                if (stackPtr-stack >= 64)
                {
                    throw std::runtime_error("btree traversal stack exhausted\n");
                }
                *stackPtr++ = node->child[Node::right]; // push
            }

            node = (traverseL) ? node->child[Node::left] : node->child[Node::right];
        }

    } while (node != nullptr);
}

} // namespace cstone