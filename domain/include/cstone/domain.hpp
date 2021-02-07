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
 * \brief A domain class to manage distributed particles and their halos.
 *
 * Particles are represented by x,y,z coordinates, interaction radii and
 * a user defined number of additional properties, such as masses or charges.
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/box_mpi.hpp"
#include "cstone/domaindecomp_mpi.hpp"
#include "cstone/halodiscovery.hpp"
#include "cstone/haloexchange.hpp"
#include "cstone/layout.hpp"
#include "cstone/octree_mpi.hpp"

#include "cstone/cuda/gather.cuh"

namespace cstone
{

template<class I, class T>
class Domain
{
public:
    /*! @brief construct empty Domain
     *
     * @param rank        executing rank
     * @param nRanks      number of ranks
     * @param bucketSize  build tree with max @a bucketSize particles per node
     * @param box         global bounding box, default is non-pbc box
     *                    for each periodic dimension in @a box, the coordinate min/max
     *                    limits will never be changed for the lifetime of the Domain
     *
     */
    explicit Domain(int rank, int nRanks, int bucketSize, const Box<T>& box = Box<T>{0,1})
        : myRank_(rank), nRanks_(nRanks), bucketSize_(bucketSize),
          particleStart_(0), particleEnd_(-1), localNParticles_(-1), box_(box)
    {}

    /*! @brief Domain update sequence for particles with coordinates x,y,z, interaction radius h and their properties
     *
     * @param x[inout]                   floating point coordinates
     * @param y[inout]
     * @param z[inout]
     * @param h[inout]                   interaction radii in SPH convention, actual interaction radius is twice
     *                                   the value in h
     * @param codes[out]                 Morton codes
     * @param particleProperties[inout]  particle properties to distribute along with the coordinates
     *                                   e.g. mass or charge
     *
     * ============================================================================================================
     * Preconditions:
     * ============================================================================================================
     *
     *   - Array sizes of x,y,z,h and particleProperties are identical
     *     AND equal to the internally stored value of localNParticles_ from the previous call, except
     *     on the first call. This is checked.
     *
     *     This means that none of the argument arrays can be resized between calls of this function.
     *     Or in other words, particles cannot be created or destroyed.
     *     (If this should ever be required though, it can be easily enabled by allowing the assigned
     *     index range from startIndex() to endIndex() to be modified from the outside.)
     *
     *   - The particle order is irrelevant
     *
     *   - Content of codes is irrelevant as it will be resized to fit x,y,z,h and particleProperties
     *
     * ============================================================================================================
     * Postconditions:
     * ============================================================================================================
     *
     *   Array sizes:
     *   ------------
     *   - All arrays, x,y,z,h, codes and particleProperties are resized with space for the newly assigned particles
     *     AND their halos.
     *
     *   Content of x,y,z and h
     *   ----------------------------
     *   - x,y,z,h at indices from startIndex() to endIndex() contain assigned particles that the executing rank owns,
     *     all other elements are _halos_ of the assigned particles, i.e. the halos for x,y,z,h and codes are already
     *     in place post-call.
     *
     *   Content of particleProperties
     *   ----------------------------
     *   - particleProperties arrays contain the updated properties at indices from startIndex() to endIndex(),
     *     i.e. index i refers to a property of the particle with coordinates (x[i], y[i], z[i]).
     *     Content of elements outside this range is _undefined_, but can be filled with the corresponding halo data
     *     by a subsequent call to exchangeHalos(particleProperty), such that also for i outside [startIndex():endIndex()],
     *     particleProperty[i] is a property of the halo particle with coordinates (x[i], y[i], z[i]).
     *
     *   Content of codes
     *   ----------------
     *   - The codes output is sorted and contains the Morton codes of assigned _and_ halo particles,
     *     i.e. all arrays will be output in Morton order.
     *
     *   Internal state of the domain
     *   ----------------------------
     *   The following members are modified by calling this function:
     *   - Update of the global octree, for use as starting guess in the next call
     *   - Update of the assigned range startIndex() and endIndex()
     *   - Update of the total local particle count, i.e. assigned + halo particles
     *   - Update of the halo exchange patterns, for subsequent use in exchangeHalos
     *   - Update of the global coordinate bounding box
     *
     * ============================================================================================================
     * Update sequence:
     * ============================================================================================================
     *      1. compute global coordinate bounding box
     *      2. compute global octree
     *      3. compute max_h per octree node
     *      4. assign octree to ranks
     *      5. discover halos
     *      6. compute particle layout, i.e. count number of halos and assigned particles
     *         and compute halo send and receive index ranges
     *      7. resize x,y,z,h,codes and properties to new number of assigned + halo particles
     *      8. exchange coordinates, h, and properties of assigned particles
     *      9. morton sort exchanged assigned particles
     *     10. exchange halo particles
     */
    template<class... Vectors>
    void sync(std::vector<T>& x, std::vector<T>& y, std::vector<T>& z, std::vector<T>& h, std::vector<I>& codes,
              Vectors&... particleProperties)
    {
        // bounds initialization on first call, use all particles
        if (particleEnd_ == -1)
        {
            particleStart_   = 0;
            particleEnd_     = x.size();
            localNParticles_ = x.size();
        }

        if (!sizesAllEqualTo(localNParticles_, x, y, z, h, particleProperties...))
        {
            throw std::runtime_error("Domain sync: input array sizes are inconsistent\n");
        }

        box_ = makeGlobalBox(cbegin(x) + particleStart_, cbegin(x) + particleEnd_,
                             cbegin(y) + particleStart_,
                             cbegin(z) + particleStart_, box_);

        // number of locally assigned particles to consider for global tree building
        int nParticles = particleEnd_ - particleStart_;

        codes.resize(nParticles);

        // compute morton codes only for particles participating in tree build
        std::vector<I> mortonCodes(nParticles);
        computeMortonCodes(cbegin(x) + particleStart_, cbegin(x) + particleEnd_,
                           cbegin(y) + particleStart_,
                           cbegin(z) + particleStart_,
                           begin(codes), box_);

        // compute the ordering that will sort the mortonCodes in ascending order
        std::vector<int> mortonOrder(nParticles);
        //sort_invert(cbegin(codes), cbegin(codes) + nParticles, begin(mortonOrder));
        deviceReorder_.setMapFromCodes(codes.data(), codes.data() + nParticles);
        deviceReorder_.getReorderMap((I*)mortonOrder.data());

        // reorder the codes according to the ordering
        // has the same net effect as std::sort(begin(mortonCodes), end(mortonCodes)),
        // but with the difference that we explicitly know the ordering, such
        // that we can later apply it to the x,y,z,h arrays or to access them in the Morton order
        //reorder(mortonOrder, codes);

        // compute the global octree in cornerstone format (leaves only)
        // the resulting tree and node counts will be identical on all ranks
        std::vector<std::size_t> nodeCounts;
        std::tie(tree_, nodeCounts) = computeOctreeGlobal(codes.data(), codes.data() + nParticles, bucketSize_,
                                                          std::move(tree_));

        // assign one single range of Morton codes each rank
        SpaceCurveAssignment<I> assignment = singleRangeSfcSplit(tree_, nodeCounts, nRanks_);
        int newNParticlesAssigned = assignment.totalCount(myRank_);

        // compute the maximum smoothing length (=halo radii) in each global node
        std::vector<T> haloRadii(nNodes(tree_));
        computeHaloRadiiGlobal(tree_.data(), nNodes(tree_), codes.data(), codes.data() + nParticles,
                               mortonOrder.data(), h.data() + particleStart_, haloRadii.data());

        // find outgoing and incoming halo nodes of the tree
        // uses 3D collision detection
        std::vector<pair<int>> haloPairs;
        findHalos(tree_, haloRadii, box_, assignment, myRank_, haloPairs);

        // group outgoing and incoming halo node indices by destination/source rank
        std::vector<std::vector<int>> incomingHaloNodes;
        std::vector<std::vector<int>> outgoingHaloNodes;
        computeSendRecvNodeList(tree_, assignment, haloPairs, incomingHaloNodes, outgoingHaloNodes);

        // compute list of local node index ranges
        std::vector<int> incomingHalosFlattened = flattenNodeList(incomingHaloNodes);
        std::vector<int> localNodeRanges        = computeLocalNodeRanges(tree_, assignment, myRank_);

        // Put all local node indices and incoming halo node indices in one sorted list.
        // and compute an offset for each node into these arrays.
        // This will be the new layout for x,y,z,h arrays.
        std::vector<int> presentNodes;
        std::vector<int> nodeOffsets;
        computeLayoutOffsets(localNodeRanges, incomingHalosFlattened, nodeCounts, presentNodes, nodeOffsets);
        localNParticles_ = *nodeOffsets.rbegin();

        int firstLocalNode = std::lower_bound(cbegin(presentNodes), cend(presentNodes), localNodeRanges[0])
                             - begin(presentNodes);

        int newParticleStart = nodeOffsets[firstLocalNode];
        int newParticleEnd   = newParticleStart + newNParticlesAssigned;

        // compute send array ranges for domain exchange
        // index ranges in domainExchangeSends are valid relative to the sorted code array mortonCodes
        // note that there is no offset applied to mortonCodes, because it was constructed
        // only with locally assigned particles
        SendList domainExchangeSends = createSendList(assignment, codes.data(), codes.data() + nParticles);

        // resize arrays to new sizes
        reallocate(localNParticles_, x,y,z,h, particleProperties...);
        reallocate(localNParticles_, codes);
        // exchange assigned particles
        exchangeParticles<T>(domainExchangeSends, Rank(myRank_), newNParticlesAssigned,
                             particleStart_, newParticleStart, mortonOrder.data(),
                             x.data(), y.data(), z.data(), h.data(), particleProperties.data()...);

        // assigned particles have been moved to their new locations starting at particleStart_
        // by the domain exchange exchangeParticles
        std::swap(particleStart_, newParticleStart);
        std::swap(particleEnd_, newParticleEnd);

        computeMortonCodes(cbegin(x) + particleStart_, cbegin(x) + particleEnd_,
                           cbegin(y) + particleStart_,
                           cbegin(z) + particleStart_,
                           begin(codes) + particleStart_, box_);

        //mortonOrder.resize(newNParticlesAssigned);
        //sort_invert(cbegin(codes) + particleStart_, cbegin(codes) + particleEnd_, begin(mortonOrder));
        deviceReorder_.setMapFromCodes(codes.data() + particleStart_, codes.data() + particleEnd_);

        // We have to reorder the locally assigned particles in the coordinate and property arrays
        // which are located in the index range [particleStart_, particleEnd_].
        // Due to the domain particle exchange, contributions from remote ranks
        // are received in arbitrary order.
        {
            std::array<std::vector<T>*, 4 + sizeof...(Vectors)> particleArrays{&x, &y, &z, &h, &particleProperties...};
            for (std::size_t i = 0; i < particleArrays.size(); ++i)
            {
                //reorder(mortonOrder, *particleArrays[i], particleStart_);
                deviceReorder_(particleArrays[i]->data() + particleStart_);
            }
            //reorder(mortonOrder, codes, particleStart_);
        }

        incomingHaloIndices_ = createHaloExchangeList(incomingHaloNodes, presentNodes, nodeOffsets);
        outgoingHaloIndices_ = createHaloExchangeList(outgoingHaloNodes, presentNodes, nodeOffsets);

        exchangeHalos(x,y,z,h);

        // compute Morton codes for halo particles just received, from 0 to particleStart_
        // and from particleEnd_ to localNParticles_
        computeMortonCodes(cbegin(x), cbegin(x) + particleStart_,
                           cbegin(y),
                           cbegin(z),
                           begin(codes), box_);
        computeMortonCodes(cbegin(x) + particleEnd_, cend(x),
                           cbegin(y) + particleEnd_,
                           cbegin(z) + particleEnd_,
                           begin(codes) + particleEnd_, box_);
    }

    /*! \brief repeat the halo exchange pattern from the previous sync operation for a different set of arrays
     *
     * @param arrays  std::vector<float or double> of size localNParticles_
     *
     * Arrays are not resized or reallocated.
     * This is used e.g. for densities.
     */
    template<class...Arrays>
    void exchangeHalos(Arrays&... arrays) const
    {
        if (!sizesAllEqualTo(localNParticles_, arrays...))
        {
            throw std::runtime_error("halo exchange array sizes inconsistent with previous sync operation\n");
        }

        haloexchange<T>(incomingHaloIndices_, outgoingHaloIndices_, arrays.data()...);
    }

    //! \brief return the index of the first particle that's part of the local assignment
    [[nodiscard]] int startIndex() const { return particleStart_; }

    //! \brief return one past the index of the last particle that's part of the local assignment
    [[nodiscard]] int endIndex() const   { return particleEnd_; }

    //! \brief return number of locally assigned particles
    [[nodiscard]] int nParticles() const { return endIndex() - startIndex(); }

    //! \brief return number of locally assigned particles plus number of halos
    [[nodiscard]] int nParticlesWithHalos() const { return localNParticles_; }

    //! \brief read only visibility of the octree to the outside
    const std::vector<I>& tree() const { return tree_; }

    //! \brief return the coordinate bounding box from the previous sync call
    Box<T> box() const { return box_; }

private:

    //! \brief return true if all array sizes are equal to value
    template<class... Arrays>
    static bool sizesAllEqualTo(std::size_t value, Arrays&... arrays)
    {
        std::array<std::size_t, sizeof...(Arrays)> sizes{arrays.size()...};
        return std::count(begin(sizes), end(sizes), value) == sizes.size();
    }

    int myRank_;
    int nRanks_;
    int bucketSize_;

    /*! \brief array index of first local particle belonging to the assignment
     *  i.e. the index of the first particle that belongs to this rank and is not a halo.
     */
    int particleStart_;
    //! \brief index (upper bound) of last particle that belongs to the assignment
    int particleEnd_;
    //! \brief number of locally present particles, = number of halos + assigned particles
    int localNParticles_;

    //! \brief coordinate bounding box, each non-periodic dimension is at a sync call
    Box<T> box_;

    SendList incomingHaloIndices_;
    SendList outgoingHaloIndices_;

    std::vector<I> tree_;

    DeviceGather<T, I> deviceReorder_;
};

} // namespace cstone
