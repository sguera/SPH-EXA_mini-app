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
 * \brief  Exchange particles between different ranks to satisfy their assignments of the global octree
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include "cstone/domaindecomp.hpp"

#include "mpi_wrappers.hpp"

namespace cstone
{

template<class T, class IndexType, class... Arrays>
void exchangeParticlesImpl(const SendList& sendList, int thisRank, std::size_t nParticlesAssigned,
                           IndexType inputOffset, IndexType outputOffset,
                           const IndexType* ordering, T* tempBuffer, Arrays... arrays)
{
    constexpr int nArrays = sizeof...(Arrays);
    std::array<T*, nArrays> sourceArrays{ (arrays + inputOffset)... };
    std::array<T*, nArrays> destinationArrays{ (arrays + outputOffset)... };

    int nRanks = sendList.size();

    std::vector<std::vector<T>> sendBuffers;
    sendBuffers.reserve(nArrays * (nRanks-1));

    std::vector<MPI_Request> sendRequests;
    sendRequests.reserve(nArrays * (nRanks-1));

    for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank)
    {
        if (destinationRank == thisRank || sendList[destinationRank].totalCount() == 0) { continue; }

        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            auto arrayBuffer = createSendBuffer(sendList[destinationRank], sourceArrays[arrayIndex], ordering);
            mpiSendAsync(arrayBuffer.data(), arrayBuffer.size(), destinationRank, arrayIndex, sendRequests);
            sendBuffers.emplace_back(std::move(arrayBuffer));
        }
    }

    // handle thisRank
    for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
    {
        // TODO: eliminate one copy by swapping the source with tempBuffer
        extractRange(sendList[thisRank], sourceArrays[arrayIndex], ordering, tempBuffer);
        std::copy(tempBuffer, tempBuffer + sendList[thisRank].totalCount(), destinationArrays[arrayIndex]);
    }

    std::size_t nParticlesPresent = sendList[thisRank].totalCount();

    while (nParticlesPresent != nParticlesAssigned)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int receiveRank  = status.MPI_SOURCE;
        //int receiveTag   = status[0].MPI_TAG;
        int receiveCount;
        MPI_Get_count(&status, MpiType<T>{}, &receiveCount);

        if (nParticlesPresent + (std::size_t)receiveCount > nParticlesAssigned)
        {
            throw std::runtime_error("Particle exchange: cannot receive more particles than assigned\n");
        }

        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            mpiRecvSync(destinationArrays[arrayIndex] + nParticlesPresent, receiveCount,
                        receiveRank, arrayIndex, &status);
        }

        nParticlesPresent += receiveCount;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(sendRequests.size(), sendRequests.data(), status);
    }

    // If this process is going to send messages with rank/tag combinations
    // already sent in this function, this can lead to messages being mixed up
    // on the receiver side. This happens e.g. with repeated consecutive calls of
    // this function. For this reason, a barrier is enacted here.
    // If there are no interfering messages going to be sent, it would be possible to
    // remove the barrier. But if that assumption turns out to be wrong, arising bugs
    // will be hard to detect.
    MPI_Barrier(MPI_COMM_WORLD);
}

//! \brief reallocate arrays to the specified size
template<class... Arrays>
void reallocate(std::size_t size, Arrays&... arrays)
{
    std::array data{ (&arrays)... };

    size_t current_capacity = data[0]->capacity();
    if (size > current_capacity)
    {
        // limit reallocation growth to 5% instead of 200%
        size_t reserve_size = double(size) * 1.05;
        for (auto array : data)
        {
            array->reserve(reserve_size);
        }
    }

    for (auto array : data)
    {
        array->resize(size);
    }
}

/*! \brief exchange array elements with other ranks according to the specified ranges
 *
 * \tparam T                  double, float or int
 * \tparam Arrays             all std::vector<T>
 * \param sendList[in]        List of index ranges to be sent to each rank, indices
 *                            are valid w.r.t to arrays present on \a thisRank relative to the \a inputOffset.
 * \param thisRank[in]        Rank of the executing process
 * \param nParticlesAssigned  New number of assigned particles for each array on \a thisRank.
 *        [in]                This serves as the stop criterion for listening to incoming particles.
 * \param inputOffset[in]     Access arrays starting from \a inputOffset when extracting particles for sending
 * \param outputOffset[in]    Incoming particles will be added to their destination arrays starting from \a outputOffset
 * \param ordering[in]        Ordering through which to access arrays
 * \param arrays[inout]       T* pointers of identical sizes. The index range based exchange operations
 *                            performed are identical for each input array. Upon completion, arrays will
 *                            contain elements from the specified ranges from all ranks.
 *                            The order in which the incoming ranges are grouped is random.
 *
 *  Example: If sendList[ri] contains the range [upper, lower), all elements (arrays+inputOffset)[ordering[upper:lower]]
 *           will be sent to rank ri. At the destination ri, any assigned particles already present,
 *           are moved to their destination arrays, starting from \a outputOffset. The incoming elements to ri
 *           will be appended to the aforementioned elements.
 *           No information about incoming particles to \a thisRank is contained in the function arguments,
 *           only their total number \a nParticlesAssigned, which also includes any assigned particles
 *           already present on \a thisRank.
 *
 *  A note on the sizes of \a arrays:
 *  Let's set
 *      int nOldAssignment = the maximum of sendList[rank].rangeEnd(range) for any rank and range combination
 *  Then inputOffset + nOldAssignment is the upper bound for read accesses in \a arrays while sending.
 *  This is assuming that ordering.size() == nOldAssignment and that ordering[i] < nOldAssignment, which
 *  is what the Morten (re)order code produces.
 *  While writing during the receive phase, the highest index is outputOffset + nParticlesAssigned and it
 *  is checked that no writes occur past that location.
 *  In summary, the conditions to check in case of access violations / segmentation faults are for all arrays:
 *
 *      array.size() >= inputOffset + nOldAssignment
 *      array.size() >= outputOffset + nParticlesAssigned
 *      ordering.size() == nOldAssignment
 *      *std::max_element(begin(ordering), end(ordering)) == nOldAssignment - 1
 */
template<class T, class IndexType, class... Arrays>
void exchangeParticles(const SendList& sendList, Rank thisRank, IndexType nParticlesAssigned,
                       IndexType inputOffset, IndexType outputOffset, const IndexType* ordering, Arrays... arrays)
{
    IndexType nParticlesAlreadyPresent = sendList[thisRank].totalCount();
    std::vector<T> tempBuffer(nParticlesAlreadyPresent);
    exchangeParticlesImpl(sendList, thisRank, nParticlesAssigned, inputOffset, outputOffset,
                          ordering, tempBuffer.data(), arrays...);
}

/*! \brief exchange array elements with other ranks according to the specified ranges
 *
 * \tparam T                  double, float or int
 * \tparam Arrays             all std::vector<T>
 * \param sendList[in]        List of index ranges assigned to each rank, indices
 *                            are valid w.r.t to arrays present on \a thisRank
 * \param thisRank[in]        Rank of the executing process
 * \param nParticlesAssigned  Number of elements that each array will hold on \a thisRank after the exchange
 * \param ordering[in]        Ordering through which to access arrays
 * \param arrays[inout]       T* pointers of identical sizes, the index range based exchange operations
 *                            performed are identical for each input array. Upon completion, arrays will
 *                            contain elements from the specified ranges from all ranks.
 *                            The order in which the incoming ranges are grouped is random.
 *
 * See documentation of exchangeParticles with the full signature
 */
template<class T, class IndexType, class... Arrays>
void exchangeParticles(const SendList& sendList, Rank thisRank, IndexType nParticlesAssigned,
                       const IndexType* ordering, Arrays... arrays)
{
    exchangeParticles<T>(sendList, thisRank, nParticlesAssigned, 0, 0, ordering, arrays...);
}

} // namespace cstone
