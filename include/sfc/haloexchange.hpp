#pragma once

#include <vector>

#include "sfc/mpi_wrappers.hpp"

#include "sfc/layout.hpp"

namespace sphexa
{

template<class T, class...Arrays>
void haloexchange(const SendList& incomingHalos,
                  const SendList& outgoingHalos,
                  Arrays... arrays)
{
    constexpr int nArrays = sizeof...(Arrays);
    std::array<T*, nArrays> data{arrays...};

    std::vector<std::vector<T>> sendBuffers;
    std::vector<MPI_Request>    sendRequests;

    for (int destinationRank = 0; destinationRank < outgoingHalos.size(); ++destinationRank)
    {
        int sendCount = outgoingHalos[destinationRank].totalCount();
        if (sendCount == 0)
            continue;

        std::vector<T> buffer(sendCount * nArrays);
        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            int outputOffset = sendCount * arrayIndex;
            for (int rangeIdx = 0; rangeIdx < outgoingHalos[destinationRank].nRanges(); ++rangeIdx)
            {
                int lowerIndex = outgoingHalos[destinationRank].rangeStart(rangeIdx);
                int upperIndex = outgoingHalos[destinationRank].rangeEnd(rangeIdx);

                std::copy(data[arrayIndex]+lowerIndex, data[arrayIndex]+upperIndex,
                          buffer.data() + outputOffset);
                outputOffset += upperIndex - lowerIndex;
            }
        }

        mpiSendAsync(buffer.data(), sendCount * nArrays, destinationRank, 0, sendRequests);
        sendBuffers.push_back(std::move(buffer));
    }

    int nMessages = 0;
    int maxReceiveSize = 0;
    for (int sourceRank = 0; sourceRank < incomingHalos.size(); ++sourceRank)
        if (incomingHalos[sourceRank].totalCount() > 0)
        {
            nMessages++;
            maxReceiveSize = std::max(maxReceiveSize, (int)incomingHalos[sourceRank].totalCount());
        }

    std::vector<T> receiveBuffer(maxReceiveSize * nArrays);

    while (nMessages > 0)
    {
        MPI_Status status;
        mpiRecvSync(receiveBuffer.data(), receiveBuffer.size(), MPI_ANY_SOURCE, 0, &status);
        int receiveRank = status.MPI_SOURCE;
        int countPerArray = incomingHalos[receiveRank].totalCount();

        for (int arrayIndex = 0; arrayIndex < nArrays; ++arrayIndex)
        {
            int inputOffset = countPerArray * arrayIndex;
            for (int rangeIdx = 0; rangeIdx < incomingHalos[receiveRank].nRanges(); ++rangeIdx)
            {
                int offset = incomingHalos[receiveRank].rangeStart(rangeIdx);
                int count  = incomingHalos[receiveRank].count(rangeIdx);

                std::copy(receiveBuffer.data() + inputOffset, receiveBuffer.data() + inputOffset + count,
                          data[arrayIndex]+offset);

                inputOffset += count;
            }
        }
        nMessages--;
    }

    if (not sendRequests.empty())
    {
        MPI_Status status[sendRequests.size()];
        MPI_Waitall(sendRequests.size(), sendRequests.data(), status);
    }

    // prevent rank from sending further messages with tag 0 while other ranks
    // are still listening to the messages sent out here
    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace sphexa