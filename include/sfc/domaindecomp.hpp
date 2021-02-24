#pragma once

#include <algorithm>
#include <vector>

#include "sfc/octree.hpp"
#include "sfc/util.hpp"
#include "sfc/zorder.hpp"


namespace sphexa
{

/*! \brief Stores ranges of local particles to be sent to another rank
 *
 * \tparam I  32- or 64-bit signed or unsigned integer to store the indices
 *
 *  Used for SendRanges with index ranges referencing elements in e.g. x,y,z,h arrays.
 *  In this case, count() equals the sum of all range differences computed as rangeEnd() - rangeStart().
 *
 *  Also used for SfcRanges with index ranges referencing parts of an SFC-octree with Morton codes.
 *  In that case, count() does NOT equal sum(rangeEnd(i) - rangeStart(i), i=0...nRanges)
 */
template<class I>
class IndexRanges
{
    struct Range
    {
        I start;
        I end;
        std::size_t count;

        friend bool operator==(const Range& a, const Range& b)
        {
            return a.start == b.start && a.end == b.end && a.count == b.count;
        }
    };

public:
    using IndexType = I;

    IndexRanges() : totalCount_(0), ranges_{} {}

    //! \brief add a local index range
    void addRange(I lower, I upper, std::size_t cnt)
    {
        assert(lower <= upper);
        ranges_.push_back({lower, upper, cnt});
        totalCount_ += cnt;
    }

    //! \brief add a local index range, infer count from difference
    void addRange(I lower, I upper)
    {
        assert(lower <= upper);
        std::size_t cnt = upper - lower;
        ranges_.push_back({lower, upper, cnt});
        totalCount_ += cnt;
    }

    [[nodiscard]] I rangeStart(int i) const
    {
        return ranges_[i].start;
    }

    [[nodiscard]] I rangeEnd(int i) const
    {
        return ranges_[i].end;
    }

    //! \brief the number of particles in range i
    [[nodiscard]] const std::size_t& count(int i) const { return ranges_[i].count; }

    //! \brief the sum of number of particles in all ranges or total send count
    [[nodiscard]] const std::size_t& totalCount() const { return totalCount_; }

    [[nodiscard]] std::size_t nRanges() const { return ranges_.size(); }

private:
    friend bool operator==(const IndexRanges& lhs, const IndexRanges& rhs)
    {
        return lhs.totalCount_ == rhs.totalCount_ && lhs.ranges_ == rhs.ranges_;
    }

    std::size_t totalCount_;
    std::vector<Range> ranges_;
};


/*! \brief a custom type for type safety in function calls
 *
 * The resulting type behaves like an int, except that explicit
 * conversion is required in function calls. Used e.g. in
 * SpaceCurveAssignment::addRange to force the caller to write
 * addRange(Rank(r), a,b,c) instead of addRange(r,a,b,c).
 * This makes it impossible to unintentionally mix up the arguments.
 */
using Rank = StrongType<int, struct RankTag>;

/*! \brief stores which parts of the SFC belong to which rank, on a per-rank basis
 *
 * \tparam I  32- or 64-bit unsigned integer
 *
 * The storage layout allows fast look-up of the Morton code ranges that a given rank
 * was assigned.
 *
 * Note: Assignment of SFC ranges to ranks should be unique, each SFC range should only
 * be assigned to one rank. This is NOT checked.
 */
template<class I>
class SpaceCurveAssignment
{
public:
    SpaceCurveAssignment() = default;

    explicit SpaceCurveAssignment(int nRanks) : rankAssignment_(nRanks) {}

    //! \brief add an index/code range to rank \a rank
    void addRange(Rank rank, I lower, I upper, std::size_t cnt)
    {
        rankAssignment_[rank].addRange(lower, upper, cnt);
    }

    [[nodiscard]] std::size_t nRanks() const { return rankAssignment_.size(); }

    [[nodiscard]] I rangeStart(int rank, int rangeIdx) const { return rankAssignment_[rank].rangeStart(rangeIdx); }

    [[nodiscard]] I rangeEnd(int rank, int rangeIdx) const { return rankAssignment_[rank].rangeEnd(rangeIdx); }

    //! \brief the number of particles in range rangeIdx of rank \a rank
    [[nodiscard]] const std::size_t& count(int rank, int rangeIdx) const { return rankAssignment_[rank].count(rangeIdx); }

    //! \brief the sum of number of particles in all ranges, i.e. total number of assigned particles per range
    [[nodiscard]] const std::size_t& totalCount(int rank) const { return rankAssignment_[rank].totalCount(); }

    //! \brief number of ranges per rank
    [[nodiscard]] std::size_t nRanges(int rank) const { return rankAssignment_[rank].nRanges(); }

    //! \brief extract morton code range pairs for a given rank, as used for octree construction
    [[nodiscard]] std::vector<I> makeRangePairs(int rank) const
    {
        std::vector<I> ret(2*nRanges(rank));
        for (int rangeIndex = 0; rangeIndex < nRanges(rank); ++rangeIndex)
        {
            ret[2*rangeIndex]   = rangeStart(rank, rangeIndex);
            ret[2*rangeIndex+1] = rangeEnd(rank, rangeIndex);
        }
        return ret;
    }

private:
    friend bool operator==(const SpaceCurveAssignment& a, const SpaceCurveAssignment& b)
    {
        return a.rankAssignment_ == b.rankAssignment_;
    }

    std::vector<IndexRanges<I>> rankAssignment_;
};


/*! \brief Stores the SFC assignment to ranks on a per-code basis
 *
 * \tparam I  32- or 64-bit unsigned integer
 *
 * The stored information is the same as the SpaceCurveAssignment, but
 * in a different layout that allows a fast lookup of the rank that a given
 * Morton code is assigned. Since this kind of lookup is only needed after the
 * SFC has been assigned, this class is non-modifiable after construction.
 *
 * Note: Construction assumes that the provided SpaceCurveAssignment only
 * contains unique assignments where each SFC range is only assigned to a single rank.
 * This is NOT checked.
 */
template<class I>
class SfcLookupKey
{
public:
    explicit SfcLookupKey(const SpaceCurveAssignment<I>& sfc)
    {
        for (int rank = 0; rank < sfc.nRanks(); ++rank)
        {
            for (int range = 0; range < sfc.nRanges(rank); ++range)
            {
                ranks_.push_back(rank);
                rangeCodeStarts_.push_back(sfc.rangeStart(rank, range));
            }
        }

        std::vector<int> order(rangeCodeStarts_.size());
        sort_invert(begin(rangeCodeStarts_), end(rangeCodeStarts_), begin(order));
        reorder(order, rangeCodeStarts_);
        reorder(order, ranks_);
    }

    //! \brief returns the rank that the argument code is assigned to
    int findRank(I code)
    {
        int index = std::upper_bound(begin(rangeCodeStarts_), end(rangeCodeStarts_), code)
                    - begin(rangeCodeStarts_);

        return ranks_[index-1];
    }

private:
    std::vector<I>   rangeCodeStarts_;
    std::vector<int> ranks_;
};


/*! \brief assign the global tree/SFC to nSplits ranks, assigning to each rank only a single Morton code range
 *
 * \tparam I                 32- or 64-bit integer
 * \param globalTree         the octree
 * \param globalCounts       counts per leaf
 * \param nSplits            divide the global tree into nSplits pieces, sensible choice e.g.: nSplits == nRanks
 * \return                   a vector with nSplit elements, each element is a vector of SfcRanges of Morton codes
 *
 * This function acts on global data. All calling ranks should call this function with identical arguments.
 * Therefore each rank will compute the same SpaceCurveAssignment and each rank will thus know the ranges that
 * all the ranks are assigned.
 *
 */
template<class I>
SpaceCurveAssignment<I> singleRangeSfcSplit(const std::vector<I>& globalTree, const std::vector<std::size_t>& globalCounts,
                                            int nSplits)
{
    // one element per rank
    SpaceCurveAssignment<I> ret(nSplits);

    std::size_t globalNParticles = std::accumulate(begin(globalCounts), end(globalCounts), std::size_t(0));

    // distribute work, every rank gets global count / nSplits,
    // the remainder gets distributed one by one
    std::vector<std::size_t> nParticlesPerSplit(nSplits, globalNParticles/nSplits);
    for (int split = 0; split < globalNParticles % nSplits; ++split)
    {
        nParticlesPerSplit[split]++;
    }

    int leavesDone = 0;
    for (int split = 0; split < nSplits; ++split)
    {
        std::size_t targetCount = nParticlesPerSplit[split];
        std::size_t splitCount = 0;
        int j = leavesDone;
        while (splitCount < targetCount && j < nNodes(globalTree))
        {
            // if adding the particles of the next leaf takes us further away from
            // the target count than where we're now, we stop
            if (targetCount < splitCount + globalCounts[j] && // overshoot
                targetCount - splitCount < splitCount + globalCounts[j] - targetCount) // overshoot more than undershoot
            { break; }

            splitCount += globalCounts[j++];
        }

        if (split < nSplits - 1)
        {
            // carry over difference of particles over/under assigned to next split
            // to avoid accumulating round off
            long int delta = (long int)(targetCount) - (long int)(splitCount);
            nParticlesPerSplit[split+1] += delta;
        }
        // afaict, j < nNodes(globalTree) can only happen if there are empty nodes at the end
        else {
            for( ; j < nNodes(globalTree); ++j)
                splitCount += globalCounts[j];
        }

        // other distribution strategies might have more than one range per rank
        ret.addRange(Rank(split), globalTree[leavesDone], globalTree[j], splitCount);
        leavesDone = j;
    }

    return ret;
}

//! \brief stores one or multiple index ranges of local particles to send out to another rank
using SendManifest = IndexRanges<int>; // works if there are < 2^31 local particles
//! \brief SendList will contain one manifest per rank
using SendList     = std::vector<SendManifest>;

/*! \brief Based on global assignment, create the list of local particle index ranges to send to each rank
 *
 * \tparam I                 32- or 64-bit integer
 * \param assignment         global space curve assignment to ranks
 * \param mortonCodes        sorted list of morton codes for local particles present on this rank
 * \return                   for each rank, a list of index ranges into \a mortonCodes to send
 *
 * Converts the global assignment Morton code ranges into particle indices with binary search
 */
template<class I>
SendList createSendList(const SpaceCurveAssignment<I>& assignment, const std::vector<I>& mortonCodes)
{
    using IndexType = SendManifest::IndexType;
    int nRanks = assignment.nRanks();

    SendList ret(nRanks);

    for (int rank = 0; rank < nRanks; ++rank)
    {
        SendManifest& manifest = ret[rank];
        for (int rangeIndex = 0; rangeIndex < assignment.nRanges(rank); ++rangeIndex)
        {
            I rangeStart = assignment.rangeStart(rank, rangeIndex);
            I rangeEnd   = assignment.rangeEnd(rank, rangeIndex);

            auto lit = std::lower_bound(cbegin(mortonCodes), cend(mortonCodes), rangeStart);
            IndexType lowerParticleIndex = std::distance(cbegin(mortonCodes), lit);

            auto uit = std::lower_bound(cbegin(mortonCodes) + lowerParticleIndex, cend(mortonCodes), rangeEnd);
            IndexType upperParticleIndex = std::distance(cbegin(mortonCodes), uit);

            IndexType count = std::distance(lit, uit);
            manifest.addRange(lowerParticleIndex, upperParticleIndex, count);
        }
    }

    return ret;
}

/*! \brief create a buffer of elements to send by extracting elements from the source array
 *
 * \tparam T         float or double
 * \param manifest   contains the index ranges of \a source to put into the send buffer
 * \param source     e.g. x,y,z,h arrays
 * \param ordering   the space curve ordering to handle unsorted source arrays
 *                   if source is space-curve-sorted, \a ordering is the trivial 0,1,...,n sequence
 * \return           the send buffer
 */
template<class T>
std::vector<T> createSendBuffer(const SendManifest& manifest, const std::vector<T>& source,
                                const std::vector<int>& ordering)
{
    int sendSize = manifest.totalCount();

    std::vector<T> sendBuffer;
    sendBuffer.reserve(sendSize);
    for (int rangeIndex = 0; rangeIndex < manifest.nRanges(); ++ rangeIndex)
    {
        for (int i = manifest.rangeStart(rangeIndex); i < manifest.rangeEnd(rangeIndex); ++i)
        {
            sendBuffer.push_back(source[ordering[i]]);
        }
    }

    return sendBuffer;
}


} // namespace sphexa
