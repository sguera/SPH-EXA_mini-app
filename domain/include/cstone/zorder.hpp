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
 * \brief Functionality for calculating the z-order and for reordering arrays according to it.
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>
#include <vector>

#include "mortoncode.hpp"

namespace cstone
{

/*! \brief calculate the sortKey that sorts the input sequence
 *
 * \param[in] inBegin input sequence start 
 * \param[in] inEnd input sequence end
 * \param[out] outBegin integer sortKey output
 *
 * upon completion of this routine, the output sequence contains the sort keys
 * that will access the input sequence in a sorted manner, i.e.
 * the sequence inBegin[outBegin[k]], k=0,1,2,...,n is sorted
 */
template <class InputIterator, class OutputIterator>
void sort_invert(InputIterator inBegin, InputIterator inEnd, OutputIterator outBegin)
{
    using ValueType = std::decay_t<decltype(*inBegin)>;
    using Integer   = std::decay_t<decltype(*outBegin)>;
    std::size_t n   = std::distance(inBegin, inEnd);

    // create index sequence 0,1,2,...,n
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
        outBegin[i] = i;

    // zip the input integer array together with the index sequence
    std::vector<std::tuple<ValueType, Integer>> keyIndexPairs(n);
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
        keyIndexPairs[i] = std::make_tuple(inBegin[i], outBegin[i]);

    // sort, comparing only the first tuple element
    std::sort(begin(keyIndexPairs), end(keyIndexPairs),
              [](const auto& t1, const auto& t2){ return std::get<0>(t1) < std::get<0>(t2); });

    // extract the resulting ordering
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < n; ++i)
        outBegin[i] = std::get<1>(keyIndexPairs[i]);
}


/*! \brief compute the Morton z-order for the input coordinate arrays
 *
 * \param[in]  [x,y,z][Begin, End] (const) input iterators for coordinate arrays
 * \param[out] order[Begin, End]  output for z-order
 * \param[in]  [x,y,z][min, max]  coordinate bounding box
 */
template<class InputIterator, class OutputIterator, class T>
void computeZorder(InputIterator  xBegin,
                   InputIterator  xEnd,
                   InputIterator  yBegin,
                   InputIterator  zBegin,
                   OutputIterator orderBegin,
                   const Box<T>&  box)
{
    std::size_t n = std::distance(xBegin, xEnd);

    std::vector<unsigned> mortonCodes(n); 
    computeMortonCodes(xBegin, xEnd, yBegin, zBegin, begin(mortonCodes), box);

    sort_invert(cbegin(mortonCodes), cend(mortonCodes), orderBegin);
}


/*! \brief reorder the input array according to the specified ordering
 *
 * \tparam I          integer type
 * \tparam ValueType  float or double
 * \param ordering    an ordering
 * \param array       an array, size >= ordering.size(), particles past ordering.size()
 *                    are copied element by element
 */
template<class I, class ValueType>
void reorder(const std::vector<I>& ordering, std::vector<ValueType>& array)
{
    assert(array.size() >= ordering.size());

    std::vector<ValueType> tmp(array.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < ordering.size(); ++i)
    {
        tmp[i] = array[ordering[i]];
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t i = ordering.size(); i < array.size(); ++i)
    {
        tmp[i] = array[i];
    }
    swap(tmp, array);
}

/*! \brief reorder the input array according to the specified ordering
 *
 * \tparam I          integer type
 * \tparam ValueType  float or double
 * \param ordering    an ordering, all indices from 0 to ordering.size() are accessed
 * \param array       an array, indices offset to offset + ordering.size() are reordered.
 *                    other elements are copied element by element
 * \param offset      access array with an offset
 */
template<class I, class ValueType>
void reorder(const std::vector<I>& ordering, std::vector<ValueType>& array, int offset)
{
    assert(array.size() >= ordering.size() + offset);

    std::vector<ValueType> tmp(array.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < offset; ++i)
    {
        tmp[i] = array[i];
    }

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < ordering.size(); ++i)
    {
        tmp[i+offset] = array[ordering[i]+offset];
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = ordering.size()+offset; i < array.size(); ++i)
    {
        tmp[i] = array[i];
    }

    swap(tmp, array);
}


/*! \brief reorder the input array according to the specified ordering, no reallocation
 *
 * \tparam LocalIndex    integer type
 * \tparam ValueType     float or double
 * \param ordering       an ordering
 * \param array          an array, size >= ordering.size()
 */
template<class LocalIndex, class ValueType>
void reorderInPlace(const std::vector<LocalIndex>& ordering, ValueType* array)
{
    std::vector<ValueType> tmp(ordering.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < ordering.size(); ++i)
    {
        tmp[i] = array[ordering[i]];
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < ordering.size(); ++i)
    {
        array[i] = tmp[i];
    }
}

//! \brief This class conforms to the same interface as the device version to allow abstraction
template<class ValueType, class CodeType, class IndexType>
class CpuGather
{
public:
    CpuGather() = default;

    /*! \brief upload the new reorder map to the device and reallocates buffers if necessary
     *
     * If the sequence [map_first:map_last] does not contain each element [0:map_last-map_first]
     * exactly once, the behavior is undefined.
     */
    void setReorderMap(const IndexType* map_first, const IndexType* map_last)
    {
        mapSize_ = std::size_t(map_last - map_first);
        ordering_.resize(mapSize_);
        std::copy(map_first, map_last, begin(ordering_));
    }

    void getReorderMap(IndexType* map_first)
    {
        std::copy(ordering_.data(), ordering_.data() + mapSize_, map_first);
    }

    /*! \brief sort given Morton codes on the device and determine reorder map based on sort order
     *
     * \param[inout] codes_first   pointer to first Morton code
     * \param[inout] codes_last    pointer to last Morton code
     *
     * Precondition:
     *   - [codes_first:codes_last] is a continues sequence of accessible elements of size N
     *
     * Postcondition
     *   - [codes_first:codes_last] is sorted
     *   - subsequent calls to operator() apply a gather operation to the input sequence
     *     with the map obtained from sort_by_key with [codes_first:codes_last] as the keys
     *     and the identity permutation as the values
     *
     *  Remarks:
     *    - reallocates space on the device if necessary to fit N elements of type LocalIndex
     *      and a second buffer of size max(2N*sizeof(T), N*sizeof(I))
     */
    void setMapFromCodes(CodeType* codes_first, CodeType* codes_last)
    {
        mapSize_ = std::size_t(codes_last - codes_first);
        ordering_.resize(mapSize_);

        sort_invert(codes_first, codes_last, begin(ordering_));
        reorderInPlace(ordering_, codes_first);
    }

    /*! \brief reorder the array \a values according to the reorder map provided previously
     *
     * \a values must have at least as many elements as the reorder map provided in the last call
     * to setReorderMap or setMapFromCodes, otherwise the behavior is undefined.
     */
    void operator()(ValueType* values)
    {
        reorderInPlace(ordering_, values);
    }

private:
    std::size_t mapSize_{0};
    std::vector<IndexType> ordering_;
};

} // namespace cstone
