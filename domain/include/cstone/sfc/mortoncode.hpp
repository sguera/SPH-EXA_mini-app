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
 * \brief  3D Morton encoding/decoding in 32- and 64-bit using the magic number method
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <cassert>
#include <cmath>       // for std::ceil
#include <cstdint>     // for uint32_t and uint64_t
#include <type_traits> // for std::enable_if_t

#include "box.hpp"
#include "common.hpp"

namespace cstone
{

namespace detail
{

//! \brief Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
CUDA_HOST_DEVICE_FUN
inline unsigned expandBits(unsigned v)
{
    v &= 0x000003ffu; // discard bit higher 10
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

/*! \brief Compacts a 30-bit integer into 10 bits by selecting only bits divisible by 3
 *         this inverts expandBits
 */
CUDA_HOST_DEVICE_FUN
inline unsigned compactBits(unsigned v)
{
    v &= 0x09249249u;
    v = (v ^ (v >>  2u)) & 0x030c30c3u;
    v = (v ^ (v >>  4u)) & 0x0300f00fu;
    v = (v ^ (v >>  8u)) & 0xff0000ffu;
    v = (v ^ (v >> 16u)) & 0x000003ffu;
    return v;
}

//! \brief Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
CUDA_HOST_DEVICE_FUN
inline uint64_t expandBits(uint64_t v)
{
    uint64_t x = v & 0x1fffffu; // discard bits higher 21
    x = (x | x << 32u) & 0x001f00000000fffflu;
    x = (x | x << 16u) & 0x001f0000ff0000fflu;
    x = (x | x << 8u)  & 0x100f00f00f00f00flu;
    x = (x | x << 4u)  & 0x10c30c30c30c30c3lu;
    x = (x | x << 2u)  & 0x1249249249249249lu;
    return x;
}

/*! \brief Compacts a 63-bit integer into 21 bits by selecting only bits divisible by 3
 *         this inverts expandBits
 */
CUDA_HOST_DEVICE_FUN
inline uint64_t compactBits(uint64_t v)
{
    v &= 0x1249249249249249lu;
    v = (v ^ (v >>  2u)) & 0x10c30c30c30c30c3lu;
    v = (v ^ (v >>  4u)) & 0x100f00f00f00f00flu;
    v = (v ^ (v >>  8u)) & 0x001f0000ff0000fflu;
    v = (v ^ (v >> 16u)) & 0x001f00000000fffflu;
    v = (v ^ (v >> 32u)) & 0x00000000001ffffflu;
    return v;
}

} // namespace detail

/*! \brief Calculates a Morton code for a 3D point in integer coordinates
 *
 * \tparam I  32- or 64 bit unsigned integer
 *
 * \param[in] ix,iy,iz input coordinates in [0:2^maxTreeLevel<I>{}]
 */
template <class I>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, I> imorton3D(unsigned ix, unsigned iy, unsigned iz)
{
    assert(ix < (1u << maxTreeLevel<I>{}));
    assert(iy < (1u << maxTreeLevel<I>{}));
    assert(iz < (1u << maxTreeLevel<I>{}));

    I xx = detail::expandBits(I(ix));
    I yy = detail::expandBits(I(iy));
    I zz = detail::expandBits(I(iz));

    // interleave the x, y, z components
    return xx * 4 + yy * 2 + zz;
}

/*! \brief Calculate morton code from n-level integer 3D coordinates
 *
 * \tparam I         32- or 64-bit unsigned integer
 * \param ix,iy,iz   input integer box coordinates, must be in the range [0, 2^treeLevel-1]
 * \param treeLevel  octree subdivison level
 * \return           the morton code
 */
template<class I>
CUDA_HOST_DEVICE_FUN
I imorton3D(unsigned ix, unsigned iy, unsigned iz, unsigned treeLevel)
{
    assert(treeLevel <= maxTreeLevel<I>{});
    unsigned shifts = maxTreeLevel<I>{} - treeLevel;
    return imorton3D<I>(ix<<shifts, iy<<shifts, iz<<shifts);
}

/*! \brief Calculates a Morton code for a 3D point in the unit cube
 *
 * \tparam I specify either a 32 or 64 bit unsigned integer to select
 *           the precision.
 *           Note: I needs to be specified explicitly.
 *           Note: not specifying an unsigned type results in a compilation error
 *
 * \param[in] x,y,z input coordinates within the unit cube [0,1]^3
 */
template <class I, class T>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, I> morton3DunitCube(T x, T y, T z)
{
    assert(x >= 0.0 && x <= 1.0);
    assert(y >= 0.0 && y <= 1.0);
    assert(z >= 0.0 && z <= 1.0);

    // normalize floating point numbers
    unsigned ix = toNBitInt<I>(x);
    unsigned iy = toNBitInt<I>(y);
    unsigned iz = toNBitInt<I>(z);

    return imorton3D<I>(ix, iy, iz);
}

/*! \brief Calculates a Morton code for a 3D point within the specified box
 *
 * \tparam I specify either a 32 or 64 bit unsigned integer to select
 *           the precision.
 *           Note: I needs to be specified explicitly.
 *           Note: not specifying an unsigned type results in a compilation error
 *
 * \param[in] x,y,z input coordinates within the unit cube [0,1]^3
 * \param[in] box   bounding for coordinates
 *
 * \return          the Morton code
 */
template <class I, class T>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, I> morton3D(T x, T y, T z, Box<T> box)
{
    return morton3DunitCube<I>(normalize(x, box.xmin(), box.xmax()),
                               normalize(y, box.ymin(), box.ymax()),
                               normalize(z, box.zmin(), box.zmax()));
}

//! \brief extract X component from a morton code
template<class I>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, I> decodeMortonX(I code)
{
    return detail::compactBits(code >> 2);
}

//! \brief extract Y component from a morton code
template<class I>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, I> decodeMortonY(I code)
{
    return detail::compactBits(code >> 1);
}

//! \brief extract Z component from a morton code
template<class I>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, I> decodeMortonZ(I code)
{
    return detail::compactBits(code);
}

/*! \brief compute range of X values that the given code can cover
 *
 * @tparam I      32- or 64-bit unsigned integer
 * @param code    A morton code, all bits except the first 2 + length
 *                bits (32-bit) or the first 1 + length bits (64-bit)
 *                are expected to be zero.
 * @param length  Number of bits to consider for calculating the upper range limit
 * @return        The range of possible X values in [0...2^10] (32-bit)
 *                or [0...2^21] (64-bit). The start of the range is the
 *                X-component of the input \a code. The length of the range
 *                only depends on the number of bits. For every X-bit, the
 *                range decreases from the maximum by a factor of two.
 */
template<class I>
CUDA_HOST_DEVICE_FUN
inline pair<int> decodeXRange(I code, int length)
{
    pair<int> ret{0, 0};

    ret[0] = decodeMortonX(code);
    ret[1] = ret[0] + (I(1) << (maxTreeLevel<I>{} - (length+2)/3));

    return ret;
}

//! \brief see decodeXRange
template<class I>
CUDA_HOST_DEVICE_FUN
inline pair<int> decodeYRange(I code, int length)
{
    pair<int> ret{0, 0};

    ret[0] = decodeMortonY(code);
    ret[1] = ret[0] + (I(1) << (maxTreeLevel<I>{} - (length+1)/3));

    return ret;
}

//! \brief see decodeXRange
template<class I>
CUDA_HOST_DEVICE_FUN
inline pair<int> decodeZRange(I code, int length)
{
    pair<int> ret{0, 0};

    ret[0] = decodeMortonZ(code);
    ret[1] = ret[0] + (I(1) << (maxTreeLevel<I>{} - length/3));

    return ret;
}

/*! \brief compute the X coordinate for a given morton code inside a given box
 *
 * @tparam I      32- or 64-bit unsigned integer
 * @tparam T      float or double
 * @param code    A morton code
 * @param box     Number of bits to consider for calculating the upper range limit
 * @return        The corresponding X coordinate for the input code in the given box
 */
template<class I, class T>
CUDA_HOST_DEVICE_FUN
inline T decodeXCoordinate(I code, const Box<T> &bbox)
{
    constexpr int maxCoord = 1u<<maxTreeLevel<I>{};
    constexpr T uL = T(1.) / maxCoord; // uL = unitLength
    int ixBox = decodeMortonX(code);
    T ret = bbox.xmin() + ixBox * uL * bbox.lx();

    return ret;
}

//! \brief see decodeXCoordinate
template<class I, class T>
CUDA_HOST_DEVICE_FUN
inline T decodeYCoordinate(I code, const Box<T> &bbox)
{
    constexpr int maxCoord = 1u<<maxTreeLevel<I>{};
    constexpr T uL = T(1.) / maxCoord; // uL = unitLength
    int iyBox = decodeMortonY(code);
    T ret = bbox.ymin() + iyBox * uL * bbox.ly();

    return ret;
}

//! \brief see decodeXCoordinate
template<class I, class T>
CUDA_HOST_DEVICE_FUN
inline T decodeZCoordinate(I code, const Box<T> &bbox)
{
    constexpr int maxCoord = 1u<<maxTreeLevel<I>{};
    constexpr T uL = T(1.) / maxCoord; // uL = unitLength
    int izBox = decodeMortonZ(code);
    T ret = bbox.zmin() + izBox * uL * bbox.lz();

    return ret;
}

/*! \brief Calculate the morton code corresponding to integer input box coordinates
 *         at a given tree subdivision level.
 *
 * \tparam I         32- or 64-bit unsigned integer
 * \param xyz        input integer box coordinates, must be in the range [0, 2^treeLevel-1]
 * \param treeLevel  octree subdivison level
 * \return           the morton code
 *
 * This function is allowed to stay here because it's used in octree.hpp
 * and does not depend on std::array.
 */
template<class I>
CUDA_HOST_DEVICE_FUN
I codeFromBox(unsigned x, unsigned y, unsigned z, unsigned treeLevel)
{
    assert(treeLevel <= maxTreeLevel<I>{});
    unsigned shifts = maxTreeLevel<I>{} - treeLevel;

    assert(x < (1u << treeLevel));
    assert(y < (1u << treeLevel));
    assert(z < (1u << treeLevel));

    I xx = detail::expandBits(I(x) << shifts);
    I yy = detail::expandBits(I(y) << shifts);
    I zz = detail::expandBits(I(z) << shifts);

    // interleave the x, y, z components
    return xx * 4 + yy * 2 + zz;
}

/*! \brief compute morton codes corresponding to neighboring octree nodes
 *         for a given input code and tree level
 *
 * \tparam I        32- or 64-bit unsigned integer type
 * \param code      input Morton code
 * \param treeLevel octree subdivision level, 0-10 for 32-bit, and 0-21 for 64-bit
 * \param dx        neighbor offset in x direction at \a treeLevel
 * \param dy        neighbor offset in y direction at \a treeLevel
 * \param dz        neighbor offset in z direction at \a treeLevel
 * \param pbcX      apply pbc in X direction
 * \param pbcY      apply pbc in Y direction
 * \param pbcZ      apply pbc in Z direction
 * \return          morton neighbor start code
 *
 * Note that the end of the neighbor range is given by the start code + nodeRange(treeLevel)
 */
template<class I>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<I>{}, I>
mortonNeighbor(I code, unsigned treeLevel, int dx, int dy, int dz,
               bool pbcX=true, bool pbcY=true, bool pbcZ=true)
{
    // maximum coordinate value per dimension 2^nBits-1
    constexpr int pbcRange = 1u << maxTreeLevel<I>{};
    constexpr int maxCoord = pbcRange - 1;

    unsigned shiftBits  = maxTreeLevel<I>{} - treeLevel;
    int shiftValue = int(1u << shiftBits);

    // zero out lower tree levels
    code = enclosingBoxCode(code, treeLevel);

    int x = decodeMortonX(code);
    int y = decodeMortonY(code);
    int z = decodeMortonZ(code);

    int newX = x + dx * shiftValue;
    if (pbcX) {
        x = pbcAdjust<pbcRange>(newX);
    }
    else {
        x = (newX < 0 || newX > maxCoord) ? x : newX;
    }

    int newY = y + dy * shiftValue;
    if (pbcY) {
        y = pbcAdjust<pbcRange>(newY);
    }
    else {
        y = (newY < 0 || newY > maxCoord) ? y : newY;
    }

    int newZ = z + dz * shiftValue;
    if (pbcZ) {
        z = pbcAdjust<pbcRange>(newZ);
    }
    else {
        z = (newZ < 0 || newZ > maxCoord) ? z : newZ;
    }

    return detail::expandBits(I(x)) * I(4)
         + detail::expandBits(I(y)) * I(2)
         + detail::expandBits(I(z));
}


/*! \brief compute the Morton codes for the input coordinate arrays
 *
 * \param[in]  [x,y,z][Begin, End] (const) input iterators for coordinate arrays
 * \param[out] order[Begin, End]  output for morton codes
 * \param[in]  [x,y,z][min, max]  coordinate bounding box
 */
template<class InputIterator, class OutputIterator, class T>
void computeMortonCodes(InputIterator  xBegin,
                        InputIterator  xEnd,
                        InputIterator  yBegin,
                        InputIterator  zBegin,
                        OutputIterator codesBegin,
                        const Box<T>& box)
{
    assert(xEnd >= xBegin);
    using CodeType = std::decay_t<decltype(*codesBegin)>;

    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < std::size_t(xEnd-xBegin); ++i)
    {
        codesBegin[i] = morton3D<CodeType>(xBegin[i], yBegin[i], zBegin[i], box);
    }
}

} // namespace cstone
