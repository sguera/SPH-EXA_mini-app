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

/*! @file
 * @brief  3D Morton encoding/decoding in 32- and 64-bit using the magic number method
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
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

//! @brief Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
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

/*! @brief Compacts a 30-bit integer into 10 bits by selecting only bits divisible by 3
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

//! @brief Expands a 21-bit integer into 63 bits by inserting 2 zeros after each bit.
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

/*! @brief Compacts a 63-bit integer into 21 bits by selecting only bits divisible by 3
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

/*! @brief Calculates a Morton code for a 3D point in integer coordinates
 *
 * @tparam KeyType  32- or 64 bit unsigned integer
 *
 * @param[in] ix,iy,iz input coordinates in [0:2^maxTreeLevel<KeyType>{}]
 */
template <class KeyType>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<KeyType>{}, KeyType> imorton3D(unsigned ix, unsigned iy, unsigned iz)
{
    assert(ix < (1u << maxTreeLevel<KeyType>{}));
    assert(iy < (1u << maxTreeLevel<KeyType>{}));
    assert(iz < (1u << maxTreeLevel<KeyType>{}));

    KeyType xx = detail::expandBits(KeyType(ix));
    KeyType yy = detail::expandBits(KeyType(iy));
    KeyType zz = detail::expandBits(KeyType(iz));

    // interleave the x, y, z components
    return xx * 4 + yy * 2 + zz;
}

/*! @brief Calculate morton code from n-level integer 3D coordinates
 *
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param ix         input integer box coordinates, must be in the range [0, 2^treeLevel-1]
 * @param iy
 * @param iz
 * @param treeLevel  octree subdivison level
 * @return           the morton code
 */
template<class KeyType>
CUDA_HOST_DEVICE_FUN
KeyType imorton3D(unsigned ix, unsigned iy, unsigned iz, unsigned treeLevel)
{
    assert(treeLevel <= maxTreeLevel<KeyType>{});
    unsigned shifts = maxTreeLevel<KeyType>{} - treeLevel;
    return imorton3D<KeyType>(ix<<shifts, iy<<shifts, iz<<shifts);
}

/*! @brief Calculates a Morton code for a 3D point in the unit cube
 *
 * @tparam KeyType specify either a 32 or 64 bit unsigned integer to select the precision.
 *                 Note: KeyType needs to be specified explicitly.
 *                 Note: not specifying an unsigned type results in a compilation error
 *
 * @param[in] x input coordinates within the unit cube [0,1]^3
 * @param[in] y
 * @param[in] z
 */
template <class KeyType, class T>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<KeyType>{}, KeyType> morton3DunitCube(T x, T y, T z)
{
    assert(x >= 0.0 && x <= 1.0);
    assert(y >= 0.0 && y <= 1.0);
    assert(z >= 0.0 && z <= 1.0);

    // normalize floating point numbers
    unsigned ix = toNBitInt<KeyType>(x);
    unsigned iy = toNBitInt<KeyType>(y);
    unsigned iz = toNBitInt<KeyType>(z);

    return imorton3D<KeyType>(ix, iy, iz);
}

/*! @brief Calculates a Morton code for a 3D point within the specified box
 *
 * @tparam KeyType specify either a 32 or 64 bit unsigned integer to select
 *           the precision.
 *           Note: KeyType needs to be specified explicitly.
 *           Note: not specifying an unsigned type results in a compilation error
 *
 * @param[in] x,y,z input coordinates within the unit cube [0,1]^3
 * @param[in] box   bounding for coordinates
 *
 * @return          the Morton code
 */
template <class KeyType, class T>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<KeyType>{}, KeyType> morton3D(T x, T y, T z, Box<T> box)
{
    return morton3DunitCube<KeyType>(normalize(x, box.xmin(), box.xmax()),
                               normalize(y, box.ymin(), box.ymax()),
                               normalize(z, box.zmin(), box.zmax()));
}

//! @brief extract X component from a morton code
template<class KeyType>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<KeyType>{}, KeyType> idecodeMortonX(KeyType code)
{
    return detail::compactBits(code >> 2);
}

//! @brief extract Y component from a morton code
template<class KeyType>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<KeyType>{}, KeyType> idecodeMortonY(KeyType code)
{
    return detail::compactBits(code >> 1);
}

//! @brief extract Z component from a morton code
template<class KeyType>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<KeyType>{}, KeyType> idecodeMortonZ(KeyType code)
{
    return detail::compactBits(code);
}

/*! @brief compute range of X values that the given code can cover
 *
 * @tparam KeyType  32- or 64-bit unsigned integer
 * @param code      A morton code, all bits except the first 2 + length
 *                  bits (32-bit) or the first 1 + length bits (64-bit)
 *                  are expected to be zero.
 * @param length    Number of bits to consider for calculating the upper range limit
 * @return          The range of possible X values in [0...2^10] (32-bit)
 *                  or [0...2^21] (64-bit). The start of the range is the
 *                  X-component of the input @p code. The length of the range
 *                  only depends on the number of bits. For every X-bit, the
 *                  range decreases from the maximum by a factor of two.
 */
template<class KeyType>
CUDA_HOST_DEVICE_FUN
inline pair<int> idecodeMortonXRange(KeyType code, int length)
{
    pair<int> ret{0, 0};

    ret[0] = idecodeMortonX(code);
    ret[1] = ret[0] + (KeyType(1) << (maxTreeLevel<KeyType>{} - (length+2)/3));

    return ret;
}

//! @brief see idecodeMortonXRange
template<class KeyType>
CUDA_HOST_DEVICE_FUN
inline pair<int> idecodeMortonYRange(KeyType code, int length)
{
    pair<int> ret{0, 0};

    ret[0] = idecodeMortonY(code);
    ret[1] = ret[0] + (KeyType(1) << (maxTreeLevel<KeyType>{} - (length+1)/3));

    return ret;
}

//! @brief see idecodeMortonXRange
template<class KeyType>
CUDA_HOST_DEVICE_FUN
inline pair<int> idecodeMortonZRange(KeyType code, int length)
{
    pair<int> ret{0, 0};

    ret[0] = idecodeMortonZ(code);
    ret[1] = ret[0] + (KeyType(1) << (maxTreeLevel<KeyType>{} - length/3));

    return ret;
}

/*! @brief returns (min,max) x-coordinate pair for a lower and upper morton code
 *
 * @tparam T         float or double
 * @tparam KeyType   32- or 64-bit unsigned integer
 * @param prefix     lowest Morton code
 * @param upperCode  upper Morton code
 * @param box        floating point coordinate bounding box used to construct
 *                   @p prefix and @p upper code
 * @return           floating point coordinate range
 *
 * Note that prefix and upperCode are assumed to delineate an octree node.
 * Therefore upperCode - prefix should be a power of 8. This saves one decode step.
 * If the difference is not a power of 8, use idecodeMortonXYZ on @p prefix and @p upperCode
 * separately.
 */
template<class T, class KeyType>
pair<T> decodeMortonXRange(KeyType prefix, KeyType upperCode, const Box<T>& box)
{
    assert(isPowerOf8(upperCode - prefix));
    constexpr int maxCoord = 1u<<maxTreeLevel<KeyType>{};
    constexpr T uL = T(1.) / maxCoord;

    int ix = idecodeMortonX(prefix);
    T xBox = box.xmin() + ix * uL * box.lx();
    int unitsPerBox = 1u<<(maxTreeLevel<KeyType>{} - treeLevel(upperCode - prefix));

    T uLx = uL * box.lx() * unitsPerBox;

    return {xBox, xBox + uLx};
}

//! @brief see decodeMortonXRange
template<class T, class KeyType>
pair<T> decodeMortonYRange(KeyType prefix, KeyType upperCode, const Box<T>& box)
{
    assert(isPowerOf8(upperCode - prefix));
    constexpr int maxCoord = 1u<<maxTreeLevel<KeyType>{};
    constexpr T uL = T(1.) / maxCoord;

    int iy = idecodeMortonY(prefix);
    T yBox = box.ymin() + iy * uL * box.ly();
    int unitsPerBox = 1u<<(maxTreeLevel<KeyType>{} - treeLevel(upperCode - prefix));

    T uLy = uL * box.ly() * unitsPerBox;

    return {yBox, yBox + uLy};
}

//! @brief see decodeMortonXRange
template<class T, class KeyType>
pair<T> decodeMortonZRange(KeyType prefix, KeyType upperCode, const Box<T>& box)
{
    assert(isPowerOf8(upperCode - prefix));
    constexpr int maxCoord = 1u<<maxTreeLevel<KeyType>{};
    constexpr T uL = T(1.) / maxCoord;

    int iz = idecodeMortonZ(prefix);
    T zBox = box.zmin() + iz * uL * box.lz();
    int unitsPerBox = 1u<<(maxTreeLevel<KeyType>{} - treeLevel(upperCode - prefix));

    T uLz = uL * box.lz() * unitsPerBox;

    return {zBox, zBox + uLz};
}

/*! @brief compute morton codes corresponding to neighboring octree nodes
 *         for a given input code and tree level
 *
 * @tparam KeyType  32- or 64-bit unsigned integer type
 * @param code      input Morton code
 * @param treeLevel octree subdivision level, 0-10 for 32-bit, and 0-21 for 64-bit
 * @param dx        neighbor offset in x direction at @p treeLevel
 * @param dy        neighbor offset in y direction at @p treeLevel
 * @param dz        neighbor offset in z direction at @p treeLevel
 * @param pbcX      apply pbc in X direction
 * @param pbcY      apply pbc in Y direction
 * @param pbcZ      apply pbc in Z direction
 * @return          morton neighbor start code
 *
 * Note that the end of the neighbor range is given by the start code + nodeRange(treeLevel)
 */
template<class KeyType>
CUDA_HOST_DEVICE_FUN
inline std::enable_if_t<std::is_unsigned<KeyType>{}, KeyType>
mortonNeighbor(KeyType code, unsigned treeLevel, int dx, int dy, int dz,
               bool pbcX=true, bool pbcY=true, bool pbcZ=true)
{
    // maximum coordinate value per dimension 2^nBits-1
    constexpr int pbcRange = 1u << maxTreeLevel<KeyType>{};
    constexpr int maxCoord = pbcRange - 1;

    unsigned shiftBits  = maxTreeLevel<KeyType>{} - treeLevel;
    int shiftValue = int(1u << shiftBits);

    // zero out lower tree levels
    code = enclosingBoxCode(code, treeLevel);

    int x = idecodeMortonX(code);
    int y = idecodeMortonY(code);
    int z = idecodeMortonZ(code);

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

    return detail::expandBits(KeyType(x)) * KeyType(4)
         + detail::expandBits(KeyType(y)) * KeyType(2)
         + detail::expandBits(KeyType(z));
}


/*! @brief compute the Morton codes for the input coordinate arrays
 *
 * @tparam     T          float or double
 * @param[in]  xBegin     input iterators for coordinate arrays
 * @param[in]  xEnd
 * @param[in]  yBegin
 * @param[in]  zBegin
 * @param[out] codeBegin  output for morton codes
 * @param[in]  box        coordinate bounding box
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
