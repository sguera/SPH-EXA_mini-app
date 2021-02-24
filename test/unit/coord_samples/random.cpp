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
 * \brief Random coordinates generation for testing
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <gtest/gtest.h>

#include "random.hpp"

using namespace sphexa;

TEST(CoordinateSamples, randomContainerIsSorted)
{
    using real = double;
    using CodeType = unsigned;
    int n = 10;

    sphexa::Box<real> box{ 0, 1, -1, 2, 0, 5 };
    RandomCoordinates<real, CodeType> c(n, box);

    std::vector<CodeType> testCodes(n);
    sphexa::computeMortonCodes(begin(c.x()), end(c.x()), begin(c.y()), begin(c.z()),
                               begin(testCodes), box);

    EXPECT_EQ(testCodes, c.mortonCodes());

    std::vector<CodeType> testCodesSorted = testCodes;
    std::sort(begin(testCodesSorted), end(testCodesSorted));

    EXPECT_EQ(testCodes, testCodesSorted);
}

template<class I>
std::vector<I> makeRegularGrid(unsigned gridSize)
{
    assert(sphexa::isPowerOf8(gridSize));
    std::vector<I> codes;

    unsigned level = sphexa::log8ceil(gridSize);

    // a regular n x n x n grid
    unsigned n = 1u << level;
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < n; ++j)
            for (unsigned k = 0; k < n; ++k)
            {
                codes.push_back(codeFromBox<I>(i, j, k, level));
            }

    std::sort(begin(codes), end(codes));

    return codes;
}

template<class I>
void testRegularGrid()
{
    unsigned gridSize = 64;

    std::vector<I> refCodes = makeRegularGrid<I>(gridSize);
    RegularGridCoordinates<double, I> coords(gridSize);

    EXPECT_EQ(refCodes, coords.mortonCodes());
}

TEST(CoordinateSamples, regularGridCodes)
{
    testRegularGrid<unsigned>();
    testRegularGrid<uint64_t>();
}