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
 * \brief Test Z-order functionality
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */


#include <vector>

#include "gtest/gtest.h"

#include "cstone/zorder.hpp"

using namespace cstone;

TEST(SFC, sortInvert)
{
    std::vector<int> v{2,1,5,4};

    // the sort keys that sorts v is {1,0,3,2}
    std::vector<int> sortKey(v.size());

    sort_invert(begin(v), end(v), begin(sortKey));

    
    std::vector<int> reference{1,0,3,2};
    EXPECT_EQ(sortKey, reference);
}

TEST(SFC, computeZorder)
{
    // assume BBox of [-1, 1]^3
    constexpr double boxMin = -1;
    constexpr double boxMax = 1;
    Box<double> box{boxMin, boxMax};

    // 8 particles, each centered in each of the 8 octants,
    // Z-indices            4    5      1     6     3     0     2    7
    // position             0    1      2     3     4     5     6    7
    std::vector<double> x{ 0.5,  0.5, -0.5,  0.5, -0.5, -0.5, -0.5, 0.5};
    std::vector<double> y{-0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5, 0.5};
    std::vector<double> z{-0.5,  0.5,  0.5, -0.5,  0.5, -0.5, -0.5, 0.5};

    // the sort order to access coordinates in ascending Z-order is
    // 5, 2, 6, 4, 0, 1, 3, 7
    std::vector<unsigned> reference{5,2,6,4,0,1,3,7};

    std::vector<unsigned> zOrder(x.size());
    computeZorder(begin(x), end(x), begin(y), begin(z), begin(zOrder), box);

    EXPECT_EQ(zOrder, reference);
}

TEST(SFC, reorder)
{
    std::vector<int> ordering{1,0,4,3,2};

    std::vector<double> x(10);
    std::iota(begin(x), end(x), 0);

    int offset = 2;

    reorder(ordering, x, offset);

    std::vector<double> refX{0,1,3,2,6,5,4,7,8,9};
    EXPECT_EQ(x, refX);
}

template<class ValueType, class CodeType, class IndexType>
void CpuGatherTest()
{
    std::vector<CodeType> codes{0, 50, 10, 60, 20, 70, 30, 80, 40, 90};

    CpuGather<ValueType, CodeType, IndexType> cpuGather;
    cpuGather.setMapFromCodes(codes.data(), codes.data() + codes.size());

    {
        std::vector<CodeType> refCodes{0,10,20,30,40,50,60,70,80,90};
        EXPECT_EQ(codes, refCodes);
    }

    std::vector<ValueType>    values{-2, -1, 0,1,2,3,4,5,6,7,8,9, 10, 11};
    cpuGather(values.data() + 2);
    std::vector<ValueType> reference{-2, -1, 0,2,4,6,8,1,3,5,7,9, 10, 11};

    EXPECT_EQ(values, reference);
}

TEST(SFC, CpuGather)
{
    CpuGatherTest<float,  unsigned, unsigned>();
    CpuGatherTest<float,  uint64_t, unsigned>();
    CpuGatherTest<double, unsigned, unsigned>();
    CpuGatherTest<double, uint64_t, unsigned>();
    CpuGatherTest<float,  unsigned, uint64_t>();
    CpuGatherTest<float,  uint64_t, uint64_t>();
    CpuGatherTest<double, unsigned, uint64_t>();
    CpuGatherTest<double, uint64_t, uint64_t>();
}