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
 * \brief  Tests a gather (reordering) operation on arrays
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "cstone/zorder.hpp"
#include "cstone/cuda/gather.cuh"

template<class I>
std::vector<I> makeRandomPermutation(std::size_t nElements)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<I> map(nElements);
    std::iota(begin(map), end(map), 0);
    std::shuffle(begin(map), end(map), gen);
    return map;
}

template<class T, class I, class IndexType>
void setFromCodeDemo()
{
    std::vector<I> codes{0, 50, 10, 60, 20, 70, 30, 80, 40, 90};

    cstone::DeviceGather<T, I, IndexType> devGather;
    devGather.setMapFromCodes(codes.data(), codes.data() + codes.size());

    std::vector<I> refCodes{0,10,20,30,40,50,60,70,80,90};
    EXPECT_EQ(codes, refCodes);

    std::vector<T> values{0,1,2,3,4,5,6,7,8,9};
    devGather(values.data());
    std::vector<T> reference{0,2,4,6,8,1,3,5,7,9};

    EXPECT_EQ(values, reference);
}

TEST(DeviceGather, smallDemo)
{
    setFromCodeDemo<float, unsigned , unsigned>();
    setFromCodeDemo<float, uint64_t , unsigned>();
    setFromCodeDemo<double, unsigned, unsigned>();
    setFromCodeDemo<double, uint64_t, unsigned>();
    setFromCodeDemo<float, unsigned , uint64_t>();
    setFromCodeDemo<float, uint64_t , uint64_t>();
    setFromCodeDemo<double, unsigned, uint64_t>();
    setFromCodeDemo<double, uint64_t, uint64_t>();
}

template<class T, class I, class IndexType>
void reorderCheck(int nElements, bool reallocate = false)
{
    cstone::DeviceGather<T, I, IndexType> devGather;
    if (reallocate)
    {
        // initialize with a small size to trigger buffer reallocation
        std::vector<IndexType> ord(10);
        devGather.setReorderMap(ord.data(), ord.data() + ord.size());
    }

    std::vector<I> codes = makeRandomPermutation<I>(nElements);

    std::vector<T> v(nElements);
    std::iota(begin(v), end(v), 0);
    std::vector<T> hv = v;

    std::vector<IndexType> h_order(nElements);
    std::iota(begin(h_order), end(h_order), 0u);
    cstone::sort_invert(begin(codes), end(codes), begin(h_order));

    auto tcpu0 = std::chrono::high_resolution_clock::now();
    cstone::reorder(h_order, hv);
    auto tcpu1 = std::chrono::high_resolution_clock::now();
    std::cout << "cpu gather Melements/s: " << T(nElements)/(1e6 * std::chrono::duration<double>(tcpu1 - tcpu0).count()) << std::endl;

    devGather.setMapFromCodes(codes.data(), codes.data() + codes.size());

    EXPECT_TRUE(std::is_sorted(begin(codes), end(codes)));

    auto tgpu0 = std::chrono::high_resolution_clock::now();
    devGather(v.data());
    auto tgpu1 = std::chrono::high_resolution_clock::now();

    std::cout << "gpu gather Melements/s: " << T(nElements)/(1e6*std::chrono::duration<double>(tgpu1 - tgpu0).count()) << std::endl;

    EXPECT_EQ(v, hv);
}

TEST(DeviceGather, matchCpu)
{
    int nElements = 320000;

    reorderCheck<float, unsigned , unsigned>(nElements);
    reorderCheck<float, uint64_t , unsigned>(nElements);
    reorderCheck<double, unsigned, unsigned>(nElements);
    reorderCheck<double, uint64_t, unsigned>(nElements);
    reorderCheck<float, unsigned , uint64_t>(nElements);
    reorderCheck<float, uint64_t , uint64_t>(nElements);
    reorderCheck<double, unsigned, uint64_t>(nElements);
    reorderCheck<double, uint64_t, uint64_t>(nElements);
}

TEST(DeviceGather, reallocate)
{
    int nElements = 32000;

    reorderCheck<float, unsigned , unsigned>(nElements, true);
    reorderCheck<float, uint64_t , unsigned>(nElements, true);
    reorderCheck<double, unsigned, unsigned>(nElements, true);
    reorderCheck<double, uint64_t, unsigned>(nElements, true);
    reorderCheck<float, unsigned , uint64_t>(nElements, true);
    reorderCheck<float, uint64_t , uint64_t>(nElements, true);
    reorderCheck<double, unsigned, uint64_t>(nElements, true);
    reorderCheck<double, uint64_t, uint64_t>(nElements, true);
}
