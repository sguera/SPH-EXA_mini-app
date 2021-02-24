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
 * \brief Test morton code implementation
 *
 * \author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <chrono>
#include <iostream>
#include <numeric>

#include "cstone/octree.hpp"

#include "cstone/halodiscovery.hpp"
#include "cstone/domaindecomp.hpp"

#include "coord_samples/random.hpp"

using namespace cstone;

int main()
{
    using CodeType = unsigned;
    Box<double> box{-1, 1};

    int nParticles = 2000000;
    int bucketSize = 10;

    RandomGaussianCoordinates<double, CodeType> randomBox(nParticles, box);

    auto tp0 = std::chrono::high_resolution_clock::now();

    auto [tree, counts] = computeOctree(randomBox.mortonCodes().data(),
                                        randomBox.mortonCodes().data() + nParticles,
                                        bucketSize);

    auto tp1  = std::chrono::high_resolution_clock::now();

    double t0 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "build time from scratch " << t0 << " nNodes(tree): " << nNodes(tree)
              << " count: " << std::accumulate(begin(counts), end(counts), 0lu) << std::endl;

    tp0  = std::chrono::high_resolution_clock::now();

    auto [tree2, counts2] = computeOctree(randomBox.mortonCodes().data(),
                                          randomBox.mortonCodes().data() + nParticles,
                                          bucketSize, std::numeric_limits<unsigned>::max(),
                                          std::move(tree));

    tp1  = std::chrono::high_resolution_clock::now();

    double t1 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "build time with guess " << t1 << " nNodes(tree): " << nNodes(tree2)
              << " count: " << std::accumulate(begin(counts2), end(counts2), 0lu) << std::endl;

    int nSplits = 4;
    SpaceCurveAssignment<CodeType> assignment = singleRangeSfcSplit(tree2, counts2, nSplits);
    std::vector<double> haloRadii(nNodes(tree2), 0.01);

    std::vector<pair<int>> haloPairs;
    int doSplit = 0;
    tp0  = std::chrono::high_resolution_clock::now();
    findHalos(tree2, haloRadii, box, assignment, doSplit, haloPairs);
    tp1  = std::chrono::high_resolution_clock::now();

    double t2 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "halo discovery: " << t2 << " nPairs: " << haloPairs.size() << std::endl;
}

