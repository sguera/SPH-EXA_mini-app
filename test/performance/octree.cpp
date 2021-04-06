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
 * @brief Test morton code implementation
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <chrono>
#include <iostream>
#include <numeric>

#include "cstone/domain/halodiscovery.hpp"
#include "cstone/domain/domaindecomp.hpp"
#include "cstone/halos/btreetraversal.hpp"
#include "cstone/tree/octree.hpp"

#include "coord_samples/random.hpp"
#include "coord_samples/plummer.hpp"

using namespace cstone;

template<class I>
std::tuple<std::vector<I>, std::vector<unsigned>>
build_tree(const I* firstCode, const I* lastCode, unsigned bucketSize)
{
    std::vector<I> tree;
    std::vector<unsigned> counts;

    auto tp0 = std::chrono::high_resolution_clock::now();
    std::tie(tree, counts) = computeOctree(firstCode, lastCode, bucketSize);
    auto tp1  = std::chrono::high_resolution_clock::now();

    double t0 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "build time from scratch " << t0 << " nNodes(tree): " << nNodes(tree)
              << " count: " << std::accumulate(begin(counts), end(counts), 0lu) << std::endl;

    tp0  = std::chrono::high_resolution_clock::now();
    updateOctree(firstCode, lastCode, bucketSize, tree, counts, std::numeric_limits<unsigned>::max());
    tp1  = std::chrono::high_resolution_clock::now();

    double t1 = std::chrono::duration<double>(tp1 - tp0).count();

    int nEmptyNodes = std::count(begin(counts), end(counts), 0);
    std::cout << "build time with guess " << t1 << " nNodes(tree): " << nNodes(tree)
              << " count: " << std::accumulate(begin(counts), end(counts), 0lu)
              << " empty nodes: " << nEmptyNodes << std::endl;

    return std::make_tuple(std::move(tree), std::move(counts));
}

template<class I>
void halo_discovery(Box<double> box, const std::vector<I>& tree, const std::vector<unsigned>& counts)
{
    int nSplits = 4;
    SpaceCurveAssignment<I> assignment = singleRangeSfcSplit(tree, counts, nSplits);
    std::vector<float> haloRadii(nNodes(tree), 0.01);

    std::vector<pair<int>> haloPairs;
    int doSplit = 0;
    auto tp0  = std::chrono::high_resolution_clock::now();
    findHalos(tree, haloRadii, box, assignment, doSplit, haloPairs);
    auto tp1  = std::chrono::high_resolution_clock::now();

    double t2 = std::chrono::duration<double>(tp1 - tp0).count();
    std::cout << "halo discovery: " << t2 << " nPairs: " << haloPairs.size() << std::endl;
}


int main()
{
    using CodeType = uint64_t;
    Box<double> box{-1, 1};

    int nParticles = 2000000;
    int bucketSize = 16;

    RandomGaussianCoordinates<double, CodeType> randomBox(nParticles, box);

    std::vector<CodeType> tree;
    std::vector<unsigned> counts;

    // tree build from random gaussian coordinates
    std::tie(tree, counts) = build_tree(randomBox.mortonCodes().data(), randomBox.mortonCodes().data() + nParticles, bucketSize);
    // halo discovery with tree
    halo_discovery(box, tree, counts);

    auto px = plummer<double>(nParticles);
    std::vector<CodeType> pxCodes(nParticles);
    Box<double> pBox(*std::min_element(begin(px[0]), end(px[0])),
                     *std::max_element(begin(px[0]), end(px[0])),
                     *std::min_element(begin(px[1]), end(px[1])),
                     *std::max_element(begin(px[1]), end(px[1])),
                     *std::min_element(begin(px[2]), end(px[2])),
                     *std::max_element(begin(px[2]), end(px[2]))
                     );

    std::cout << "plummer box: " << pBox.xmin() << " " << pBox.xmax() << " "
                                 << pBox.ymin() << " " << pBox.ymax() << " "
                                 << pBox.zmin() << " " << pBox.zmax() << std::endl;

    computeMortonCodes(begin(px[0]), end(px[0]), begin(px[1]), begin(px[2]), begin(pxCodes), pBox);
    std::sort(begin(pxCodes), end(pxCodes));

    std::tie(tree, counts) = build_tree(pxCodes.data(), pxCodes.data() + nParticles, bucketSize);
}
