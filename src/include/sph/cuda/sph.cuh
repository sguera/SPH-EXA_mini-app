#pragma once

#include <vector>

#include "SqPatch.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{
template <typename T, class Dataset>
extern void computeMomentumAndEnergy(const std::vector<int> &l, Dataset &dataset);

template <typename T, class Dataset>
extern void computeDensity(const std::vector<int> &l, Dataset &dataset);

template <typename T, class Dataset>
extern void computeIAD(const std::vector<int> &l, Dataset &dataset);

template <typename T, class Dataset>
extern void computeMomentumAndEnergyIAD(const std::vector<int> &l, Dataset &dataset);
}
} // namespace sph
} // namespace sphexa
