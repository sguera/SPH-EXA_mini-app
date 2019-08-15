#pragma once

#include <vector>

#include "SqPatch.hpp"

namespace sphexa
{
namespace sph
{

template <typename T, class Dataset>
extern void cudaComputeMomentumAndEnergy(const std::vector<int> &l, Dataset &dataset);

} // namespace sph
} // namespace sphexa
