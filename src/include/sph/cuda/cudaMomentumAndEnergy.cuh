#pragma once

#include <vector>

#include "SqPatch.hpp"

namespace sphexa
{
namespace sph
{

template <typename T>
extern void cudaComputeMomentumAndEnergy(const std::vector<int> &l, SqPatch<T> &dataset);

} // namespace sph
} // namespace sphexa
