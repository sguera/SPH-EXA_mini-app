#pragma once

#if defined(USE_CUDA)
#include <vector>

#include "Task.hpp"
#include "LinearOctree.hpp"
#include "cudaParticlesData.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{
namespace kernels
{
template <typename T>
__global__ void findNeighbors(const DeviceLinearOctree<T> o, const int *clist, const int n, const T *x, const T *y, const T *z, const T *h, const T displx,
                              const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount);
} // namespace kernels

template <typename T, class Dataset>
extern void computeFindNeighbors2(const LinearOctree<T> &o, std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeDensity(std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeMomentumAndEnergy(const std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeIAD(const std::vector<Task> &taskList, Dataset &d);

template <typename T, class Dataset>
extern void computeMomentumAndEnergyIAD(const std::vector<Task> &taskList, Dataset &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa

#endif
