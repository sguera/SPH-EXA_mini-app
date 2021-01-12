#include <cuda.h>
#include <algorithm>

#include "sph.cuh"
#include "BBox.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"
#include "../kernels.hpp"
#include "../lookupTables.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{
namespace kernels
{
template <typename T>
__global__ void density(const int n, const T sincIndex, const T K, const int ngmax, const BBox<T> *bbox, const int *clist,
                        const int *neighbors, const int *neighborsCount, const T *x, const T *y, const T *z, const T *h, const T *m, T *ro)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    const int i = clist[tid];
    const int nn = neighborsCount[tid];

    T roloc = 0.0;

    for (int pj = 0; pj < nn; ++pj)
    {
        const int j = neighbors[tid * ngmax + pj];
        const T dist = distancePBC(*bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]);
        const T vloc = dist / h[i];
        const T w = K * math_namespace::pow(wharmonic(vloc), (int)sincIndex);
        const T value = w / (h[i] * h[i] * h[i]);
        roloc += value * m[j];
    }

    ro[i] = roloc + m[i] * K / (h[i] * h[i] * h[i]);
}
} // namespace kernels

template <typename T, class ParticleData>
void computeDensity(const std::vector<Task> &taskList, ParticleData &d)
{
    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);
    const T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    const auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    const size_t size_largerNeighborsChunk_int = largestChunkSize * ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);
    const size_t size_bbox = sizeof(BBox<T>);

    // device pointers - d_ prefix stands for device
    int *d_clist, *d_neighbors, *d_neighborsCount;
    T *d_x, *d_y, *d_z, *d_m, *d_h;
    T *d_ro;
    BBox<T> *d_bbox;

    // input data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m, d_ro));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d_bbox));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist, d_neighborsCount));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    for (const auto &t : taskList)
    {
        const size_t n = t.clist.size();
        const size_t size_n_int = n * sizeof(int);
        const size_t size_nNeighbors = n * ngmax * sizeof(int);

        CHECK_CUDA_ERR(cudaMemcpy(d_clist, t.clist.data(), size_n_int, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_neighbors, t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_neighborsCount, t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice));

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        kernels::density<<<blocksPerGrid, threadsPerBlock>>>(n, d.sincIndex, d.K, t.ngmax, d_bbox, d_clist, d_neighbors, d_neighborsCount,
                                                             d_x, d_y, d_z, d_h, d_m, d_ro);
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    CHECK_CUDA_ERR(cudaMemcpy(d.ro.data(), d_ro, size_np_T, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(utils::cudaFree(d_clist, d_neighbors, d_neighborsCount, d_x, d_y, d_z, d_h, d_m, d_bbox, d_ro));
}

template void computeDensity<double, ParticlesData<double>>(const std::vector<Task> &taskList, ParticlesData<double> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
