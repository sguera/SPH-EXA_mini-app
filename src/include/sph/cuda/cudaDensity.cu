#include <cuda.h>
#include <algorithm>

#include "../kernels.hpp"
#include "sph.cuh"
#include "utils.cuh"

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

    ro[tid] = roloc + m[i] * K / (h[i] * h[i] * h[i]);
}
} // namespace kernels

template void computeDensity<double, Wharmonic<double>, SqPatch<double>>(const std::vector<ParticleIdxChunk> &chunksToCompute,
                                                                         const Wharmonic<double> &kernelFun, SqPatch<double> &d);

template <typename T, class KernelFun, class Dataset>
void computeDensity(const std::vector<ParticleIdxChunk> &chunksToCompute, const KernelFun &kernelFun, Dataset &d)
{
    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);

    const auto largestChunkSize =
        std::max_element(chunksToCompute.cbegin(), chunksToCompute.cend(),
                         [](const std::vector<int> &lhs, const std::vector<int> &rhs) { return lhs.size() < rhs.size(); })
            ->size();

    const size_t size_largerNeighborsChunk_int = largestChunkSize * d.ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);
    const size_t size_largerNChunk_T = largestChunkSize * sizeof(T);
    const size_t size_bbox = sizeof(BBox<T>);

    // device pointers - d_ prefix stands for device
    int *d_clist, *d_neighbors, *d_neighborsCount;
    T *d_x, *d_y, *d_z, *d_m, *d_h;
    T *d_ro;
    BBox<T> *d_bbox;

    // input data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d_bbox));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist, d_neighborsCount));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors));

    // output data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_T, d_ro));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    for (const auto &clist : chunksToCompute)
    {
        const size_t n = clist.size();
        const size_t size_n_T = n * sizeof(T);
        const size_t size_n_int = n * sizeof(int);
        const size_t size_nNeighbors = n * d.ngmax * sizeof(int);

        const size_t neighborsOffset = clist.front() * d.ngmax;
        const int *neighbors = d.neighbors.data() + neighborsOffset;

        const size_t neighborsCountOffset = clist.front();
        const int *neighborsCount = d.neighborsCount.data() + neighborsCountOffset;

        CHECK_CUDA_ERR(cudaMemcpy(d_clist, clist.data(), size_n_int, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_neighbors, neighbors, size_nNeighbors, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(d_neighborsCount, neighborsCount, size_n_int, cudaMemcpyHostToDevice));

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        kernels::density<<<blocksPerGrid, threadsPerBlock>>>(n, d.sincIndex, d.K, d.ngmax, d_bbox, d_clist, d_neighbors, d_neighborsCount,
                                                             d_x, d_y, d_z, d_h, d_m, d_ro);
        CHECK_CUDA_ERR(cudaGetLastError());

        CHECK_CUDA_ERR(cudaMemcpy(d.ro.data() + clist.front(), d_ro, size_n_T, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA_ERR(utils::cudaFree(d_clist, d_neighbors, d_neighborsCount, d_x, d_y, d_z, d_h, d_m, d_bbox, d_ro));
}

} // namespace cuda
} // namespace sph
} // namespace sphexa
