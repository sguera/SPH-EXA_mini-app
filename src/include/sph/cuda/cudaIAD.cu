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
__global__ void computeIAD(const int n, const T sincIndex, const T K, const int ngmax, const BBox<T> *bbox, const int *clist,
                           const int *neighbors, const int *neighborsCount, const T *x, const T *y, const T *z, const T *h, const T *m,
                           const T *ro, T *c11, T *c12, T *c13, T *c22, T *c23, T *c33)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    const int i = clist[tid];
    const int nn = neighborsCount[tid];

    T tau11 = 0.0, tau12 = 0.0, tau13 = 0.0, tau22 = 0.0, tau23 = 0.0, tau33 = 0.0;
    for (int pj = 0; pj < nn; ++pj)
    {
        const int j = neighbors[tid * ngmax + pj];

        const T dist = distancePBC(*bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]);
        const T vloc = dist / h[i];

        const T w = K * math_namespace::pow(wharmonic(vloc), (int)sincIndex);
        const T W = w / (h[i] * h[i] * h[i]);

        T r_ijx = (x[i] - x[j]);
        T r_ijy = (y[i] - y[j]);
        T r_ijz = (z[i] - z[j]);

        applyPBC(*bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);

        tau11 += r_ijx * r_ijx * m[j] / ro[j] * W;
        tau12 += r_ijx * r_ijy * m[j] / ro[j] * W;
        tau13 += r_ijx * r_ijz * m[j] / ro[j] * W;
        tau22 += r_ijy * r_ijy * m[j] / ro[j] * W;
        tau23 += r_ijy * r_ijz * m[j] / ro[j] * W;
        tau33 += r_ijz * r_ijz * m[j] / ro[j] * W;
    }

    const T det =
        tau11 * tau22 * tau33 + 2.0 * tau12 * tau23 * tau13 - tau11 * tau23 * tau23 - tau22 * tau13 * tau13 - tau33 * tau12 * tau12;

    c11[tid] = (tau22 * tau33 - tau23 * tau23) / det;
    c12[tid] = (tau13 * tau23 - tau33 * tau12) / det;
    c13[tid] = (tau12 * tau23 - tau22 * tau13) / det;
    c22[tid] = (tau11 * tau33 - tau13 * tau13) / det;
    c23[tid] = (tau13 * tau12 - tau11 * tau23) / det;
    c33[tid] = (tau11 * tau22 - tau12 * tau12) / det;
}
} // namespace kernels

template void computeIAD<double, ParticlesData<double>>(const std::vector<ParticleIdxChunk> &chunksToCompute, ParticlesData<double> &d);

template <typename T, class Dataset>
void computeIAD(const std::vector<ParticleIdxChunk> &chunksToCompute, Dataset &d)
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
    T *d_x, *d_y, *d_z, *d_m, *d_h, *d_ro;
    BBox<T> *d_bbox;
    T *d_c11, *d_c12, *d_c13, *d_c22, *d_c23, *d_c33;

    // input data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m, d_ro));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d_bbox));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist, d_neighborsCount));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors));

    // output data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_T, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_ro, d.ro.data(), size_np_T, cudaMemcpyHostToDevice));
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

        // printf("CUDA IAD kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        kernels::computeIAD<<<blocksPerGrid, threadsPerBlock>>>(n, d.sincIndex, d.K, d.ngmax, d_bbox, d_clist, d_neighbors,
                                                                d_neighborsCount, d_x, d_y, d_z, d_h, d_m, d_ro, d_c11, d_c12, d_c13, d_c22,
                                                                d_c23, d_c33);
        CHECK_CUDA_ERR(cudaGetLastError());

        const auto outputOffset = clist.front();

        CHECK_CUDA_ERR(cudaMemcpy(d.c11.data() + outputOffset, d_c11, size_n_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c12.data() + outputOffset, d_c12, size_n_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c13.data() + outputOffset, d_c13, size_n_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c22.data() + outputOffset, d_c22, size_n_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c23.data() + outputOffset, d_c23, size_n_T, cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERR(cudaMemcpy(d.c33.data() + outputOffset, d_c33, size_n_T, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA_ERR(utils::cudaFree(d_bbox, d_clist, d_neighbors, d_neighborsCount, d_x, d_y, d_z, d_h, d_m, d_ro, d_c11, d_c12, d_c13,
                                   d_c22, d_c23, d_c33));
}

} // namespace cuda
} // namespace sph
} // namespace sphexa
