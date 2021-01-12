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

    c11[i] = (tau22 * tau33 - tau23 * tau23) / det;
    c12[i] = (tau13 * tau23 - tau33 * tau12) / det;
    c13[i] = (tau12 * tau23 - tau22 * tau13) / det;
    c22[i] = (tau11 * tau33 - tau13 * tau13) / det;
    c23[i] = (tau13 * tau12 - tau11 * tau23) / det;
    c33[i] = (tau11 * tau22 - tau12 * tau12) / det;
}
} // namespace kernels

template <typename T, class Dataset>
void computeIAD(const std::vector<Task> &taskList, Dataset &d)
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
    T *d_x, *d_y, *d_z, *d_m, *d_h, *d_ro;
    BBox<T> *d_bbox;
    T *d_c11, *d_c12, *d_c13, *d_c22, *d_c23, *d_c33;

    // input data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m, d_ro, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d_bbox));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist, d_neighborsCount));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_ro, d.ro.data(), size_np_T, cudaMemcpyHostToDevice));
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

        // printf("CUDA IAD kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        kernels::computeIAD<<<blocksPerGrid, threadsPerBlock>>>(n, d.sincIndex, d.K, ngmax, d_bbox, d_clist, d_neighbors,
                                                                d_neighborsCount, d_x, d_y, d_z, d_h, d_m, d_ro, d_c11, d_c12, d_c13, d_c22,
                                                                d_c23, d_c33);
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    CHECK_CUDA_ERR(cudaMemcpy(d.c11.data(), d_c11, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c12.data(), d_c12, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c13.data(), d_c13, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c22.data(), d_c22, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c23.data(), d_c23, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c33.data(), d_c33, size_np_T, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(utils::cudaFree(d_bbox, d_clist, d_neighbors, d_neighborsCount, d_x, d_y, d_z, d_h, d_m, d_ro, d_c11, d_c12, d_c13,
                                   d_c22, d_c23, d_c33));
}

template void computeIAD<double, ParticlesData<double>>(const std::vector<Task> &taskList, ParticlesData<double> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
