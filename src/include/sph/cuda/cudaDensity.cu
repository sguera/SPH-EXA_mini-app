#include "../kernels.hpp"
#include "sph.cuh"
#include "utils.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{
template void computeDensity<double, SqPatch<double>>(const std::vector<int> &clist, SqPatch<double> &d);

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
        const T value = wharmonic(vloc, h[i], sincIndex, K);
        roloc += value * m[j];
    }

    ro[tid] = roloc + m[i] * K / (h[i] * h[i] * h[i]);
}

template <typename T, class Dataset>
void computeDensity(const std::vector<int> &clist, Dataset &d)
{
    const size_t n = clist.size();
    const size_t np = d.x.size();
    const size_t n_chunk = n / d.noOfGpuLoopSplits;
    const size_t n_lastChunk = n / d.noOfGpuLoopSplits + n % d.noOfGpuLoopSplits; // in case n is not dividable by noOfGpuLoopSplits
    const size_t allNeighbors_chunk = n_chunk * d.ngmax;
    const size_t allNeighbors_lastChunk = n_lastChunk * d.ngmax;

    const size_t size_np_T = np * sizeof(T);
    const size_t size_bbox = sizeof(BBox<T>);
    const size_t size_allNeighbors_chunk = allNeighbors_chunk * sizeof(int);
    const size_t size_allNeighbors_lastChunk = allNeighbors_lastChunk * sizeof(int);
    const size_t size_n_T_chunk = n_chunk * sizeof(T);
    const size_t size_n_T_lastChunk = n_lastChunk * sizeof(T);
    const size_t size_n_int_chunk = n_chunk * sizeof(int);
    const size_t size_n_int_lastChunk = n_lastChunk * sizeof(int);

    int *d_clist, *d_neighbors, *d_neighborsCount;
    T *d_x, *d_y, *d_z, *d_m, *d_h;
    T *d_ro;
    BBox<T> *d_bbox;

    // input data
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_clist, size_n_int_lastChunk));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_neighbors, size_allNeighbors_lastChunk));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_neighborsCount, size_n_int_lastChunk));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_x, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_y, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_z, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_h, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_m, size_np_T));
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_bbox, size_bbox));

    // output data
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_ro, size_n_T_lastChunk));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    for (ushort s = 0; s < d.noOfGpuLoopSplits; ++s)
    {
        const int threadsPerBlock = 256;

        if (s == d.noOfGpuLoopSplits - 1)
        {
            CHECK_CUDA_ERR(cudaMemcpy(d_clist, clist.data() + (s * n_chunk), size_n_int_lastChunk, cudaMemcpyHostToDevice));
            CHECK_CUDA_ERR(cudaMemcpy(d_neighbors, d.neighbors.data() + (s * allNeighbors_chunk), size_allNeighbors_lastChunk,
                                      cudaMemcpyHostToDevice));
            CHECK_CUDA_ERR(
                cudaMemcpy(d_neighborsCount, d.neighborsCount.data() + (s * n_chunk), size_n_int_lastChunk, cudaMemcpyHostToDevice));

            const int blocksPerGrid = (n_lastChunk + threadsPerBlock - 1) / threadsPerBlock;

            // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

            density<<<blocksPerGrid, threadsPerBlock>>>(n_lastChunk, d.sincIndex, d.K, d.ngmax, d_bbox, d_clist, d_neighbors,
                                                        d_neighborsCount, d_x, d_y, d_z, d_h, d_m, d_ro);
            CHECK_CUDA_ERR(cudaGetLastError());

            CHECK_CUDA_ERR(cudaMemcpy(d.ro.data() + (s * n_chunk), d_ro, size_n_T_lastChunk, cudaMemcpyDeviceToHost));
        }
        else
        {
            CHECK_CUDA_ERR(cudaMemcpy(d_clist, clist.data() + (s * n_chunk), size_n_int_chunk, cudaMemcpyHostToDevice));
            CHECK_CUDA_ERR(
                cudaMemcpy(d_neighbors, d.neighbors.data() + (s * allNeighbors_chunk), size_allNeighbors_chunk, cudaMemcpyHostToDevice));
            CHECK_CUDA_ERR(cudaMemcpy(d_neighborsCount, d.neighborsCount.data() + (s * n_chunk), size_n_int_chunk, cudaMemcpyHostToDevice));

            const int blocksPerGrid = (n_chunk + threadsPerBlock - 1) / threadsPerBlock;

            // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

            density<<<blocksPerGrid, threadsPerBlock>>>(n_chunk, d.sincIndex, d.K, d.ngmax, d_bbox, d_clist, d_neighbors, d_neighborsCount,
                                                        d_x, d_y, d_z, d_h, d_m, d_ro);
            CHECK_CUDA_ERR(cudaGetLastError());

            CHECK_CUDA_ERR(cudaMemcpy(d.ro.data() + (s * n_chunk), d_ro, size_n_T_chunk, cudaMemcpyDeviceToHost));
        }
    }

    CHECK_CUDA_ERR(cudaFree(d_clist));
    CHECK_CUDA_ERR(cudaFree(d_neighbors));
    CHECK_CUDA_ERR(cudaFree(d_neighborsCount));
    CHECK_CUDA_ERR(cudaFree(d_x));
    CHECK_CUDA_ERR(cudaFree(d_y));
    CHECK_CUDA_ERR(cudaFree(d_z));
    CHECK_CUDA_ERR(cudaFree(d_h));
    CHECK_CUDA_ERR(cudaFree(d_m));
    CHECK_CUDA_ERR(cudaFree(d_bbox));
    CHECK_CUDA_ERR(cudaFree(d_ro));
}

} // namespace cuda
} // namespace sph
} // namespace sphexa
