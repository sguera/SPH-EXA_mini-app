#include <cuda.h>
#include <algorithm>
#include <omp.h>

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

    const auto largestChunkSize = std::max_element(taskList.cbegin(), taskList.cend(),
                                                   [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
                                      ->clist.size();

    const size_t size_largerNeighborsChunk_int = largestChunkSize * Task::ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);
    const size_t size_bbox = sizeof(BBox<T>);

    // device pointers - d_ prefix stands for device
    int ts = taskList.size();
    int *d_clist[ts], *d_neighbors[ts], *d_neighborsCount[ts];
    T *d_x, *d_y, *d_z, *d_m, *d_h;
    T *d_ro;
    BBox<T> *d_bbox;

    // input data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m, d_ro));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d_bbox));
    // CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist, d_neighborsCount));
    // CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    const int nStreams = taskList.size();
    cudaStream_t streams[nStreams];
    int i = 0;
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    // #pragma omp parallel
    // #pragma omp single
    for (const auto &t : taskList)
    {
        // #pragma omp task
        {
            const size_t n = t.clist.size();
            const size_t size_n_int = n * sizeof(int);
            const size_t size_nNeighbors = n * Task::ngmax * sizeof(int);
            CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist[i], d_neighborsCount[i]));
            CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors[i]));

            // CHECK_CUDA_ERR(cudaMemcpy(d_clist, t.clist.data(), size_n_int, cudaMemcpyHostToDevice));
            // CHECK_CUDA_ERR(cudaMemcpy(d_neighbors, t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice));
            // CHECK_CUDA_ERR(cudaMemcpy(d_neighborsCount, t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice));

            // if (i != 0) cudaStreamSynchronize(streams[i - 1]);

            CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist[i], t.clist.data(), size_n_int, cudaMemcpyHostToDevice, streams[i]));
            CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighbors[i], t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice, streams[i]));
            CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighborsCount[i], t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice, streams[i]));

            const int threadsPerBlock = 256;
            const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

            // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

            // kernels::density<<<blocksPerGrid, threadsPerBlock>>>(n, d.sincIndex, d.K, t.ngmax, d_bbox, d_clist, d_neighbors,
            //                                                      d_neighborsCount, d_x, d_y, d_z, d_h, d_m, d_ro);
            kernels::density<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
                n, d.sincIndex, d.K, t.ngmax, d_bbox, d_clist[i], d_neighbors[i], d_neighborsCount[i], d_x, d_y, d_z, d_h, d_m, d_ro);
            CHECK_CUDA_ERR(cudaGetLastError());

            ++i;
        }
    }
    // #pragma omp taskwait

    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamSynchronize(streams[i]);
        CHECK_CUDA_ERR(utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
    }

    CHECK_CUDA_ERR(cudaMemcpy(d.ro.data(), d_ro, size_np_T, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(utils::cudaFree(d_x, d_y, d_z, d_h, d_m, d_bbox, d_ro));
}

template <typename T, class ParticleData>
void computeDensity(const Task &task, ParticleData &d)
{
    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);

    const auto largestChunkSize = task.clist.size();
    // std::max_element(taskList.cbegin(), taskList.cend(),
    //                                                [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
    //                                   ->clist.size();

    const size_t size_largerNeighborsChunk_int = largestChunkSize * Task::ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);
    const size_t size_bbox = sizeof(BBox<T>);

    // device pointers - d_ prefix stands for device
    int *d_clist, *d_neighbors, *d_neighborsCount;
    // T *d_x, *d_y, *d_z, *d_m, *d_h;
    // T *d_ro;
    // BBox<T> *d_bbox;

    // input data
    // CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m, d_ro));
    // CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d_bbox));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist, d_neighborsCount));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors));

    // CHECK_CUDA_ERR(cudaMemcpyAsync(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpyAsync(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpyAsync(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpyAsync(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpyAsync(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpyAsync(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    // const int nStreams = taskList.size();
    // cudaStream_t streams[nStreams];
    // int i = 0;
    // for (int i = 0; i < nStreams; i++)
    //     cudaStreamCreate(&streams[i]);

    const size_t n = task.clist.size();
    const size_t size_n_int = n * sizeof(int);
    const size_t size_nNeighbors = n * Task::ngmax * sizeof(int);

    // CHECK_CUDA_ERR(cudaMemcpy(d_clist, task.clist.data(), size_n_int, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_neighbors, task.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_neighborsCount, task.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist, task.clist.data(), size_n_int, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighbors, task.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighborsCount, task.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    kernels::density<<<blocksPerGrid, threadsPerBlock>>>(n, d.sincIndex, d.K, task.ngmax, d.d_bbox, d_clist, d_neighbors, d_neighborsCount,
                                                         d.d_x, d.d_y, d.d_z, d.d_h, d.d_m, d.d_ro);
    CHECK_CUDA_ERR(cudaGetLastError());

    // CHECK_CUDA_ERR(cudaMemcpyAsync(d.ro.data() + task.clist.front(), d_ro + task.clist.front(), task.clist.size() * sizeof(T),
    // cudaMemcpyDeviceToHost));

    // CHECK_CUDA_ERR(utils::cudaFree(d_clist, d_neighbors, d_neighborsCount, d_x, d_y, d_z, d_h, d_m, d_bbox, d_ro));
    CHECK_CUDA_ERR(utils::cudaFree(d_clist, d_neighbors, d_neighborsCount));
}

template void computeDensity<double, ParticlesData<double>>(const std::vector<Task> &taskList, ParticlesData<double> &d);
template void computeDensity<double, ParticlesData<double>>(const Task &task, ParticlesData<double> &d);

template void copyInDensity<double, ParticlesData<double>>(ParticlesData<double> &d);
template void copyOutDensity<double, ParticlesData<double>>(ParticlesData<double> &d);
template <typename T, class Dataset>
void copyInDensity(Dataset &d)

{
    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);
    const size_t size_bbox = sizeof(BBox<T>);

    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d.d_x, d.d_y, d.d_z, d.d_h, d.d_m, d.d_ro));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d.d_bbox));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d.d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d.d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d.d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d.d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d.d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d.d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));
}

template <typename T, class Dataset>
void copyOutDensity(Dataset &d)
{
    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);
    // CHECK_CUDA_ERR(cudaMemcpyAsync(d.ro.data() + task.clist.front(), d.d_ro + task.clist.front(), task.clist.size() * sizeof(T),
    // cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpyAsync(d.ro.data(), d.d_ro, size_np_T, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(utils::cudaFree(d.d_x, d.d_y, d.d_z, d.d_h, d.d_m, d.d_bbox, d.d_ro));
}

#include <cstdlib>
#include <new>
#include <limits>
template <class T>
struct CudaAllocator
{
    typedef T value_type;
    CudaAllocator() = default;
    template <class U>
    constexpr CudaAllocator(const CudaAllocator<U> &) noexcept {}[[nodiscard]] T *allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
        if (auto p = static_cast<T *>(std::malloc(n * sizeof(T)))) return p;
        throw std::bad_alloc();
    }
    void deallocate(T *p, std::size_t) noexcept { std::free(p); }
};
template <class T, class U>
bool operator==(const CudaAllocator<T> &, const CudaAllocator<U> &)
{
    return true;
}
template <class T, class U>
bool operator!=(const CudaAllocator<T> &, const CudaAllocator<U> &)
{
    return false;
}

} // namespace cuda
} // namespace sph
} // namespace sphexa
