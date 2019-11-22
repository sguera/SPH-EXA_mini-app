#include <cuda.h>
#include <algorithm>
#include <omp.h>
#include <numeric>

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
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    int i = 0;

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

            // CHECK_CUDA_ERR(cudaMemcpy(d_clist[i], t.clist.data(), size_n_int, cudaMemcpyHostToDevice));
            // CHECK_CUDA_ERR(cudaMemcpy(d_neighbors[i], t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice));
            // CHECK_CUDA_ERR(cudaMemcpy(d_neighborsCount[i], t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice));

            // if (i != 0) cudaStreamSynchronize(streams[i - 1]);

            CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist[i], t.clist.data(), size_n_int, cudaMemcpyHostToDevice, streams[i]));
            CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighbors[i], t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice, streams[i]));
            CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighborsCount[i], t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice, streams[i]));

            const int threadsPerBlock = 256;
            const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

            // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

            // kernels::density<<<blocksPerGrid, threadsPerBlock>>>(n, d.sincIndex, d.K, t.ngmax, d_bbox, d_clist[i], d_neighbors[i],
            //                                                      d_neighborsCount[i], d_x, d_y, d_z, d_h, d_m, d_ro);
            kernels::density<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
                n, d.sincIndex, d.K, t.ngmax, d_bbox, d_clist[i], d_neighbors[i], d_neighborsCount[i], d_x, d_y, d_z, d_h, d_m, d_ro);
            CHECK_CUDA_ERR(cudaGetLastError());

            const size_t offset = t.clist.front();
            // printf("CUDA Density offset: %lu\n", offset);

            // CHECK_CUDA_ERR(cudaMemcpyAsync(d.ro.data() + offset, d_ro + offset, t.clist.size() * sizeof(T), cudaMemcpyDeviceToHost,
            // streams[i])); // It works but prevents memory transfer overlapping, pinned memory needed probably
            ++i;
        }
    }
    // #pragma omp taskwait

    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        CHECK_CUDA_ERR(utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
    }

    // const size_t tSize = offset; // TODO THIS IS DUMB RANDOM VARIABLE
    // CHECK_CUDA_ERR(cudaMemcpy(d.ro.data(), d_ro, size_np_T, cudaMemcpyDeviceToHost));
    const size_t beg = taskList.front().clist.front();
    const size_t size =
        std::accumulate(taskList.begin(), taskList.end(), 0, [](int total, const Task &t) { return total + t.clist.size(); });
    CHECK_CUDA_ERR(cudaMemcpy(d.ro.data() + beg, d_ro + beg, size * sizeof(T), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(utils::cudaFree(d_x, d_y, d_z, d_h, d_m, d_bbox, d_ro));
}

template <typename T, class ParticleData>
void computeDensity(std::vector<Task>::iterator tbegin, std::vector<Task>::iterator tend, ParticleData &d)
{
    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);

    int ts = std::distance(tbegin, tend);
    printf("TaskList has a size of %d\n", ts);

    const auto largestChunkSize = // tbegin->clist.size();
        std::max_element(tbegin, tend, [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })->clist.size();

    const size_t size_largerNeighborsChunk_int = largestChunkSize * Task::ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);
    const size_t size_bbox = sizeof(BBox<T>);

    // device pointers - d_ prefix stands for device

    int *d_clist[ts], *d_neighbors[ts], *d_neighborsCount[ts];
    T *d_x, *d_y, *d_z, *d_m, *d_h;
    T *d_ro;
    BBox<T> *d_bbox;

    // input data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m, d_ro));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d_bbox));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    const int nStreams = ts;
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    int i = 0;

    for (auto it = tbegin; it != tend; ++it)
    {

        const auto t = *it;
        const size_t n = t.clist.size();
        const size_t size_n_int = n * sizeof(int);
        const size_t size_nNeighbors = n * Task::ngmax * sizeof(int);
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist[i], d_neighborsCount[i]));
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors[i]));

        CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist[i], t.clist.data(), size_n_int, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighbors[i], t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighborsCount[i], t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice, streams[i]));

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        kernels::density<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
            n, d.sincIndex, d.K, t.ngmax, d_bbox, d_clist[i], d_neighbors[i], d_neighborsCount[i], d_x, d_y, d_z, d_h, d_m, d_ro);
        CHECK_CUDA_ERR(cudaGetLastError());

        const size_t offset = t.clist.front();

        // CHECK_CUDA_ERR(cudaMemcpyAsync(d.ro.data() + offset, d_ro + offset, t.clist.size() * sizeof(T), cudaMemcpyDeviceToHost,
        // streams[i])); // It works but prevents memory transfer overlapping, pinned memory needed probably
        ++i;
    }

    // #pragma omp taskwait

    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        CHECK_CUDA_ERR(utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
    }

    const size_t beg = tbegin->clist.front();
    const size_t size = std::accumulate(tbegin, tend, 0, [](int total, const Task &t) { return total + t.clist.size(); });
    CHECK_CUDA_ERR(cudaMemcpy(d.ro.data() + beg, d_ro + beg, size * sizeof(T), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(utils::cudaFree(d_x, d_y, d_z, d_h, d_m, d_bbox, d_ro));
}

template <typename T, class ParticleData>
void computeDensity(::sphexa::TaskQueue &taskQueue, ParticleData &d)
{
    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);

    const auto &tl = taskQueue.taskList;

    const auto largestChunkSize =
        std::max_element(tl.cbegin(), tl.cend(), [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    const size_t size_largerNeighborsChunk_int = largestChunkSize * Task::ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);
    const size_t size_bbox = sizeof(BBox<T>);

    // device pointers - d_ prefix stands for device
    int ts = tl.size();
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

    const int nStreams = 4;
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++)
        cudaStreamCreate(&streams[i]);
    int i = 0;
    int si = 0;

    for (int si = 0; si < nStreams; ++si)
    {
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist[si], d_neighborsCount[si]));
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors[si]));
    }

    for (int ti = 0; ti < taskQueue.size(); ++ti)
    {
        for (int si = 0; si < nStreams; ++si)
        {
            if (taskQueue.areAllProcessed()) break;

            const auto &t = taskQueue.pop();
            printf("CUDA: Taken task %d\n", taskQueue.lastProcessedTask - 1);

            const size_t n = t.clist.size();
            const size_t size_n_int = n * sizeof(int);
            const size_t size_nNeighbors = n * Task::ngmax * sizeof(int);

            CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist[si], t.clist.data(), size_n_int, cudaMemcpyHostToDevice, streams[si]));
            CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighbors[si], t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice, streams[si]));
            CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighborsCount[si], t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice, streams[si]));

            const int threadsPerBlock = 256;
            const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

            kernels::density<<<blocksPerGrid, threadsPerBlock, 0, streams[si]>>>(
                n, d.sincIndex, d.K, t.ngmax, d_bbox, d_clist[si], d_neighbors[si], d_neighborsCount[si], d_x, d_y, d_z, d_h, d_m, d_ro);
            CHECK_CUDA_ERR(cudaGetLastError());

            const size_t offset = t.clist.front();
            // printf("CUDA Density offset: %lu\n", offset);

            CHECK_CUDA_ERR(
                cudaMemcpyAsync(d.ro.data() + offset, d_ro + offset, t.clist.size() * sizeof(T), cudaMemcpyDeviceToHost,
                                streams[si])); // It works but prevents memory transfer overlapping, pinned memory needed probably
        }
        for (int i = 0; i < nStreams; i++)
        {
            cudaStreamSynchronize(streams[i]);
        }
    }

    for (int i = 0; i < nStreams; i++)
    {
        cudaStreamDestroy(streams[i]);
        CHECK_CUDA_ERR(utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
    }

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
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);

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
template void computeDensity<double, ParticlesData<double>>(TaskQueue &tq, ParticlesData<double> &d);
template void computeDensity<double, ParticlesData<double>>(std::vector<Task>::iterator tend, std::vector<Task>::iterator tbegin,
                                                            ParticlesData<double> &d);

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
    CHECK_CUDA_ERR(cudaMemcpy(d.ro.data(), d.d_ro, size_np_T, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(utils::cudaFree(d.d_x, d.d_y, d.d_z, d.d_h, d.d_m, d.d_bbox, d.d_ro));
}

#include <cstdlib>
#include <new>
#include <limits>

template <class T>
CudaAllocator<T>::CudaAllocator() = default;

template <class T>
// constexpr CudaAllocator<T>::CudaAllocator(const CudaAllocator<T> &)// noexcept {}[[nodiscard]] T *allocate(std::size_t n)
[[nodiscard]] T *CudaAllocator<T>::allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
    // if (auto p = static_cast<T *>(cudaMallocHost(n * sizeof(T)))) return p;
    T *ptr;
    CHECK_CUDA_ERR(cudaMallocHost(ptr, n * sizeof(T)));
    if (ptr) return ptr;

    throw std::bad_alloc();
} template <class T>
void CudaAllocator<T>::deallocate(T *p, std::size_t) noexcept
{
    std::free(p);
} // TODO CudaMallocHostFree

// template CudaAllocator<double>;
/*
template <class T, class U>
bool CudaAllocator<T>::operator==(const CudaAllocator<T> &, const CudaAllocator<U> &)
{
    return true;
}
template <class T, class U>
bool CudaAllocator<T>::operator!=(const CudaAllocator<T> &, const CudaAllocator<U> &)
{
    return false;
}
*/
} // namespace cuda
} // namespace sph
} // namespace sphexa
