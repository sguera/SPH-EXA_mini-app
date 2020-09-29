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
                           const T *ro, const T *wh, const T *whd, const size_t ltsize, T *c11, T *c12, T *c13, T *c22, T *c23, T *c33)
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

        const T w = K * math_namespace::pow(lt::wharmonic_lt_with_derivative(wh, whd, ltsize, vloc), (int)sincIndex);
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

template <typename T>
__global__ void findNeighbors(const DeviceLinearOctree<T> o, const int *clist, const int n, const T *x, const T *y, const T *z, const T *h, const T displx,
                              const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount);

} // namespace kernels

template <typename T, class Dataset>
void computeIAD(const LinearOctree<T> &o, const std::vector<Task> &taskList, Dataset &d)
{
    const int maz = d.bbox.PBCz ? 2 : 0;
    const int may = d.bbox.PBCy ? 2 : 0;
    const int max = d.bbox.PBCx ? 2 : 0;
    
    const T displx = o.xmax[0] - o.xmin[0];
    const T disply = o.ymax[0] - o.ymin[0];
    const T displz = o.zmax[0] - o.zmin[0];

    const size_t np = d.x.size();
    const size_t size_np_T = np * sizeof(T);
    const T ngmax = taskList.empty() ? 0 : taskList.front().ngmax;

    const auto largestChunkSize =
        std::max_element(taskList.cbegin(), taskList.cend(),
                         [](const Task &lhs, const Task &rhs) { return lhs.clist.size() < rhs.clist.size(); })
            ->clist.size();

    const size_t size_largerNeighborsChunk_int = largestChunkSize * ngmax * sizeof(int);
    const size_t size_largerNChunk_int = largestChunkSize * sizeof(int);

    // number of CUDA streams to use
    const int NST = 2;

    // device pointers - d_ prefix stands for device
    int *d_clist[NST], *d_neighbors[NST], *d_neighborsCount[NST]; // work arrays per stream

    const size_t ltsize = d.wh.size();

    // input data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d.devicePtrs.d_c11, d.devicePtrs.d_c12, d.devicePtrs.d_c13, d.devicePtrs.d_c22, d.devicePtrs.d_c23, d.devicePtrs.d_c33));

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist[i], d_neighborsCount[i]));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors[i]));

    
    // CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d.devicePtrs.d_ro, d.ro.data(), size_np_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_wh, d.wh.data(), size_lt_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_whd, d.whd.data(), size_lt_T, cudaMemcpyHostToDevice));
    // CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));
    

    cudaStream_t streams[NST];
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamCreate(&streams[i]));

    //DeviceLinearOctree<T> d_o;
    //d.devicePtrs.d_o.mapLinearOctreeToDevice(o);
    
    for (int i = 0; i < taskList.size(); ++i)
    {
        const auto &t = taskList[i];

        const int sIdx = i % NST;
        cudaStream_t stream = streams[sIdx];

        int *d_clist_use = d_clist[sIdx];
        int *d_neighbors_use = d_neighbors[sIdx];
        int *d_neighborsCount_use = d_neighborsCount[sIdx];

        const size_t n = t.clist.size();
        const size_t size_n_int = n * sizeof(int);
        // const size_t size_nNeighbors = n * ngmax * sizeof(int);

        CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, cudaMemcpyHostToDevice, stream));
        //CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighbors_use, t.neighbors.data(), size_nNeighbors, cudaMemcpyHostToDevice, stream));
        //CHECK_CUDA_ERR(cudaMemcpyAsync(d_neighborsCount_use, t.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice, stream));

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        kernels::findNeighbors<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d.devicePtrs.d_o, d_clist_use, n, d.devicePtrs.d_x, d.devicePtrs.d_y, d.devicePtrs.d_z, d.devicePtrs.d_h, displx, disply, displz, max, may, maz, ngmax, d_neighbors_use, d_neighborsCount_use
        );

        // printf("CUDA IAD kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        kernels::computeIAD<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(n, d.sincIndex, d.K, ngmax, d.devicePtrs.d_bbox, d_clist_use, d_neighbors_use,
            d_neighborsCount_use, d.devicePtrs.d_x, d.devicePtrs.d_y, d.devicePtrs.d_z, d.devicePtrs.d_h, d.devicePtrs.d_m, d.devicePtrs.d_ro, d.devicePtrs.d_wh, d.devicePtrs.d_whd, ltsize, d.devicePtrs.d_c11, d.devicePtrs.d_c12, d.devicePtrs.d_c13, d.devicePtrs.d_c22,
            d.devicePtrs.d_c23, d.devicePtrs.d_c33);
        CHECK_CUDA_ERR(cudaGetLastError());
    }

    
    // d.d_o.unmapLinearOctreeFromDevice();

    CHECK_CUDA_ERR(cudaMemcpy(d.c11.data(), d.devicePtrs.d_c11, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c12.data(), d.devicePtrs.d_c12, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c13.data(), d.devicePtrs.d_c13, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c22.data(), d.devicePtrs.d_c22, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c23.data(), d.devicePtrs.d_c23, size_np_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.c33.data(), d.devicePtrs.d_c33, size_np_T, cudaMemcpyDeviceToHost));
    

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamDestroy(streams[i]));

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
}

template void computeIAD<double, ParticlesData<double>>(const LinearOctree<double> &o, const std::vector<Task> &taskList, ParticlesData<double> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
