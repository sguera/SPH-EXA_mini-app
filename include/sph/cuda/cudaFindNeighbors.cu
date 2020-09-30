#include <vector>
#include "Task.hpp"
#include "LinearOctree.hpp"
#include "ParticlesData.hpp"
#include "cudaUtils.cuh"

namespace sphexa
{
namespace sph
{
namespace cuda
{

namespace kernels
{

template <typename T>
__device__ T normalize(T d, T min, T max) { return (d - min) / (max - min); }

template <typename T>
__device__ void findNeighborsDispl(const DeviceLinearOctree<T> o, const int pi, const int *clist, const T *x, const T *y, const T *z,  const T *h, const T displx, const T disply, const T displz, const int ngmax,
                      int *neighbors, int *neighborsCount)
{
    const int i = clist[pi];

    // 64 is not enough... Depends on the bucket size and h...
    // This can be created and stored on the GPU directly.
    // For a fixed problem and size, if it works then it will always work
    int collisionsCount = 0;
    int collisionNodes[128];

    const T xi = x[i] + displx;
    const T yi = y[i] + disply;
    const T zi = z[i] + displz;
    const T ri = 2.0 * h[i];

    constexpr int nX = 2;
    constexpr int nY = 2;
    constexpr int nZ = 2;

    int stack[64];
    int stackptr = 0;
    stack[stackptr++] = -1;

    int node = 0;

    do
    {
        if(o.ncells[node] == 8)
        {
            int mix = std::max((int)(normalize(xi - ri, o.xmin[node], o.xmax[node]) * nX), 0);
            int miy = std::max((int)(normalize(yi - ri, o.ymin[node], o.ymax[node]) * nY), 0);
            int miz = std::max((int)(normalize(zi - ri, o.zmin[node], o.zmax[node]) * nZ), 0);
            int max = std::min((int)(normalize(xi + ri, o.xmin[node], o.xmax[node]) * nX), nX - 1);
            int may = std::min((int)(normalize(yi + ri, o.ymin[node], o.ymax[node]) * nY), nY - 1);
            int maz = std::min((int)(normalize(zi + ri, o.zmin[node], o.zmax[node]) * nZ), nZ - 1);
           
            // Maximize threads sync
            for (int hz = 0; hz < 2; hz++)
            {
                for (int hy = 0; hy < 2; hy++)
                {
                    for (int hx = 0; hx < 2; hx++)
                    {
                        // if overlap
                        if(hz >= miz && hz <= maz && hy >= miy && hy <= may && hx >= mix && hx <= max)
                        {
                            // int l = hz * nX * nY + hy * nX + hx;
                            // stack[stackptr++] = o.cells[node * 8 +l];
                            const int l = hz * nX * nY + hy * nX + hx;
                            const int child = o.cells[node * 8 + l];
                            if(o.localParticleCount[child] > 0)
                                stack[stackptr++] = child;
                        }
                    }
                }
            }
        }

        if(o.ncells[node] != 8)
            collisionNodes[collisionsCount++] = node;
        
        node = stack[--stackptr]; // Pop next
    }
    while(node > 0);

    //__syncthreads();

    int ngc = neighborsCount[pi];
    
    for(int ni=0; ni<collisionsCount; ni++)
    {
        int node = collisionNodes[ni];
        T r2 = ri * ri;

        for (int pj = 0; pj < o.localParticleCount[node]; pj++)
        {
            int j = o.localPadding[node] + pj;

            T xj = x[j];
            T yj = y[j];
            T zj = z[j];

            T xx = xi - xj;
            T yy = yi - yj;
            T zz = zi - zj;

            T dist = xx * xx + yy * yy + zz * zz;

            if (dist < r2 && i != j && ngc < ngmax)
                neighbors[ngc++] = j;
        }
    }

    neighborsCount[pi] = ngc;

    //__syncthreads();
}

template <typename T>
__global__ void findNeighbors(const DeviceLinearOctree<T> o, const int *clist, const int n, const T *x, const T *y, const T *z, const T *h, const T displx,
                              const T disply, const T displz, const int max, const int may, const int maz, const int ngmax, int *neighbors, int *neighborsCount)
{
    const int pi = blockDim.x * blockIdx.x + threadIdx.x;
    if (pi >= n) return;

    T dispx[3], dispy[3], dispz[3];

    dispx[0] = 0;       dispy[0] = 0;       dispz[0] = 0;
    dispx[1] = -displx; dispy[1] = -disply; dispz[1] = -displz;
    dispx[2] = displx;  dispy[2] = disply;  dispz[2] = displz;

    neighborsCount[pi] = 0;

    for (int hz = 0; hz <= maz; hz++)
        for (int hy = 0; hy <= may; hy++)
            for (int hx = 0; hx <= max; hx++)
                findNeighborsDispl(o, pi, clist, x, y, z, h, dispx[hx], dispy[hy], dispz[hz], ngmax, &neighbors[pi*ngmax], neighborsCount);
}
}

template <typename T, class Dataset>
void computeFindNeighbors2(const LinearOctree<T> &o, std::vector<Task> &taskList, Dataset &d)
{
    const T *h = d.h.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();

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

    const int NST = 2;

    // Device pointers
    int *d_clist[NST], *d_neighbors[NST], *d_neighborsCount[NST]; // work arrays per stream
    T *d_x, *d_y, *d_z, *d_h;

    CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNChunk_int, d_clist[i], d_neighborsCount[i]));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaMalloc(size_largerNeighborsChunk_int, d_neighbors[i]));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));

    DeviceLinearOctree<T> d_o;
    d_o.mapLinearOctreeToDevice(o);

    cudaStream_t streams[NST];
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamCreate(&streams[i]));

    for (int i = 0; i < taskList.size(); ++i)
    {
        auto &t = taskList[i];

        const int sIdx = i % NST;
        cudaStream_t stream = streams[sIdx];

        int *d_clist_use = d_clist[sIdx];
        int *d_neighbors_use = d_neighbors[sIdx];
        int *d_neighborsCount_use = d_neighborsCount[sIdx];

        const size_t n = t.clist.size();
        const size_t size_n_int = n * sizeof(int);
        const size_t size_nNeighbors = n * ngmax * sizeof(int);

        //CHECK_CUDA_ERR(cudaMemcpy(d_clist, t.clist.data(), size_n_int, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpyAsync(d_clist_use, t.clist.data(), size_n_int, cudaMemcpyHostToDevice, stream));

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // printf("CUDA Density kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

        kernels::findNeighbors<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            d_o, d_clist_use, n, d_x, d_y, d_z, d_h, displx, disply, displz, max, may, maz, ngmax, d_neighbors_use, d_neighborsCount_use
        );

        CHECK_CUDA_ERR(cudaMemcpyAsync(t.neighbors.data(), d_neighbors_use, size_nNeighbors, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERR(cudaMemcpyAsync(t.neighborsCount.data(), d_neighborsCount_use, size_n_int, cudaMemcpyDeviceToHost, stream));
    }

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamSynchronize(streams[i]));

    d_o.unmapLinearOctreeFromDevice();

    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(cudaStreamDestroy(streams[i]));

    CHECK_CUDA_ERR(utils::cudaFree(d_x, d_y, d_z, d_h));
    for (int i = 0; i < NST; ++i)
        CHECK_CUDA_ERR(utils::cudaFree(d_clist[i], d_neighbors[i], d_neighborsCount[i]));
}

template void computeFindNeighbors2<double, ParticlesData<double>>(const LinearOctree<double> &o, std::vector<Task> &taskList, ParticlesData<double> &d);

} // namespace cuda
} // namespace sph
} // namespace sphexa
