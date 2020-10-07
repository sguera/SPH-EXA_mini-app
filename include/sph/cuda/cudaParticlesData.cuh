#pragma once

#include "LinearOctree.hpp"
#include "cudaUtils.cuh"
#include "BBox.hpp"

namespace sphexa
{
namespace sph
{
namespace cuda
{

template <typename T>
struct DeviceLinearOctree
{
    int size;
    int *ncells;
    int *cells;
    int *localPadding;
    int *localParticleCount;
    T *xmin, *xmax, *ymin, *ymax, *zmin, *zmax;
    T xmin0, xmax0, ymin0, ymax0, zmin0, zmax0;

    void mapLinearOctreeToDevice(const LinearOctree<T> &o)
    {
        size_t size_int = o.size * sizeof(int);
        size_t size_T = o.size * sizeof(T);

        size = o.size;

        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaMalloc(size_int * 8, cells));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaMalloc(size_int, ncells, localPadding, localParticleCount));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaMalloc(size_T, xmin, xmax, ymin, ymax, zmin, zmax));

        CHECK_CUDA_ERR(cudaMemcpy(cells, o.cells.data(), size_int * 8, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(ncells, o.ncells.data(), size_int, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(localPadding, o.localPadding.data(), size_int, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(localParticleCount, o.localParticleCount.data(), size_int, cudaMemcpyHostToDevice));

        CHECK_CUDA_ERR(cudaMemcpy(xmin, o.xmin.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(xmax, o.xmax.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(ymin, o.ymin.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(ymax, o.ymax.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(zmin, o.zmin.data(), size_T, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(zmax, o.zmax.data(), size_T, cudaMemcpyHostToDevice));

        xmin0 = o.xmin[0];
        xmax0 = o.xmax[0];
        ymin0 = o.ymin[0];
        ymax0 = o.ymax[0];
        zmin0 = o.zmin[0];
        zmax0 = o.zmax[0];
    }

    void unmapLinearOctreeFromDevice()
    {
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaFree(cells));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaFree(ncells, localPadding, localParticleCount));
        CHECK_CUDA_ERR(sphexa::sph::cuda::utils::cudaFree(xmin, xmax, ymin, ymax, zmin, zmax));
    }
};

template <typename T, class ParticleData>
struct DeviceParticlesData
{
    // number of CUDA streams to use
    static const int NST = 2;

    struct neighbors_stream {
        cudaStream_t streams;
        int *d_clist, *d_neighbors, *d_neighborsCount;
    };

    struct neighbors_stream d_stream[NST];

    T *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_m, *d_h, *d_ro, *d_p, *d_c, *d_c11, *d_c12, *d_c13, *d_c22, *d_c23, *d_c33, *d_wh, *d_whd;
    BBox<T> *d_bbox;
    T *d_grad_P_x, *d_grad_P_y, *d_grad_P_z, *d_du, *d_maxvsignal;

    DeviceLinearOctree<T> d_o;

    DeviceParticlesData(const ParticleData &pd)
    {
        printf("[DEBUG] -- ALLOCATING: d_x before %p\n", d_x);
        const size_t np = pd.x.size();
        const size_t size_np_T = np * sizeof(T);
        
        const size_t size_bbox = sizeof(BBox<T>);

        const size_t ltsize = pd.wh.size();
        const size_t size_lt_T = ltsize * sizeof(T);

        CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_h, d_m, d_ro));
        CHECK_CUDA_ERR(utils::cudaMalloc(size_lt_T, d_wh, d_whd));
        CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d_bbox));
        CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));
        CHECK_CUDA_ERR(utils::cudaMalloc(size_np_T, d_vx, d_vy, d_vz, d_p, d_c, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du, d_maxvsignal));
        
        printf("[DEBUG] -- ALLOCATING: d_x after %p\n", d_x);
    }

    ~DeviceParticlesData()
    {
        printf("[DEBUG] -- ALLOCATING: d_x before %p\n", d_x);
        CHECK_CUDA_ERR(utils::cudaFree(d_bbox, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_m, d_ro, d_p,
            d_c, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du, d_maxvsignal, d_wh, d_whd));
        printf("[DEBUG] -- ALLOCATING: d_x after %p\n", d_x);
    }
};
} // namespace cuda
} // namespace sph
} // namespace sphexa
