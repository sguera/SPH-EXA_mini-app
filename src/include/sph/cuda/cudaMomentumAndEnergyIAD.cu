#include <cuda.h>

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
const double gradh_i = 1.0;
const double gradh_j = 1.0;

template <typename T>
__global__ void computeMomentumAndEnergyIAD(const int n, const T sincIndex, const T K, const int ngmax, const BBox<T> *bbox,
                                            const int *clist, const int *neighbors, const int *neighborsCount, const T *x, const T *y,
                                            const T *z, const T *vx, const T *vy, const T *vz, const T *h, const T *m, const T *ro,
                                            const T *p, const T *c, const T *c11, const T *c12, const T *c13, const T *c22, const T *c23,
                                            const T *c33, T *grad_P_x, T *grad_P_y, T *grad_P_z, T *du)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n) return;

    const int i = clist[tid];
    const int nn = neighborsCount[tid];

    T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0, energyAV = 0.0;
    for (int pj = 0; pj < nn; ++pj)
    {
        const int j = neighbors[tid * ngmax + pj];

        T r_ijx = (x[i] - x[j]);
        T r_ijy = (y[i] - y[j]);
        T r_ijz = (z[i] - z[j]);

        T r_jix = (x[j] - x[i]);
        T r_jiy = (y[j] - y[i]);
        T r_jiz = (z[j] - z[i]);

        applyPBC(*bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);
        applyPBC(*bbox, 2.0 * h[i], r_jix, r_jiy, r_jiz);

        const T dist = std::sqrt(r_ijx * r_ijx + r_ijy * r_ijy + r_ijz * r_ijz);

        const T v_ijx = (vx[i] - vx[j]);
        const T v_ijy = (vy[i] - vy[j]);
        const T v_ijz = (vz[i] - vz[j]);

        const T v1 = dist / h[i];
        const T v2 = dist / h[j];

        const T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

        const T w1 = K * math_namespace::pow(wharmonic(v1), (int)sincIndex);
        const T w2 = K * math_namespace::pow(wharmonic(v2), (int)sincIndex);

        const T W1 = w1 / (h[i] * h[i] * h[i]);
        const T W2 = w2 / (h[j] * h[j] * h[j]);

        const T kern11_i = c11[i] * r_jix;
        const T kern12_i = c12[i] * r_jiy;
        const T kern13_i = c13[i] * r_jiz;
        const T kern21_i = c12[i] * r_jix;
        const T kern22_i = c22[i] * r_jiy;
        const T kern23_i = c23[i] * r_jiz;
        const T kern31_i = c13[i] * r_jix;
        const T kern32_i = c23[i] * r_jiy;
        const T kern33_i = c33[i] * r_jiz;

        const T kern11_j = c11[j] * r_jix;
        const T kern12_j = c12[j] * r_jiy;
        const T kern13_j = c13[j] * r_jiz;
        const T kern21_j = c12[j] * r_jix;
        const T kern22_j = c22[j] * r_jiy;
        const T kern23_j = c23[j] * r_jiz;
        const T kern31_j = c13[j] * r_jix;
        const T kern32_j = c23[j] * r_jiy;
        const T kern33_j = c33[j] * r_jiz;

        const T termA1_i = (kern11_i + kern12_i + kern13_i) * W1;
        const T termA2_i = (kern21_i + kern22_i + kern23_i) * W1;
        const T termA3_i = (kern31_i + kern32_i + kern33_i) * W1;

        const T termA1_j = (kern11_j + kern12_j + kern13_j) * W2;
        const T termA2_j = (kern21_j + kern22_j + kern23_j) * W2;
        const T termA3_j = (kern31_j + kern32_j + kern33_j) * W2;

        const T pro_i = p[i] / (gradh_i * ro[i] * ro[i]);
        const T pro_j = p[j] / (gradh_j * ro[j] * ro[j]);

        const T r_square = dist * dist;
        const T viscosity_ij = artificial_viscosity(ro[i], ro[j], h[i], h[j], c[i], c[j], rv, r_square);

        const T grad_Px_AV = 0.5 * (m[i] / ro[i] * viscosity_ij * termA1_i + m[j] / ro[j] * viscosity_ij * termA1_j);
        const T grad_Py_AV = 0.5 * (m[i] / ro[i] * viscosity_ij * termA2_i + m[j] / ro[j] * viscosity_ij * termA2_j);
        const T grad_Pz_AV = 0.5 * (m[i] / ro[i] * viscosity_ij * termA3_i + m[j] / ro[j] * viscosity_ij * termA3_j);

        momentum_x += m[j] * (pro_i * termA1_i + pro_j * termA1_j) + grad_Px_AV;
        momentum_y += m[j] * (pro_i * termA2_i + pro_j * termA2_j) + grad_Py_AV;
        momentum_z += m[j] * (pro_i * termA3_i + pro_j * termA3_j) + grad_Pz_AV;

        energy += m[j] * 2.0 * pro_i * (v_ijx * termA1_i + v_ijy * termA2_i + v_ijz * termA3_i);
        energyAV += grad_Px_AV * v_ijx + grad_Py_AV * v_ijy + grad_Pz_AV * v_ijz;
    }

    du[tid] = 0.5 * (energy + energyAV);
    grad_P_x[tid] = momentum_x;
    grad_P_y[tid] = momentum_y;
    grad_P_z[tid] = momentum_z;
}
} // namespace kernels

template void computeMomentumAndEnergyIAD<double, SqPatch<double>>(const std::vector<int> &clist, SqPatch<double> &d);

template <typename T, class Dataset>
void computeMomentumAndEnergyIAD(const std::vector<int> &clist, Dataset &d)
{
    const size_t n = clist.size();
    const size_t np = d.x.size();

    const size_t allNeighbors = n * d.ngmax;

    const size_t size_bbox = sizeof(BBox<T>);
    const size_t size_np_T = np * sizeof(T);
    const size_t size_n_int = n * sizeof(int);
    const size_t size_n_T = n * sizeof(T);
    const size_t size_allNeighbors = allNeighbors * sizeof(int);

    int *d_clist, *d_neighbors, *d_neighborsCount;
    T *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_m, *d_h, *d_ro, *d_p, *d_c, *d_c11, *d_c12, *d_c13, *d_c22, *d_c23, *d_c33;
    BBox<T> *d_bbox;
    T *d_grad_P_x, *d_grad_P_y, *d_grad_P_z, *d_du;

    // input data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_n_int, d_clist, d_neighborsCount));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_allNeighbors, d_neighbors));
    CHECK_CUDA_ERR(utils::cudaMalloc(size_bbox, d_bbox));
    CHECK_CUDA_ERR(
        utils::cudaMalloc(size_np_T, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_m, d_ro, d_p, d_c, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33));

    // oputput data
    CHECK_CUDA_ERR(utils::cudaMalloc(size_n_T, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du));

    CHECK_CUDA_ERR(cudaMemcpy(d_x, d.x.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_y, d.y.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_z, d.z.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_vx, d.vx.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_vy, d.vy.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_vz, d.vz.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_h, d.h.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_m, d.m.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_ro, d.ro.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_p, d.p.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_c, d.c.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_c11, d.c11.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_c12, d.c12.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_c13, d.c13.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_c22, d.c22.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_c23, d.c23.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_c33, d.c33.data(), size_np_T, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_bbox, &d.bbox, size_bbox, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERR(cudaMemcpy(d_clist, clist.data(), size_n_int, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_neighbors, d.neighbors.data(), size_allNeighbors, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_neighborsCount, d.neighborsCount.data(), size_n_int, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    kernels::computeMomentumAndEnergyIAD<<<blocksPerGrid, threadsPerBlock>>>(
        n, d.sincIndex, d.K, d.ngmax, d_bbox, d_clist, d_neighbors, d_neighborsCount, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_m, d_ro, d_p,
        d_c, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du);

    CHECK_CUDA_ERR(cudaGetLastError());

    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_x.data(), d_grad_P_x, size_n_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_y.data(), d_grad_P_y, size_n_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.grad_P_z.data(), d_grad_P_z, size_n_T, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(d.du.data(), d_du, size_n_T, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERR(utils::cudaFree(d_clist, d_neighborsCount, d_neighbors, d_bbox, d_x, d_y, d_z, d_vx, d_vy, d_vz, d_h, d_m, d_ro, d_p,
                                   d_c, d_c11, d_c12, d_c13, d_c22, d_c23, d_c33, d_grad_P_x, d_grad_P_y, d_grad_P_z, d_du));
}

} // namespace cuda
} // namespace sph
} // namespace sphexa
