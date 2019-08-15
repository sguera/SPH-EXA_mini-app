#pragma once

#include <vector>
#include <cmath>

#include "kernels.hpp"
#include "cuda/cudaMomentumAndEnergy.cuh"

namespace sphexa
{
namespace sph
{
template <typename T, class Dataset>
void computeMomentumAndEnergy(const std::vector<int> &l, Dataset &d)
{
#if defined(USE_CUDA)
    cudaComputeMomentumAndEnergy<T>(l, d);
    return;
#endif

    const size_t n = l.size();
    const size_t ngmax = d.ngmax;
    const int *clist = l.data();
    const int *neighborsCount = d.neighborsCount.data();

    const T *h = d.h.data();
    const T *m = d.m.data();
    const T *x = d.x.data();
    const T *y = d.y.data();
    const T *z = d.z.data();
    const T *vx = d.vx.data();
    const T *vy = d.vy.data();
    const T *vz = d.vz.data();
    const T *ro = d.ro.data();
    const T *c = d.c.data();
    const T *p = d.p.data();

    T *du = d.du.data();
    T *grad_P_x = d.grad_P_x.data();
    T *grad_P_y = d.grad_P_y.data();
    T *grad_P_z = d.grad_P_z.data();

    const BBox<T> bbox = d.bbox;

    const T dx = d.dx;
    const T sincIndex = d.sincIndex;
    const T K = d.K;

    const T gradh_i = 1.0;
    const T gradh_j = 1.0;
    const T ep1 = 0.2, ep2 = 0.02;
    const int mre = 4;

#if defined(USE_OMP_TARGET)
    const int np = d.x.size();
    const size_t allNeighbors = n * ngmax;

    for (ushort s = 0; s < d.noOfGpuLoopSplits; ++s)
    {
        const size_t begin_n = s * n / d.noOfGpuLoopSplits;
        const size_t end_n = (s + 1) * n / d.noOfGpuLoopSplits;
        const size_t begin_neighbors = begin_n * ngmax;
        const size_t end_neighbors = end_n * ngmax;
        const size_t neighCount = end_neighbors - begin_neighbors;
        const size_t neighborsOffset = begin_neighbors;
        const size_t neighborsChunkSize = end_neighbors - begin_neighbors;

        const int *neighbors = d.neighbors.data() + neighborsOffset;

// clang-format off
#pragma omp target map(to                                                                                                                  \
		       : clist [:n], neighbors[:neighborsChunkSize], neighborsCount[:n], x [0:np], y [0:np], z [0:np], \
                         vx [0:np], vy [0:np], vz [0:np], h [0:np], m [0:np], ro [0:np], p [0:np], c [0:np])                               \
                   map(from                                                                                                                \
                       : grad_P_x [:n], grad_P_y [:n], grad_P_z [:n], du [:n])
// clang-format on
#pragma omp teams distribute parallel for // dist_schedule(guided)
#elif defined(USE_ACC)
    const size_t neighborsOffset = 0;
    const int *neighbors = d.neighbors.data();
    const int np = d.x.size();
    const size_t allNeighbors = n * ngmax;
    const size_t begin_n = 0;
    const size_t end_n = n;
#pragma acc parallel loop copyin(clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], x [0:np], y [0:np], z [0:np], vx [0:np],   \
                                 vy [0:np], vz [0:np], h [0:np], m [0:np], ro [0:np], p [0:np], c [0:np])                                  \
    copyout(grad_P_x [0:n], grad_P_y [0:n], grad_P_z [0:n], du [0:n])
#else
    const size_t neighborsOffset = 0;
    const int *neighbors = d.neighbors.data();
    const size_t begin_n = 0;
    const size_t end_n = n;
#pragma omp parallel for schedule(guided)
#endif
        for (size_t pi = begin_n; pi < end_n; pi++)
        {
            const int i = clist[pi];
            const int nn = neighborsCount[pi];

            T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;

            const T A_i = p[i] < 0.0 ? 1.0 : 0.0;

            // int converstion to avoid a bug that prevents vectorization with some compilers

            for (int pj = 0; pj < nn; pj++)
            {
                const int j = neighbors[pi * ngmax + pj - neighborsOffset];
                // calculate the scalar product rv = rij * vij

                T r_ijx = (x[i] - x[j]);
                T r_ijy = (y[i] - y[j]);
                T r_ijz = (z[i] - z[j]);

                applyPBC(bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);

                const T v_ijx = (vx[i] - vx[j]);
                const T v_ijy = (vy[i] - vy[j]);
                const T v_ijz = (vz[i] - vz[j]);

                const T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

                const T r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

                const T r_ij = std::sqrt(r_square);
                const T rv_i = r_ij / h[i];
                const T rv_j = r_ij / h[j];

                const T viscosity_ij = artificial_viscosity(ro[i], ro[j], h[i], h[j], c[i], c[j], rv, r_square);

                const T derivative_kernel_i = wharmonic_derivative(rv_i, h[i], sincIndex, K);
                const T derivative_kernel_j = wharmonic_derivative(rv_j, h[j], sincIndex, K);

                // divide by r_ij? missing h?
                const T grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
                const T grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
                const T grad_v_kernel_z_i = r_ijz * derivative_kernel_i;

                const T grad_v_kernel_x_j = r_ijx * derivative_kernel_j;
                const T grad_v_kernel_y_j = r_ijy * derivative_kernel_j;
                const T grad_v_kernel_z_j = r_ijz * derivative_kernel_j;

                const T grad_v_kernel_x_ij = (grad_v_kernel_x_i + grad_v_kernel_x_j) / 2.0;
                const T grad_v_kernel_y_ij = (grad_v_kernel_y_i + grad_v_kernel_y_j) / 2.0;
                const T grad_v_kernel_z_ij = (grad_v_kernel_z_i + grad_v_kernel_z_j) / 2.0;

                const T force_i_j_r = std::exp(-(rv_i * rv_i)) * std::exp((dx * dx) / (h[i] * h[i]));

                const T A_j = p[j] < 0.0 ? 1.0 : 0.0;

                const T delta_pos_i_j = (p[i] > 0.0 && p[j] > 0.0) ? 1.0 : 0.0;

                const T R_i_j =
                    ep1 * (A_i * std::abs(p[i]) + A_j * std::abs(p[j])) + ep2 * delta_pos_i_j * (std::abs(p[i]) + std::abs(p[j]));

                const T r_force_i_j = R_i_j * sphexa::math::pow(force_i_j_r, (int)mre);

                const T partial_repulsive_force = (r_force_i_j / (ro[i] * ro[j]));

                const T pro_i = p[i] / (gradh_i * ro[i] * ro[i]);
                const T pro_j = p[j] / (gradh_j * ro[j] * ro[j]);

                momentum_x += m[j] * (pro_i * grad_v_kernel_x_i + pro_j * grad_v_kernel_x_j +
                                      (partial_repulsive_force + viscosity_ij) * grad_v_kernel_x_ij);
                momentum_y += m[j] * (pro_i * grad_v_kernel_y_i + pro_j * grad_v_kernel_y_j +
                                      (partial_repulsive_force + viscosity_ij) * grad_v_kernel_y_ij);
                momentum_z += m[j] * (pro_i * grad_v_kernel_z_i + pro_j * grad_v_kernel_z_j +
                                      (partial_repulsive_force + viscosity_ij) * grad_v_kernel_z_ij);

                energy += m[j] * (pro_i + 0.5 * viscosity_ij) *
                          (v_ijx * grad_v_kernel_x_i + v_ijy * grad_v_kernel_y_i + v_ijz * grad_v_kernel_z_i);
            }

            du[i] = energy;

#ifndef NDEBUG
            if (std::isnan(momentum_x) || std::isnan(momentum_y) || std::isnan(momentum_z))
                printf("ERROR::MomentumEnergy(%d) MomentumEnergy (%f %f %f)\n", i, momentum_x, momentum_y, momentum_z);
            if (std::isnan(du[i])) printf("ERROR:Energy du %f energy %f p_i %f gradh_i %f ro_i %f\n", du[i], energy, p[i], gradh_i, ro[i]);
#endif

            grad_P_x[i] = momentum_x;
            grad_P_y[i] = momentum_y;
            grad_P_z[i] = momentum_z;
        }
#if defined(USE_OMP_TARGET)
    }
#endif
}
} // namespace sph
} // namespace sphexa
