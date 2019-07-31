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
    cudaComputeMomentumAndEnergy(l, d);
    return;
#endif

    const int64_t n = l.size();
    const int64_t ngmax = d.ngmax;
    const int *clist = l.data();
    const int *neighbors = d.neighbors.data();
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
    const int64_t allNeighbors = n * ngmax;
#pragma omp target map(to                                                                                                                  \
                       : clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], x [0:np], y [0:np], z [0:np], vx [0:np],           \
                         vy [0:np], vz [0:np], h [0:np], m [0:np], ro [0:np], p [0:np], c [0:np])                                          \
    map(from                                                                                                                               \
        : grad_P_x [0:n], grad_P_y [0:n], grad_P_z [0:n], du [0:n])
#pragma omp teams distribute parallel for // dist_schedule(guided)// parallel for
#elif defined(USE_ACC)
    const int np = d.x.size();
    const int64_t allNeighbors = n * ngmax;
#pragma acc parallel loop copyin(clist [0:n], neighbors [0:allNeighbors], neighborsCount [0:n], x [0:np], y [0:np], z [0:np], vx [0:np],   \
                                 vy [0:np], vz [0:np], h [0:np], m [0:np], ro [0:np], p [0:np], c [0:np])                                  \
    copyout(grad_P_x [0:n], grad_P_y [0:n], grad_P_z [0:n], du [0:n])
#else
#pragma omp parallel for schedule(guided)
#endif
    for (int pi = 0; pi < n; pi++)
    {
        const int i = clist[pi];
        const int nn = neighborsCount[pi];

        T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;

        T A_i = 0.0;
        if (p[i] < 0.0) A_i = 1.0;

        // int converstion to avoid a bug that prevents vectorization with some compilers
        for (int pj = 0; pj < nn; pj++)
        {
            const int j = neighbors[pi * ngmax + pj];

            // calculate the scalar product rv = rij * vij

            T r_ijx = (x[i] - x[j]);
            T r_ijy = (y[i] - y[j]);
            T r_ijz = (z[i] - z[j]);

            applyPBC(bbox, 2.0 * h[i], r_ijx, r_ijy, r_ijz);

            T v_ijx = (vx[i] - vx[j]);
            T v_ijy = (vy[i] - vy[j]);
            T v_ijz = (vz[i] - vz[j]);

            T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

            T r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

            T r_ij = std::sqrt(r_square);
            T rv_i = r_ij / h[i];
            T rv_j = r_ij / h[j];

            T viscosity_ij = artificial_viscosity(ro[i], ro[j], h[i], h[j], c[i], c[j], rv, r_square);

            T derivative_kernel_i = wharmonic_derivative(rv_i, h[i], sincIndex, K);
            T derivative_kernel_j = wharmonic_derivative(rv_j, h[j], sincIndex, K);

            // divide by r_ij? missing h?
            T grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
            T grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
            T grad_v_kernel_z_i = r_ijz * derivative_kernel_i;

            T grad_v_kernel_x_j = r_ijx * derivative_kernel_j;
            T grad_v_kernel_y_j = r_ijy * derivative_kernel_j;
            T grad_v_kernel_z_j = r_ijz * derivative_kernel_j;

            T grad_v_kernel_x_ij = (grad_v_kernel_x_i + grad_v_kernel_x_j) / 2.0;
            T grad_v_kernel_y_ij = (grad_v_kernel_y_i + grad_v_kernel_y_j) / 2.0;
            T grad_v_kernel_z_ij = (grad_v_kernel_z_i + grad_v_kernel_z_j) / 2.0;

            T force_i_j_r = std::exp(-(rv_i * rv_i)) * std::exp((dx * dx) / (h[i] * h[i]));

            T A_j = 0.0;
            if (p[j] < 0.0) A_j = 1.0;

            T delta_pos_i_j = 0.0;
            if (p[i] > 0.0 && p[j] > 0.0) delta_pos_i_j = 1.0;

            T R_i_j = ep1 * (A_i * std::abs(p[i]) + A_j * std::abs(p[j])) + ep2 * delta_pos_i_j * (std::abs(p[i]) + std::abs(p[j]));

            T r_force_i_j = R_i_j * sphexa::math::pow(force_i_j_r, (int)mre);

            T partial_repulsive_force = (r_force_i_j / (ro[i] * ro[j]));

            T pro_i = p[i] / (gradh_i * ro[i] * ro[i]);
            T pro_j = p[j] / (gradh_j * ro[j] * ro[j]);

            momentum_x += m[j] * (pro_i * grad_v_kernel_x_i + pro_j * grad_v_kernel_x_j +
                                  (partial_repulsive_force + viscosity_ij) * grad_v_kernel_x_ij);
            momentum_y += m[j] * (pro_i * grad_v_kernel_y_i + pro_j * grad_v_kernel_y_j +
                                  (partial_repulsive_force + viscosity_ij) * grad_v_kernel_y_ij);
            momentum_z += m[j] * (pro_i * grad_v_kernel_z_i + pro_j * grad_v_kernel_z_j +
                                  (partial_repulsive_force + viscosity_ij) * grad_v_kernel_z_ij);

            energy +=
                m[j] * (pro_i + 0.5 * viscosity_ij) * (v_ijx * grad_v_kernel_x_i + v_ijy * grad_v_kernel_y_i + v_ijz * grad_v_kernel_z_i);
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
}
} // namespace sph
} // namespace sphexa
