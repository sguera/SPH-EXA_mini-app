#pragma once

#include <vector>
#include "kernels.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class MomentumEnergySqPatch
{
public:
    MomentumEnergySqPatch(const T dx, const T sincIndex = 6.0, const T K = compute_3d_k(6.0)) : dx(dx), sincIndex(sincIndex), K(K) {}

    template<class Dataset>
    void compute(const std::vector<int> &clist, Dataset &d)
    {
        const int n = clist.size();
        const int *c_clist = clist.data();

        const T *c_h = d.h.data();
        const T *c_m = d.m.data();
        const T *c_x = d.x.data();
        const T *c_y = d.y.data();
        const T *c_z = d.z.data();
        const T *c_vx = d.vx.data();
        const T *c_vy = d.vy.data();
        const T *c_vz = d.vz.data();
        const T *c_ro = d.ro.data();
        const T *c_c = d.c.data();
        const T *c_p = d.p.data();
        
        T *c_du = d.du.data();
        T *c_grad_P_x = d.grad_P_x.data();
        T *c_grad_P_y = d.grad_P_y.data();
        T *c_grad_P_z = d.grad_P_z.data();

        const int *c_neighbors = d.neighbors.data();
        const int *c_neighborsCount = d.neighborsCount.data();

        const BBox<T> bbox = d.bbox;
        const int ngmax = d.ngmax;

        const T gradh_i = 1.0;
        const T gradh_j = 1.0;
        const T ep1 = 0.2, ep2 = 0.02, mre = 4.0;
        
        #pragma omp target
        #pragma omp parallel for
        for(int pi=0; pi<n; pi++)
        {
            const int i = c_clist[pi];
            const int nn = c_neighborsCount[pi];

            T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;
            
            T A_i = 0.0;
            if(c_p[i] < 0.0)
                A_i = 1.0;

            // int converstion to avoid a bug that prevents vectorization with some compilers
            for(int pj=0; pj<nn; pj++)
            {
                const int j = c_neighbors[pi*ngmax+pj];

                // calculate the scalar product rv = rij * vij
                T r_ijx = (c_x[i] - c_x[j]);
                T r_ijy = (c_y[i] - c_y[j]);
                T r_ijz = (c_z[i] - c_z[j]);

                applyPBC(bbox, 2.0*c_h[i], r_ijx, r_ijy, r_ijz);

                T v_ijx = (c_vx[i] - c_vx[j]);
                T v_ijy = (c_vy[i] - c_vy[j]);
                T v_ijz = (c_vz[i] - c_vz[j]);

                T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

                T r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

                T r_ij = std::sqrt(r_square);
                T rv_i = r_ij / c_h[i];
                T rv_j = r_ij / c_h[j];

                T viscosity_ij = artificial_viscosity(c_ro[i], c_ro[j], c_h[i], c_h[j], c_c[i], c_c[j], rv, r_square);

                T derivative_kernel_i = wharmonic_derivative(rv_i, c_h[i], sincIndex, K);
                T derivative_kernel_j = wharmonic_derivative(rv_j, c_h[j], sincIndex, K);
                
                // divide by r_ij? missing h?
                T grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
                T grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
                T grad_v_kernel_z_i = r_ijz * derivative_kernel_i;

                T grad_v_kernel_x_j = r_ijx * derivative_kernel_j;
                T grad_v_kernel_y_j = r_ijy * derivative_kernel_j;
                T grad_v_kernel_z_j = r_ijz * derivative_kernel_j;
                
                T grad_v_kernel_x_ij = (grad_v_kernel_x_i + grad_v_kernel_x_j)/2.0;
                T grad_v_kernel_y_ij = (grad_v_kernel_y_i + grad_v_kernel_y_j)/2.0;
                T grad_v_kernel_z_ij = (grad_v_kernel_z_i + grad_v_kernel_z_j)/2.0;

                T force_i_j_r = std::exp(-(rv_i * rv_i)) * std::exp((dx*dx) / (c_h[i] * c_h[i]));

                T A_j = 0.0;
                if(c_p[j] < 0.0) A_j = 1.0;

                T delta_pos_i_j = 0.0;
                if(c_p[i] > 0.0 && c_p[j] > 0.0) delta_pos_i_j = 1.0;

                T R_i_j = ep1 * (A_i * std::abs(c_p[i]) + A_j * std::abs(c_p[j])) + ep2 * delta_pos_i_j * (std::abs(c_p[i]) + std::abs(c_p[j]));

                T r_force_i_j = R_i_j * std::pow(force_i_j_r, mre);

                T partial_repulsive_force = (r_force_i_j / (c_ro[i] * c_ro[j]));

                T pro_i = c_p[i]/(gradh_i * c_ro[i] * c_ro[i]);
                T pro_j = c_p[j]/(gradh_j * c_ro[j] * c_ro[j]);
                
                momentum_x += c_m[j] * (pro_i * grad_v_kernel_x_i + pro_j * grad_v_kernel_x_j + (partial_repulsive_force + viscosity_ij) * grad_v_kernel_x_ij);
                momentum_y += c_m[j] * (pro_i * grad_v_kernel_y_i + pro_j * grad_v_kernel_y_j + (partial_repulsive_force + viscosity_ij) * grad_v_kernel_y_ij);
                momentum_z += c_m[j] * (pro_i * grad_v_kernel_z_i + pro_j * grad_v_kernel_z_j + (partial_repulsive_force + viscosity_ij) * grad_v_kernel_z_ij);

                energy += c_m[j] * (pro_i + 0.5 * viscosity_ij) * (v_ijx * grad_v_kernel_x_i + v_ijy * grad_v_kernel_y_i + v_ijz * grad_v_kernel_z_i);
            }

            c_du[i] = energy;

            #ifndef NDEBUG
                if(std::isnan(momentum_x) || std::isnan(momentum_y) || std::isnan(momentum_z))
                    printf("ERROR::MomentumEnergy(%d) MomentumEnergy (%f %f %f)\n", i, momentum_x, momentum_y, momentum_z);
                if(std::isnan(du[i]))
                    printf("ERROR:Energy du %f energy %f p_i %f gradh_i %f ro_i %f\n", c_du[i], energy, c_p[i], gradh_i, c_ro[i]);
            #endif

            c_grad_P_x[i] = momentum_x;
            c_grad_P_y[i] = momentum_y;
            c_grad_P_z[i] = momentum_z;
        }
    }

private:
    const T dx, sincIndex, K;
};

}

