#pragma once

#include <vector>
#include "kernels.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class MomentumEnergySqPatch
{
public:
    MomentumEnergySqPatch(const int stabilizationTimesteps = -1, const T sincIndex = 6.0, const T K = compute_3d_k(6.0)) : 
    stabilizationTimesteps(stabilizationTimesteps), sincIndex(sincIndex), K(K) {}

    void compute(const std::vector<int> &clist, const BBox<T> &bbox, const int iteration, const std::vector<std::vector<int>> &neighbors, 
        const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h,
        const ArrayT &vx, const ArrayT &vy, const ArrayT &vz, 
        const ArrayT &ro, const ArrayT &p, const ArrayT &c, const ArrayT &m,
        ArrayT &grad_P_x, ArrayT &grad_P_y, ArrayT &grad_P_z, ArrayT &du)
    {
        const int n = clist.size();

        const T gradh_i = 1.0;
        const T gradh_j = 1.0;
        //const T delta_x_i = 0.01; // Initial inter-particule distance
        const T delta_x_i = 1.0;
        const T ep1 = 0.2, ep2 = 0.02, mre = 4.0;
        
        #pragma omp parallel for
        for(int pi=0; pi<n; pi++)
        {
            const int i = clist[pi];
            const int nn = neighbors[pi].size();

            T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;
            
            T A_i = 0.0;
            if(p[i] < 0.0)
                A_i = 1.0;

            // int converstion to avoid a bug that prevents vectorization with some compilers
            for(int pj=0; pj<nn; pj++)
            {
                const int j = neighbors[pi][pj];

                // calculate the scalar product rv = rij * vij
                T r_ijx = (x[i] - x[j]);
                T r_ijy = (y[i] - y[j]);
                T r_ijz = (z[i] - z[j]);

                applyPBC(bbox, 2.0*h[i], r_ijx, r_ijy, r_ijz);

                T v_ijx = (vx[i] - vx[j]);
                T v_ijy = (vy[i] - vy[j]);
                T v_ijz = (vz[i] - vz[j]);

                T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

                T r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

                T r_ij = sqrt(r_square);
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
                
                T grad_v_kernel_x_ij = (grad_v_kernel_x_i + grad_v_kernel_x_j)/2.0;
                T grad_v_kernel_y_ij = (grad_v_kernel_y_i + grad_v_kernel_y_j)/2.0;
                T grad_v_kernel_z_ij = (grad_v_kernel_z_i + grad_v_kernel_z_j)/2.0;

                T force_i_j_r = exp(-(rv_i * rv_i)) * exp((delta_x_i*delta_x_i) / (h[i] * h[i]));

                if(iteration < stabilizationTimesteps)
                   force_i_j_r = 0.0;

                T A_j = 0.0;
                if(p[j] < 0.0) A_j = 1.0;

                T delta_pos_i_j = 0.0;
                if(p[i] > 0.0 && p[j] > 0.0) delta_pos_i_j = 1.0;

                T R_i_j = ep1 * (A_i * std::abs(p[i]) + A_j * std::abs(p[j])) + ep2 * delta_pos_i_j * (std::abs(p[i]) + std::abs(p[j]));

                T r_force_i_j = R_i_j * pow(force_i_j_r, mre);

                T partial_repulsive_force = (r_force_i_j / (ro[i] * ro[j]));

                T pro_i = p[i]/(gradh_i * ro[i] * ro[i]);
                T pro_j = p[j]/(gradh_j * ro[j] * ro[j]);

                momentum_x += m[j] * (pro_i * grad_v_kernel_x_i + pro_j * grad_v_kernel_x_j + (partial_repulsive_force + viscosity_ij) * grad_v_kernel_x_ij);
                momentum_y += m[j] * (pro_i * grad_v_kernel_y_i + pro_j * grad_v_kernel_y_j + (partial_repulsive_force + viscosity_ij) * grad_v_kernel_y_ij);
                momentum_z += m[j] * (pro_i * grad_v_kernel_z_i + pro_j * grad_v_kernel_z_j + (partial_repulsive_force + viscosity_ij) * grad_v_kernel_z_ij);

                energy += m[j] * (pro_i + 0.5 * viscosity_ij) * (v_ijx * grad_v_kernel_x_i + v_ijy * grad_v_kernel_y_i + v_ijz * grad_v_kernel_z_i);
            }

            du[i] = energy;

            #ifndef NDEBUG
                if(std::isnan(momentum_x) || std::isnan(momentum_y) || std::isnan(momentum_z))
                    printf("ERROR::MomentumEnergy(%d) MomentumEnergy (%f %f %f)\n", i, momentum_x, momentum_y, momentum_z);
                if(std::isnan(du[i]))
                    printf("ERROR:Energy du %f energy %f p_i %f gradh_i %f ro_i %f\n", du[i], energy, p[i], gradh_i, ro[i]);
            #endif

            grad_P_x[i] = momentum_x;
            grad_P_y[i] = momentum_y;
            grad_P_z[i] = momentum_z;
        }
    }

private:
    const int stabilizationTimesteps;
    const T sincIndex, K;
};

}

