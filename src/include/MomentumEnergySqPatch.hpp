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
        int n = clist.size();

        const T gradh_i = 1.0;
        const T gradh_j = 1.0;
        const T delta_x_i = 1.0;
        const T ep1 = 0.2, ep2 = 0.02, mre = 4.0;

        #pragma omp parallel for
        for(int pi=0; pi<n; pi++)
        {
            int i = clist[pi];

            T momentum_x = 0.0, momentum_y = 0.0, momentum_z = 0.0, energy = 0.0;
            
            T A_i = 0.0;
            if(p[i] < 0.0)
                A_i = 1.0;

            for(unsigned int j=0; j<neighbors[pi].size(); j++)
            {
                // retrive the id of a neighbor
                int nid = neighbors[pi][j];
                if(nid == i) continue;

                // calculate the scalar product rv = rij * vij
                T r_ijx = (x[i] - x[nid]);
                T r_ijy = (y[i] - y[nid]);
                T r_ijz = (z[i] - z[nid]);

                // Handling PBC
                if(bbox.PBCx && r_ijx > 2*h[i]) r_ijx -= (bbox.xmax-bbox.xmin);
                else if(r_ijx < -2*h[i]) r_ijx += (bbox.xmax-bbox.xmin);
                
                if(bbox.PBCy && r_ijy > 2*h[i]) r_ijy -= (bbox.ymax-bbox.ymin);
                else if(r_ijy < -2*h[i]) r_ijy += (bbox.ymax-bbox.ymin);
                
                if(bbox.PBCz && r_ijz > 2*h[i]) r_ijz -= (bbox.zmax-bbox.zmin);
                else if(r_ijz < -2*h[i]) r_ijz += (bbox.zmax-bbox.zmin);

                T v_ijx = (vx[i] - vx[nid]);
                T v_ijy = (vy[i] - vy[nid]);
                T v_ijz = (vz[i] - vz[nid]);

                T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

                T r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

                T viscosity_ij = artificial_viscosity(ro[i], ro[nid], h[i], h[nid], c[i], c[nid], rv, r_square);
                
                T r_ij = sqrt(r_square);
                T rv_i = r_ij / h[i];
                T rv_j = r_ij / h[nid];

                T derivative_kernel_i = wharmonic_derivative(rv_i, h[i], sincIndex, K);
                T derivative_kernel_j = wharmonic_derivative(rv_j, h[nid], sincIndex, K);
                
                T grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
                T grad_v_kernel_x_j = r_ijx * derivative_kernel_j;
                T grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
                T grad_v_kernel_y_j = r_ijy * derivative_kernel_j;
                T grad_v_kernel_z_i = r_ijz * derivative_kernel_i;
                T grad_v_kernel_z_j = r_ijz * derivative_kernel_j;
                
                T grad_v_kernel_x_i_j = (grad_v_kernel_x_i + grad_v_kernel_x_j)/2.0;
                T grad_v_kernel_y_i_j = (grad_v_kernel_y_i + grad_v_kernel_y_j)/2.0;
                T grad_v_kernel_z_i_j = (grad_v_kernel_z_i + grad_v_kernel_z_j)/2.0;

                T force_i_j_r = exp(-(rv_i * rv_i)) * exp(delta_x_i / (h[i] * h[i]));

                if(iteration < stabilizationTimesteps)
                   force_i_j_r = 0.0;

                T A_j = 0.0;
                if(p[nid] < 0.0) A_j = 1.0;

                T delta_pos_i_j = 0.0;
                if(p[i] > 0.0 && p[nid] > 0.0) delta_pos_i_j = 1.0;

                T R_i_j = ep1 * (A_i * abs(p[i]) + A_j * abs(p[nid])) + ep2 * delta_pos_i_j * (abs(p[i]) + abs(p[nid]));

                T r_force_i_j = R_i_j * pow(force_i_j_r, mre);

                T partial_repulsive_force = (r_force_i_j / (ro[i] * ro[nid])) * m[nid];

                momentum_x += (p[i]/(gradh_i * ro[i] * ro[i]) * grad_v_kernel_x_i) 
                    + (p[nid]/(gradh_j * ro[nid] * ro[nid]) * grad_v_kernel_x_j) 
                    + grad_v_kernel_x_i_j * (viscosity_ij + partial_repulsive_force);
                momentum_y += (p[i]/(gradh_i * ro[i] * ro[i]) * grad_v_kernel_y_i) 
                    + (p[nid]/(gradh_j * ro[nid] * ro[nid]) * grad_v_kernel_y_j) 
                    + grad_v_kernel_y_i_j * (viscosity_ij + partial_repulsive_force);
                momentum_z += (p[i]/(gradh_i * ro[i] * ro[i]) * grad_v_kernel_z_i) 
                    + (p[nid]/(gradh_j * ro[nid] * ro[nid]) * grad_v_kernel_z_j) 
                    + grad_v_kernel_z_i_j * (viscosity_ij + partial_repulsive_force);
                
                energy += m[nid] * (1.0 + 0.5 * viscosity_ij) * (v_ijx * grad_v_kernel_x_i + v_ijy * grad_v_kernel_y_i + v_ijz * grad_v_kernel_z_i);
            }

            if(std::isnan(momentum_x) || std::isnan(momentum_y) || std::isnan(momentum_z))
                printf("ERROR::MomentumEnergy(%d) MomentumEnergy (%f %f %f)\n", i, momentum_x, momentum_y, momentum_z);

            // 2.0 * p[i]???
            du[i] = energy;
            //du[i] =  energy * (-p[i]/(gradh_i * ro[i] * ro[i]));

            if(std::isnan(du[i]))
                printf("ERROR:Energy du %f energy %f p_i %f gradh_i %f ro_i %f\n", du[i], energy, p[i], gradh_i, ro[i]);

            grad_P_x[i] = momentum_x * m[i];
            grad_P_y[i] = momentum_y * m[i];
            grad_P_z[i] = momentum_z * m[i];
        }
    }

private:
    const int stabilizationTimesteps;
    const T sincIndex, K;
};

}

