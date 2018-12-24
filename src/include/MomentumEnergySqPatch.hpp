#pragma once

#include <vector>
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class MomentumEnergySqPatch : public TaskLoop
{
public:
    struct Params
    {
        Params(const T K = compute_3d_k(6.0), const T delta_x_i = 1.0, const T A_i = 0.0, 
            const T ep1 = 0.2, const T ep2 = 0.02, const T mre = 4.0, const int init_timesteps = -1) : K(K), delta_x_i(delta_x_i),
            A_i(A_i), ep1(ep1), ep2(ep2), mre(mre), init_timesteps(init_timesteps) {}
        const T K, delta_x_i, A_i, ep1, ep2, mre;
        const int init_timesteps;
    };
public:

    MomentumEnergySqPatch(const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h,
        const ArrayT &vx, const ArrayT &vy, const ArrayT &vz, 
        const ArrayT &ro, const ArrayT &p, const ArrayT &c, const ArrayT &m,
        const int &iteration, const std::vector<std::vector<int>> &neighbors, 
        ArrayT &grad_P_x, ArrayT &grad_P_y, ArrayT &grad_P_z, ArrayT &du, Params params = Params()) : 
    TaskLoop(x.size()),
    x(x), y(y), z(z), h(h), vx(vx), vy(vy), vz(vz), ro(ro), p(p), c(c), m(m), iteration(iteration), neighbors(neighbors),
    grad_P_x(grad_P_x), grad_P_y(grad_P_y), grad_P_z(grad_P_z), du(du), params(params) {}

    virtual void compute(int i)
    {
        T K = params.K;
        T ro_i = ro[i];
        T p_i = p[i];
        T x_i = x[i];
        T y_i = y[i];
        T z_i = z[i];
        T vx_i = vx[i];
        T vy_i = vy[i];
        T vz_i = vz[i];
        T h_i = h[i];
        T momentum_x = 0.0;
        T momentum_y = 0.0;
        T momentum_z = 0.0;

        T energy = 0.0;

        T delta_x_i = params.delta_x_i;
        T A_i = params.A_i;
        T ep1 = params.ep1;
        T ep2 = params.ep2;
        T mre = params.mre;
        int init_timesteps = params.init_timesteps;

        if((p_i) < 0.0)
            A_i = 1.0;

        for(unsigned int j=0; j<neighbors[i].size(); j++)
        {
            // retrive the id of a neighbor
            int nid = neighbors[i][j];
            if(nid == i) continue;

            T ro_j = ro[nid];
            T p_j = p[nid];
            T x_j = x[nid];
            T y_j = y[nid];
            T z_j = z[nid];
            T h_j = h[nid];
            T m_j = m[nid];

            // calculate the scalar product rv = rij * vij
            T r_ijx = (x_i - x_j);
            T r_ijy = (y_i - y_j);
            T r_ijz = (z_i - z_j);

            T v_ijx = (vx_i - vx[nid]);
            T v_ijy = (vy_i - vy[nid]);
            T v_ijz = (vz_i - vz[nid]);

            T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

            T r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

            T viscosity_ij = artificial_viscosity(ro_i, ro_j, h[i], h[nid], c[i], c[nid], rv, r_square);

            T r_ij = sqrt(r_square);
            T v_i = r_ij / h_i;
            T v_j = r_ij / h_j;

            T derivative_kernel_i = wharmonic_derivative(v_i, h_i, K);
            T derivative_kernel_j = wharmonic_derivative(v_j, h_j, K);
            
            T grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
            T grad_v_kernel_x_j = r_ijx * derivative_kernel_j;
            T grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
            T grad_v_kernel_y_j = r_ijy * derivative_kernel_j;
            T grad_v_kernel_z_i = r_ijz * derivative_kernel_i;
            T grad_v_kernel_z_j = r_ijz * derivative_kernel_j;

            T grad_v_kernel_x_i_j = (grad_v_kernel_x_i + grad_v_kernel_x_j)/2.0;
            T grad_v_kernel_y_i_j = (grad_v_kernel_y_i + grad_v_kernel_y_j)/2.0;
            T grad_v_kernel_z_i_j = (grad_v_kernel_z_i + grad_v_kernel_z_j)/2.0;

            T force_i_j_r = exp(-(v_i * v_i)) * exp(delta_x_i / (h_i * h_i));

            if (iteration < init_timesteps)
               force_i_j_r = 0.0;

            T A_j = 0.0;

            if(p_j < 0.0)
                A_j = 1.0;

            T delta_pos_i_j = 0.0;

            if(p_i > 0.0 && p_j > 0.0)
                delta_pos_i_j = 1.0;

            T R_i_j = ep1 * (A_i * abs(p_i) + A_j * abs(p_j)) + ep2 * delta_pos_i_j * (abs(p_i) + abs(p_j));

            // if(std::isnan(ep1))
            //     std::cout << "Ep1 is nan" << std::endl;

            T r_force_i_j = R_i_j * pow(force_i_j_r, mre);

            // if(std::isnan(r_force_i_j))
            //     std::cout << "r_force_i_j is nan" << std::endl;

            T partial_repulsive_force = (r_force_i_j / (ro_i * ro_j)) * m_j;

            // if(std::isnan(partial_repulsive_force))
            //     printf("partial_repulsive_force: %f %f %f %f\n", ro_i, ro_j, m_j, r_force_i_j);

            T repulsive_force_x = partial_repulsive_force * grad_v_kernel_x_i_j;
            T repulsive_force_y = partial_repulsive_force * grad_v_kernel_y_i_j;
            T repulsive_force_z = partial_repulsive_force * grad_v_kernel_z_i_j;

            //if(std::isnan(repulsive_force_x) && std::isnan(repulsive_force_y) && std::isnan(repulsive_force_z))
            //    std::cout << "repulsive_force_x y or z is nan" << std::endl;
            
            momentum_x +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_x_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_x_j) + viscosity_ij * grad_v_kernel_x_i_j + repulsive_force_x;
            momentum_y +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_y_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_y_j) + viscosity_ij * grad_v_kernel_y_i_j + repulsive_force_y;
            momentum_z +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_z_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_z_j) + viscosity_ij * grad_v_kernel_z_i_j + repulsive_force_z;
            
            energy +=  m_j * (1 + 0.5 * viscosity_ij) * (v_ijx * grad_v_kernel_x_i + v_ijy * grad_v_kernel_y_i + v_ijz * grad_v_kernel_z_i);
        }

        if(std::isnan(momentum_x) || std::isnan(momentum_y) || std::isnan(momentum_z))
            printf("ERROR::Momentum(%d) momentum (%f %f %f)\n", i, momentum_x, momentum_y, momentum_z);

        du[i] =  energy * (-p_i/(gradh_i * ro_i * ro_i));

        if(std::isnan(du[i]))
            printf("ERROR:Energy du %f energy %f p_i %f gradh_i %f ro_i %f\n", du[i], energy, p_i, gradh_i, ro_i);

        grad_P_x[i] = momentum_x * m[i];
        grad_P_y[i] = momentum_y * m[i];
        grad_P_z[i] = momentum_z * m[i];
    }

private:
    const T gradh_i = 1.0;
    const T gradh_j = 1.0;
    const ArrayT &x, &y, &z, &h, &vx, &vy, &vz, &ro, &p, &c, &m;
    const int &iteration;
    const std::vector<std::vector<int>> &neighbors;

    ArrayT &grad_P_x, &grad_P_y, &grad_P_z, &du;

    Params params;
};

}

