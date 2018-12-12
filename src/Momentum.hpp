#pragma once

#include <vector>
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class Momentum : public TaskLoop
{
public:
	struct Params
	{
		Params(const T K = compute_3d_k(5.0)) : K(K) {}
		const T K;
	};
public:

	Momentum(const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h,
		const ArrayT &vx, const ArrayT &vy, const ArrayT &vz, 
		const ArrayT &ro, const ArrayT &p, const ArrayT &c, const ArrayT &m, 
		const std::vector<std::vector<int>> &neighbors, 
		ArrayT &grad_P_x, ArrayT &grad_P_y, ArrayT &grad_P_z, Params params = Params()) : 
	TaskLoop(x.size()),
	x(x), y(y), z(z), h(h), vx(vx), vy(vy), vz(vz), ro(ro), p(p), c(c), m(m), neighbors(neighbors),
	grad_P_x(grad_P_x), grad_P_y(grad_P_y), grad_P_z(grad_P_z), params(params) {}

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
			
			momentum_x +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_x_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_x_j) + viscosity_ij * (grad_v_kernel_x_i + grad_v_kernel_x_j)/2.0;
	        momentum_y +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_y_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_y_j) + viscosity_ij * (grad_v_kernel_y_i + grad_v_kernel_y_j)/2.0;
	        momentum_z +=  (p_i/(gradh_i * ro_i * ro_i) * grad_v_kernel_z_i) + (p_j/(gradh_j * ro_j * ro_j) * grad_v_kernel_z_j) + viscosity_ij * (grad_v_kernel_z_i + grad_v_kernel_z_j)/2.0;
	    }

	    grad_P_x[i] = momentum_x * m[i];
	    grad_P_y[i] = momentum_y * m[i];
	    grad_P_z[i] = momentum_z * m[i];
	}

private:
	const T gradh_i = 1.0;
	const T gradh_j = 1.0;
	const ArrayT &x, &y, &z, &h, &vx, &vy, &vz, &ro, &p, &c, &m;
	const std::vector<std::vector<int>> &neighbors;

	ArrayT &grad_P_x, &grad_P_y, &grad_P_z;

	Params params;
};

}

