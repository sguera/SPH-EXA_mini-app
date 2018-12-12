#pragma once

#include <vector>
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class Energy : public TaskLoop
{
public:
	struct Params
	{
		Params(const T K = compute_3d_k(5.0)) : K(K) {}
		const T K;
	};
public:

	Energy(const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h,
		const ArrayT &vx, const ArrayT &vy, const ArrayT &vz, 
		const ArrayT &ro, const ArrayT &p, const ArrayT &c, const ArrayT &m, 
		const std::vector<std::vector<int>> &neighbors, 
		ArrayT &du, Params params = Params()) : 
	TaskLoop(x.size()), x(x), y(y), z(z), h(h), vx(vx), vy(vy), vz(vz), ro(ro), p(p), c(c), m(m), neighbors(neighbors), du(du), params(params) {}

	virtual void compute(int i)
	{
		T K = params.K;

		// note that practically all these variables are already calculated in
	    // computeMomentum, so it would make sens to fuse momentum and energy
	    T ro_i = ro[i];
	    T p_i = p[i];
	    T vx_i = vx[i];
	    T vy_i = vy[i];
	    T vz_i = vz[i];
	    T x_i = x[i];
	    T y_i = y[i];
	    T z_i = z[i];
	    T h_i = h[i];

	    T energy = 0.0;

	    for(unsigned int j=0; j<neighbors[i].size(); j++)
	    {
	        // retrive the id of a neighbor
	        int nid = neighbors[i][j];
	        if(nid == i) continue;

	        T ro_j = ro[nid];
	        T m_j = m[nid];
	        T x_j = x[nid];
	        T y_j = y[nid];
	        T z_j = z[nid];

	        //calculate the velocity difference
	        T v_ijx = (vx_i - vx[nid]);
	        T v_ijy = (vy_i - vy[nid]);
	        T v_ijz = (vz_i - vz[nid]);

	        T r_ijx = (x_i - x_j);
	        T r_ijy = (y_i - y_j);
	        T r_ijz = (z_i - z_j);

	        T rv = r_ijx * v_ijx + r_ijy * v_ijy + r_ijz * v_ijz;

	        T r_square = (r_ijx * r_ijx) + (r_ijy * r_ijy) + (r_ijz * r_ijz);

	        T viscosity_ij = artificial_viscosity(ro_i, ro_j, h[i], h[nid], c[i], c[nid], rv, r_square);

	        T r_ij = sqrt(r_square);
	        T v_i = r_ij / h_i;

	        T derivative_kernel_i = wharmonic_derivative(v_i, h_i, K);

	        T grad_v_kernel_x_i = r_ijx * derivative_kernel_i;
	        T grad_v_kernel_y_i = r_ijy * derivative_kernel_i;
	        T grad_v_kernel_z_i = r_ijz * derivative_kernel_i;

	        energy +=  m_j * (1 + 0.5 * viscosity_ij) * (v_ijx * grad_v_kernel_x_i + v_ijy * grad_v_kernel_y_i + v_ijz * grad_v_kernel_z_i);
	    }

    	du[i] =  energy * (-p_i/(gradh_i * ro_i * ro_i));

	}

private:
	const T gradh_i = 1.0;
	const ArrayT &x, &y, &z, &h, &vx, &vy, &vz, &ro, &p, &c, &m;
	const std::vector<std::vector<int>> &neighbors;

	ArrayT &du;

	Params params;
};

}

