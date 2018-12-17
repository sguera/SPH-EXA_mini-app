#pragma once

#include <vector>
#include <cmath>
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class UpdateQuantities : public TaskLoop
{
public:
	struct Params
	{
		Params(const int STABILIZATION_TIMESTEPS = -1) : 
			STABILIZATION_TIMESTEPS(STABILIZATION_TIMESTEPS) {}
		const int STABILIZATION_TIMESTEPS;
	};
public:

	UpdateQuantities(const ArrayT &grad_P_x, const ArrayT &grad_P_y, const ArrayT &grad_P_z, const ArrayT &dt, const ArrayT &du, const int &iteration, 
		ArrayT &x, ArrayT &y, ArrayT &z, ArrayT &vx, ArrayT &vy, ArrayT &vz, ArrayT &x_m1, ArrayT &y_m1, ArrayT &z_m1, ArrayT &u, ArrayT &du_m1, 
		ArrayT &dt_m1, Params params = Params()) : 
		TaskLoop(x.size()),	grad_P_x(grad_P_x), grad_P_y(grad_P_y), grad_P_z(grad_P_z), dt(dt), du(du), iteration(iteration), 
		x(x), y(y), z(z), vx(vx), vy(vy), vz(vz), x_m1(x_m1), y_m1(y_m1), z_m1(z_m1), u(u), du_m1(du_m1), dt_m1(dt_m1), params(params) {}

	virtual void compute(int i)
	{
		int stabilization_timesteps = params.STABILIZATION_TIMESTEPS;
		T t_m1 = dt_m1[i];
	    T t_0 = dt[i];
	    T x_loc = x[i];
	    T y_loc = y[i];
	    T z_loc = z[i];

	    // ADD COMPONENT DUE TO THE GRAVITY HERE
	    T ax = - (grad_P_x[i]); //-G * fx
	    T ay = - (grad_P_y[i]); //-G * fy
	    T az = - (grad_P_z[i]); //-G * fz

	    if(iteration < stabilization_timesteps)
	    {
	        ax = 0.0;
	        ay = 0.0;
	        az = 0.0;
	    }

	    if(std::isnan(ax) || std::isnan(ay) || std::isnan(az))
	        std::cout << "ERROR: " << ax << ' ' << ay << ' ' << az << std::endl;

	    //update positions according to Press (2nd order)
	    T deltaA = t_0 + 0.5 * t_m1;
	    T deltaB = 0.5 * (t_0 + t_m1);

	    T valx = (x_loc - x_m1[i]) / t_m1;
	    T valy = (y_loc - y_m1[i]) / t_m1;
	    T valz = (z_loc - z_m1[i]) / t_m1;

	    T vx_loc = valx + ax * deltaA;
	    T vy_loc = valy + ay * deltaA;
	    T vz_loc = valz + az * deltaA;
   
	    vx[i] = vx_loc;
	    vy[i] = vy_loc;
	    vz[i] = vz_loc;

	    x[i] = x_loc + t_0 * valx + (vx_loc - valx) * t_0 * deltaB / deltaA;
	    //x[i] = x + t_0 * valx + ax * t_0 * deltaB;
	    y[i] = y_loc + t_0 * valy + (vy_loc - valy) * t_0 * deltaB / deltaA;
	    z[i] = z_loc + t_0 * valz + (vz_loc - valz) * t_0 * deltaB / deltaA;

	    //update the energy according to Adams-Bashforth (2nd order)
	    deltaA = 0.5 * t_0 * t_0 / t_m1;
	    deltaB = t_0 + deltaA;

	    u[i] = u[i] + 0.5 * du[i] * deltaB - 0.5 * du_m1[i] * deltaA;
	    du_m1[i] = du[i];

	    //update positions
	    x_m1[i] = x_loc;
	    y_m1[i] = y_loc;
	    z_m1[i] = z_loc;

	    dt_m1[i] = t_0;
	}


private:
	const ArrayT &grad_P_x, &grad_P_y, &grad_P_z, &dt, &du;
	const int &iteration;
	ArrayT &x, &y, &z, &vx, &vy, &vz, &x_m1, &y_m1, &z_m1, &u, &du_m1, &dt_m1;

	Params params;
};

}

