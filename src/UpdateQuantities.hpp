#pragma once

#include <vector>
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class UpdateQuantities : public TaskLoop
{
public:
	struct Params
	{
		Params(const int STABILIZATION_TIMESTEPS = 15) : 
			STABILIZATION_TIMESTEPS(STABILIZATION_TIMESTEPS) {}
		const int STABILIZATION_TIMESTEPS;
	};
public:

	UpdateQuantities(const ArrayT &grad_P_x, const ArrayT &grad_P_y, const ArrayT &grad_P_z, const ArrayT &dt, const ArrayT &d_u, const int iteration, 
		ArrayT &x, ArrayT &y, ArrayT &z, ArrayT &vx, ArrayT &vy, ArrayT &vz, ArrayT &x_m1, ArrayT &y_m1, ArrayT &z_m1, ArrayT &u, ArrayT &d_u_m1, 
		ArrayT &dt_m1, Params params = Params()) : TaskLoop(x.size()),	grad_P_x(grad_P_x), grad_P_y(grad_P_y), grad_P_z(grad_P_z), 
		dt(dt), du(du), dt_m1(dt_m1), iteration(iteration), vx(vx), vy(vy), vz(vz), x(x), y(y), z(z), params(params) {}

	virtual void compute(int i)
	{
		int stabilization_timesteps = params.STABILIZATION_TIMESTEPS;
		double t_m1 = dt_m1[i];
	    double t_0 = dt[i];
	    double x_loc = x[i];
	    double y_loc = y[i];
	    double z_loc = z[i];



	    // ADD COMPONENT DUE TO THE GRAVITY HERE
	    double ax = - (grad_P_x[i]); //-G * fx
	    double ay = - (grad_P_y[i]); //-G * fy
	    double az = - (grad_P_z[i]); //-G * fz

	    if(iteration < stabilization_timesteps){
	        ax = 0.0;
	        ay = 0.0;
	        az = 0.0;
	    }

	    if (isnan(ax) || isnan(ay) || isnan(az))
	        std::cout << "ERROR: " << ax << ' ' << ay << ' ' << az << std::endl;

	    //update positions according to Press (2nd order)
	    double deltaA = t_0 + 0.5 * t_m1;
	    double deltaB = 0.5 * (t_0 + t_m1);

	    double valx = (x_loc - x_m1[i]) / t_m1;
	    double valy = (y_loc - y_m1[i]) / t_m1;
	    double valz = (z_loc - z_m1[i]) / t_m1;

	    double vx_loc = valx + ax * deltaA;
	    double vy_loc = valy + ay * deltaA;
	    double vz_loc = valz + az * deltaA;
   
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

	    u[i] = u[i] + 0.5 * d_u[i] * deltaB - 0.5 * d_u_m1[i] * deltaA;
	    d_u_m1[i] = d_u[i];


	    //update positions
	    x_m1[i] = x_loc;
	    y_m1[i] = y_loc;
	    z_m1[i] = z_loc;

	    dt_m1[i] = t_0;

	}


private:
	vx(vx), vy(vy), vz(vz), x(x), y(y), z(z), params(params)
	const ArrayT &grad_P_x, &grad_P_y, &grad_P_z, &dt, &d_u;
	const int iteration;
	ArrayT &x, &y, &z, &x_m1, &y_m1, &z_m1, &vx, &vy, &vz, &dt_m1, &u, &d_u_m1;

	Params params;
};

}