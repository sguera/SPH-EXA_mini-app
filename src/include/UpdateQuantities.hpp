#pragma once

#include <vector>
#include <cmath>

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class UpdateQuantities
{
public:

	UpdateQuantities(const int stabilizationTimesteps = -1) : stabilizationTimesteps(stabilizationTimesteps) {}

	void compute(const std::vector<int> &clist, const int iteration, const ArrayT &grad_P_x, const ArrayT &grad_P_y, const ArrayT &grad_P_z, const ArrayT &dt, const ArrayT &du, 
		const BBox<T> &bbox, ArrayT &x, ArrayT &y, ArrayT &z, ArrayT &vx, ArrayT &vy, ArrayT &vz, ArrayT &x_m1, ArrayT &y_m1, ArrayT &z_m1, ArrayT &u, ArrayT &du_m1, ArrayT &dt_m1)
	{
		int n = clist.size();

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];

			T t_m1 = dt_m1[i];
		    T t_0 = dt[i];
		    T x_loc = x[i];
		    T y_loc = y[i];
		    T z_loc = z[i];
		    T u_i = u[i];

		    // ADD COMPONENT DUE TO THE GRAVITY HERE
		    T ax = - (grad_P_x[i]); //-G * fx
		    T ay = - (grad_P_y[i]); //-G * fy
		    T az = - (grad_P_z[i]); //-G * fz

		    if(iteration < stabilizationTimesteps)
		    {
		        ax = 0.0;
		        ay = 0.0;
		        az = 0.0;
		    }

		    if(std::isnan(ax) || std::isnan(ay) || std::isnan(az))
		    	printf("ERROR::UpdateQuantities(%d) acceleration: (%f %f %f)\n", i, ax, ay, az);

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

		    if(bbox.PBCx && x[i] < bbox.xmin) x[i] += (bbox.xmax-bbox.xmin);
		    if(bbox.PBCx && x[i] > bbox.xmax) x[i] -= (bbox.xmax-bbox.xmin);
		    if(bbox.PBCy && y[i] < bbox.ymin) y[i] += (bbox.ymax-bbox.ymin);
		    if(bbox.PBCy && y[i] > bbox.ymax) y[i] -= (bbox.ymax-bbox.ymin);
		    if(bbox.PBCz && z[i] < bbox.zmin) z[i] += (bbox.zmax-bbox.zmin);
		    if(bbox.PBCz && z[i] > bbox.zmax) z[i] -= (bbox.zmax-bbox.zmin);

		    //update the energy according to Adams-Bashforth (2nd order)
		    deltaA = 0.5 * t_0 * t_0 / t_m1;
		    deltaB = t_0 + deltaA;

		    u[i] = u_i + 0.5 * du[i] * deltaB - 0.5 * du_m1[i] * deltaA;

		    if(std::isnan(u[i]))
		    	printf("ERROR::UpdateQuantities(%d) internal energy: new_u %f u %f du %f dB %f du_m1 %f dA %f\n", i, u[i], u_i, du[i], deltaB, du_m1[i], deltaA);

		    du_m1[i] = du[i];

		    //update positions
		    x_m1[i] = x_loc;
		    y_m1[i] = y_loc;
		    z_m1[i] = z_loc;

		    dt_m1[i] = t_0;
		}
	}


private:
	const int stabilizationTimesteps;
};

}

