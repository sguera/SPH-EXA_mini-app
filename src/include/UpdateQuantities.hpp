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

		    // Update positions according to Press (2nd order)
		    T deltaA = dt[i] + 0.5 * dt_m1[i];
		    T deltaB = 0.5 * (dt[i] + dt_m1[i]);

		    T valx = (x[i] - x_m1[i]) / dt_m1[i];
		    T valy = (y[i] - y_m1[i]) / dt_m1[i];
		    T valz = (z[i] - z_m1[i]) / dt_m1[i];

		    vx[i] = valx + ax * deltaA;
		    vy[i] = valy + ay * deltaA;
		    vz[i] = valz + az * deltaA;

		    x_m1[i] = x[i];
		    y_m1[i] = y[i];
		    z_m1[i] = z[i];

		   	//x[i] = x + dt[i] * valx + ax * dt[i] * deltaB;
		    x[i] += dt[i] * valx + (vx[i] - valx) * dt[i] * deltaB / deltaA;
		    y[i] += dt[i] * valy + (vy[i] - valy) * dt[i] * deltaB / deltaA;
		    z[i] += dt[i] * valz + (vz[i] - valz) * dt[i] * deltaB / deltaA;

		    if(bbox.PBCx && x[i] < bbox.xmin) x[i] += (bbox.xmax-bbox.xmin);
		    else if(bbox.PBCx && x[i] > bbox.xmax) x[i] -= (bbox.xmax-bbox.xmin);
		    if(bbox.PBCy && y[i] < bbox.ymin) y[i] += (bbox.ymax-bbox.ymin);
		    else if(bbox.PBCy && y[i] > bbox.ymax) y[i] -= (bbox.ymax-bbox.ymin);
		    if(bbox.PBCz && z[i] < bbox.zmin) z[i] += (bbox.zmax-bbox.zmin);
		    else if(bbox.PBCz && z[i] > bbox.zmax) z[i] -= (bbox.zmax-bbox.zmin);

		    // Update the energy according to Adams-Bashforth (2nd order)
		    deltaA = 0.5 * dt[i] * dt[i] / dt_m1[i];
		    deltaB = dt[i] + deltaA;

		    u[i] += 0.5 * du[i] * deltaB - 0.5 * du_m1[i] * deltaA;

		    if(std::isnan(u[i]))
		    	printf("ERROR::UpdateQuantities(%d) internal energy: u %f du %f dB %f du_m1 %f dA %f\n", i, u[i], du[i], deltaB, du_m1[i], deltaA);

		    du_m1[i] = du[i];
		    dt_m1[i] = dt[i];
		}
	}


private:
	const int stabilizationTimesteps;
};

}

