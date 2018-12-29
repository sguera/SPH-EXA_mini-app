#pragma once

#include <vector>

#include "kernels.hpp"

#ifdef USE_MPI
	#include "mpi.h"
#endif

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class Timestep
{
public:
	Timestep(const T chi = compute_3d_k(5.0), const T maxDtIncrease = 1.1) : chi(chi), maxDtIncrease(maxDtIncrease) {}

	void compute(const std::vector<int> &clist, const ArrayT &h, const ArrayT &c, const ArrayT &dt_m1, ArrayT &dt)
	{
		int n =  clist.size();

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];
		    dt[i] = chi * (h[i]/c[i]);
		}

        T min = INFINITY;
        for(unsigned int i = 0; i < dt.size(); ++i)
            if(dt[i] < min)
                min = dt[i];

        min = std::min(min, maxDtIncrease * dt_m1[0]);

        #ifdef USE_MPI
        	MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        #endif

        #pragma omp parallel for
        for(unsigned int i=0; i<dt.size(); i++)
        	dt[i] = min;
	}

private:
	const T chi, maxDtIncrease;
};

}

