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
	Timestep(const T Kcour = 0.2, const T maxDtIncrease = 1.1) : Kcour(Kcour), maxDtIncrease(maxDtIncrease) {}

	void compute(const std::vector<int> &clist, const ArrayT &h, const ArrayT &c, const ArrayT &dt_m1, ArrayT &dt, T &ttot)
	{
		int n =  clist.size();

		// Time-scheme according to Press (2nd order)
		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];
		    dt[i] = Kcour * (h[i]/c[i]);
		}

        T min = INFINITY;
        for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];
            if(dt[i] < min)
                min = dt[i];
        }

        min = std::min(min, maxDtIncrease * dt_m1[0]);

        #ifdef USE_MPI
        	MPI_Allreduce(MPI_IN_PLACE, &min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        #endif

        #pragma omp parallel for
        for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];
        	dt[i] = min;
        }

        ttot += min;
	}

private:
	const T Kcour, maxDtIncrease;
};

}

