#pragma once

#include <vector>
#include <iostream>

#ifdef USE_MPI
    #include "mpi.h"
#endif

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class EnergyConservation
{
public:
	void compute(const std::vector<int> &clist, const ArrayT &u, const ArrayT &vx, const ArrayT &vy, const ArrayT &vz, const ArrayT &m, T &etot, T &ecin, T &eint)
	{
		int n = clist.size();

		T ecintmp = 0.0, einttmp = 0.0;
		#pragma omp parallel for reduction (+:ecintmp,einttmp)
        for(int pi=0; pi<n; pi++)
        {
            int i = clist[pi];

            T vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
            ecintmp += 0.5 * m[i] * vmod2;
            einttmp += u[i] * m[i]; 
        }

        #ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &ecintmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &einttmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        #endif

        ecin = ecintmp;
        eint = einttmp;
        etot = ecin + eint;
    }
};

}

