#pragma once

#include <vector>

#ifdef USE_MPI
    #include "mpi.h"
#endif

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class EnergyConservation
{
public:
	EnergyConservation() {}

	void compute(const std::vector<int> &clist, const ArrayT &u, const ArrayT &vx, const ArrayT &vy, const ArrayT &vz, const ArrayT &m, T &etot, T &ecin, T &eint)
	{
		int n = clist.size();

		etot = ecin = eint = 0.0;
		#pragma omp parallel for reduction (+:ecin,eint)
        for(int pi=0; pi<n; pi++)
        {
            int i = clist[pi];

            T vmod2 = 0.0;
            vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
            ecin += 0.5 * m[i] * vmod2;
            eint += u[i] * m[i]; 
        }

        #ifdef USE_MPI
            //MPI_Allreduce(MPI_IN_PLACE, &etot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &ecin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &eint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	   #endif

        etot = ecin + eint;
    }
};

}

