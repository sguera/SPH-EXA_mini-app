#pragma once

#include <vector>
#include <iostream>

#ifdef USE_MPI
    #include "mpi.h"
#endif

namespace sphexa
{
namespace sph
{
	template<typename T, class Dataset>
    void computeTotalEnergy(const std::vector<int> &l, Dataset &d)
    {
        const int n = l.size();
        const int *clist = l.data();

        const T *u =  d.u.data();
        const T *vx =  d.vx.data();
        const T *vy =  d.vy.data();
        const T *vz =  d.vz.data();
        const T *m =  d.m.data();
        T &etot = d.etot;
        T &ecin = d.ecin;
        T &eint = d.eint;

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
}
}

