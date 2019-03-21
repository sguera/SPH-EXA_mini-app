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

    void compute(const std::vector<int> &clist, const ArrayT &h, const ArrayT &c, const ArrayT &dt_m1, ArrayT &dt)
    {
        const int n = clist.size();

        T mini = INFINITY;

        #ifdef SPEC_OPENMP
        #pragma omp parallel for reduction(min:mini)
        #endif
        for(int pi=0; pi<n; pi++)
        {
            int i = clist[pi];
            // Time-scheme according to Press (2nd order)
            dt[i] = Kcour * (h[i]/c[i]);
            if(dt[i] < mini)
                mini = dt[i];
        }

        mini = std::min(mini, maxDtIncrease * dt_m1[0]);

        #ifdef USE_MPI
            MPI_Allreduce(MPI_IN_PLACE, &mini, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        #endif

        #ifdef SPEC_OPENMP
        #pragma omp parallel for
        #endif
        for(int pi=0; pi<n; pi++)
        {
            int i = clist[pi];
            dt[i] = mini;
        }
    }

private:
    const T Kcour, maxDtIncrease;
};

}

