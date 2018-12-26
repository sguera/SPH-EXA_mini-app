#pragma once

#include <vector>

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class EnergyConservation
{
public:
	EnergyConservation() {}

	void compute(const ArrayT &u, const ArrayT &vx, const ArrayT &vy, const ArrayT &vz, const ArrayT &m, T &etot, T &ecin, T &eint)
	{
		int n = u.size();

		etot = ecin = eint = 0.0;
		#pragma omp parallel for reduction (+:etot,ecin,eint)
        for(int i=0; i<n; i++)
        {
            T vmod2 = 0.0;
            vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
            ecin += 0.5 * m[i] * vmod2;
            eint += u[i] * m[i]; 
        }
        etot += ecin + eint;
	}
};

}

