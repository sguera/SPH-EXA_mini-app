#pragma once

#include <vector>
#include "Task.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class EnergyConservation : public Task
{
public:
	EnergyConservation(const ArrayT &u, const ArrayT &vx, const ArrayT &vy, const ArrayT &vz, const ArrayT &m,
		T &etot, T &ecin, T &eint) : u(u), vx(vx), vy(vy), vz(vz), m(m), 
		etot(etot), ecin(ecin), eint(eint) {}

	virtual void compute() override
	{
		etot = ecin = eint = 0.0;
        for(unsigned int i=0; i<u.size(); i++)
        {
            T vmod2 = 0.0;
            vmod2 = vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i];
            ecin += 0.5 * m[i] * vmod2;
            eint += u[i] * m[i]; 
        }
        etot += ecin + eint;
	}

private:
	const ArrayT &u, &vx, &vy, &vz, &m;
	T &etot, &ecin, &eint;
};

}

