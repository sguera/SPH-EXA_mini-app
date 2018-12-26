#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class Timestep
{
public:
	Timestep(const T chi = compute_3d_k(5.0), const T maxDtIncrease = 1.1) : chi(chi), maxDtIncrease(maxDtIncrease) {}

	void compute(const ArrayT &h, const ArrayT &c, const ArrayT &dt_m1, ArrayT &dt)
	{
		int n =  h.size();

		#pragma omp parallel for
		for(int i=0; i<n; i++)
		    dt[i] = chi * (h[i]/c[i]);

        T min = INFINITY;
        for(unsigned int i = 0; i < dt.size(); ++i)
            if(dt[i] < min)
                min = dt[i];

        min = std::min(min, maxDtIncrease * dt_m1[0]);

        #pragma omp parallel for
        for(unsigned int i=0; i<dt.size(); i++)
        	dt[i] = min;
	}

private:
	const T chi, maxDtIncrease;
};

}

