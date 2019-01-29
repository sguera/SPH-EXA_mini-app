#pragma once

#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{

template<typename T, typename ArrayT = std::vector<T>>
class EquationOfStateSqPatch
{
public:
	EquationOfStateSqPatch(const int stabilizationTimesteps = -1) : 
		stabilizationTimesteps(stabilizationTimesteps) {}

	void compute(const std::vector<int> &clist, const int iteration, ArrayT &ro_0, const ArrayT &p_0, ArrayT &ro, ArrayT &p, ArrayT &u, ArrayT &c)
	{
		int n = clist.size();

		const T chi = (1.0 / 7.0) * (3500.0 * 3500.0);

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];

		    if(iteration < stabilizationTimesteps)
		    {
		        p[i] = p_0[i];
		    }
		    else if(iteration == stabilizationTimesteps)
		    {
		        p[i] = p_0[i];
		        ro_0[i] = ro[i];
		    }
		    else
		    {
		        p[i] = chi * (pow((ro[i] / ro_0[i]), 7) - 1.0) + p_0[i];
		    }

		    c[i] = 3500.0;
		    u[i] = 1.0;
		}
	}

private:
	const int stabilizationTimesteps;
};

}

