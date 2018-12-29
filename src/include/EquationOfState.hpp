#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{

template<typename T, typename ArrayT = std::vector<T>>
class EquationOfState
{
public:
	void compute(const std::vector<int> &clist, const ArrayT &ro, const ArrayT &mui, ArrayT &temp, ArrayT &u, ArrayT &p, ArrayT &c, ArrayT &cv)
	{
		int n = clist.size();

		const T R = 8.317e7, gamma = (5.0/3.0);

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];

			cv[i] = (gamma - 1) * R / mui[i];
		    temp[i] = u[i] / cv[i];
		    T tmp = u[i] * (gamma - 1);
		    p[i] = ro[i] * tmp;
		    c[i] = sqrt(tmp);

			if(std::isnan(c[i]) || std::isnan(cv[i]))
	        	printf("ERROR:equation_of_state c %f cv %f temp %f u %f p %f\n", c[i], cv[i], temp[i], u[i], p[i]);
		}
	}
};

}

