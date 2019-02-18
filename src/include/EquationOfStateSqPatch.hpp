#pragma once

#include <vector>

#include "kernels.hpp"

namespace sphexa
{

template<typename T, typename ArrayT = std::vector<T>>
class EquationOfStateSqPatch
{
public:
	void compute(const std::vector<int> &clist, const ArrayT &ro_0, const ArrayT &p_0, const ArrayT &ro, ArrayT &p, ArrayT &u, ArrayT &c)
	{
		int n = clist.size();

		// (ro_0 / 7.0) * c^2
		const T chi = (1000.0 / 7.0) * (35.0 * 35.0);
		//const T chi = (1.0 / 7.0) * (3500.0 * 3500.0);

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];

		    p[i] = chi * (pow(ro[i] / ro_0[i], 7.0) - 1.0) + p_0[i];
		    c[i] = 35.0;//c[i] = 35.0;
		    //u[i] = 1.0;//
		    u[i] = 1e-10;
		    // 1e7 per unit of mass (1e-3 or 1g)
		}
	}
};

}

