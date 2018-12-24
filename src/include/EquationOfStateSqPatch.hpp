#pragma once

#include <cmath>
#include <vector>

#include "common.hpp"
#include "kernels.hpp"
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class EquationOfStateSquarePatch : public TaskLoop
{
public:
	struct Params
	{
		Params(const T chi = (1 / 7) * (3500 * 3500)) : 
			chi(chi) {}
		const T chi;
	};
public:

	EquationOfStateSquarePatch(const ArrayT &ro_0, const ArrayT &p_0, const int &iteration,
		ArrayT &p, ArrayT &u, ArrayT &ro, ArrayT &c, Params params = Params()) : 
			TaskLoop(ro_0.size()), ro_0(ro_0), p_0(p_0), iteration(iteration), p(p), u(u), ro(ro), c(c), params(params) {}

	virtual void compute(int i)
	{
		T chi = params.chi;

	    if(iteration < 15)
	    {
	        p[i] = p_0[i];
	    }
	    else if(iteration == 15)
	    {
	        p[i] = p_0[i];
	        ro[i] = ro_0[i];
	    }
	    else
	    {
	        p[i] = chi * (pow((ro[i] / ro_0[i]), 7) - 1) + p_0[i];
	    }

	    c[i] = 3500.0;
	    u[i] = 1.0;
	}

	const ArrayT &ro_0, &p_0;
	const int &iteration;
	ArrayT &p, &u, &ro, &c;

	Params params;
};

}

