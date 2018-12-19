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
		sphexa::equation_of_state_square_patch(chi, p_0[i], ro_0[i], iteration, p[i], u[i], ro[i],  c[i]);
	}

	const ArrayT &ro_0, &p_0;
	const int &iteration;
	ArrayT &p, &u, &ro, &c;

	Params params;
};

}

