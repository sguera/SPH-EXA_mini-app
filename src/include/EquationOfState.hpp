#pragma once

#include <vector>

#include "kernels.hpp"
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class EquationOfState : public TaskLoop
{
public:
	struct Params
	{
		Params(const T R = 8.317e7, const T gamma = (5.0/3.0)) : 
			R(R), gamma(gamma) {}
		const T R, gamma;
	};
public:

	EquationOfState(const ArrayT &ro, const ArrayT &u, const ArrayT &mui, 
		ArrayT &p, ArrayT &temp, ArrayT &c, ArrayT &cv, Params params = Params()) : 
			TaskLoop(ro.size()), ro(ro), u(u), mui(mui), p(p), temp(temp), c(c), cv(cv), params(params) {}

	virtual void compute(int i) override
	{
		T R = params.R;
		T gamma = params.gamma;

		cv[i] = (gamma - 1) * R / mui[i];
	    temp[i] = u[i] / cv[i];
	    T tmp = u[i] * (gamma - 1);
	    p[i] = ro[i] * tmp;
	    c[i] = sqrt(tmp);

		if(std::isnan(c[i]) || std::isnan(cv[i]))
        	printf("ERROR:equation_of_state c %f cv %f temp %f u %f p %f\n", c[i], cv[i], temp[i], u[i], p[i]);
	}

	const ArrayT &ro, &u, &mui;
	ArrayT &p, &temp, &c, &cv;

	Params params;
};

}

