#pragma once

#include <cmath>
#include <vector>

#include "common.hpp"
#include "kernels.hpp"
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class EOS : public TaskLoop
{
public:
	struct Params
	{
		Params(const T R = 8.317e7, const T gamma = (5.0/3.0)) : 
			R(R), gamma(gamma) {}
		const T R, gamma;
	};
public:

	EOS(const ArrayT &ro, const ArrayT &u, const ArrayT &mui, 
		ArrayT &p, ArrayT &temp, ArrayT &c, ArrayT &cv, Params params = Params()) : 
			TaskLoop(ro.size()), ro(ro), u(u), mui(mui), p(p), temp(temp), c(c), cv(cv), params(params) {}

	virtual void compute(int i)
	{
		T R = params.R;
		T gamma = params.gamma;
		sphexa::eos(R, gamma, ro[i], u[i], mui[i], p[i], temp[i], c[i], cv[i]);
	}

	const ArrayT &ro, &u, &mui;
	ArrayT &p, &temp, &c, &cv;

	Params params;
};

}

