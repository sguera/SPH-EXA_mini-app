#pragma once

#include <vector>
#include <cmath>
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class H : public TaskLoop
{
public:
	struct Params
	{
		Params(const int NV0 = 100, const T c0 = 7.0, const T exp = 1.0/3.0) : NV0(NV0), c0(c0), exp(exp) {}
		const int NV0;
		const T c0, exp;
	};
public:

	H(const std::vector<std::vector<int>> &neighbors, ArrayT &h, Params params = Params()) : 
		TaskLoop(h.size()), neighbors(neighbors), h(h), params(params) {}

	virtual void compute(int i)
	{
		const int NV0 = params.NV0;
	    const T c0 = params.c0;
	    const T exp = params.exp;

	    T ka = pow((1 + c0 * NV0 / neighbors[i].size()), exp);

	    h[i] = h[i] * 0.5 * ka;
	}

private:
	const std::vector<std::vector<int>> &neighbors;
	ArrayT &h;
	Params params;
};

}

