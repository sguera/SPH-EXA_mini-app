#pragma once

#include <vector>
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class Timestep : public TaskLoop
{
public:
	struct Params
	{
		Params(const T CHI = compute_3d_k(5.0), const T MAX_DT_INCREASE = 1.1) : CHI(CHI), MAX_DT_INCREASE(MAX_DT_INCREASE) {}
		const T CHI, MAX_DT_INCREASE;
	};
public:

	Timestep(const ArrayT &h, const ArrayT &c, const ArrayT &dt_m1, ArrayT &dt, Params params = Params()) : 
		TaskLoop(h.size()), h(h), c(c), dt_m1(dt_m1), dt(dt), params(params) {}

	virtual void compute(int i)
	{
		T CHI = params.CHI;
	    dt[i] = CHI * (h[i]/c[i]);
	}

	virtual void postprocess()
	{
		T MAX_DT_INCREASE = params.MAX_DT_INCREASE;

        T min = INFINITY;
        for(unsigned int i = 0; i < dt.size(); ++i)
            if(dt[i] < min)
                min = dt[i];

        min = std::min(min, MAX_DT_INCREASE * dt_m1[0]);

        for(unsigned int i=0; i<dt.size(); i++)
        	dt[i] = min;
	}

private:
	const ArrayT &h, &c, &dt_m1;
	ArrayT &dt;
	Params params;
};

}

