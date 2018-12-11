//         ms = task_loop(evrard.n, [&](int i)
//         {
//             computeTimestep(i, evrard);
//         });
//         //find the minimum timestep between the ones of each particle and use that one as new timestep
//         TimePoint start1 = Clock::now();
//         auto result = *std::min_element(evrard.timestep.begin(), evrard.timestep.end());
//         //int index = std::distance(evrard.timestep.begin(), it);
//         // double min = evrard.timestep[index];

//         //all particles have the same time-step so we just take the one of particle 0
//         double min = std::min(result, MAX_DT_INCREASE * evrard.timestep_m1[0]);

//         // double min = 10.0;
//         //int minIndex = evrard.n + 1;
//         // for (int i = 0; i < evrard.n; ++i){
//         //     if (evrard.timestep[i] < min){
//         //         min = evrard.timestep[i];
//         // //        minIndex = i;
//         //     }
//         // }
//         std::fill(evrard.timestep.begin(), evrard.timestep.end(), min);

//         //cout << "# Total Time (s) to compute the Timestep : " << ms << endl;
//         TimePoint stop1 = Clock::now();
//         cout << "# Total Time (s) to compute the Timestep : " << ms + duration_cast<duration<float>>(stop1-start1).count() << endl;

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