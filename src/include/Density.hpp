#pragma once

#include <cmath>
#include <vector>

#include "common.hpp"
#include "kernels.hpp"
#include "TaskLoop.hpp"

namespace sphexa
{

template<typename T = double, typename ArrayT = std::vector<T>>
class Density : public TaskLoop
{
public:
	struct Params
	{
		Params(const T K = compute_3d_k(5.0)) : K(K) {}
		const T K;
	};
public:

	Density(const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h, 
		const ArrayT &m, const std::vector<std::vector<int>> &neighbors,
		ArrayT &ro, Params params = Params()) : 
			TaskLoop(x.size()), x(x), y(y), z(z), h(h), m(m), neighbors(neighbors), ro(ro), params(params) {}

	virtual void compute(int i) override
	{
		T K = params.K;

	    T roloc = 0.0;
	    ro[i] = 0.0;

	    for(unsigned int j=0; j<neighbors[i].size(); j++)
	    {
	        // retrive the id of a neighbor
	        int nid = neighbors[i][j];
	        if(nid == i) continue;

	        // later can be stores into an array per particle
	        T dist =  distance(x[i], y[i], z[i], x[nid], y[nid], z[nid]); //store the distance from each neighbor

	        // calculate the v as ratio between the distance and the smoothing length
	        T vloc = dist / h[i];
	        
	        //assert(vloc<=2);
	        T value = wharmonic(vloc, h[i], K);
	        roloc += value * m[nid];
	    }

	    ro[i] = roloc + m[i] * K/(h[i]*h[i]*h[i]);
	}

	const ArrayT &x, &y, &z, &h, &m;
	const std::vector<std::vector<int>> &neighbors;
	ArrayT &ro;

	Params params;
};

}

