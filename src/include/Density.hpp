#pragma once

#include <cmath>
#include <vector>

#include "kernels.hpp"

namespace sphexa
{

template<typename T, typename ArrayT = std::vector<T>>
class Density
{
public:
	Density(T K = compute_3d_k(5.0)) : K(K) {}

	void compute(const std::vector<int> &clist, const std::vector<std::vector<int>> &neighbors, const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h, const ArrayT &m, ArrayT &ro)
	{
		int n = clist.size();

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];

		    T roloc = 0.0;
		    ro[i] = 0.0;

		    for(unsigned int j=0; j<neighbors[pi].size(); j++)
		    {
		        // retrive the id of a neighbor
		        int nid = neighbors[pi][j];
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

		    if(std::isnan(ro[i]))
		    {
		    	printf("ERROR::Density(%d) density %f, position: (%f %f %f), h: %f\n", i, ro[i], x[i], y[i], z[i], h[i]);
		    }
		}
	}

private:
	const double K;
};

}

