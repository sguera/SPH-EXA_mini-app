#pragma once

#include <cmath>
#include <vector>
#include <cassert>

#include "kernels.hpp"

namespace sphexa
{

template<typename T, typename ArrayT = std::vector<T>>
class Density
{
public:
	Density(T sincIndex = 5.0, T K = compute_3d_k(5.0)) : sincIndex(sincIndex), K(K) {}

	void compute(const std::vector<int> &clist, const BBox<T> &bbox, const std::vector<std::vector<int>> &neighbors, const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h, const ArrayT &m, ArrayT &ro)
	{
		const int n = clist.size();

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			const int i = clist[pi];
			const int nn = (int)neighbors[pi].size();
		
		    T roloc = 0.0;
		    ro[i] = 0.0;

		    // int converstion to avoid a bug that prevents vectorization with some compilers
		    for(int pj=0; pj<nn; pj++)
		    {
		    	const int j = neighbors[pi][pj];

		        // later can be stores into an array per particle
		        T dist =  distancePBC(bbox, h[i], x[i], y[i], z[i], x[j], y[j], z[j]); //store the distance from each neighbor

		        // calculate the v as ratio between the distance and the smoothing length
		        T vloc = dist / h[i];
		        
		        #ifndef NDEBUG
			        if(vloc > 2.0+1e-6 || vloc < 0.0)
			        	printf("ERROR:Density(%d,%d) vloc %f -- x %f %f %f -- %f %f %f -- dist %f -- hi %f\n", i, j, vloc, x[i], y[i], z[i], x[j], y[j], z[j], dist, h[i]);
			    #endif

		        T value = wharmonic(vloc, h[i], sincIndex, K);
		        roloc += value * m[j];
		    }

		    ro[i] = roloc + m[i] * K/(h[i]*h[i]*h[i]);

		    #ifndef NDEBUG
			    if(std::isnan(ro[i]))
			    	printf("ERROR::Density(%d) density %f, position: (%f %f %f), h: %f\n", i, ro[i], x[i], y[i], z[i], h[i]);
		    #endif
		}
	}

private:
	const T sincIndex, K;
};

}

