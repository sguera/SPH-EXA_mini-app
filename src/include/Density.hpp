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

	template<class Dataset>
	void compute(const std::vector<int> &clist, Dataset &d)
	{
		const int n = clist.size();
		const int *c_clist = clist.data();

		const T *c_h = d.h.data();
		const T *c_m = d.m.data();
		const T *c_x = d.x.data();
		const T *c_y = d.y.data();
		const T *c_z = d.z.data();
		T *c_ro = d.ro.data();

		const int *c_neighbors = d.neighbors.data();
		const int *c_neighborsCount = d.neighborsCount.data();

		const BBox<T> bbox = d.bbox;
		const int ngmax = d.ngmax;
		
		#pragma omp target
		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			const int i = c_clist[pi];
			const int nn = c_neighborsCount[pi];

		    T roloc = 0.0;
		    c_ro[i] = 0.0;

		    // int converstion to avoid a bug that prevents vectorization with some compilers
		    for(int pj=0; pj<nn; pj++)
		    {
		    	const int j = c_neighbors[pi*ngmax+pj];

		        // later can be stores into an array per particle
		        T dist =  distancePBC(bbox, c_h[i], c_x[i], c_y[i], c_z[i], c_x[j], c_y[j], c_z[j]); //store the distance from each neighbor

		        // calculate the v as ratio between the distance and the smoothing length
		        T vloc = dist / c_h[i];
		        
		        #ifndef NDEBUG
			        if(vloc > 2.0+1e-6 || vloc < 0.0)
			        	printf("ERROR:Density(%d,%d) vloc %f -- x %f %f %f -- %f %f %f -- dist %f -- hi %f\n", i, j, vloc, c_x[i], c_y[i], c_z[i], c_x[j], c_y[j], c_z[j], dist, c_h[i]);
			    #endif

		        T value = wharmonic(vloc, c_h[i], sincIndex, K);
		        roloc += value * c_m[j];
		    }

		    c_ro[i] = roloc + c_m[i] * K/(c_h[i]*c_h[i]*c_h[i]);

		    #ifndef NDEBUG
			    if(std::isnan(c_ro[i]))
			    	printf("ERROR::Density(%d) density %f, position: (%f %f %f), h: %f\n", i, c_ro[i], c_x[i], c_y[i], c_z[i], c_h[i]);
		    #endif
		}
	}

private:
	const T sincIndex, K;
};

}
