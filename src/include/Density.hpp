#pragma once

#include <cmath>
#include <vector>

#include "kernels.hpp"

namespace sphexa
{

template<typename T>
inline T distance(const BBox<T> &bbox, const T hi, const T x1, const T y1, const T z1, const T x2, const T y2, const T z2)
{
    T xx = x1 - x2;
    T yy = y1 - y2;
    T zz = z1 - z2;

    if(xx > 2*hi) xx -= (bbox.xmax-bbox.xmin);
    else if(xx < -2*hi) xx += (bbox.xmax-bbox.xmin);
    
    if(yy > 2*hi) yy -= (bbox.ymax-bbox.ymin);
    else if(yy < -2*hi) yy += (bbox.ymax-bbox.ymin);
    
    if(zz > 2*hi) zz -= (bbox.zmax-bbox.zmin);
    else if(zz < -2*hi) zz += (bbox.zmax-bbox.zmin);

    return sqrt(xx*xx + yy*yy + zz*zz);
}

template<typename T, typename ArrayT = std::vector<T>>
class Density
{
public:
	Density(T sincIndex = 5.0, T K = compute_3d_k(5.0)) : sincIndex(sincIndex), K(K) {}

	void compute(const std::vector<int> &clist, const BBox<T> &bbox, const std::vector<std::vector<int>> &neighbors, const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h, const ArrayT &m, ArrayT &ro)
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
		        T dist =  distance(bbox, h[i], x[i], y[i], z[i], x[nid], y[nid], z[nid]); //store the distance from each neighbor

		        // calculate the v as ratio between the distance and the smoothing length
		        T vloc = dist / h[i];
		        
		        //assert(vloc<=2 && vloc >= 0);
		        T value = wharmonic(vloc, h[i], sincIndex, K);
		        roloc += value * m[nid];
		    }

		    ro[i] = roloc + m[i] * K/(h[i]*h[i]*h[i]);

		    if(std::isnan(ro[i]))
		    	printf("ERROR::Density(%d) density %f, position: (%f %f %f), h: %f\n", i, ro[i], x[i], y[i], z[i], h[i]);
		}
	}

private:
	const T sincIndex, K;
};

}

