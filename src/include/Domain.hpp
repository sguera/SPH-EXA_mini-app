#pragma once

#include <vector>

#ifdef USE_MPI
	#include "mpi.h"
#endif

#include "config.hpp"

namespace sphexa
{

template<typename T, class Tree = Octree<T>>
class Domain
{
public:
	Domain(int ngmin, int ng0, int ngmax, unsigned int bucketSize = 128) : 
		ngmin(ngmin), ng0(ng0), ngmax(ngmax), bucketSize(bucketSize) {}

    void reorderSwap(const std::vector<int> &ordering, Array<T> &data)
    {
        std::vector<T> tmp(ordering.size());
        for(unsigned int i=0; i<ordering.size(); i++)
            tmp[i] = data[ordering[i]];
        tmp.swap(data);
    }

	void reorder(std::vector<Array<T>*> &data)
    {
        for(unsigned int i=0; i<data.size(); i++)
            reorderSwap(*tree.ordering, *data[i]);
    }

	void findNeighbors(const std::vector<int> &clist, const BBox<T> &bbox, const Array<T> &x, const Array<T> &y, const Array<T> &z, Array<T> &h, std::vector<std::vector<int>> &neighbors)
	{
		int n = clist.size();
		neighbors.resize(n);

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];

            neighbors[pi].resize(0);
            tree.findNeighbors(i, x[i], y[i], z[i], 2*h[i], ngmax, neighbors[pi], bbox.PBCx, bbox.PBCy, bbox.PBCz);
            
            #ifndef NDEBUG
	            if(neighbors[pi].size() == 0)
	            	printf("ERROR::FindNeighbors(%d) x %f y %f z %f h = %f ngi %zu\n", i, x[i], y[i], z[i], h[i], neighbors[pi].size());
	        #endif
	    }
	}

	long long int neighborsSum(const std::vector<int> &clist, const std::vector<std::vector<int>> &neighbors)
	{
	    long long int sum = 0;
	    #pragma omp parallel for reduction (+:sum)
	    for(unsigned int i=0; i<clist.size(); i++)
	        sum += neighbors[i].size();

	    #ifdef USE_MPI
	        MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
	    #endif

	    return sum;
	}

	inline T update_smoothing_length(const int ng0, const int ngi, const T hi)
	{
	    const T c0 = 7.0;
	    const T exp = 1.0/3.0;

	    T ka = pow((1.0 + c0 * ng0 / ngi), exp);

	    return hi * 0.5 * ka;
	}

	void updateSmoothingLength(const std::vector<int> &clist, const std::vector<std::vector<int>> &neighbors, Array<T> &h)
	{
		const T c0 = 7.0;
		const T exp = 1.0/3.0;

		int n = clist.size();

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];
			int ngi = std::max((int)neighbors[pi].size(),1);
			
		    h[i] = h[i] * 0.5 * pow((1.0 + c0 * ng0 / ngi), exp);

		    #ifndef NDEBUG
		        if(std::isinf(h[i]) || std::isnan(h[i]))
		        	printf("ERROR::h(%d) ngi %d h %f\n", i, ngi, h[i]);
		    #endif
	    }
	}

    inline void buildTree(const BBox<T> &bbox, const Array<T> &x, const Array<T> &y, const Array<T> &z, const Array<T> &h)
    {
    	tree.build(bbox, x, y, z, h, bucketSize);
    }

    virtual void build(const std::vector<int> &clist, const Array<T> &x, const Array<T> &y, const Array<T> &z, const Array<T> &h, BBox<T> &bbox)
	{
		bbox.compute(clist, x, y, z);
		buildTree(bbox, x, y, z, h);
	}

private:
	Tree tree;
	const int ngmin, ng0, ngmax;
	const unsigned int bucketSize;
};

}
