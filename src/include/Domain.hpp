#pragma once

namespace sphexa
{

template<typename T>
static inline T update_smoothing_length(const int ng0, const int ngi, const T hi)
{
    const T c0 = 7.0;
    const T exp = 1.0/3.0;

    T ka = pow((1.0 + c0 * ng0 / ngi), exp);

    return hi * 0.5 * ka;
}

template<typename T, class Tree = Octree<T>, class ArrayT = std::vector<T>>
class Domain
{
public:
	Domain(int ngmin, int ng0, int ngmax, unsigned int bucketSize = 128) : 
		ngmin(ngmin), ng0(ng0), ngmax(ngmax), bucketSize(bucketSize) {}

	void computeBBox(const ArrayT &x, const ArrayT &y, const ArrayT &z, BBox<T> &bbox)
	{
		int n = x.size();

        if(!bbox.PBCx) bbox.xmin = INFINITY;
        if(!bbox.PBCx) bbox.xmax = -INFINITY;
        if(!bbox.PBCy) bbox.ymin = INFINITY;
        if(!bbox.PBCy) bbox.ymax = -INFINITY;
        if(!bbox.PBCz) bbox.zmin = INFINITY;
        if(!bbox.PBCz) bbox.zmax = -INFINITY;

        for(int i=0; i<n; i++)
        {
            if(!bbox.PBCx && x[i] < bbox.xmin) bbox.xmin = x[i];
            if(!bbox.PBCx && x[i] > bbox.xmax) bbox.xmax = x[i];
            if(!bbox.PBCy && y[i] < bbox.ymin) bbox.ymin = y[i];
            if(!bbox.PBCy && y[i] > bbox.ymax) bbox.ymax = y[i];
            if(!bbox.PBCz && z[i] < bbox.zmin) bbox.zmin = z[i];
            if(!bbox.PBCz && z[i] > bbox.zmax) bbox.zmax = z[i];
        }
	}

    void reorderSwap(const std::vector<int> &ordering, ArrayT &data)
    {
        std::vector<T> tmp(ordering.size());
        for(unsigned int i=0; i<ordering.size(); i++)
            tmp[i] = data[ordering[i]];
        tmp.swap(data);
    }

	void reorder(std::vector<ArrayT*> &data)
    {
        for(unsigned int i=0; i<data.size(); i++)
            reorderSwap(*tree.ordering, *data[i]);
    }

	void buildTree(const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h, BBox<T> &bbox)
	{
		computeBBox(x, y, z, bbox);
		tree.build(bbox, x, y, z, h, bucketSize);
	}

	void findNeighbors(const std::vector<int> &clist, const BBox<T> &bbox, const ArrayT &x, const ArrayT &y, const ArrayT &z, ArrayT &h, std::vector<std::vector<int>> &neighbors)
	{
		int n = clist.size();
		neighbors.resize(n);

		#pragma omp parallel for
		for(int pi=0; pi<n; pi++)
		{
			int i = clist[pi];
			int ngi = neighbors[i].size();
			
			if(ngi > 0)
				h[i] = update_smoothing_length(ng0, ngi, h[i]);

	        do
	        {
	            neighbors[i].resize(0);
	            tree.findNeighbors(x[i], y[i], z[i], 2*h[i], ngmax, neighbors[i], bbox.PBCx, bbox.PBCy, bbox.PBCz);

	            ngi = neighbors[i].size();

	            if(ngi < ngmin || ngi > ngmax)
	                h[i] = update_smoothing_length(ng0, ngi, h[i]);
	        }
	        while(ngi < ngmin || ngi > ngmax);
	    }
	}

private:
	Tree tree;
	const int ngmin, ng0, ngmax;
	const unsigned int bucketSize;
};

}