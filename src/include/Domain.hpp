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
	Domain(int ngmin, int ng0, int ngmax, bool PBCx = false, bool PBCy = false, bool PBCz = false, unsigned int bucketSize = 128) : 
		ngmin(ngmin), ng0(ng0), ngmax(ngmax), PBCx(PBCx), PBCy(PBCy), PBCz(PBCz), bucketSize(bucketSize) {}

	void computeBBox(const ArrayT &x, const ArrayT &y, const ArrayT &z, BBox<T> &bbox)
	{
		int n = x.size();

        if(!PBCx) bbox.xmin = INFINITY;
        if(!PBCx) bbox.xmax = -INFINITY;
        if(!PBCy) bbox.ymin = INFINITY;
        if(!PBCy) bbox.ymax = -INFINITY;
        if(!PBCz) bbox.zmin = INFINITY;
        if(!PBCz) bbox.zmax = -INFINITY;

        for(int i=0; i<n; i++)
        {
            if(!PBCx && x[i] < bbox.xmin) bbox.xmin = x[i];
            if(!PBCx && x[i] > bbox.xmax) bbox.xmax = x[i];
            if(!PBCy && y[i] < bbox.ymin) bbox.ymin = y[i];
            if(!PBCy && y[i] > bbox.ymax) bbox.ymax = y[i];
            if(!PBCz && z[i] < bbox.zmin) bbox.zmin = z[i];
            if(!PBCz && z[i] > bbox.zmax) bbox.zmax = z[i];
        }
	}

	template<class Dataset>
	void reorderParticles(Dataset &d)
	{
		d.reorder(*tree.ordering);
	    for(int i=0; i<d.n; i++)
	       (*tree.ordering)[i] = i;
	}

	void buildTree(const ArrayT &x, const ArrayT &y, const ArrayT &z, const ArrayT &h, BBox<T> &bbox)
	{
		computeBBox(x, y, z, bbox);
		tree.build(bbox, x, y, z, h, bucketSize);
	}

	void findNeighbors(const ArrayT &x, const ArrayT &y, const ArrayT &z, ArrayT &h, std::vector<std::vector<int>> &neighbors)
	{
		int n = x.size();

		#pragma omp parallel for
		for(int i=0; i<n; i++)
		{
			int ngi = neighbors[i].size();
			
			if(ngi > 0)
				h[i] = update_smoothing_length(ng0, ngi, h[i]);

	        do
	        {
	            neighbors[i].resize(0);
	            tree.findNeighbors(x[i], y[i], z[i], 2*h[i], ngmax, neighbors[i], PBCx, PBCy, PBCz);

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
	const bool PBCx, PBCy, PBCz;
	const unsigned int bucketSize;
};

}