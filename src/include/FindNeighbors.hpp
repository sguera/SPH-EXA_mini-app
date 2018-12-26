#pragma once

#include <vector>

namespace sphexa
{

template<class Tree, typename T = double, typename ArrayT = std::vector<T>>
class FindNeighbors
{
public:
	struct Params
	{
		Params(const int ngmin = 50, const int ng0 = 100, const int ngmax = 150) : ngmin(ngmin), ng0(ng0), ngmax(ngmax) {}
		const int ngmin, ng0, ngmax;
	};
public:
	
	FindNeighbors(const Tree &tree, std::vector<std::vector<int>> &neighbors, ArrayT &h, const Params params = Params()) : 
		tree(tree), neighbors(neighbors), h(h), params(params) {}

	void compute(int n)
	{
		#pragma omp parallel for
		for(int i=0; i<n; i++)
		{
			int ngi = neighbors[i].size();
			
			if(ngi > 0)
				h[i] = update_smoothing_length(params.ng0, ngi, h[i]);

	        do
	        {
	            neighbors[i].resize(0);
	            tree.findNeighbors(i, neighbors[i]);

	            ngi = neighbors[i].size();

	            if(ngi < params.ngmin || ngi > params.ngmax)
	                h[i] = update_smoothing_length(params.ng0, ngi, h[i]);
	        }
	        while(ngi < params.ngmin || ngi > params.ngmax);
	    }
	}

private:
	const Tree &tree;
	std::vector<std::vector<int>> &neighbors;
	ArrayT &h;

	Params params;
};



}