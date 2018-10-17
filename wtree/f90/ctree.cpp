#include <omp.h>
#include <stdio.h>

#include "Tree.hpp"

#define START omp_get_wtime()
#define STOP (double)(omp_get_wtime()-start)

using namespace std;

extern "C" {
	static Tree ctree;

	void tree_build_c(const int *n, const double *xmin, const double *xmax, const double *ymin, const double *ymax, const double *zmin, const double *zmax, const double *x, const double *y, const double *z)
	{
		ctree.init(*xmin, *xmax, *ymin, *ymax, *zmin, *zmax);
		ctree.build(*n, x, y, z);
	}

	void tree_find_neighbors_c(const int *i, const double *x, const double *y, const double *z, const double *r, const int *ngmax, int *ng, int *nvi, int *PBCx, int *PBCy, int *PBCz)
	{
		ctree.findNeighbors(*i-1, x, y, z, *r, *ngmax, ng, *nvi, *PBCx, *PBCy, *PBCz);
	}
}
