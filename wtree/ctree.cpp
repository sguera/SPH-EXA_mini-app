#include <omp.h>
#include <stdio.h>

#include "Tree.hpp"

#define START omp_get_wtime()
#define STOP (double)(omp_get_wtime()-start)

using namespace std;

extern "C" {
	static Tree ctree;

	void tree_set_box_c(const double *xmin, const double *xmax, const double *ymin, const double *ymax, const double *zmin, const double *zmax)
	{
		ctree.setBox(*xmin, *xmax, *ymin, *ymax, *zmin, *zmax);
	}

	void tree_build_c(const int *n, const double *x, const double *y, const double *z)
	{
		ctree.build(*n, x, y, z);
	}

	void tree_find_neighbors_c(const double xi, const double yi, const double zi, const double ri, const int *ngmax, int *ng, int *nvi, int *PBCx, int *PBCy, int *PBCz)
	{
		ctree.findNeighbors(xi, yi, zi, ri, *ngmax, ng, *nvi, *PBCx, *PBCy, *PBCz);
	}
}
