#include <cstdlib>
#include <cstdio>
#include <vector>
#include <omp.h>

#include "common.hpp"

using namespace sphexa;

#define START omp_get_wtime()
#define STOP (double)(omp_get_wtime()-start)

using namespace std;

int main()
{
	int n = 0, ngmax = 0;
	double *x = NULL, *y = NULL, *z = NULL, *h = NULL;
	int *ng = NULL, *nvi = NULL;

	#ifdef EVRARD
		readfileEvrard("../../bigfiles/Evrard3D.bin", n, ngmax, x, y, z, h, ng, nvi);
	#elif SQPATCH
		readfileSquarePatch("../../bigfiles/squarepatch3D.bin", n, ngmax, x, y, z, h, ng, nvi);
	#endif

	double start;
	double xmin, xmax, ymin, ymax, zmin, zmax;
	computeBox(xmin, xmax, ymin, ymax, zmin, zmax, x, y, z, n);
	
	double buildAvg = 0.0;
	double findAvg = 0.0;
	double cleanAvg = 0.0;

	for(int repeat=0; repeat<=REPEAT; repeat++)
	{
		for(int i=0; i<n; i++)
			nvi[i] = 0;

		TREEINTERFACE tree;

		start = START;
		tree.setBox(xmin, xmax, ymin, ymax, zmin, zmax);
		#ifdef USE_H
			tree.build(n, x, y, z, h);
		#else
			tree.build(n, x, y, z);
		#endif
		buildAvg += STOP;

		start = START;
		#pragma omp parallel for schedule(guided)
		for(int i=0; i<n; i++)
			tree.findNeighbors(x[i], y[i], z[i], 2.0*h[i], ngmax, &ng[(long)i*ngmax], nvi[i], false, false, false);
		findAvg += STOP;

		start = START;
		tree.clean();
		cleanAvg += STOP;
	}

	buildAvg /= REPEAT;
	findAvg /= REPEAT;
	cleanAvg /= REPEAT;

	printf("%f %f %f\n", buildAvg, findAvg, cleanAvg);

	delete[] x;
	delete[] y;
	delete[] z;
	delete[] h;
	delete[] ng;
	delete[] nvi;

	return 0;
}
