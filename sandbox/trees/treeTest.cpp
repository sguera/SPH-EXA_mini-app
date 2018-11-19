#include <cstdlib>
#include <cstdio>
#include <vector>
#include <omp.h>

#include "common.hpp"

using namespace sphexa;

#define START omp_get_wtime()
#define STOP (double)(omp_get_wtime()-start)

// giving some context:
#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)
#define CURRENT_TREEINTERFACE (TO_STR(TREEINTERFACE))

#ifdef EVRARD
#define CURRENT_TESTCASE "EVRARD"
#endif

#ifdef SQPATCH
#define CURRENT_TESTCASE "SQPATCH"
#endif

#ifndef CURRENT_TESTCASE
#define CURRENT_TESTCASE "SQPATCH"
#endif

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
    cout << "CURRENT_TESTCASE:" << CURRENT_TESTCASE << endl;
    cout << "CURRENT_TREEINTERFACE:" << CURRENT_TREEINTERFACE << endl;
	
	double start, tbuild, tfind;
	double xmin, xmax, ymin, ymax, zmin, zmax;
	computeBox(xmin, xmax, ymin, ymax, zmin, zmax, x, y, z, n);

	printf("Domain x[%f %f]\n", xmin, xmax);
	printf("Domain y[%f %f]\n", ymin, ymax);
	printf("Domain z[%f %f]\n", zmin, zmax);
	
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
	tbuild = STOP;

	//printf("CELLS: %d\n", tree.cellCount());
	//printf("BUCKETS: %d\n", tree.bucketCount());
	printf("BUILD TIME: %f\n", tbuild);

	start = START;
	#pragma omp parallel for schedule(guided)
	for(int i=0; i<n; i++)
		tree.findNeighbors(x[i], y[i], z[i], 2.0*h[i], ngmax, &ng[(long)i*ngmax], nvi[i], false, false, false);
	//tree.findNeighbors(x[0], y[0], z[0], 2.0*h[0], ngmax, &ng[0], nvi[0], false, false, false);
	tfind = STOP;
	
	printf("FIND TIME: %f\n", tfind);

	long int sum = 0;
	for(int i=0; i<n; i++)
		sum += nvi[i];

	printf("Total neighbors found: %lu\n", sum);
	printf("Total cells: %d\n", tree.cellCount());
	printf("Total buckets: %d\n", tree.bucketCount());

	tree.clean();

	delete[] x;
	delete[] y;
	delete[] z;
	delete[] h;
	delete[] ng;
	delete[] nvi;

	return 0;
}
