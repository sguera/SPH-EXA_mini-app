#include <cstdlib>
#include <cstdio>
#include <vector>
#include <omp.h>

#include "../../src/tree/BroadTree.hpp"
#include "../../src/tree/KdTree.hpp"
#include "../../src/tree/Octree.hpp"
#include "../../src/tree/NNFTree.hpp"

using namespace sphexa;

#define START omp_get_wtime()
#define STOP (double)(omp_get_wtime()-start)

using namespace std;

void computeBox(double &xmin, double &xmax, double &ymin, double &ymax, double &zmin, double &zmax, const double *x, const double *y, const double *z, const int n)
{
	xmin = INFINITY;
	xmax = -INFINITY;
	ymin = INFINITY;
	ymax = -INFINITY;
	zmin = INFINITY;
	zmax = -INFINITY;
	for(int i=0; i<n; i++)
	{
		if(x[i] < xmin) xmin = x[i];
		if(x[i] > xmax) xmax = x[i];
		if(y[i] < ymin) ymin = y[i];
		if(y[i] > ymax) ymax = y[i];
		if(z[i] < zmin) zmin = z[i];
		if(z[i] > zmax) zmax = z[i];
	}
}

void readfileEvrard(const char *filename, int &n, int &ngmax, double *&x, double *&y, double *&z, double *&h, int *&ng, int *&nvi)
{
	n = 1000000;
	ngmax = 1510;
	
	FILE *f = fopen(filename, "rb");
	if(f)
	{
		x = new double[n];
		y = new double[n];
		z = new double[n];
		h = new double[n];

		nvi = new int[n]; 
		ng = new int[(long)n*ngmax];

		fread(x, sizeof(double), n, f);
		fread(y, sizeof(double), n, f);
		fread(z, sizeof(double), n, f);
		fread(h, sizeof(double), n, f);
		fclose(f);
	}
	else
	{
		printf("Error opening file.\n");
		exit(1);
	}
}

void readfileSquarePatch(const char *filename, int &n, int &ngmax, double *&x, double *&y, double *&z, double *&h, int *&ng, int *&nvi)
{
	n = 10077696;
	ngmax = 550;

	FILE *f = fopen(filename, "rb");
	if(f)
	{
		x = new double[n];
		y = new double[n];
		z = new double[n];
		h = new double[n];

		nvi = new int[n]; 
		ng = new int[(long)n*ngmax];

		int u1;
		double gm, gh, gd;
		fread(&u1, sizeof(int), 1, f);
		fread(&gm, sizeof(double), 1, f);
		fread(&gh, sizeof(double), 1, f);
		fread(&gd, sizeof(double), 1, f);

		fread(x, sizeof(double), n, f);
		fread(y, sizeof(double), n, f);
		fread(z, sizeof(double), n, f);

		for(int i=0; i<n; i++)
			h[i] = gh*1.5916455;

		fclose(f);
	}
	else
	{
		printf("Error opening file.\n");
		exit(1);
	}
}

int main()
{
	int n = 0, ngmax = 0;
	double *x = NULL, *y = NULL, *z = NULL, *h = NULL;
	int *ng = NULL, *nvi = NULL;

	#ifdef EVRARD
		readfileEvrard("../../bigfiles/evrard_1M.bin", n, ngmax, x, y, z, h, ng, nvi);
	#elif SQPATCH
		readfileSquarePatch("../../bigfiles/squarepatch3D.bin", n, ngmax, x, y, z, h, ng, nvi);
	#endif
	
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

	tree.clean();

	delete[] x;
	delete[] y;
	delete[] z;
	delete[] h;
	delete[] ng;
	delete[] nvi;

	return 0;
}
