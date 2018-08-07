#include <cstdlib>
#include <cstdio>
#include <vector>
#include <omp.h>

#include "Tree.hpp"

#define START omp_get_wtime()
#define STOP (double)(omp_get_wtime()-start)

using namespace std;

void readfileEvrard(const char *filename, int n, double *x, double *y, double *z, double *h)
{
	FILE *f = fopen(filename, "rb");
	if(f)
	{
		fread(x, sizeof(double), n, f);
		fread(y, sizeof(double), n, f);
		fread(z, sizeof(double), n, f);
		fread(h, sizeof(double), n, f);
		fclose(f);
	}
	else
		printf("Error opening file.\n");
}

void readfileSquarePatch(const char *filename, int n, double *x, double *y, double *z, double *h)
{
	FILE *f = fopen(filename, "rb");
	if(f)
	{
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
		printf("Error opening file.\n");
}

int main()
{
	int n = 1000000;
	//int n = 10077696;
	int ngmax = 550;

	double *x = new double[n];
	double *y = new double[n];
	double *z = new double[n];
	double *h = new double[n];

	int *nvi = new int[n]; 
	int *ng = new int[(long)n*ngmax];
	for(int i=0; i<n; i++)
		nvi[i] = 0;
	
	double start, tbuild, tfind;

	readfileEvrard("../bigfiles/evrard_1M.bin", n, x, y, z, h);
	//readfileSquarePatch("../bigfiles/squarepatch3D.bin", n, x, y, z, h);

	double xmin = 1000, xmax = -1000, ymin = 1000, ymax = -1000, zmin = 1000, zmax = -1000;
	for(int i=0; i<n; i++)
	{
		if(x[i] < xmin) xmin = x[i];
		if(x[i] > xmax) xmax = x[i];
		if(y[i] < ymin) ymin = y[i];
		if(y[i] > ymax) ymax = y[i];
		if(z[i] < zmin) zmin = z[i];
		if(z[i] > zmax) zmax = z[i];
	}

	printf("Domain x[%f %f]\n", xmin, xmax);
	printf("Domain y[%f %f]\n", ymin, ymax);
	printf("Domain z[%f %f]\n", zmin, zmax);
  
	Tree tree;

	start = START;
	tree.init(xmin, xmax, ymin, ymax, zmin, zmax);
	tree.build(n, x, y, z);
	tbuild = STOP;

	printf("CELLS: %d\n", tree.cellCount());
	printf("BUILD TIME: %f\n", tbuild);

	start = START;
	#pragma omp parallel for schedule(runtime)
	for(int i=0; i<n; i++)
		tree.findNeighbors(i, x, y, z, 2.0*h[i], ngmax, &ng[(long)i*ngmax], nvi[i], false, false, true);
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