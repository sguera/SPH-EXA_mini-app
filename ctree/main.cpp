#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <omp.h>

#include "Tree.hpp"

#define START chrono::high_resolution_clock::now(); //omp_get_wtime()
#define STOP chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now()-start).count(); //(double)(omp_get_wtime()-start)

using namespace std;

void readfileEvrard(const char *filename, int &n, int &ngmax, double *&x, double *&y, double *&z, double *&h)
{
	n = 1000000;
	ngmax = 150;

	x = new double[n];
	y = new double[n];
	z = new double[n];
	h = new double[n];

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

void readfileSquarePatch(const char *filename, int &n, int &ngmax, double *&x, double *&y, double *&z, double *&h)
{
	n = 10077696;
	ngmax = 550;

	x = new double[n];
	y = new double[n];
	z = new double[n];
	h = new double[n];

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

void reorder(const int n, double *&x, int *ordering)
{
	double *xtmp = new double[n];
	for(int i=0; i<n; i++)
		xtmp[i] = x[ordering[i]];
	swap(x, xtmp);
}

int main(int argc, char **argv)
{
	int n = 0;
	int ngmax = 0;

	double *x, *y, *z, *h;
	
	double tbuild, tfind;
	chrono::time_point<chrono::high_resolution_clock> start;

	bool PBCx = false, PBCy = false, PBCz = false;

	if(argc > 1 && atoi(argv[1]) == 0)
		readfileEvrard("../bigfiles/evrard_1M.bin", n, ngmax, x, y, z, h);
	else if(argc > 1 && atoi(argv[1]) == 1)
	{
		readfileSquarePatch("../bigfiles/squarepatch3D.bin", n, ngmax, x, y, z, h);
		PBCz = true;
	}
	else
		readfileEvrard("../bigfiles/evrard_1M.bin", n, ngmax, x, y, z, h);
	// else
	// {
	// 	printf("%s input(0: Evrard1M, 1: SquarePatch10M) MAXP(32) TREE(1), RATIO(0.5)\n", argv[0]);
	// 	return 0;
	// }

	if(argc > 2)
		TREE = atoi(argv[2]);
	else
		TREE = 1;

	if(argc > 3)
		MAXP = atoi(argv[3]);
	else
		MAXP = 32;

	if(argc > 4)
		BLOCK_SIZE = atof(argv[4]);
	else
		BLOCK_SIZE = 8;

	if(argc > 5)
		RATIO = atof(argv[5]);
	else
		RATIO = 0.5;

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

	int *ordering = 0;

	start = START;
	tree.setBox(xmin, xmax, ymin, ymax, zmin, zmax);
	tree.buildSort(n, x, y, z, &ordering);
	tbuild = STOP;

	reorder(n, x, ordering);
	reorder(n, y, ordering);
	reorder(n, z, ordering);
	reorder(n, h, ordering);

	for(int i=0; i<n; i++)
		ordering[i] = i;

	printf("CELLS: %d\n", tree.cellCount());
	printf("BUCKETS: %d\n", tree.bucketCount());
	printf("BUILD TIME: %f\n", tbuild);


	int *nvi = new int[n]; 
	int *ng = new int[(long)n*ngmax];
	for(int i=0; i<n; i++)
		nvi[i] = 0;

	vector<double> thtime(omp_get_max_threads());

	start = START;
	#pragma omp parallel for schedule(static)
	for(int i=0; i<n; i++)
		tree.findNeighbors(x[i], y[i], z[i], 2.0*h[i], ngmax, &ng[(long)i*ngmax], nvi[i], PBCx, PBCy, PBCz);
	tfind = STOP;

	printf("FIND TIME: %f\n", tfind);

	// for(unsigned int i=0; i<thtime.size(); i++)
	// 	printf("%d: %f\n", i, thtime[i]);

	long int sum = 0;
	for(int i=0; i<n; i++)
		sum += nvi[i];

	printf("Total neighbors found: %ld\n", sum);

	// FILE *ftree = 0;

	// if(TREE == 0)
	// {
	// 	if(argc > 1 && atoi(argv[1]) == 0)
	// 		ftree = fopen("trees/octree-evrard-2d.txt", "w");
	// 	else if(argc > 1 && atoi(argv[1]) == 1)
	// 		ftree = fopen("trees/octree-square-2d.txt", "w");
	// }

	// if(TREE == 1)
	// {
	// 	if(argc > 1 && atoi(argv[1]) == 0)
	// 		ftree = fopen("trees/log-evrard-2d.txt", "w");
	// 	else if(argc > 1 && atoi(argv[1]) == 1)
	// 		ftree = fopen("trees/log-square-2d.txt", "w");
	// }

	// tree.print2d(ftree);
	// fclose(ftree);

	delete[] x;
	delete[] y;
	delete[] z;
	delete[] h;
	delete[] ng;
	delete[] nvi;

	return 0;
}
