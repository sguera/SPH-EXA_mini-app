#pragma once

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <omp.h>

#include "../../src/tree/BroadTree.hpp"
#include "../../src/tree/KdTree.hpp"
#include "../../src/tree/Octree.hpp"
#include "../../src/tree/NNFTree.hpp"

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
	ngmax = 150;
	
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

        double *dummy = new double[n];
        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);
        fread(h, sizeof(double), n, f);
        fread(dummy, sizeof(double), n, f);

        fclose(f);

        delete[] dummy;
	}
	else
	{
		printf("Error opening file.\n");
		exit(1);
	}
}