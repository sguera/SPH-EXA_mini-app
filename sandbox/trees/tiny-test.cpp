#include <cstdlib>
#include <cstdio>
#include <vector>
#include <chrono>
#include <omp.h>
#include <iostream>
#include <cmath>
#include <cassert>

#include "../../src/tree/BroadTree.hpp"
#include "../../src/tree/KdTree.hpp"
#include "../../src/tree/Octree.hpp"
#include "../../src/tree/NNFTree.hpp"

#define START chrono::high_resolution_clock::now(); //omp_get_wtime()
#define STOP chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now()-start).count(); //(double)(omp_get_wtime()-start)

using namespace sphexa;
using namespace std;

void initUnitCube(int n, double *x, double *y, double *z, double *h)
{
	assert(n == 8);
	x[0]=0.; y[0]=0.; z[0]=0.;
	x[1]=1.; y[1]=0.; z[1]=0.;
	x[2]=1.; y[2]=0.; z[2]=1.;
	x[3]=0.; y[3]=0.; z[3]=1.;
	x[4]=0.; y[4]=1.; z[4]=0.;
	x[5]=1.; y[5]=1.; z[5]=0.;
	x[6]=1.; y[6]=1.; z[6]=1.;
	x[7]=0.; y[7]=1.; z[7]=1.;
	for(int i=0; i<8; i++)
		h[i] = 1.1;
}

void initUniquePosition(int n, double *x, double *y, double *z, double *h)
{
	for(int i=0; i<n; ++i)
	{
		x[i] = 1.;
		y[i] = 0.;
		z[i] = 0.;
		h[i] = 2.;
	}
}

void computeBox(int n, double *x, double *y, double *z,
	double &xmin, double &xmax, double &ymin, double &ymax, double &zmin, double &zmax)
{
	xmin = 1000; xmax = -1000; ymin = 1000; ymax = -1000; zmin = 1000; zmax = -1000;
	for(int i=0; i<n; i++)
	{
		if(x[i] < xmin) xmin = x[i];
		if(x[i] > xmax) xmax = x[i];
		if(y[i] < ymin) ymin = y[i];
		if(y[i] > ymax) ymax = y[i];
		if(z[i] < zmin) zmin = z[i];
		if(z[i] > zmax) zmax = z[i];
	}

  	// epsilion to avoid case where the box is null
	auto boxEpsilon = [](auto const &mi, auto const &ma){ return mi == ma ? max(abs(ma)*0.001,0.000001) : 0.;};
	xmax += boxEpsilon(xmin, xmax);
	ymax += boxEpsilon(ymin, ymax);
	zmax += boxEpsilon(zmin, zmax);

	printf("Domain x[%f %f]\n", xmin, xmax);
	printf("Domain y[%f %f]\n", ymin, ymax);
	printf("Domain z[%f %f]\n", zmin, zmax);
}

int main()
{
	int n = 8;
	int ngmax = 256;

	double *x = new double[n];
	double *y = new double[n];
	double *z = new double[n];
	double *h = new double[n];

	int *nvi = new int[n];
	int *ng = new int[(long)n*ngmax];

	double tbuild, tfind;
	chrono::time_point<chrono::high_resolution_clock> start;

  	// Unit cube
	{
		cout << endl << "UNIT CUBE: " << endl;

		for(int i=0; i<n; i++)
			nvi[i] = 0;

		initUnitCube(n, x, y, z, h);

		double xmin, xmax, ymin, ymax, zmin, zmax;
		computeBox(n, &x[0], &y[0], &z[0], xmin, xmax, ymin, ymax, zmin, zmax);

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
		printf("BUILD TIME: %f\n", tbuild);

		start = START;
		tree.findNeighbors(x[0], y[0], z[0], h[0], ngmax, &ng[0], nvi[0], false, false, false);
		tfind = STOP;

		printf("FIND TIME: %f\n", tfind);

		long int sum = 0;
		cout << "Particle 0's neighbors: ";
		for (int k=0; k<nvi[0]; k++)
			cout << ng[k] << ",";
		cout << endl;
		sum += nvi[0];
		printf("Total neighbors found: %lu\n", sum);

		tree.clean();

		if (sum != 4)
		{
			cout << "TEST FAILED! Origin of a unit cube should have exactly 4 neighbors inside a radius of size " << h << endl;
			return 1;
		}
		else
			cout << "TEST PASSED!" << endl;
	}

  	// Unique position
	{
		cout << endl << "UNIQUE POSITION: " << endl;
		initUniquePosition(n, x, y, z, h);

		for(int i=0; i<n; i++)
			nvi[i] = 0;

		double xmin, xmax, ymin, ymax, zmin, zmax;
		computeBox(n, &x[0], &y[0], &z[0], xmin, xmax, ymin, ymax, zmin, zmax);

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
		printf("BUILD TIME: %f\n", tbuild);

		start = START;
		tree.findNeighbors(x[0], y[0], z[0], h[0], ngmax, &ng[0], nvi[0], false, false, false);
		tfind = STOP;

		printf("FIND TIME: %f\n", tfind);

		long int sum = 0;
		cout << "Particle 0's neighbors: ";
		for (int k=0; k<nvi[0]; k++)
			cout << ng[k] << ",";
		cout << endl;
		sum += nvi[0];
		printf("Total neighbors found: %lu\n", sum);

		tree.clean();

		if (sum != n)
		{
			cout << "TEST FAILED! All particles on the same position {1,0,0} should work with h = " << h << endl;
			return 1;
		}
		else
			cout << "TEST PASSED!" << endl;
	}


	delete[] x;
	delete[] y;
	delete[] z;
	delete[] ng;
	delete[] nvi;

	return 0;
}
