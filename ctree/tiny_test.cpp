#include <cstdlib>
#include <cstdio>
#include <vector>
#include <omp.h>
#include <iostream>
#include "Tree.hpp"
#include <cmath>

#define START omp_get_wtime()
#define STOP (double)(omp_get_wtime()-start)

using namespace std;

void initUnitCube(int n, double *x, double *y, double *z, double &h)
{
  x[0]=0.; y[0]=0.; z[0]=0.;
  x[1]=1.; y[1]=0.; z[1]=0.;
  x[2]=1.; y[2]=0.; z[2]=1.;
  x[3]=0.; y[3]=0.; z[3]=1.;
  x[4]=0.; y[4]=1.; z[4]=0.;
  x[5]=1.; y[5]=1.; z[5]=0.;
  x[6]=1.; y[6]=1.; z[6]=1.;
  x[7]=0.; y[7]=1.; z[7]=1.;
  h = 1.1;
}

int main()
{
  int n = 8;
  int ngmax = 256;

  double *x = new double[n];
  double *y = new double[n];
  double *z = new double[n];
  double h;

  int *nvi = new int[n];
  int *ng = new int[(long)n*ngmax];
  for(int i=0; i<n; i++)
    nvi[i] = 0;

  double start, tbuild, tfind;

  initUnitCube(n, &x[0], &y[0], &z[0], h);

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
  tree.findNeighbors(0, x, y, z, h, ngmax, &ng[0], nvi[0], true, true, true);
  tfind = STOP;

  printf("FIND TIME: %f\n", tfind);

  long int sum = 0;
  std::cout << "Particle 0's neighbors: ";
  for (int k=0; k<nvi[0]; k++)
    std::cout << ng[k] << ",";
  std::cout << std::endl;
  sum += nvi[0];
  printf("Total neighbors found: %lu\n", sum);

  if (sum != 4)
    std::cout << "TEST FAILED! Origin of a unit cube should have exactly 4 neighbors inside a radius of size " << h << std::endl;

  tree.clean();

  delete[] x;
  delete[] y;
  delete[] z;
  delete[] ng;
  delete[] nvi;

  return 0;
}
