#!/bin/bash

export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=4

make clean
make -f Makefile -j 2 mpi+omp+cuda SRCDIR=. BUILDDIR=build BINDIR=bin NVCC="nvcc" NVCCARCH=sm_52 TESTCASE=sedov
mv bin/* bin/app
