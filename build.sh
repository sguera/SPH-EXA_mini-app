#!/bin/bash

export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=1

make clean
make -f Makefile -j 2 mpi+omp+cuda SRCDIR=. BUILDDIR=build BINDIR=bin NVCC="nvcc"
