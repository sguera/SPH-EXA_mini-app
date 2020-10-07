#!/bin/bash

export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=1

make clean
make -f Makefile NVCC="nvcc -g -G" CXXFLAGS="-std=c++14 -g -O0 -DNDEBUG" mpi+omp+cuda SRCDIR=. BUILDDIR=build BINDIR=bin NVCCFLAGS="-std=c++14 -rdc=true -arch=sm_52 -g -G --expt-relaxed-constexpr" NVCCARCH=sm_52
