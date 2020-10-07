#!/bin/bash

export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=1

rm -f report
#srun nvprof -o report --metrics achieved_occupancy ./bin/mpi+omp+cuda.app -n 100 -s 1 2>&1
./bin/mpi+omp+cuda.app -n 100 -s 200 -w 200 2>&1
