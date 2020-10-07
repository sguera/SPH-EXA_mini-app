#!/bin/bash

export CRAYPE_LINK_TYPE=dynamic
export OMP_NUM_THREADS=1

# -> Add -o nvprof.%h.%p.%q{SLURM_NODEID}.%q{SLURM_PROCID}.nvvp
# to create 1 file per MPI rank for the Visual Profiler:
# nvvp -i nvprof.nid*
# -> Add --kernels "::put_kernel_name_here:" --analysis-metrics
# to collect advanced nvvp performance data, for example:
# "::computeMomentumAndEnergyIAD:" 

rm -f report
#nvprof -o report --metrics achieved_occupancy ./bin/mpi+omp+cuda.app -n 100 -s 1 2>&1
nvprof --profile-child-processes -f -o report-%p ./bin/mpi+omp+cuda.app -n 30 -s 2 2>&1
