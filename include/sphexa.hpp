#pragma once

#ifdef USE_MPI
#include "mpi.h"
#include "DistributedDomain.hpp"
#include "mpiFileUtils.hpp"
#endif

#include "Domain.hpp"
#include "Octree.hpp"
#include "BBox.hpp"
#include "Task.hpp"
#include "ArgParser.hpp"
#include "config.hpp"
#include "Timer.hpp"
#include "Printer.hpp"

#if defined(USE_CUDA)
#include "sph/cuda/sph.cuh"
#endif

#include "sph/findNeighbors.hpp"
#include "sph/density.hpp"
#include "sph/newtonRaphson.hpp"
#include "sph/gradhTerms.hpp"
#include "sph/IAD.hpp"
#include "sph/momentumAndEnergyIAD.hpp"
#include "sph/kernels.hpp"
#include "sph/equationOfState.hpp"
#include "sph/timestep.hpp"
#include "sph/positions.hpp"
#include "sph/totalEnergy.hpp"
#include "sph/updateSmoothingLength.hpp"
#include "sph/gravityTreeWalk.hpp"
