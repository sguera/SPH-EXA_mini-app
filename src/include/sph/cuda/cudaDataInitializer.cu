#include <cuda.h>

#include "../../math.hpp"

__device__ double sphexa::math::fast_cossin_table[MAX_CIRCLE_ANGLE];
sphexa::math::GpuCosSinLookupTableInitializer<double> sincosInit;
