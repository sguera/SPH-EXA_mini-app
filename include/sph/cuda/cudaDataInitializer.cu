#include <cuda.h>

#include "../kernels.hpp"
#include "../lookupTables.hpp"

namespace sphexa
{
namespace lookup_tables
{
__device__ double wharmonicLookupTable[lookup_tables::wharmonicLookupTableSize];
__device__ double wharmonicDerivativeLookupTable[lookup_tables::wharmonicLookupTableSize];
GpuLookupTableInitializer<double> ltinit;

} // namespace lookup_tables
} // namespace sphexa