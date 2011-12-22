#ifndef __dax_cont_internal_ConvertTypes_h
#define __dax_cont_internal_ConvertTypes_h

#include <dax/cont/StructuredGrid.h>
#include <dax/internal/GridStructures.h>

#include <dax/internal/DataArray.h>
#include <dax/cont/Array.h>

namespace dax {
namespace cont {
namespace internal {

dax::internal::StructureUniformGrid convert(
    const dax::cont::StructuredGrid& g)
  {
  dax::internal::StructureUniformGrid grid;
  grid.Origin = g.Origin;
  grid.Spacing = g.Spacing;
  grid.Extent.Min = g.Extent.Min;
  grid.Extent.Max = g.Extent.Max;
  return grid;
  }





} } }

#endif // __dax_cuda_cont_internal_ConvertTypes_h
