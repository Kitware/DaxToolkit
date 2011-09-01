/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_internal_FieldAccess_h
#define __dax_exec_internal_FieldAccess_h

#include <dax/exec/Field.h>

namespace dax { namespace exec { class CellVoxel; }}

namespace dax { namespace exec { namespace internal {

template<typename T>
__device__ T fieldAccessNormalGet(const dax::exec::Field<T> &field,
                                  dax::Id index)
{
  return field.GetArray().GetValue(index);
}

template<typename T>
__device__ void fieldAccessNormalSet(dax::exec::Field<T> &field,
                                     dax::Id index,
                                     const T &value)
{
  field.GetArray().SetValue(index, value);
}

__device__ dax::Vector3 fieldAccessStructuredCoordinatesGet(
  const dax::StructuredPointsMetaData &gridStructure,
  dax::Id index)
{
  dax::Int3 ijk = flatIndexToInt3Index(index, gridStructure.Extent);
  dax::Vector3 coords;
  coords = (  gridStructure.Origin
            + (ijk + gridStructure.Extent.Min) * gridStructure.Spacing );
  return coords;
}

}}}

#endif //__dax_exec_internal_FieldAccess_h
