/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_internal_FieldAccess_h
#define __dax_exec_internal_FieldAccess_h

#include <dax/exec/Field.h>

#include <dax/internal/GridStructures.h>

namespace dax { namespace exec { class CellVoxel; }}

namespace dax { namespace exec { namespace internal {

template<typename T>
DAX_EXEC_EXPORT const T &fieldAccessNormalGet(const dax::exec::Field<T> &field,
                                              dax::Id index)
{
  return field.GetArray().GetValue(index);
}

template<typename T>
DAX_EXEC_EXPORT void fieldAccessNormalSet(dax::exec::Field<T> &field,
                                          dax::Id index,
                                          const T &value)
{
  field.GetArray().SetValue(index, value);
}

DAX_EXEC_EXPORT dax::Vector3 fieldAccessUniformCoordinatesGet(
  const dax::internal::StructureUniformGrid &gridStructure,
  dax::Id index)
{
  dax::Id3 ijk = flatIndexToIndex3(index, gridStructure.Extent);
  dax::Vector3 coords;
  coords.x = gridStructure.Origin.x + ijk.x * gridStructure.Spacing.x;
  coords.y = gridStructure.Origin.y + ijk.y * gridStructure.Spacing.y;
  coords.z = gridStructure.Origin.z + ijk.z * gridStructure.Spacing.z;
  return coords;
}

}}}

#endif //__dax_exec_internal_FieldAccess_h
