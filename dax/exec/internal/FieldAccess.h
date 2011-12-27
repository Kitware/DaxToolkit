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
DAX_EXEC_EXPORT inline const T &fieldAccessNormalGet(
  const dax::exec::Field<T> &field,
  dax::Id index)
{
  return field.GetArray().GetValue(index);
}

template<typename T>
DAX_EXEC_EXPORT inline void fieldAccessNormalSet(dax::exec::Field<T> &field,
                                                 dax::Id index,
                                                 const T &value)
{
  field.GetArray().SetValue(index, value);
}

DAX_EXEC_EXPORT inline dax::Vector3 fieldAccessUniformCoordinatesGet(
  const dax::internal::StructureUniformGrid &gridStructure,
  dax::Id index)
{
  dax::Id3 ijk = flatIndexToIndex3(index, gridStructure.Extent);
  dax::Vector3 coords;
  coords[0] = gridStructure.Origin[0] + ijk[0] * gridStructure.Spacing[0];
  coords[1] = gridStructure.Origin[1] + ijk[1] * gridStructure.Spacing[1];
  coords[2] = gridStructure.Origin[2] + ijk[2] * gridStructure.Spacing[2];
  return coords;
}

}}}

#endif //__dax_exec_internal_FieldAccess_h
