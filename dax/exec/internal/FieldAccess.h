/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax_exec_internal_FieldAccess_h
#define __dax_exec_internal_FieldAccess_h

#include <dax/exec/Field.h>

#include <dax/internal/GridTopologys.h>

namespace dax { namespace exec { class CellVoxel; }}

namespace dax { namespace exec { namespace internal {

template<typename T>
DAX_EXEC_EXPORT const T &fieldAccessNormalGet(
  const dax::exec::Field<T> &field,
  dax::Id index)
{
  return field.GetArray().GetValue(index);
}

template<typename T, int Size>
DAX_EXEC_EXPORT dax::Tuple<T,Size> fieldAccessNormalGet(
  const dax::exec::Field<T> &field,
  dax::Tuple<dax::Id,Size> indices)
{
  dax::Tuple<T,Size> result;
  const dax::internal::DataArray<T> data = field.GetArray();
  for(int i=0; i < Size; ++i)
    {
    result[i] = data.GetValue(indices[i]);
    }
  return result;
}


template<typename T>
DAX_EXEC_EXPORT void fieldAccessNormalSet(dax::exec::Field<T> &field,
                                          dax::Id index,
                                          const T &value)
{
  field.GetArray().SetValue(index, value);
}

template<typename Grid>
DAX_EXEC_EXPORT dax::Vector3 fieldAccessUniformCoordinatesGet(
  const Grid &GridTopology,
  dax::Id index)
{
  return dax::internal::pointCoordiantes(GridTopology, index);
}

template<typename Grid, typename T, int Size>
DAX_EXEC_EXPORT dax::Tuple<T,Size> fieldAccessUniformCoordinatesGet(
  const Grid &GridTopology,
  const dax::Tuple<Id,Size>& indices)
{
  dax::Tuple<T,Size> result;
  for(int i=0; i < Size; ++i)
    {
    result[i] = dax::internal::pointCoordiantes(GridTopology, indices[i]);
    }
  return result;
}


}}}

#endif //__dax_exec_internal_FieldAccess_h
