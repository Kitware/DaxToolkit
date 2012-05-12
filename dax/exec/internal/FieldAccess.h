//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_exec_internal_FieldAccess_h
#define __dax_exec_internal_FieldAccess_h

#include <dax/exec/Field.h>

#include <dax/internal/GridTopologys.h>

namespace dax { namespace exec { class CellVoxel; }}

namespace dax { namespace exec { namespace internal {

template<typename T, class ExecutionAdapter, class WorkType>
DAX_EXEC_EXPORT const T &fieldAccessNormalGet(
    const dax::exec::Field<T, ExecutionAdapter> &field,
    dax::Id index,
    WorkType &work)
{
  typename dax::exec::Field<T, ExecutionAdapter>::IteratorConstType iterator
      = dax::exec::internal::FieldAccessor::GetBeginIterator(field, work);
  return *(iterator + index);
}

template<typename T, int Size, class ExecutionAdapter, class WorkType>
DAX_EXEC_EXPORT dax::Tuple<T,Size> fieldAccessNormalMultiGet(
  const dax::exec::Field<T, ExecutionAdapter> &field,
  dax::Tuple<dax::Id,Size> indices,
    WorkType &work)
{
  typename dax::exec::Field<T, ExecutionAdapter>::IteratorConstType iterator
      = dax::exec::internal::FieldAccessor::GetBeginIterator(field, work);
  dax::Tuple<T,Size> result;
  for(int i=0; i < Size; ++i)
    {
    result[i] = *(iterator + indices[i]);
    }
  return result;
}


template<typename T, class ExecutionAdapter, class WorkType>
DAX_EXEC_EXPORT void fieldAccessNormalSet(
    dax::exec::Field<T, ExecutionAdapter> &field,
    dax::Id index,
    const T &value,
    WorkType &work)
{
  typename dax::exec::Field<T, ExecutionAdapter>::IteratorType iterator
      = dax::exec::internal::FieldAccessor::GetBeginIterator(field, work);
  iterator += index;
  *iterator = value;
}

template<typename Grid>
DAX_EXEC_EXPORT dax::Vector3 fieldAccessUniformCoordinatesGet(
  const Grid &GridTopology,
  dax::Id index)
{
  return dax::internal::pointCoordiantes(GridTopology, index);
}

template<typename Grid, typename T, int Size>
DAX_EXEC_EXPORT dax::Tuple<T,Size> fieldAccessUniformCoordinatesMultiGet(
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
