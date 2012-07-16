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

#include <dax/exec/internal/GridTopologies.h>

namespace dax { namespace exec { class CellVoxel; }}

namespace dax { namespace exec { namespace internal {

struct FieldAccess
{
  /// Using normal field semantics (that is, a field is just a pointer to an
  /// array), get the value at the given index.
  ///
  template<class T,
           class ExecutionAdapter,
           class Access,
           class Association,
           class WorkType>
  DAX_EXEC_EXPORT static T GetField(
      dax::exec::internal::FieldBase<Access, Association, T, ExecutionAdapter> field,
      dax::Id index,
      WorkType work)
  {
    DAX_ASSERT_EXEC(field.Portal.GetNumberOfValues() > 0, work);
    return field.Portal.Get(index);
  }

  /// Using normal field semantics (that is, a field is just a pointer to an
  /// array), set the value at the given index.
  ///
  template<class T, class ExecAdapter, class Association, class WorkType>
  DAX_EXEC_EXPORT static void SetField(
      dax::exec::internal::FieldBase<
          dax::exec::internal::FieldAccessOutputTag,
          Association,
          T,
          ExecAdapter> field,
      dax::Id index,
      T value,
      WorkType work)
  {
    DAX_ASSERT_EXEC(field.Portal.GetNumberOfValues() > 0, work);
    field.Portal.Set(index, value);
  }

  /// Using normal field semantics (that is, a field is just a pointer to an
  /// array), get several values from the field.
  ///
  template<int Size, class T, class ExecutionAdapter, class Access, class Association, class WorkType>
  DAX_EXEC_EXPORT static dax::Tuple<T, Size> GetMultiple(
      dax::exec::internal::FieldBase<Access, Association, T, ExecutionAdapter> field,
      dax::Tuple<dax::Id,Size> indices,
      WorkType work)
  {
    dax::Tuple<T, Size> result;
    for (int i = 0; i < Size; i++)
      {
      result[i] = GetField(field, indices[i], work);
      }
    return result;
  }

  /// Get the coordinates from a point coordinate field (which may require
  /// some computations on the topology).
  ///
  template<class ExecutionAdapter, class WorkType>
  DAX_EXEC_EXPORT static
  dax::Vector3 GetCoordinates(
      dax::exec::internal::FieldBase<
          FieldAccessInputTag,
          dax::exec::internal::FieldAssociationCoordinatesTag,
          dax::Vector3,
          ExecutionAdapter>,
      dax::Id index,
      const dax::exec::internal::TopologyUniform &topology,
      WorkType)
  {
    return dax::exec::internal::pointCoordiantes(topology, index);
  }

  /// Get the coordinates from a point coordinate field (which may require
  /// some computations on the topology).
  ///
  template<class ExecutionAdapter, class CellType, class WorkType>
  DAX_EXEC_EXPORT static
  dax::Vector3 GetCoordinates(
      dax::exec::internal::FieldBase<
          FieldAccessInputTag,
          dax::exec::internal::FieldAssociationCoordinatesTag,
          dax::Vector3,
          ExecutionAdapter> field,
      dax::Id index,
      const dax::exec::internal::TopologyUnstructured<
          CellType,ExecutionAdapter> &daxNotUsed(topology),
      WorkType work)
  {
    return GetField(field, index, work);
  }

  /// Get the coordinates from a point coordinate field (which may require
  /// some computations on the topology).
  ///
  template<int Size, class ExecutionAdapter, class WorkType>
  DAX_EXEC_EXPORT static
  dax::Tuple<dax::Vector3,Size> GetCoordinatesMultiple(
      dax::exec::internal::FieldBase<
          FieldAccessInputTag,
          dax::exec::internal::FieldAssociationCoordinatesTag,
          dax::Vector3,
          ExecutionAdapter> field,
      dax::Tuple<dax::Id,Size> indices,
      const dax::exec::internal::TopologyUniform &topology,
      WorkType work)
  {
    dax::Tuple<dax::Vector3,Size> result;
    for (int i = 0; i < Size; i++)
      {
      result[i] = GetCoordinates(field, indices[i], topology, work);
      }
    return result;
  }

  /// Get the coordinates from a point coordinate field (which may require
  /// some computations on the topology).
  ///
  template<int Size, class ExecutionAdapter, class CellType, class WorkType>
  DAX_EXEC_EXPORT static
  dax::Tuple<dax::Vector3,Size> GetCoordinatesMultiple(
      dax::exec::internal::FieldBase<
          FieldAccessInputTag,
          dax::exec::internal::FieldAssociationCoordinatesTag,
          dax::Vector3,
          ExecutionAdapter> field,
      dax::Tuple<dax::Id,Size> indices,
      const dax::exec::internal::TopologyUnstructured<
          CellType,ExecutionAdapter> &daxNotUsed(topology),
      WorkType work)
  {
    return GetMultiple(field, indices, work);
  }
};

}}}

#endif //__dax_exec_internal_FieldAccess_h
