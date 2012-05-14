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

#include <dax/internal/TopologyStructured.h>

namespace dax { namespace exec { class CellVoxel; }}

namespace dax { namespace exec { namespace internal {

struct FieldAccess
{
private:
  template <typename T>
  DAX_EXEC_EXPORT static bool NotDefaultConstructor(T &x) { return (x != T()); }

public:
  /// Using normal field semantics (that is, a field is just a pointer to an
  /// array), get the value at the given index.
  ///
  template<class Access, class Association, class WorkType>
  DAX_EXEC_EXPORT static
  typename Access::ValueType GetNormal(
      dax::exec::internal::FieldBase<Access, Association> field,
      dax::Id index,
      WorkType work)
  {
    DAX_ASSERT_EXEC(NotDefaultConstructor(field.BeginIterator), work);
    return *(field.BeginIterator + index);
  }

  /// Using normal field semantics (that is, a field is just a pointer to an
  /// array), set the value at the given index.
  ///
  template<class T, class ExecutionAdapter, class Association, class WorkType>
  DAX_EXEC_EXPORT static void SetNormal(
      dax::exec::internal::FieldBase<
          dax::exec::internal::FieldAccessPolicyOutput<T, ExecutionAdapter>,
          Association> field,
      dax::Id index,
      T value,
      WorkType work)
  {
    DAX_ASSERT_EXEC(NotDefaultConstructor(field.BeginIterator), work);
    *(field.BeginIterator + index) = value;
  }

  /// Using normal field semantics (that is, a field is just a pointer to an
  /// array), get several values from the field.
  ///
  template<int Size, class Access, class Association, class WorkType>
  DAX_EXEC_EXPORT static
  dax::Tuple<typename Access::ValueType, Size> GetMultiple(
      dax::exec::internal::FieldBase<Access, Association> field,
      dax::Tuple<dax::Id,Size> indices,
      WorkType work)
  {
    dax::Tuple<typename Access::ValueType, Size> result;
    for (int i = 0; i < Size; i++)
      {
      result[i] = GetNormal(field, indices[i], work);
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
          FieldAccessPolicyInput<dax::Vector3, ExecutionAdapter>,
          dax::exec::internal::FieldAssociationCoordinatesTag>,
      dax::Id index,
      const dax::internal::TopologyUniform topology,
      WorkType)
  {
    return dax::internal::pointCoordiantes(topology, index);
  }

  /// Get the coordinates from a point coordinate field (which may require
  /// some computations on the topology).
  ///
  template<int Size, class ExecutionAdapter, class WorkType>
  DAX_EXEC_EXPORT static
  dax::Tuple<dax::Vector3,Size> GetCoordinatesMultiple(
      dax::exec::internal::FieldBase<
          FieldAccessPolicyInput<dax::Vector3, ExecutionAdapter>,
          dax::exec::internal::FieldAssociationCoordinatesTag> field,
      dax::Tuple<dax::Id,Size> indices,
      const dax::internal::TopologyUniform topology,
      WorkType work)
  {
    dax::Tuple<dax::Vector3,Size> result;
    for (int i = 0; i < Size; i++)
      {
      result[i] = GetCoordinates(field, indices[i], topology, work);
      }
    return result;
  }
};

}}}

#endif //__dax_exec_internal_FieldAccess_h
