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

#ifndef __dax_cont_internal_ExecutionPackageField_h
#define __dax_cont_internal_ExecutionPackageField_h

#include <dax/Types.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/UniformGrid.h>

#include <dax/exec/Field.h>

namespace dax {
namespace cont {
namespace internal {

class ExecutionPackageField
{
private:
  template<class ArrayHandleType, class GridType>
  DAX_CONT_EXPORT static
  dax::Id GetFieldSize(const GridType &grid,
                       dax::exec::internal::FieldAssociationCellTag)
  {
    return grid.GetNumberOfCells();
  }
  template<class ArrayHandleType, class GridType>
  DAX_CONT_EXPORT static
  dax::Id GetFieldSize(const GridType &grid,
                       dax::exec::internal::FieldAssociationPointTag)
  {
    return grid.GetNumberOfPoints();
  }

  template<class FieldType, class ArrayHandleType>
  DAX_CONT_EXPORT static
  typename FieldType::IteratorType
  GetExecutionIterator(const ArrayHandleType &arrayHandle,
                       dax::Id numValues,
                       dax::exec::internal::FieldAccessInputTag)
  {
    if (arrayHandle.GetNumberOfValues() != numValues)
      {
      throw dax::cont::ErrorControlBadValue(
            "Received an array that is the wrong size to represent the field "
            "that it is associated with.");
      }
    return arrayHandle.PrepareForInput().first;
  }
  template<class FieldType, class ArrayHandleType>
  DAX_CONT_EXPORT static
  typename FieldType::IteratorType
  GetExecutionIterator(ArrayHandleType &arrayHandle,
                       dax::Id numValues,
                       dax::exec::internal::FieldAccessOutputTag)
  {
    return arrayHandle.PrepareForOutput(numValues).first;
  }

  // Special case for point array of uniform grids.  Needs no actual data.
  template<class FieldType>
  DAX_CONT_EXPORT static
  typename FieldType::IteratorType
  GetExecutionIterator(
      const dax::cont::UniformGrid::PointCoordinatesArrayPlaceholder &,
      dax::Id,
      dax::exec::internal::FieldAccessInputTag)
  {
    return FieldType::IteratorType();
  }

public:
  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<class FieldType, class ArrayHandleType>
  DAX_CONT_EXPORT static
  FieldType GetExecutionObject(ArrayHandleType &arrayHandle,
                               dax::Id numValues)
  {
    typename FieldType::IteratorType fieldIterator
        = GetExecutionIterator<FieldType>(arrayHandle,
                                          numValues,
                                          typename FieldType::AccessTag());
    return FieldType(fieldIterator);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<class FieldType, class ArrayHandleType, class GridType>
  DAX_CONT_EXPORT static
  FieldType GetExecutionObject(ArrayHandleType &arrayHandle,
                               const GridType &grid)
  {
    dax::Id numValues = GetFieldSize(grid, FieldType::AssociationTag());
    return GetExecutionObject<FieldType>(arrayHandle, numValues);
  }
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ExecutionPackageField_h
