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

#include <dax/exec/Field.h>

namespace dax {
namespace cont {
namespace internal {

class ExecutionPackageField
{
private:
  template<class ArrayHandleType, class GridType>
  DAX_CONT_EXPORT static
  dax::Id CheckArraySize(const ArrayHandleType &arrayHandle,
                         const GridType &grid,
                         dax::exec::internal::FieldAssociationCellTag)
  {
    dax::Id numValues = grid.GetNumberOfCells();
    if (arrayHandle.GetNumberOfValues() != numValues)
      {
      throw dax::cont::ErrorControlBadValue(
            "Received an array that is the wrong size to represent a "
            "cell-associated field with the given grid.");
      }
    return numValues;
  }
  template<class ArrayHandleType, class GridType>
  DAX_CONT_EXPORT static
  dax::Id CheckArraySize(const ArrayHandleType &arrayHandle,
                         const GridType &grid,
                         dax::exec::internal::FieldAssociationPointTag)
  {
    dax::Id numValues= grid.GetNumberOfPoints();
    if (arrayHandle.GetNumberOfValues() != numValues)
      {
      throw dax::cont::ErrorControlBadValue(
            "Received an array that is the wrong size to represent a "
            "point-associated field with the given grid.");
      }
    return numValues;
  }

  template<class FieldType, class ArrayHandleType>
  DAX_CONT_EXPORT static
  typename FieldType::IteratorType
  GetExecutionIterator(const ArrayHandleType &arrayHandle,
                       dax::Id,
                       dax::exec::internal::FieldAccessInputTag)
  {
    return arrayHandle.PrepareForInput().first;
  }
  template<class FieldType, class ArrayHandleType>
  DAX_CONT_EXPORT static
  typename FieldType::IteratorType
  GetExecutionIterator(const ArrayHandleType &arrayHandle,
                       dax::Id numValues,
                       dax::exec::internal::FieldAccessOutputTag)
  {
    return arrayHandle.PrepareForOutput(numValues).first;
  }

public:
  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<class FieldType, class ArrayHandleType, class GridType>
  DAX_CONT_EXPORT static
  FieldType GetExecutionObject(const ArrayHandleType &arrayHandle,
                               const GridType &grid)
  {
    dax::Id numValues
        = CheckArraySize(arrayHandle, grid, FieldType::AssociationTag());
    typename FieldType::IteratorType fieldIterator
        = GetExecutionIterator<FieldType>(arrayHandle,
                                          numValues,
                                          FieldType::AccessPolicy());
    return FieldType(fieldIterator);
  }
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ExecutionPackageField_h
