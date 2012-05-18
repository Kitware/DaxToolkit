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

  template<class FieldType,
           template <typename> class Container,
           class DeviceAdapter>
  DAX_CONT_EXPORT static
  FieldType GetExecutionObjectInternal(
      const dax::cont::ArrayHandle<FieldType, Container, DeviceAdapter> &arrayHandle,
      dax::Id numValues,
      dax::exec::internal::FieldAccessInputTag)
  {
    typename FieldType::IteratorType fieldIterator
        = GetExecutionIterator<FieldType>(arrayHandle,
                                          numValues,
                                          typename FieldType::AccessTag());
    return FieldType(fieldIterator);
  }

public:
  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<typename FieldType,
           template <typename> class Container,
           class DeviceAdapter>
  DAX_CONT_EXPORT static
  FieldType GetExecutionObject(
      dax::cont::ArrayHandle<typename FieldType::ValueType,
                             Container,
                             DeviceAdapter> &arrayHandle,
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
  template<template <typename, class> class FieldType,
           typename ValueType,
           template <typename> class Container,
           class DeviceAdapter>
  DAX_CONT_EXPORT static
  FieldType<ValueType, typename dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>::ExecutionAdapter>
  GetExecutionObject(
      dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter> &arrayHandle,
      dax::Id numValues)
  {
    typedef typename DeviceAdapter::template ExeutionAdapter<Container>
        ExecutionAdapter;
    return GetExecutionObject<FieldType<ValueType, ExecutionAdapter> >(
          arrayHandle, numValues);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<template <class> class FieldType,
           typename ValueType,
           template <typename> class Container,
           class DeviceAdapter>
  DAX_CONT_EXPORT static
  FieldType<typename dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>::ExecutionAdapter>
  GetExecutionObject(
      dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter> &arrayHandle,
      dax::Id numValues)
  {
    typedef typename DeviceAdapter::template ExecutionAdapter<Container>
        ExecutionAdapter;
    return GetExecutionObject<FieldType<ExecutionAdapter> >(
          arrayHandle, numValues);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<class FieldType,
           template <typename> class Container,
           class DeviceAdapter>
  DAX_CONT_EXPORT static
  FieldType GetExecutionObject(
      const dax::cont::ArrayHandle<FieldType, Container, DeviceAdapter>
          &arrayHandle,
      dax::Id numValues)
  {
    return GetExecutionObjectInternal(arrayHandle,
                                      numValues,
                                      typename FieldType::AccessTag());
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<template <typename, class> class FieldType,
           typename ValueType,
           template <typename> class Container,
           class DeviceAdapter>
  DAX_CONT_EXPORT static
  FieldType<ValueType, typename dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>::ExecutionAdapter>
  GetExecutionObject(
      const dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>
          &arrayHandle,
      dax::Id numValues)
  {
    typedef typename DeviceAdapter::template ExeutionAdapter<Container>
        ExecutionAdapter;
    return GetExecutionObject<FieldType<ValueType, ExecutionAdapter> >(
          arrayHandle, numValues);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<template <class> class FieldType,
           typename ValueType,
           template <typename> class Container,
           class DeviceAdapter>
  DAX_CONT_EXPORT static
  FieldType<typename dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>::ExecutionAdapter>
  GetExecutionObject(
      const dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>
          &arrayHandle,
      dax::Id numValues)
  {
    typedef typename DeviceAdapter::template ExecutionAdapter<Container>
        ExecutionAdapter;
    return GetExecutionObject<FieldType<ExecutionAdapter> >(
          arrayHandle, numValues);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<typename FieldType,
           template <typename> class Container,
           class DeviceAdapter,
           class GridType>
  DAX_CONT_EXPORT static
  FieldType GetExecutionObject(
      dax::cont::ArrayHandle<typename FieldType::ValueType,
                             Container,
                             DeviceAdapter> &arrayHandle,
      const GridType &grid)
  {
    dax::Id numValues = GetFieldSize(grid, FieldType::AssociationTag());
    return GetExecutionObject<FieldType>(arrayHandle, numValues);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<template <typename, class> class FieldType,
           typename ValueType,
           template <typename> class Container,
           class DeviceAdapter,
           class GridType>
  DAX_CONT_EXPORT static
  FieldType<ValueType, typename dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>::ExecutionAdapter>
  GetExecutionObject(
      dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter> &arrayHandle,
      const GridType &grid)
  {
    typedef typename DeviceAdapter::template ExeutionAdapter<Container>
        ExecutionAdapter;
    return GetExecutionObject<FieldType<ValueType, ExecutionAdapter> >(
          arrayHandle, grid);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<template <class> class FieldType,
           typename ValueType,
           template <typename> class Container,
           class DeviceAdapter,
           class GridType>
  DAX_CONT_EXPORT static
  FieldType<typename dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>::ExecutionAdapter>
  GetExecutionObject(
      dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter> &arrayHandle,
      const GridType &grid)
  {
    typedef typename DeviceAdapter::template ExecutionAdapter<Container>
        ExecutionAdapter;
    return GetExecutionObject<FieldType<ExecutionAdapter> >(
          arrayHandle, grid);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<typename FieldType,
           template <typename> class Container,
           class DeviceAdapter,
           class GridType>
  DAX_CONT_EXPORT static
  FieldType GetExecutionObject(
      const dax::cont::ArrayHandle<typename FieldType::ValueType,
                                   Container,
                                   DeviceAdapter> &arrayHandle,
      const GridType &grid)
  {
    dax::Id numValues = GetFieldSize(grid, FieldType::AssociationTag());
    return GetExecutionObject<FieldType>(arrayHandle, numValues);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<template <typename, class> class FieldType,
           typename ValueType,
           template <typename> class Container,
           class DeviceAdapter,
           class GridType>
  DAX_CONT_EXPORT static
  FieldType<ValueType, typename dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>::ExecutionAdapter>
  GetExecutionObject(
      const dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>
          &arrayHandle,
      const GridType &grid)
  {
    typedef typename DeviceAdapter::template ExeutionAdapter<Container>
        ExecutionAdapter;
    return GetExecutionObject<FieldType<ValueType, ExecutionAdapter> >(
          arrayHandle, grid);
  }

  /// Given an ArrayHandle, returns a Field object that can be used in the
  /// execution environment.
  ///
  template<template <class> class FieldType,
           typename ValueType,
           template <typename> class Container,
           class DeviceAdapter,
           class GridType>
  DAX_CONT_EXPORT static
  FieldType<typename dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>::ExecutionAdapter>
  GetExecutionObject(
      const dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>
          &arrayHandle,
      const GridType &grid)
  {
    typedef typename DeviceAdapter::template ExecutionAdapter<Container>
        ExecutionAdapter;
    return GetExecutionObject<FieldType<ExecutionAdapter> >(
          arrayHandle, grid);
  }

  /// Special case for uniform grid.
  ///
  template<template <typename> class Container, class DeviceAdapter>
  DAX_CONT_EXPORT static
  dax::exec::FieldCoordinatesIn<
      typename dax::cont::ArrayHandle<dax::Vector3, Container, DeviceAdapter>::ExecutionAdapter>
  GetExecutionObject(
      const typename dax::cont::UniformGrid<Container, DeviceAdapter>
      ::PointCoordinatesArrayPlaceholder &,
      const dax::cont::UniformGrid<Container, DeviceAdapter> &)
  {
    typedef typename DeviceAdapter::template ExecutionAdapter<Container>
        ExecutionAdapter;
    return dax::exec::FieldCoordinatesIn<ExecutionAdapter>();
  }
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ExecutionPackageField_h
