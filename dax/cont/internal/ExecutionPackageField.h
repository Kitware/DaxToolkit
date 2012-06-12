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

#include <boost/concept_check.hpp>

namespace dax {
namespace cont {
namespace internal {

namespace detail {

template<class FieldType>
struct FieldIsInput
{
  typedef typename FieldType::AccessTag AccessTag;

  BOOST_CONCEPT_ASSERT((boost::Convertible<AccessTag,
                        dax::exec::internal::FieldAccessInputTag>));
};

template<class FieldType>
struct FieldIsCoordinates
{
  typedef typename FieldType::AssociationTag AssociationTag;

  BOOST_CONCEPT_ASSERT((boost::Convertible<AssociationTag,
                        dax::exec::internal::FieldAssociationCoordinatesTag>));
};

template<class GridType>
DAX_CONT_EXPORT
dax::Id GetFieldSize(const GridType &grid,
                     dax::exec::internal::FieldAssociationCellTag)
{
  return grid.GetNumberOfCells();
}
template<class GridType>
DAX_CONT_EXPORT
dax::Id GetFieldSize(const GridType &grid,
                     dax::exec::internal::FieldAssociationPointTag)
{
  return grid.GetNumberOfPoints();
}

template<class FieldType, class ArrayHandleType>
DAX_CONT_EXPORT
typename FieldType::IteratorType
ExecutionFieldIterator(const ArrayHandleType &arrayHandle,
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
DAX_CONT_EXPORT
typename FieldType::IteratorType
ExecutionFieldIterator(ArrayHandleType &arrayHandle,
                       dax::Id numValues,
                       dax::exec::internal::FieldAccessOutputTag)
{
  return arrayHandle.PrepareForOutput(numValues).first;
}

template<class FieldType,
         class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT
FieldType ExecutionPackageFieldInternal(
    const dax::cont::ArrayHandle<typename FieldType::ValueType,
                                 Container,
                                 DeviceAdapter> &arrayHandle,
    dax::Id numValues,
    dax::exec::internal::FieldAccessInputTag)
{
  typename FieldType::IteratorType fieldIterator
      = ExecutionFieldIterator<FieldType>(arrayHandle,
                                          numValues,
                                          typename FieldType::AccessTag());
  return FieldType(fieldIterator);
}

} // namespace detail

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<typename FieldType,
         class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT
FieldType ExecutionPackageFieldArray(
    dax::cont::ArrayHandle<typename FieldType::ValueType,
                           Container,
                           DeviceAdapter> &arrayHandle,
    dax::Id numValues)
{
  typename FieldType::IteratorType fieldIterator
      = dax::cont::internal::detail::ExecutionFieldIterator<FieldType>(
        arrayHandle,
        numValues,
        typename FieldType::AccessTag());
  return FieldType(fieldIterator);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<template <typename, class> class FieldType,
         typename ValueType,
         class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT
FieldType<ValueType,
          dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldArray(
    dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter> &arrayHandle,
    dax::Id numValues)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return ExecutionPackageFieldArray<FieldType<ValueType, ExecutionAdapter> >(
        arrayHandle, numValues);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<template <class> class FieldType,
         typename ValueType,
         class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT
FieldType<dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldArray(
    dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter> &arrayHandle,
    dax::Id numValues)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return ExecutionPackageFieldArray<FieldType<ExecutionAdapter> >(
        arrayHandle, numValues);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<class FieldType,
         class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT
FieldType ExecutionPackageFieldArrayConst(
    const dax::cont::ArrayHandle<
        typename FieldType::ValueType, Container, DeviceAdapter> &arrayHandle,
    dax::Id numValues)
{
  BOOST_CONCEPT_ASSERT((detail::FieldIsInput<FieldType>));

  return
      dax::cont::internal::detail
      ::ExecutionPackageFieldInternal<FieldType,Container,DeviceAdapter>(
        arrayHandle,
        numValues,
        typename FieldType::AccessTag());
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<template <typename, class> class FieldType,
         typename ValueType,
         class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT
FieldType<ValueType,
          dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldArrayConst(
    const dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>
        &arrayHandle,
    dax::Id numValues)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return ExecutionPackageFieldArrayConst
        <FieldType<ValueType, ExecutionAdapter> >(arrayHandle, numValues);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<template <class> class FieldType,
         typename ValueType,
         class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT
FieldType<dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldArrayConst(
    const dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>
        &arrayHandle,
    dax::Id numValues)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return ExecutionPackageFieldArrayConst<FieldType<ExecutionAdapter> >(
        arrayHandle, numValues);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<typename FieldType,
         class Container,
         class DeviceAdapter,
         class GridType>
DAX_CONT_EXPORT
FieldType ExecutionPackageFieldGrid(
    dax::cont::ArrayHandle<typename FieldType::ValueType,
                           Container,
                           DeviceAdapter> &arrayHandle,
    const GridType &grid)
{
  dax::Id numValues
      = detail::GetFieldSize(grid, typename FieldType::AssociationTag());
  return ExecutionPackageFieldArray<FieldType>(arrayHandle, numValues);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<template <typename, class> class FieldType,
         typename ValueType,
         class Container,
         class DeviceAdapter,
         class GridType>
DAX_CONT_EXPORT
FieldType<ValueType,
          dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldGrid(
    dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter> &arrayHandle,
    const GridType &grid)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return ExecutionPackageFieldGrid<FieldType<ValueType, ExecutionAdapter> >(
        arrayHandle, grid);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<template <class> class FieldType,
         typename ValueType,
         class Container,
         class DeviceAdapter,
         class GridType>
DAX_CONT_EXPORT
FieldType<dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldGrid(
    dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter> &arrayHandle,
    const GridType &grid)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return ExecutionPackageFieldGrid<FieldType<ExecutionAdapter> >(
        arrayHandle, grid);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<typename FieldType,
         class Container,
         class DeviceAdapter,
         class GridType>
DAX_CONT_EXPORT
FieldType ExecutionPackageFieldGridConst(
    const dax::cont::ArrayHandle<typename FieldType::ValueType,
                                 Container,
                                 DeviceAdapter> &arrayHandle,
    const GridType &grid)
{
  dax::Id numValues
      = detail::GetFieldSize(grid, typename FieldType::AssociationTag());
  return ExecutionPackageFieldArrayConst<FieldType>(arrayHandle, numValues);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<template <typename, class> class FieldType,
         typename ValueType,
         class Container,
         class DeviceAdapter,
         class GridType>
DAX_CONT_EXPORT
FieldType<ValueType,
          dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldGridConst(
    const dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>
        &arrayHandle,
    const GridType &grid)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return ExecutionPackageFieldGridConst
        <FieldType<ValueType, ExecutionAdapter> >(arrayHandle, grid);
}

/// Given an ArrayHandle, returns a Field object that can be used in the
/// execution environment.
///
template<template <class> class FieldType,
         typename ValueType,
         class Container,
         class DeviceAdapter,
         class GridType>
DAX_CONT_EXPORT
FieldType<dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldGridConst(
    const dax::cont::ArrayHandle<ValueType, Container, DeviceAdapter>
        &arrayHandle,
    const GridType &grid)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return ExecutionPackageFieldGridConst<FieldType<ExecutionAdapter> >(
        arrayHandle, grid);
}

template<template <typename, class> class FieldType,
         class Container,
         class DeviceAdapter>
DAX_CONT_EXPORT
dax::exec::FieldCoordinatesIn<
    dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldGrid(
    const typename dax::cont::UniformGrid<Container, DeviceAdapter>
    ::PointCoordinatesArrayPlaceholder &,
    const dax::cont::UniformGrid<Container, DeviceAdapter> &);

/// Special case for uniform grid.
///
template<class FieldType, class Container, class DeviceAdapter>
DAX_CONT_EXPORT
dax::exec::FieldCoordinatesIn<
    dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldGridConst(
    const typename dax::cont::UniformGrid<Container, DeviceAdapter>
    ::PointCoordinatesArrayPlaceholder &,
    const dax::cont::UniformGrid<Container, DeviceAdapter> &)
{
  BOOST_CONCEPT_ASSERT((detail::FieldIsInput<FieldType>));
  BOOST_CONCEPT_ASSERT((detail::FieldIsCoordinates<FieldType>));

  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return dax::exec::FieldCoordinatesIn<ExecutionAdapter>();
}

/// Special case for uniform grid.
///
template<template <class> class FieldType, class Container, class DeviceAdapter>
DAX_CONT_EXPORT
dax::exec::FieldCoordinatesIn<
    dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter> >
ExecutionPackageFieldGridConst(
    const typename dax::cont::UniformGrid<Container, DeviceAdapter>
    ::PointCoordinatesArrayPlaceholder &coords,
    const dax::cont::UniformGrid<Container, DeviceAdapter> &grid)
{
  typedef dax::exec::internal::ExecutionAdapter<Container,DeviceAdapter>
      ExecutionAdapter;
  return ExecutionPackageFieldGridConst<FieldType<ExecutionAdapter> >
      (coords, grid);
}

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ExecutionPackageField_h
