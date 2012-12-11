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
#ifndef __dax_cont_ArrayContainerControlConstantValue_h
#define __dax_cont_ArrayContainerControlConstantValue_h

#include <dax/cont/ArrayPortal.h>
#include <dax/cont/IteratorFromArrayPortal.h>
#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/ErrorControlBadValue.h>

namespace dax {
namespace cont {

/// \brief An array portal that returns an constant value
///
/// This array portal is similar to an implicit array i.e an array that is
/// defined functionally rather than actually stored in memory. The array
/// comprises of a single constant value for each index. If the array is asked
/// to hold constant value 10 then the values are [10, 10, 10, 10,...].
///
/// The ArrayPortalConstantValue is used in an ArrayHandle with an
/// ArrayContainerControlTagConstantValue container.
///
template <class ConstantValueType>
class ArrayPortalConstantValue
{
public:
  typedef ConstantValueType ValueType;

  DAX_EXEC_CONT_EXPORT
  ArrayPortalConstantValue() : ConstantValue(0),LastIndex(0) {  }

  DAX_EXEC_CONT_EXPORT
  ArrayPortalConstantValue(ValueType constantValue,dax::Id numValues) :
    ConstantValue(constantValue),LastIndex(numValues)
  {  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const { return this->LastIndex; }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id daxNotUsed(index)) const { return this->ConstantValue; }

  typedef dax::cont::IteratorFromArrayPortal < ArrayPortalConstantValue
                                               < ConstantValueType > >
  IteratorType;

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const
  {
    return IteratorType(*this);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const
  {
    return IteratorType(*this, this->LastIndex);
  }

private:
  // The current implementation of this class only allows you to specify the
  // number of indices and then gives indices 0..(size-1). This is the most
  // common case although there may be uses for offset indices (starting at
  // something other than 0) or different strides. These are easy enough to
  // implement, but I have not because I currently have no need for them and
  // they require a bit more state (that must be copied around). If these extra
  // features are needed, they can be added in the future.

  // The last index given by this portal (exclusive).
  dax::Id LastIndex;

  ValueType ConstantValue;
};

/// \brief An array portal that returns an constant value
///
/// This array portal is similar to an implicit array i.e an array that is
/// defined functionally rather than actually stored in memory. The array
/// comprises of a single constant value for each index. If the array is asked
/// to hold constant value 10 then the values are [10, 10, 10, 10,...].
///
/// When creating an ArrayHandle with an ArrayContainerControlTagConstantValue
/// container, use an ArrayPortalConstantValue to establish the array.
///
template < template <class ConstantValueType> class ArrayPortalType,
           class ConstantValueType>
struct ArrayContainerControlTagConstantValue{
  typedef ArrayPortalType<ConstantValueType> PortalType;
};

namespace internal {

template<template <class ConstantValueType> class ArrayPortalType,
         class ConstantValueType>
class ArrayContainerControl<
    typename ArrayPortalType<ConstantValueType>::ValueType,
    ArrayContainerControlTagConstantValue <ArrayPortalType,
                                           ConstantValueType> >
{
public:
  typedef typename ArrayPortalType<ConstantValueType>::ValueType ValueType;
  typedef ArrayPortalType<ConstantValueType> PortalConstType;

  // This is meant to be invalid. Because ConstantValue arrays are read only, you
  // should only be able to use the const version.
  struct PortalType {
    typedef void *ValueType;
    typedef void *IteratorType;
  };

  // All these methods do nothing but raise errors.
  PortalType GetPortal() {
    throw dax::cont::ErrorControlBadValue("ConstantValue arrays are read-only.");
  }
  PortalConstType GetPortalConst() const {
    // This does not work because the ArrayHandle holds the constant
    // ArrayPortal, not the container.
    throw dax::cont::ErrorControlBadValue(
          "ConstantValue container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }
  dax::Id GetNumberOfValues() const {
    // This does not work because the ArrayHandle holds the constant
    // ArrayPortal, not the container.
    throw dax::cont::ErrorControlBadValue(
          "ConstantValue container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }
  void Allocate(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlBadValue("ConstantValue arrays are read-only.");
  }
  void Shrink(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlBadValue("ConstantValue arrays are read-only.");
  }
  void ReleaseResources() {
    throw dax::cont::ErrorControlBadValue("ConstantValue arrays are read-only.");
  }
};
} // internal
} // cont
} // dax

#endif //__dax_cont_ArrayContainerControlConstantValue_h
