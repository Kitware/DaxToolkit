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
#ifndef __dax_cont_ArrayHandleConstant_h
#define __dax_cont_ArrayHandleConstant_h


#include <dax/cont/ArrayContainerControlImplicit.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/IteratorFromArrayPortal.h>

namespace dax {
namespace cont {

namespace internal {


/// \brief An array portal that returns an constant value
///
/// This array portal is similar to an implicit array i.e an array that is
/// defined functionally rather than actually stored in memory. The array
/// comprises of a single constant value for each index. If the array is asked
/// to hold constant value 10 then the values are [10, 10, 10, 10,...].
///
/// The ArrayPortalConstant is used in an ArrayHandle with an
/// ArrayContainerControlTagConstant container.
///
template <class ConstantValueType>
class ArrayPortalConstant
{
public:
  typedef ConstantValueType ValueType;

  DAX_EXEC_CONT_EXPORT
  ArrayPortalConstant() :
    ConstantValue( ),
    NumberOfValues(0) {  }

  DAX_EXEC_CONT_EXPORT
  ArrayPortalConstant(ValueType constantValue,dax::Id numValues) :
    ConstantValue(constantValue),
    NumberOfValues(numValues)
  {  }

  template<typename OtherValueType>
  DAX_EXEC_CONT_EXPORT
  ArrayPortalConstant(const ArrayPortalConstant<OtherValueType> &src)
    : ConstantValue(src.ConstantValue),
      NumberOfValues(src.NumberOfValues)
  {  }

  template<typename OtherValueType>
  DAX_EXEC_CONT_EXPORT
  ArrayPortalConstant<ValueType> &operator=(
      const ArrayPortalConstant<OtherValueType> &src)
  {
    this->ConstantValue = src.ConstantValue;
    this->NumberOfValues = src.NumberOfValues;
    return *this;
  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const { return this->NumberOfValues; }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id daxNotUsed(index)) const { return this->ConstantValue; }

  typedef dax::cont::internal::IteratorFromArrayPortal < ArrayPortalConstant
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
    return IteratorType(*this, this->NumberOfValues);
  }

private:
  ValueType ConstantValue;

  dax::Id NumberOfValues;
};

/// A convenience class that provides a typedef to the appropriate tag for
/// a constant array container.
template<typename ConstantValueType>
struct ArrayHandleConstantTraits
{
  typedef dax::cont::ArrayContainerControlTagImplicit<
      dax::cont::internal::ArrayPortalConstant<ConstantValueType> > Tag;
};

} // namespace internal

/// ArrayHandleConstants are a specialization of ArrayHandles. By default
/// it contains a constant value array portal. A ConstantValueArrayPortal is an
/// array portal with a function generating a defined constant value. Like an
/// implicit array, this too holds no memory.  The ConstantValue array simply
/// returns a single value for each of the index.
template <typename ConstantValueType,
          class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class ArrayHandleConstant
    : public dax::cont::ArrayHandle<
          ConstantValueType,
          typename internal::ArrayHandleConstantTraits<ConstantValueType>::Tag,
          DeviceAdapterTag>
{
  typedef dax::cont::ArrayHandle<
      ConstantValueType,
      typename internal::ArrayHandleConstantTraits<ConstantValueType>::Tag,
      DeviceAdapterTag> Superclass;

public:
  ArrayHandleConstant(ConstantValueType value, dax::Id length)
    :Superclass(typename Superclass::PortalConstControl(value,length))
  {
  }

  ArrayHandleConstant():Superclass() {}
};

/// A convenience function for creating a ArrayHandleConstant. It only
/// takes constant value and the lenght of the array and constructs a
/// ArrayHandleConstant of the specified length. The array returns a
/// specified constant value for each index.
template<typename ConstantValueType, typename DeviceAdapter>
DAX_CONT_EXPORT
ArrayHandleConstant<ConstantValueType,DeviceAdapter>
make_ArrayHandleConstant(ConstantValueType value,
                         dax::Id length,
                         DeviceAdapter)
{
  return ArrayHandleConstant<ConstantValueType, DeviceAdapter>(value,length);
}

template<typename ConstantValueType>
DAX_CONT_EXPORT
dax::cont::ArrayHandleConstant<ConstantValueType,DAX_DEFAULT_DEVICE_ADAPTER_TAG>
make_ArrayHandleConstant(ConstantValueType value,
                         dax::Id length)
{
  return dax::cont::make_ArrayHandleConstant(
        value, length, DAX_DEFAULT_DEVICE_ADAPTER_TAG());
}

}
} // namespace dax::cont

#endif //__dax_cont_ArrayHandleConstant_h
