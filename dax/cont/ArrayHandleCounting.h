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
#ifndef __dax_cont_ArrayHandleCounting_h
#define __dax_cont_ArrayHandleCounting_h

#include <dax/cont/ArrayContainerControlImplicit.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/IteratorFromArrayPortal.h>

namespace dax {
namespace cont {

namespace internal {

/// \brief An implicit array portal that returns an counting value.
template <class CountingValueType>
class ArrayPortalCounting
{
public:
  typedef CountingValueType ValueType;

  DAX_EXEC_CONT_EXPORT
  ArrayPortalCounting() :
  StartingValue(),
  NumberOfValues(0)
  {  }

  DAX_EXEC_CONT_EXPORT
  ArrayPortalCounting(ValueType startingValue, dax::Id numValues) :
  StartingValue(startingValue),
  NumberOfValues(numValues)
  {  }

  template<typename OtherValueType>
  DAX_EXEC_CONT_EXPORT
  ArrayPortalCounting(const ArrayPortalCounting<OtherValueType> &src)
    : StartingValue(src.StartingValue),
      NumberOfValues(src.NumberOfValues)
  {  }

  template<typename OtherValueType>
  DAX_EXEC_CONT_EXPORT
  ArrayPortalCounting<ValueType> &operator=(
      const ArrayPortalCounting<OtherValueType> &src)
  {
    this->StartingValue = src.StartingValue;
    this->NumberOfValues = src.NumberOfValues;
    return *this;
  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const { return this->NumberOfValues; }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id index) const { return StartingValue+index; }

  typedef dax::cont::internal::IteratorFromArrayPortal<
          ArrayPortalCounting < CountingValueType> > IteratorType;

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
  CountingValueType StartingValue;
  dax::Id NumberOfValues;
};

/// A convenience class that provides a typedef to the appropriate tag for
/// a counting array container.
template<typename ConstantValueType>
struct ArrayHandleCountingTraits
{
  typedef dax::cont::ArrayContainerControlTagImplicit<
      dax::cont::internal::ArrayPortalCounting<ConstantValueType> > Tag;
};

} // namespace internal

/// ArrayHandleCountings is a specialization of ArrayHandle. By default it
/// contains a increment value, that is increment for each step between zero
/// and the passed in length
template <typename CountingValueType,
          class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class ArrayHandleCounting
    : public dax::cont::ArrayHandle <
          CountingValueType,
          typename internal::ArrayHandleCountingTraits<CountingValueType>::Tag,
          DeviceAdapterTag >
{
  typedef dax::cont::ArrayHandle <
          CountingValueType,
          typename internal::ArrayHandleCountingTraits<CountingValueType>::Tag,
          DeviceAdapterTag > Superclass;
public:

  ArrayHandleCounting(CountingValueType startingValue, dax::Id length)
    :Superclass(typename Superclass::PortalConstControl(startingValue, length))
  {
  }

  ArrayHandleCounting():Superclass() {}
};

/// A convenience function for creating an ArrayHandleCounting. It takes the
/// value to start counting from and and the number of times to increment.
template<typename CountingValueType, typename DeviceAdapterTag>
DAX_CONT_EXPORT
dax::cont::ArrayHandleCounting<CountingValueType,DeviceAdapterTag>
make_ArrayHandleCounting(CountingValueType startingValue,
                         dax::Id length,
                         DeviceAdapterTag)
{
  return dax::cont::ArrayHandleCounting<CountingValueType,DeviceAdapterTag>(
        startingValue, length);
}

template<typename CountingValueType>
DAX_CONT_EXPORT
dax::cont::ArrayHandleCounting<CountingValueType,DAX_DEFAULT_DEVICE_ADAPTER_TAG>
make_ArrayHandleCounting(CountingValueType startingValue,
                         dax::Id length)
{
  return make_ArrayHandleCounting(startingValue,
                                  length,
                                  DAX_DEFAULT_DEVICE_ADAPTER_TAG());
}

}
} // namespace dax::cont

#endif //__dax_cont_ArrayHandleCounting_h
