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

#include <dax/cont/internal/ArrayContainerControlCounting.h>
#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {
/// ArrayHandleCountings is a specialization of ArrayHandle. By default it
/// contains a increment value, that is increment for each step between zero
/// and the passed in length
template <typename CountingValueType,
          class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class ArrayHandleCounting : public ArrayHandle < CountingValueType,
                         dax::cont::internal::ArrayContainerControlTagCounting,
                         DeviceAdapterTag >
{
public:
  typedef dax::cont::ArrayHandle < CountingValueType,
                        dax::cont::internal::ArrayContainerControlTagCounting,
                        DeviceAdapterTag> superclass;
  typedef dax::cont::internal::ArrayPortalCounting<
                                              CountingValueType> PortalType;

  ArrayHandleCounting(CountingValueType startingValue, dax::Id length)
    :superclass(PortalType(startingValue, length))
  {
  }
};

/// A convenience function for creating an ArrayHandleCounting. It takes the
/// value to start counting from and and the number of times to increment.
template<typename CountingValueType>
DAX_CONT_EXPORT
dax::cont::ArrayHandleCounting<CountingValueType> make_ArrayHandleCounting(
                                               CountingValueType startingValue,
                                               dax::Id length)
{
  return dax::cont::ArrayHandleCounting<CountingValueType>(startingValue,
                                                           length);
}


}
}

#endif //__dax_cont_ArrayHandleCounting_h
