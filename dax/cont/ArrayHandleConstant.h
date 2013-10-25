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


#include <dax/cont/internal/ArrayContainerControlConstant.h>
#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {

/// ArrayHandleConstants are a specialization of ArrayHandles. By default
/// it contains a constant value array portal. A ConstantValueArrayPortal is an
/// array portal with a function generating a defined constant value. Like an
/// implicit array, this too holds no memory.  The ConstantValue array simply
/// returns a single value for each of the index.
template <typename ConstantValueType,
          class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class ArrayHandleConstant: public ArrayHandle < ConstantValueType,
  dax::cont::internal::ArrayContainerControlTagConstant,
  DeviceAdapterTag >
{
public:
  typedef dax::cont::ArrayHandle < ConstantValueType,
      dax::cont::internal::ArrayContainerControlTagConstant,
      DeviceAdapterTag > Superclass;
  typedef typename dax::cont::internal::ArrayPortalConstant<
                                          ConstantValueType> PortalType;

  ArrayHandleConstant(ConstantValueType value, dax::Id length)
    :Superclass(PortalType(value,length))
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
ArrayHandleConstant<ConstantValueType,DAX_DEFAULT_DEVICE_ADAPTER_TAG>
make_ArrayHandleConstant(ConstantValueType value,
                         dax::Id length)
{
  return make_ArrayHandleConstant(value,length,DAX_DEFAULT_DEVICE_ADAPTER_TAG());
}

}
}

#endif //__dax_cont_ArrayHandleConstant_h
