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
#ifndef __dax_cont_ArrayHandleConstantValue_h
#define __dax_cont_ArrayHandleConstantValue_h


#include <dax/cont/internal/ArrayContainerControlConstantValue.h>
#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {

/// ArrayHandleConstantValues are a specialization of ArrayHandles. By default
/// it contains a constant value array portal. A ConstantValueArrayPortal is an
/// array portal with a function generating a defined constant value. Like an
/// implicit array, this too holds no memory.  The ConstantValue array simply
/// returns a single value for each of the index.
template <typename ConstantValueType,
          class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class ArrayHandleConstantValue: public ArrayHandle < ConstantValueType,
  dax::cont::internal::ArrayContainerControlTagConstantValue<ConstantValueType>,
  DeviceAdapterTag >
{
public:
  typedef dax::cont::ArrayHandle < ConstantValueType,
    dax::cont::internal::ArrayContainerControlTagConstantValue<ConstantValueType>,
    DeviceAdapterTag > superclass;
  typedef typename dax::cont::internal::ArrayPortalConstantValue<ConstantValueType> PortalType;

  ArrayHandleConstantValue(ConstantValueType value,dax::Id length)
    :superclass(PortalType(value,length))
  {
  }
};

/// A convenience function for creating a ArrayHandleConstantValue. It only
/// takes constant value and the lenght of the array and constructs a
/// ArrayHandleConstantValue of the specified length. The array returns a
/// specified constant value for each index.
template<typename ConstantValueType>
DAX_CONT_EXPORT
ArrayHandleConstantValue<ConstantValueType>
make_ArrayHandleConstantValue(ConstantValueType value,dax::Id length)
{
  return ArrayHandleConstantValue<ConstantValueType>(value,length);
}

}
}

#endif //__dax_cont_ArrayHandleConstantValue_h
