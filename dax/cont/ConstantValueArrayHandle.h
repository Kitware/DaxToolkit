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
#ifndef __dax_cont_ConstantValueArrayHandle_h
#define __dax_cont_ConstantValueArrayHandle_h


#include <dax/cont/ArrayContainerControlConstantValue.h>
#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {

/// ConstantValueArrayHandles are a specialization of ArrayHandles. By default
/// it contains a constant value array portal. A ConstantValueArrayPortal is an
/// array portal with a function generating a defined constant value. Like an
/// implicit array, this too holds no memory.  The ConstantValue array simply
/// returns a single value for each of the index.
template <typename ConstantValueType>
class ConstantValueArrayHandle: public ArrayHandle < ConstantValueType,
    dax::cont::ArrayContainerControlTagConstantValue
     <dax::cont::ArrayPortalConstantValue,ConstantValueType> >
{
public:
  typedef ArrayHandle < ConstantValueType,
      dax::cont::ArrayContainerControlTagConstantValue
       <dax::cont::ArrayPortalConstantValue,ConstantValueType> >
  superclass;
  typedef typename dax::cont::ArrayPortalConstantValue<ConstantValueType> PortalType;

  ConstantValueArrayHandle(ConstantValueType value,dax::Id length)
    :superclass(PortalType(value,length))
  {
  }
};

/// A convenience function for creating a ConstantValueArrayHandle. It only
/// takes constant value and the lenght of the array and constructs a
/// ConstantValueArrayHandle of the specified length. The array returns a
/// specified constant value for each index.
template<typename ConstantValueType>
DAX_CONT_EXPORT
ConstantValueArrayHandle<ConstantValueType>
make_ConstantValueArrayHandle(ConstantValueType value,dax::Id length)
{
  return ConstantValueArrayHandle<ConstantValueType>(value,length);
}

}
}

#endif //__dax_cont_ConstantValueArrayHandle_h