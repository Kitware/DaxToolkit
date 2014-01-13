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
#ifndef __dax_cont_ArrayHandleTransform_h
#define __dax_cont_ArrayHandleTransform_h

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/ArrayContainerControlTransform.h>

namespace dax {
namespace cont {

/// ArrayHandleTransforms is a specialization of ArrayHandle.
/// It takes a delegate array handle and makes a new handle that calls
/// a given functor with the element at a given index, and returns the
/// result of the functor as the value of this array at that position
///

template <typename ValueType,
          class ArrayHandleType,
          class FunctorType,
          class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class ArrayHandleTransform
    : public dax::cont::ArrayHandle <
          ValueType,
          typename internal::ArrayHandleTransformTraits<ValueType,
                                                        ArrayHandleType,
                                                        FunctorType>::Tag,
          DeviceAdapterTag >
{
private:
  typedef typename internal::ArrayHandleTransformTraits<ValueType,
                                ArrayHandleType,
                                FunctorType> ArrayTraits;

  typedef typename ArrayTraits::Tag Tag;
  typedef typename ArrayTraits::ContainerType ContainerType;

 public:
  typedef dax::cont::ArrayHandle < ValueType,
                                   Tag,
                                   DeviceAdapterTag > Superclass;

  ArrayHandleTransform()
    : Superclass( )
  {
  }

  ArrayHandleTransform(const ArrayHandleType& handle)
    : Superclass( typename Superclass::PortalConstControl(handle,
                                                          FunctorType()) )
  {
  }

  ArrayHandleTransform(const ArrayHandleType& handle, FunctorType f)
    : Superclass( typename Superclass::PortalConstControl(handle,f) )
  {
  }

};

/// make_ArrayHandleTransform is convenience function to generate an
/// ArrayHandleTransform.  It takes in an ArrayHandle and a functor
/// to apply to each element of the Handle.

template <typename T, typename HandleType, typename FunctorType>
DAX_CONT_EXPORT
dax::cont::ArrayHandleTransform<T, HandleType, FunctorType>
make_ArrayHandleTransform(HandleType handle, FunctorType functor)
{
  return ArrayHandleTransform<T,HandleType,FunctorType>(handle,functor);
}


}
} // namespace dax::cont

#endif //__dax_cont_ArrayHandleTransform_h
