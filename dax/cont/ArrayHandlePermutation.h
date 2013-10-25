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
#ifndef __dax_cont_ArrayHandlePermutation_h
#define __dax_cont_ArrayHandlePermutation_h

#include <dax/Types.h>

#include <dax/cont/internal/ArrayContainerControlPermutation.h>
#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {

/// ArrayHandlePermutation are a specialization of ArrayHandles.
/// It takes two delegate array handle and makes a new handle that access
/// the corresponding entries in the second handle given the re-indexing scheme
/// of the first array. This generally requires that the key (first param) is
/// of an integer type.
///

template <typename IndicesHandleType,
          typename ValueHandleType,
          class DeviceAdapterTag_ = DAX_DEFAULT_DEVICE_ADAPTER_TAG >
class ArrayHandlePermutation
    : public ArrayHandle <
      typename dax::cont::internal::ArrayContainerControlPermutationTypes<
               IndicesHandleType,ValueHandleType>::ValueType,
      typename dax::cont::internal::ArrayContainerControlPermutationTypes<
               IndicesHandleType,ValueHandleType>::ArrayContainerControlTag,
      DeviceAdapterTag_>
{
private:
  typedef dax::cont::internal::ArrayContainerControlPermutationTypes<
      IndicesHandleType,ValueHandleType> PermTypes;

public:
  typedef typename PermTypes::ValueType ValueType;
  typedef typename PermTypes::ArrayContainerControlTag ArrayContainerControlTag;
  typedef DeviceAdapterTag_ DeviceAdapterTag;

   typedef dax::cont::ArrayHandle< ValueType, ArrayContainerControlTag,
                                   DeviceAdapterTag> Superclass;
private:
  typedef dax::cont::internal::ArrayTransfer<
      ValueType,ArrayContainerControlTag,DeviceAdapterTag> ArrayTransferType;

public:
  ArrayHandlePermutation(const IndicesHandleType& keyHandle,
                         const ValueHandleType& valueHandle)
    : Superclass(
        typename PermTypes::ArrayContainerControlType(keyHandle,valueHandle),
        true,
        ArrayTransferType(keyHandle,valueHandle),
        false)
  {
  }

};

/// make_ArrayHandlePermutation is convenience function to generate an
/// ArrayHandlePermutation.  It takes in a Indices Handle and Value Handle as
/// inputs to generate a ArrayHandlePermutation.
template <typename IndicesHandle, typename ValueHandle>
DAX_CONT_EXPORT
dax::cont::ArrayHandlePermutation<IndicesHandle,ValueHandle>
make_ArrayHandlePermutation(IndicesHandle key, ValueHandle value)
{
  return ArrayHandlePermutation<IndicesHandle,ValueHandle>(key,value);
}

}
}

#endif //__dax_cont_ArrayHandlePermutation_h
