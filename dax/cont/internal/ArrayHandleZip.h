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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_cont_internal_ArrayHandleZip_h
#define __dax_cont_internal_ArrayHandleZip_h

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/ArrayContainerControlZip.h>

namespace dax {
namespace cont {
namespace internal {

/// ArrayHandleZip is a specialization of ArrayHandle. It takes two delegate
/// array handle and makes a new handle that access the corresponding entries
/// in these arrays as a pair.
///
template<class FirstArrayHandleType,
         class SecondArrayHandleType>
class ArrayHandleZip
    : public ArrayHandle<
        typename ArrayContainerControlZipTypes<
          FirstArrayHandleType,SecondArrayHandleType>::ValueType,
        typename ArrayContainerControlZipTypes<
          FirstArrayHandleType,SecondArrayHandleType>::ArrayContainerControlTag,
        typename FirstArrayHandleType::DeviceAdapterTag>
{
public:
  typedef typename FirstArrayHandleType::DeviceAdapterTag DeviceAdapterTag;

private:
  typedef dax::cont::internal::ArrayContainerControlZipTypes<
      FirstArrayHandleType,SecondArrayHandleType> ZipTypes;

  typedef dax::cont::ArrayHandle<
      typename ZipTypes::ValueType,
      typename ZipTypes::ArrayContainerControlTag,
      DeviceAdapterTag> Superclass;

public:
  typedef typename ZipTypes::ValueType ValueType;
  typedef dax::cont::internal::ArrayContainerControlTagZip<
      FirstArrayHandleType,SecondArrayHandleType> ArrayContainerControlTag;

private:
  typedef dax::cont::internal::ArrayTransfer<
      ValueType,ArrayContainerControlTag,DeviceAdapterTag> ArrayTransferType;

public:
  ArrayHandleZip(const FirstArrayHandleType &firstArray,
                 const SecondArrayHandleType &secondArray)
    : Superclass(
        typename ZipTypes::ArrayContainerControlType(firstArray,secondArray),
        true,
        ArrayTransferType(firstArray,secondArray),
        false)
  {
  }
};

/// A convenience function for creating an ArrayHandleZip. It takes the two
/// arrays to be zipped together.
///
template<class FirstArrayHandleType,
         class SecondArrayHandleType>
DAX_CONT_EXPORT
dax::cont::internal::ArrayHandleZip<FirstArrayHandleType,
                                    SecondArrayHandleType>
make_ArrayHandleZip(const FirstArrayHandleType &first,
                    const SecondArrayHandleType &second)
{
  typedef typename FirstArrayHandleType::DeviceAdapterTag DeviceAdapterTag;
  return dax::cont::internal::ArrayHandleZip<
      FirstArrayHandleType,SecondArrayHandleType>(first, second);
}

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayHandleZip_h
