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
/// ArrayHandleCountings are a specialization of ArrayHandles. By default it
/// contains a counting implicit array portal. An implicit array is a array
/// portal with a function generating values and holds no memory. The counting
/// implicit array simply returns the value of the index.
template <class DeviceAdapterTag = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class ArrayHandleCounting : public ArrayHandle < dax::Id,
                         dax::cont::internal::ArrayContainerControlTagCounting,
                         DeviceAdapterTag >
{
public:
  typedef dax::cont::ArrayHandle < dax::Id,
                        dax::cont::internal::ArrayContainerControlTagCounting,
                        DeviceAdapterTag> superclass;
  typedef dax::cont::internal::ArrayPortalCounting PortalType;

  ArrayHandleCounting(dax::Id length)
    :superclass(PortalType(length))
  {
  }
};

/// A convenience function for creating an ArrayHandleCounting. It only takes
/// the length of the array and constructs a ArrayHandleCounting of that length.
DAX_CONT_EXPORT
dax::cont::ArrayHandleCounting< > make_ArrayHandleCounting(dax::Id length)
{
  return dax::cont::ArrayHandleCounting< >(length);
}

}
}

#endif //__dax_cont_ArrayHandleCounting_h
