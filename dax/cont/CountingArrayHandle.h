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
#ifndef __dax_cont_CountingArrayHandle_h
#define __dax_cont_CountingArrayHandle_h

#include <dax/cont/ArrayContainerControlCounting.h>
#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {

/// CountingArrayHandles are a specialization of ArrayHandles. By default it
/// contains a counting implicit array portal. An implicit array is a array
/// portal with a function generating values and holds no memory. The counting
/// implicit array simply returns the value of the index.
typedef dax::cont::ArrayHandle < dax::Id,
                                 dax::cont::ArrayContainerControlTagCounting>
CountingArrayHandle;

/// A convenience function for creating an CountingArrayHandle. It only takes
/// the length of the array and constructs a CountingArrayHandle of that length.
DAX_CONT_EXPORT
CountingArrayHandle make_CountingArrayHandle(dax::Id length)
{
  typedef dax::cont::ArrayPortalCounting PortalType;
  typedef CountingArrayHandle CountingArrayHandleType;
  return CountingArrayHandleType(PortalType(length));
}

}
}

#endif //__dax_cont_CountingArrayHandle_h
