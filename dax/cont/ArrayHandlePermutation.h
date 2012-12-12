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

#include <dax/cont/ArrayContainerControlPermutation.h>
#include <dax/cont/ArrayHandle.h>

namespace dax {
namespace cont {

/// ArrayHandlePermutation are a speclization of ArrayHandles. By default it
/// contains an ArrayPortalPermutation. An ArrayPortalPermutation is an array
/// portal which takes in two array portals namely KeyPortal and
/// ValuePortal. The KeyPortal has the holds the index into the ValuePotal. So
/// for example if we want to access element 10 from a PermutationArrayHandle
/// names array we get array[10] = ValuePotal[ KeyPortal[10]]. Like an implicit
/// array this too does not hold any memory.
template <typename KeyPortalType, typename ValuePortalType>
class ArrayHandlePermutation: public ArrayHandle <dax::Id,
      ArrayContainerControlTagPermutation
        <KeyPortalType,ValuePortalType> >
{
public:
  typedef ArrayHandle <dax::Id,ArrayContainerControlTagPermutation
    <KeyPortalType,ValuePortalType> >
  superclass;

  typedef typename dax::cont::ArrayPortalPermutation <KeyPortalType,ValuePortalType>
    PortalType;

  ArrayHandlePermutation(KeyPortalType keyPortal, ValuePortalType valuePortal)
    :superclass(PortalType(keyPortal,valuePortal))
  {
  }
};

/// make_ArrayHandlePermutation is convenience funciton to generate an
/// ArrayHandlePermutation.  It takes in a KeyPortal and a Value portal as
/// inputs to generate a ArrayHandlePermutation.
template <typename KeyPortalType, typename ValuePortalType>
DAX_CONT_EXPORT
ArrayHandlePermutation<KeyPortalType,ValuePortalType>
make_ArrayHandlePermutation(KeyPortalType key,ValuePortalType value)
{
  return ArrayHandlePermutation<KeyPortalType,ValuePortalType>(key,value);
}

}
}

#endif //__dax_cont_ArrayHandlePermutation_h
