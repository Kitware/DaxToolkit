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

#ifndef __dax_thrust_cont_InclusiveScan_h
#define __dax_thrust_cont_InclusiveScan_h

#include <dax/Types.h>
#include <dax/thrust/cont/internal/ArrayContainerExecutionThrust.h>
#include <thrust/scan.h>

namespace dax {
namespace thrust {
namespace cont {

template<typename T>
DAX_CONT_EXPORT T inclusiveScan(
    const dax::thrust::cont::internal::ArrayContainerExecutionThrust<T> &from,
    dax::thrust::cont::internal::ArrayContainerExecutionThrust<T> &to)
{
  typedef typename dax::thrust::cont::internal::ArrayContainerExecutionThrust<T>::iterator Iter;
  Iter result = ::thrust::inclusive_scan(from.GetBeginThrustIterator(),
                                         from.GetEndThrustIterator(),
                                         to.GetBeginThrustIterator());

  //return the value at the last index in the array, as that is the size
  if(::thrust::distance(to.GetBeginThrustIterator(),result) > 0)
    {
    return *(--result);
    }
  return T(0);
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_cont_InclusiveScan_h
