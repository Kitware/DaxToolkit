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

#ifndef __dax_cont_InclusiveScanDebug_h
#define __dax_cont_InclusiveScanDebug_h

#include <dax/cont/internal/ArrayContainerExecutionCPU.h>
#include <numeric>

namespace dax {
namespace cont {

template<typename T>
DAX_CONT_EXPORT T inclusiveScanDebug(
    const dax::cont::internal::ArrayContainerExecutionCPU<T> &from,
    dax::cont::internal::ArrayContainerExecutionCPU<T> &to)
{
  typedef typename dax::cont::internal::ArrayContainerExecutionCPU<T>::iterator Iter;
  Iter result = std::partial_sum(from.begin(),from.end(),to.begin());

  //return the value at the last index in the array, as that is the size
  if(std::distance(to.begin(),result) > 0)
    {
    return *(--result);
    }
  return T(0);
}

}
} // namespace dax::cont

#endif //__dax_cont_InclusiveScanDebug_h
