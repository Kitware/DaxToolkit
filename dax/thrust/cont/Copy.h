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

#ifndef __dax_thrust_cont_Copy_h
#define __dax_thrust_cont_Copy_h

#include <dax/Types.h>
#include <dax/thrust/cont/internal/ArrayContainerExecutionThrust.h>

#include <thrust/copy.h>

namespace dax {
namespace thrust {
namespace cont {


template<typename T>
DAX_CONT_EXPORT void copy(
    const dax::thrust::cont::internal::ArrayContainerExecutionThrust<T> &from,
    dax::thrust::cont::internal::ArrayContainerExecutionThrust<T> &to)
{
  ::thrust::copy(from.GetBeginThrustIterator(),
                 from.GetEndThrustIterator(),
                 to.GetBeginThrustIterator());
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_Copy_h
