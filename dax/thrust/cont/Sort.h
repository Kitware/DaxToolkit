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

#ifndef __dax_thrust_cont_Sort_h
#define __dax_thrust_cont_Sort_h

#include <dax/Types.h>
#include <dax/thrust/cont/internal/ArrayContainerExecutionThrust.h>

#include <thrust/sort.h>

namespace dax {
namespace thrust {
namespace cont {


template<typename T>
DAX_CONT_EXPORT void sort(
    dax::thrust::cont::internal::ArrayContainerExecutionThrust<T> &values)
{
  ::thrust::sort(values.GetBeginThrustIterator(),values.GetEndThrustIterator());
}

}
}
} // namespace dax::thrust::cont

#endif //__dax_thrust_cont_Sort_h
