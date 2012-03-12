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

#ifndef __dax_openmp_cont_internal_ArrayContainerExecutionThrust_h
#define __dax_openmp_cont_internal_ArrayContainerExecutionThrust_h

#include <dax/openmp/cont/internal/SetThrustForOpenMP.h>

#include <dax/thrust/cont/internal/ArrayContainerExecutionThrust.h>

namespace dax {
namespace openmp {
namespace cont {
namespace internal {

/// Manages an OpenMP device array. Can allocate the array of the given type on
/// the device, copy data do and from it, and release the memory. The memory is
/// also released when this object goes out of scope. This class is currently
/// dumb about the copying. It might make unnecessary copies.
///
template<typename T>
class ArrayContainerExecutionThrust
    : public dax::thrust::cont::internal::ArrayContainerExecutionThrust<T>
{
public:
  typedef T ValueType;
};

}
}
}
}

#endif // __dax_openmp_cont_internal_ArrayContainerExecutionThrust_h
