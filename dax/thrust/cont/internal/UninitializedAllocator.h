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
#ifndef __dax_thrust_cont_internal_UninitializedAllocator_h
#define __dax_thrust_cont_internal_UninitializedAllocator_h

#include <dax/thrust/cont/internal/CheckThrustBackend.h>

// Disable GCC warnings we check Dax for but Thrust does not.
#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#endif // gcc version >= 4.6
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 2)
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif // gcc version >= 4.2
#endif // gcc && !CUDA

#include <thrust/device_malloc_allocator.h>


#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif // gcc version >= 4.6
#endif // gcc && !CUDA


namespace dax {
namespace thrust {
namespace cont {
namespace internal {

// UninitializedAllocator is an allocator which
// derives from device_allocator and which has a
// no-op construct member function
template<typename T>
  struct UninitializedAllocator
    : ::thrust::device_malloc_allocator<T>
{
  // note that construct is annotated as
  // a __host__ __device__ function
  __host__ __device__
  void construct(T *p)
  {
    // no-op
  }
};

}
}
}
}

#endif
