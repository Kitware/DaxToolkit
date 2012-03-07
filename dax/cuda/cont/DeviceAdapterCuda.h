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

#ifndef __dax_cuda_cont_DeviceAdapterCuda_h
#define __dax_cuda_cont_DeviceAdapterCuda_h

#ifdef DAX_DEFAULT_DEVICE_ADAPTER
#undef DAX_DEFAULT_DEVICE_ADAPTER
#endif

#define DAX_DEFAULT_DEVICE_ADAPTER ::dax::cuda::cont::DeviceAdapterCuda

//#define DAX_CUDA_NATIVE_SCHEDULE

#ifdef DAX_CUDA_NATIVE_SCHEDULE
#include <dax/cuda/cont/ScheduleCuda.h>
#else
#include <dax/cuda/cont/ScheduleThrust.h>
#endif

#include <dax/cuda/cont/internal/ArrayContainerExecutionThrust.h>

namespace dax {
namespace cuda {
namespace cont {

/// An implementation of DeviceAdapter that will schedule execution on a GPU
/// using CUDA.
///
struct DeviceAdapterCuda
{
  template<class Functor, class Parameters>
  static void Schedule(Functor functor,
                       Parameters parameters,
                       dax::Id numInstances)
  {
#ifdef DAX_CUDA_NATIVE_SCHEDULE
    dax::cuda::cont::scheduleCuda(functor, parameters, numInstances);
#else
    dax::cuda::cont::scheduleThrust(functor, parameters, numInstances);
#endif
  }

  template<typename T>
  class ArrayContainerExecution
      : public dax::cuda::cont::internal::ArrayContainerExecutionThrust<T>
  { };
};

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_DeviceAdapterCuda_h
