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
#ifndef __dax_cuda_cont_internal_DeviceAdapterAlgorithmCuda_h
#define __dax_cuda_cont_internal_DeviceAdapterAlgorithmCuda_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>

#include <dax/cuda/cont/internal/DeviceAdapterTagCuda.h>
#include <dax/cuda/cont/internal/ArrayManagerExecutionCuda.h>

#include <dax/cont/internal/DeviceAdapterAlgorithm.h>

// Here are the actual implementation of the algorithms.
#include <dax/thrust/cont/internal/DeviceAdapterAlgorithmThrust.h>

#include <cuda.h>

namespace dax {
namespace cont {
namespace internal {

template<>
struct DeviceAdapterAlgorithm<dax::cuda::cont::DeviceAdapterTagCuda>
    : public dax::thrust::cont::internal::DeviceAdapterAlgorithmThrust<
          dax::cuda::cont::DeviceAdapterTagCuda>
{

  DAX_CONT_EXPORT static void Synchronize()
  {
    cudaError_t error = cudaDeviceSynchronize();
  }

};

/// CUDA contains its own high resolution timer.
///
template<>
class DeviceAdapterTimerImplementation<dax::cuda::cont::DeviceAdapterTagCuda>
{
public:
  DAX_CONT_EXPORT DeviceAdapterTimerImplementation()
  {
    cudaEventCreate(&this->StartEvent);
    cudaEventCreate(&this->EndEvent);
    this->Reset();
  }
  DAX_CONT_EXPORT ~DeviceAdapterTimerImplementation()
  {
    cudaEventDestroy(this->StartEvent);
    cudaEventDestroy(this->EndEvent);
  }

  DAX_CONT_EXPORT void Reset()
  {
    cudaEventRecord(this->StartEvent, 0);
  }

  DAX_CONT_EXPORT dax::Scalar GetElapsedTime()
  {
    cudaEventRecord(this->EndEvent, 0);
    cudaEventSynchronize(this->EndEvent);
    float elapsedTimeMilliseconds;
    cudaEventElapsedTime(&elapsedTimeMilliseconds,
                         this->StartEvent,
                         this->EndEvent);
    return static_cast<dax::Scalar>(0.001f*elapsedTimeMilliseconds);
  }

private:
  // Copying CUDA events is problematic.
  DeviceAdapterTimerImplementation(const DeviceAdapterTimerImplementation<dax::cuda::cont::DeviceAdapterTagCuda> &);
  void operator=(const DeviceAdapterTimerImplementation<dax::cuda::cont::DeviceAdapterTagCuda> &);

  cudaEvent_t StartEvent;
  cudaEvent_t EndEvent;
};

}
}
} // namespace dax::cont::internal

#endif //__dax_cuda_cont_internal_DeviceAdapterAlgorithmCuda_h
