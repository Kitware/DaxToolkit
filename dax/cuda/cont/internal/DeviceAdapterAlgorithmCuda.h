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

}
}
} // namespace dax::cont::internal

namespace dax {
namespace cont {

// Add prototype for Timer template, which might not be defined yet.
template<class DeviceAdapter> class Timer;

/// CUDA contains its own high resolution timer.
///
template<>
class Timer<dax::cuda::cont::DeviceAdapterTagCuda>
{
public:
  DAX_CONT_EXPORT Timer()
  {
    cudaEventCreate(&this->StartEvent);
    cudaEventCreate(&this->EndEvent);
    this->Reset();
  }
  DAX_CONT_EXPORT ~Timer()
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
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, this->StartEvent, this->EndEvent);
    return static_cast<dax::Scalar>(elapsedTime);
  }

private:
  // Copying CUDA events is problematic.
  Timer(const Timer<dax::cuda::cont::DeviceAdapterTagCuda> &);
  void operator=(const Timer<dax::cuda::cont::DeviceAdapterTagCuda> &);

  cudaEvent_t StartEvent;
  cudaEvent_t EndEvent;
};

}
} // namespace dax::cont

#endif //__dax_cuda_cont_internal_DeviceAdapterAlgorithmCuda_h
