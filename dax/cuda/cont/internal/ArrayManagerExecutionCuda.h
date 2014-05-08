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
#ifndef __dax_cuda_cont_internal_ArrayManagerExecutionCuda_h
#define __dax_cuda_cont_internal_ArrayManagerExecutionCuda_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>

#include <dax/cuda/cont/internal/DeviceAdapterTagCuda.h>

#include <dax/cont/internal/ArrayManagerExecution.h>
#include <dax/thrust/cont/internal/ArrayManagerExecutionThrustDevice.h>

// These must be placed in the dax::cont::internal namespace so that
// the template can be found.

namespace dax {
namespace cont {
namespace internal {

template <typename T, class ArrayContainerTag>
class ArrayManagerExecution
    <T, ArrayContainerTag, dax::cuda::cont::DeviceAdapterTagCuda>
    : public dax::thrust::cont::internal::ArrayManagerExecutionThrustDevice
        <T, ArrayContainerTag>
{
public:
  typedef dax::thrust::cont::internal::ArrayManagerExecutionThrustDevice
      <T, ArrayContainerTag> Superclass;
  typedef typename Superclass::ValueType ValueType;
  typedef typename Superclass::PortalType PortalType;
  typedef typename Superclass::PortalConstType PortalConstType;

  template<class PortalControl>
  DAX_CONT_EXPORT void LoadDataForInput(PortalControl arrayPortal)
  {
    try
      {
      this->Superclass::LoadDataForInput(arrayPortal);
      }
    catch (dax::cont::ErrorControlOutOfMemory error)
      {
      // Thrust does not seem to be clearing the CUDA error, so do it here.
      cudaError_t cudaError = cudaPeekAtLastError();
      if (cudaError == cudaErrorMemoryAllocation)
        {
        cudaGetLastError();
        }
      throw error;
      }
  }

  DAX_CONT_EXPORT void AllocateArrayForOutput(
      dax::cont::internal::ArrayContainerControl<ValueType,ArrayContainerTag>
        &container,
      dax::Id numberOfValues)
  {
    try
      {
      this->Superclass::AllocateArrayForOutput(container, numberOfValues);
      }
    catch (dax::cont::ErrorControlOutOfMemory error)
      {
      // Thrust does not seem to be clearing the CUDA error, so do it here.
      cudaError_t cudaError = cudaPeekAtLastError();
      if (cudaError == cudaErrorMemoryAllocation)
        {
        cudaGetLastError();
        }
      throw error;
      }
  }
};

}
}
} // namespace dax::cont::internal


#endif //__dax_cuda_cont_internal_ArrayManagerExecutionCuda_h
