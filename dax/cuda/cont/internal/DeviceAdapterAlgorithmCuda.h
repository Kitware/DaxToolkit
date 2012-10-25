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

// Here are the actual implementation of the algorithms.
#include <dax/thrust/cont/internal/DeviceAdapterAlgorithmThrust.h>

namespace dax {
namespace cont {
namespace internal {

template<>
struct DeviceAdapterAlgorithm<dax::cuda::cont::DeviceAdapterTagCuda>
    : public dax::thrust::cont::internal::DeviceAdapterAlgorithmThrust<
          dax::cuda::cont::DeviceAdapterTagCuda>
{  };

}
}
} // namespace dax::cont::internal

#endif //__dax_cuda_cont_internal_DeviceAdapterAlgorithmCuda_h
