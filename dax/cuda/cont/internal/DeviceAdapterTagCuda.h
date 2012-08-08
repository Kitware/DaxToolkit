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
#ifndef __dax_cuda_cont_internal_DeviceAdapterTagCuda_h
#define __dax_cuda_cont_internal_DeviceAdapterTagCuda_h

#include <dax/cuda/cont/internal/SetThrustForCuda.h>

#include <dax/thrust/cont/internal/DeviceAdapterTagThrust.h>

namespace dax {
namespace cuda {
namespace cont {

/// A DeviceAdapter that uses Cuda.  To use this adapter, the code must be
/// compiled with nvcc.
///
struct DeviceAdapterTagCuda
    : public dax::thrust::cont::internal::DeviceAdapterTagThrust
{  };

}
}
} // namespace dax::cuda::cont

#endif //__dax_cuda_cont_internal_DeviceAdapterTagCuda_h
