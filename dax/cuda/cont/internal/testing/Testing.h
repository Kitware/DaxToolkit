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
#ifndef __dax_cuda_cont_internal_Testing_h
#define __dax_cuda_cont_internal_Testing_h

#include <dax/cont/internal/testing/Testing.h>

#include <cuda.h>

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

struct Testing
{
public:
  static DAX_CONT_EXPORT int CheckCudaBeforeExit(int result)
  {
    cudaError_t cudaError = cudaPeekAtLastError();
    if (cudaError != cudaSuccess)
      {
      std::cout << "***** Unchecked Cuda error." << std::endl
                << cudaGetErrorString(cudaError) << std::endl;
      return 1;
      }
    else
      {
      std::cout << "No Cuda error detected." << std::endl;
      }
    return result;
  }

  template<class Func>
  static DAX_CONT_EXPORT int Run(Func function)
  {
    int result = dax::cont::internal::Testing::Run(function);
    return CheckCudaBeforeExit(result);
  }
};

}
}
}
} // namespace dax::cuda::cont::internal

#endif //__dax_cuda_cont_internal_Testing_h
