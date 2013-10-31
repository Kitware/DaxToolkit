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
#ifndef __dax__cuda__opengl__SetOpenGLDevice_h
#define __dax__cuda__opengl__SetOpenGLDevice_h

#include <cuda.h>
#include <cuda_gl_interop.h>

#include <dax/cont/ErrorExecution.h>

namespace dax{
namespace cuda{
namespace opengl {


static void SetCudaGLDevice(int id)
{
  cudaError_t cError = cudaGLSetGLDevice(id);
  if(cError != cudaSuccess)
    {
    std::string cuda_error_msg("Unable to setup cuda/opengl interop. Error: ");
    cuda_error_msg.append(cudaGetErrorString(cError));
    throw dax::cont::ErrorExecution(cuda_error_msg);
    }
}


}
}
} //namespace

#endif
