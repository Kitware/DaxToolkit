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

namespace dax{
namespace cuda{
namespace opengl {


void SetCudaGLDevice(int id)
{
  cudaGLSetGLDevice(id);
}


}
}
} //namespace

#endif
