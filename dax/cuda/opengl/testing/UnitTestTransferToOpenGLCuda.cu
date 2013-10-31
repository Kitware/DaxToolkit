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

//silence boost threading warnings when using cuda
#define BOOST_SP_DISABLE_THREADS

//This sets up testing with the cuda device adapter
#include <dax/cuda/cont/DeviceAdapterCuda.h>
#include <dax/cuda/cont/internal/testing/Testing.h>

#include <dax/opengl/testing/TestingOpenGLInterop.h>

int UnitTestTransferToOpenGLCuda(int, char *[])
{
  int result = 1;
  result = dax::opengl::testing::TestingOpenGLInterop
                           <dax::cuda::cont::DeviceAdapterTagCuda >::Run();
  return dax::cuda::cont::internal::Testing::CheckCudaBeforeExit(result);
}
