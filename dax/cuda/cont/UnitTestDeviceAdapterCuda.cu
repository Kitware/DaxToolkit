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

#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR
#define BOOST_SP_DISABLE_THREADS

#include <dax/cuda/cont/DeviceAdapterCuda.h>

#include <dax/cont/internal/TestingDeviceAdapter.h>
#include <dax/cuda/cont/internal/Testing.h>

int UnitTestDeviceAdapterCuda(int, char *[])
{
  int result =  dax::cont::internal::TestingDeviceAdapter
      <dax::cuda::cont::DeviceAdapterTagCuda>::Run();
  return dax::cuda::cont::internal::Testing::CheckCudaBeforeExit(result);
}
