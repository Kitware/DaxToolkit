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

#include <dax/cont/DeviceAdapterSerial.h>

#include <dax/cont/internal/testing/TestingDeviceAdapter.h>

int UnitTestDeviceAdapterSerial(int, char *[])
{
  return dax::cont::internal::TestingDeviceAdapter
      <dax::cont::DeviceAdapterTagSerial>::Run();
}
