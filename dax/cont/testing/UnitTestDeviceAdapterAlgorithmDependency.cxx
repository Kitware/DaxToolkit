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

// This tests a previous problem where code templated on the device adapter and
// used one of the device adapter algorithms (for example, the dispatcher) had
// to be declared after any device adapter it was ever used with.

#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR

#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/DispatcherMapField.h>

#include <dax/cont/testing/Testing.h>

// Important for this test!
//This file must be included after DispatcherMapField.h
#include <dax/cont/DeviceAdapterSerial.h>

namespace {

struct ExampleWorklet : public dax::exec::WorkletMapField
{
  typedef void ControlSignature(FieldIn);
  typedef void ExecutionSignature(_1);

  template <typename T>
  void operator()(T daxNotUsed(v)) const {  }
};

void CheckPostDefinedDeviceAdapter()
{
  dax::cont::DispatcherMapField<ExampleWorklet,
        dax::cont::DeviceAdapterTagSerial> dispatcher;
  dispatcher.Invoke(1);

  // Nothing to really check. If this compiles, then the test is probably
  // successful. UnitTestSchedule and the DeviceAdapter tests will check the
  // actual operation of scheduling.
}

} // anonymous namespace

int UnitTestDeviceAdapterAlgorithmDependency(int, char *[])
{
  return dax::cont::testing::Testing::Run(CheckPostDefinedDeviceAdapter);
}
