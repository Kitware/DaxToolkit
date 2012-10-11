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

#include <dax/worklet/testing/Assert.worklet>

#include <dax/Types.h>
#include <dax/VectorTraits.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/testing/Testing.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
static void TestAssert()
{
  std::vector<dax::Scalar> array(DIM);
  dax::cont::ArrayHandle<dax::Scalar> arrayHandle =
      dax::cont::make_ArrayHandle(array);

  std::cout << "Running field map worklet that errors" << std::endl;
  bool gotError = false;
  try
    {
    dax::cont::Schedule<>()(dax::worklet::testing::Assert(),arrayHandle);
    }
  catch (dax::cont::ErrorExecution error)
    {
    std::cout << "Got expected ErrorExecution object." << std::endl;
    std::cout << error.GetMessage() << std::endl;
    gotError = true;
    }

//The kernel using a NDEBUG check to determine if the assert is checked
//so we need to make sure that in release mode the test still passes by
//making sure an error isn't thrown.
#ifndef NDEBUG
  DAX_TEST_ASSERT(gotError, "Never got the error thrown.");
#else
  DAX_TEST_ASSERT(!gotError, "Debug Assert in release mode, not allowed.");
#endif


}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletAssert(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestAssert);
}
