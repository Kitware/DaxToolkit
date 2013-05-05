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

#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/testing/TestingGridGenerator.h>
#include <dax/cont/testing/Testing.h>

#include <dax/worklet/Magnitude.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
struct TestMagnitudeWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
  {
  dax::cont::testing::TestGrid<GridType> grid(DIM);

  dax::cont::ArrayHandle<dax::Scalar,
                         dax::cont::ArrayContainerControlTagBasic,
                         dax::cont::DeviceAdapterTagSerial> magnitudeHandle;

  std::cout << "Running Magnitude worklet" << std::endl;
  dax::cont::Scheduler<> scheduler;
  scheduler.Invoke(dax::worklet::Magnitude(),
                   grid->GetPointCoordinates(),
                   magnitudeHandle);

  std::cout << "Checking result" << std::endl;
  dax::Id numPoints = grid->GetNumberOfPoints();
  std::vector<dax::Scalar> magnitude(numPoints);
  magnitudeHandle.CopyInto(magnitude.begin());
  for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
    {
    dax::Scalar magnitudeValue = magnitude[pointIndex];
    dax::Vector3 pointCoordinates = grid.GetPointCoordinates(pointIndex);
    dax::Scalar magnitudeExpected =
        sqrt(dax::dot(pointCoordinates, pointCoordinates));
    DAX_TEST_ASSERT(test_equal(magnitudeValue, magnitudeExpected),
                    "Got bad magnitude.");
    }
  }
};

//-----------------------------------------------------------------------------
void TestMagnitude()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes(TestMagnitudeWorklet());
  }



} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletMagnitude(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestMagnitude);
}
