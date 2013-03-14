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

#include <dax/cont/internal/testing/TestingGridGenerator.h>
#include <dax/cont/internal/testing/Testing.h>

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
  dax::cont::internal::TestGrid<GridType> grid(DIM);

  dax::cont::ArrayHandle<dax::Scalar,
                         dax::cont::ArrayContainerControlTagBasic,
                         dax::cont::DeviceAdapterTagSerial> magnitudeHandle;

  std::cout << "Running Magnitude worklet" << std::endl;
  dax::cont::Scheduler<> scheduler;
  scheduler.Invoke(dax::worklet::Magnitude(),
                   grid->GetPointCoordinates(),
                   magnitudeHandle);

  std::cout << "Checking result" << std::endl;
  std::vector<dax::Scalar> magnitude(grid->GetNumberOfPoints());
  magnitudeHandle.CopyInto(magnitude.begin());
  dax::Id3 ijk;
  for (ijk[2] = 0; ijk[2] < DIM; ijk[2]++)
    {
    for (ijk[1] = 0; ijk[1] < DIM; ijk[1]++)
      {
      for (ijk[0] = 0; ijk[0] < DIM; ijk[0]++)
        {
        dax::Id pointIndex = grid->ComputePointIndex(ijk);
        dax::Scalar magnitudeValue = magnitude[pointIndex];
        dax::Vector3 pointCoordinates =grid->ComputePointCoordinates(pointIndex);
        // Wrong, but what is currently computed.
        dax::Scalar magnitudeExpected =
            sqrt(dax::dot(pointCoordinates, pointCoordinates));
        DAX_TEST_ASSERT(test_equal(magnitudeValue, magnitudeExpected),
                        "Got bad magnitude.");
        }
      }
    }
  }
};

//-----------------------------------------------------------------------------
void TestMagnitude()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(TestMagnitudeWorklet(),
                     dax::cont::internal::GridTesting::TypeCheckUniformGrid());
  }



} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletMagnitude(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestMagnitude);
}
