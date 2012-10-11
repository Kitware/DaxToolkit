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

#include <dax/worklet/Elevation.worklet>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
struct TestElevationWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
  {
  dax::cont::internal::TestGrid<GridType> grid(DIM);

  dax::cont::ArrayHandle<dax::Scalar,
                         dax::cont::ArrayContainerControlTagBasic,
                         dax::cont::DeviceAdapterTagSerial> elevationHandle;

  std::cout << "Running Elevation worklet" << std::endl;
  dax::worklet::Elevation elev(dax::make_Vector3(DIM, DIM, DIM),
                               dax::make_Vector3(0.0, 0.0, 0.0));
  dax::cont::Scheduler<> scheduler;
  scheduler.invoke(elev,
                   grid->GetPointCoordinates(),
                   elevationHandle);

  std::cout << "Checking result" << std::endl;
  std::vector<dax::Scalar> elevation(grid->GetNumberOfPoints());
  elevationHandle.CopyInto(elevation.begin());
  dax::Id3 ijk;
  for (ijk[2] = 0; ijk[2] < DIM; ijk[2]++)
    {
    for (ijk[1] = 0; ijk[1] < DIM; ijk[1]++)
      {
      for (ijk[0] = 0; ijk[0] < DIM; ijk[0]++)
        {
        dax::Id pointIndex = grid->ComputePointIndex(ijk);
        dax::Scalar elevationValue = elevation[pointIndex];
        dax::Vector3 pointCoordinates =grid->ComputePointCoordinates(pointIndex);
        // Wrong, but what is currently computed.
        dax::Scalar elevationExpected =
            1.0 - (dax::dot(pointCoordinates, dax::make_Vector3(1.0, 1.0, 1.0))
                   /(3*DIM));
        DAX_TEST_ASSERT(test_equal(elevationValue, elevationExpected),
                        "Got bad elevation.");
        }
      }
    }
  }
};

//-----------------------------------------------------------------------------
void TestElevation()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(TestElevationWorklet(),
                     dax::cont::internal::GridTesting::TypeCheckUniformGrid());
  }



} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletElevation(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestElevation);
}
