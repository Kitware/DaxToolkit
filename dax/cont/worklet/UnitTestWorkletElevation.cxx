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

// These header files help tease out when the default template arguments to
// ArrayHandle are inappropriately used.
#include <dax/cont/internal/ArrayContainerControlError.h>
#include <dax/cont/internal/DeviceAdapterError.h>

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/DeviceAdapterSerial.h>

#include <dax/cont/internal/TestingGridGenerator.h>
#include <dax/cont/internal/Testing.h>

#include <dax/cont/worklet/Elevation.h>
#include <dax/cont/ArrayHandle.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

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
    dax::cont::worklet::Elevation(grid.GetRealGrid(),
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
          dax::Scalar elevationExpected
              = sqrt(dax::dot(pointCoordinates, pointCoordinates));
          DAX_TEST_ASSERT(test_equal(elevationValue, elevationExpected),
                          "Got bad elevation.");
          }
        }
      }
    }
};

//-----------------------------------------------------------------------------
static void TestElevation()
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
