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

#include <dax/worklet/Square.h>

#include <dax/VectorTraits.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/testing/TestingGridGenerator.h>
#include <dax/cont/testing/Testing.h>

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
  dax::cont::testing::TestGrid<GridType> grid(DIM);
  dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

  std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
  for (dax::Id pointIndex = 0;
       pointIndex < grid->GetNumberOfPoints();
       pointIndex++)
    {
    field[pointIndex]
        = dax::dot(grid->ComputePointCoordinates(pointIndex), trueGradient);
    }
  dax::cont::ArrayHandle<dax::Scalar> fieldHandle =
      dax::cont::make_ArrayHandle(field);

  dax::cont::ArrayHandle<dax::Scalar> squareHandle;

  std::cout << "Running Square worklet" << std::endl;
  dax::cont::Scheduler<> scheduler;
  scheduler.Invoke(dax::worklet::Square(),fieldHandle, squareHandle);

  std::cout << "Checking result" << std::endl;
  std::vector<dax::Scalar> square(grid->GetNumberOfPoints());
  squareHandle.CopyInto(square.begin());
  for (dax::Id pointIndex = 0;
       pointIndex < grid->GetNumberOfPoints();
       pointIndex++)
    {
    dax::Scalar squareValue = square[pointIndex];
    dax::Scalar squareTrue = field[pointIndex]*field[pointIndex];
    DAX_TEST_ASSERT(test_equal(squareValue, squareTrue),
                    "Got bad square");
    }
  }
};


//-----------------------------------------------------------------------------
static void TestSquare()
{
  dax::cont::testing::GridTesting::TryAllGridTypes(TestElevationWorklet());
}
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletSquare(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestSquare);
}
