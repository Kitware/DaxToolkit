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

#include <dax/cont/testing/TestingGridGenerator.h>
#include <dax/cont/testing/Testing.h>

#include <dax/worklet/CellGradient.h>

#include <dax/CellTraits.h>
#include <dax/Types.h>
#include <dax/VectorTraits.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>

#include <vector>

namespace {

const dax::Id DIM = 8;

//-----------------------------------------------------------------------------
template<typename CellTag>
void verifyGradient(
    const dax::cont::testing::CellCoordinates<CellTag> &pointCoordinates,
    const dax::Vector3& computedGradient,
    const dax::Vector3& trueGradient)
{
  //the true gradient needs to be fixed based on the topological dimensions
  const int TOPOLOGICAL_DIMENSIONS =
      dax::CellTraits<CellTag>::TOPOLOGICAL_DIMENSIONS;
  dax::Vector3 expectedGradient;
  if (TOPOLOGICAL_DIMENSIONS == 3)
    {
    expectedGradient = trueGradient;
    }
  else if (TOPOLOGICAL_DIMENSIONS == 2)
    {
    dax::Vector3 normal = dax::math::TriangleNormal(
          pointCoordinates[0], pointCoordinates[1], pointCoordinates[2]);
    dax::math::Normalize(normal);
    expectedGradient =
        trueGradient - dax::dot(trueGradient,normal)*normal;
    }
  else if (TOPOLOGICAL_DIMENSIONS == 1)
    {
    dax::Vector3 direction =
        dax::math::Normal(pointCoordinates[1]-pointCoordinates[0]);
    expectedGradient = direction * dax::dot(direction, trueGradient);
    }
  else if (TOPOLOGICAL_DIMENSIONS == 0)
    {
    expectedGradient = dax::make_Vector3(0, 0, 0);
    }
  else
    {
    DAX_TEST_FAIL("Unknown cell dimension.");
    }

  DAX_TEST_ASSERT(test_equal(computedGradient,expectedGradient),
                  "Got bad gradient");
}

//-----------------------------------------------------------------------------
struct TestCellGradientWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  DAX_CONT_EXPORT
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

    dax::cont::ArrayHandle<dax::Vector3> gradientHandle;

    std::cout << "Running CellGradient worklet" << std::endl;
    dax::cont::Scheduler< > scheduler;
    scheduler.Invoke(dax::worklet::CellGradient(),
                    grid.GetRealGrid(),
                    grid->GetPointCoordinates(),
                    fieldHandle,
                    gradientHandle);

    std::cout << "Checking result" << std::endl;
    std::vector<dax::Vector3> gradient(grid->GetNumberOfCells());
    gradientHandle.CopyInto(gradient.begin());
    for (dax::Id cellIndex = 0;
         cellIndex < grid->GetNumberOfCells();
         cellIndex++)
      {
      verifyGradient(grid.GetCellVertexCoordinates(cellIndex),
                     gradient[cellIndex],
                     trueGradient);
      }
    }
};

//-----------------------------------------------------------------------------
void TestCellGradient()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes( TestCellGradientWorklet() );
  }

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletCellGradient(int, char *[])
  {
  return dax::cont::testing::Testing::Run(TestCellGradient);
  }
