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

// These macros help tease out when the default template arguments to
// ArrayHandle are inappropriately used.
#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/internal/testing/TestingGridGenerator.h>
#include <dax/cont/internal/testing/Testing.h>

#include <dax/worklet/CellGradient.worklet>

#include <dax/Types.h>
#include <dax/VectorTraits.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
template<typename CellType>
void verifyGradient(const dax::Tuple<dax::Vector3,CellType::NUM_POINTS> &pointCoordinates,
                    const dax::Vector3& computedGradient,
                    const dax::Vector3& trueGradient)
{
  //the true gradient needs to be fixed based on the toplogical demensions
  dax::Vector3 expectedGradient;
  if (CellType::TOPOLOGICAL_DIMENSIONS == 3)
    {
    expectedGradient = trueGradient;
    }
  else if (CellType::TOPOLOGICAL_DIMENSIONS == 2)
    {
    dax::Vector3 normal = dax::math::TriangleNormal(
          pointCoordinates[0], pointCoordinates[1], pointCoordinates[2]);
    dax::math::Normalize(normal);
    expectedGradient =
        trueGradient - dax::dot(trueGradient,normal)*normal;
    }
  else if (CellType::TOPOLOGICAL_DIMENSIONS == 1)
    {
    dax::Vector3 direction =
        dax::math::Normal(pointCoordinates[1]-pointCoordinates[0]);
    expectedGradient = direction * dax::dot(direction, trueGradient);
    }
  else if (CellType::TOPOLOGICAL_DIMENSIONS == 0)
    {
    expectedGradient = dax::make_Vector3(0, 0, 0);
    }
  else
    {
    DAX_TEST_FAIL("Unknown cell dimension.");
    }

  DAX_TEST_ASSERT(test_equal(computedGradient,expectedGradient),"Got bad gradient");
}

//-----------------------------------------------------------------------------
struct TestCellGradientWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
    {
    typedef typename GridType::CellType CellType;
    dax::cont::internal::TestGrid<
        GridType,
        dax::cont::ArrayContainerControlTagBasic,
        dax::cont::DeviceAdapterTagSerial> grid(DIM);


    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      field[pointIndex]
          = dax::dot(grid->ComputePointCoordinates(pointIndex), trueGradient);
      }
    dax::cont::ArrayHandle<dax::Scalar,
      dax::cont::ArrayContainerControlTagBasic,
      dax::cont::DeviceAdapterTagSerial> fieldHandle =
        dax::cont::make_ArrayHandle(field,
                                    dax::cont::ArrayContainerControlTagBasic(),
                                    dax::cont::DeviceAdapterTagSerial());

    dax::cont::ArrayHandle<dax::Vector3,
        dax::cont::ArrayContainerControlTagBasic,
        dax::cont::DeviceAdapterTagSerial> gradientHandle;

    std::cout << "Running CellGradient worklet" << std::endl;
    dax::cont::Scheduler<> scheduler;
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
      verifyGradient<CellType>(
                               grid.GetCellVertexCoordinates(cellIndex),
                               gradient[cellIndex],
                               trueGradient);
      }
    }
};

//-----------------------------------------------------------------------------
void TestCellGradient()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(
        TestCellGradientWorklet(),
        dax::cont::ArrayContainerControlTagBasic(),
        dax::cont::DeviceAdapterTagSerial());
  }


} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletCellGradient(int, char *[])
  {
  return dax::cont::internal::Testing::Run(TestCellGradient);
  }
