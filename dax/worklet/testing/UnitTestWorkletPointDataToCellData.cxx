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

#include <dax/worklet/PointDataToCellData.worklet>

#include <dax/Types.h>
#include <dax/VectorTraits.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
template<typename CellTag>
void verifyCellData(const dax::exec::CellVertices<CellTag> &cellVertices,
                    const dax::Scalar& computedCellData)
{
  dax::Scalar expectedCellData = 0.0;
  for (int vertexIndex = 0;
       vertexIndex < cellVertices.NUM_VERTICES;
       ++vertexIndex)
  {
    expectedCellData += cellVertices[vertexIndex];
  }
  expectedCellData /= cellVertices.NUM_VERTICES;
  DAX_TEST_ASSERT(test_equal(computedCellData,expectedCellData),"Got bad average");
}

//-----------------------------------------------------------------------------
struct TestPointDataToCellDataWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
    {
    dax::cont::internal::TestGrid<
        GridType,
        dax::cont::ArrayContainerControlTagBasic,
        dax::cont::DeviceAdapterTagSerial> grid(DIM);


    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      field[pointIndex] = pointIndex;
      }

    dax::cont::ArrayHandle<dax::Scalar,
      dax::cont::ArrayContainerControlTagBasic,
      dax::cont::DeviceAdapterTagSerial> fieldHandle =
        dax::cont::make_ArrayHandle(field,
                                    dax::cont::ArrayContainerControlTagBasic(),
                                    dax::cont::DeviceAdapterTagSerial());

    dax::cont::ArrayHandle<dax::Scalar,
        dax::cont::ArrayContainerControlTagBasic,
        dax::cont::DeviceAdapterTagSerial> resultHandle;

    std::cout << "Running PointDataToCellData worklet" << std::endl;
    dax::cont::Scheduler<> scheduler;
    scheduler.Invoke(dax::worklet::PointDataToCellData(),
                     grid.GetRealGrid(),
                     fieldHandle,
                     resultHandle);

    std::cout << "Checking result" << std::endl;
    std::vector<dax::Scalar> cellData(grid->GetNumberOfCells());
    resultHandle.CopyInto(cellData.begin());
    for (dax::Id cellIndex = 0;
         cellIndex < grid->GetNumberOfCells();
         cellIndex++)
      {
      verifyCellData(grid.GetCellConnections(cellIndex),
                     cellData[cellIndex]);
      }
    }
};

//-----------------------------------------------------------------------------
void TestPointDataToCellData()
  {
  dax::cont::internal::GridTesting::TryAllGridTypes(
        TestPointDataToCellDataWorklet(),
        dax::cont::ArrayContainerControlTagBasic(),
        dax::cont::DeviceAdapterTagSerial());
  }


} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletPointDataToCellData(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestPointDataToCellData);
}
