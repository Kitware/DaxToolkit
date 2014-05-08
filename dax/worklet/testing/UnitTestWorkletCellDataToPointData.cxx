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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#include <dax/cont/testing/TestingGridGenerator.h>
#include <dax/cont/testing/Testing.h>

#include <dax/worklet/CellDataToPointData.h>

#include <dax/CellTraits.h>
#include <dax/Types.h>
#include <dax/VectorTraits.h>
#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleConstant.h>
#include <dax/cont/DeviceAdapterSerial.h>
#include <dax/cont/DispatcherGenerateKeysValues.h>
#include <dax/cont/DispatcherReduceKeysValues.h>

#include <iostream>
#include <algorithm>
#include <vector>

namespace {

const dax::Id DIM = 3;

//-----------------------------------------------------------------------------
template<typename TestGrid>
void verifyPointData(const TestGrid &grid,
                     const std::vector<dax::Scalar> cellData,
                     const std::vector<dax::Scalar> computedPointData)
{
  typedef typename TestGrid::CellTag CellTag;

  std::vector<dax::Id> numConnections(grid->GetNumberOfPoints());
  std::fill(numConnections.begin(), numConnections.end(), 0);

  std::vector<dax::Scalar> pointDataSums(grid->GetNumberOfPoints());
  std::fill(pointDataSums.begin(), pointDataSums.end(), 0);

  for (dax::Id cellIndex = 0; cellIndex < grid->GetNumberOfCells(); cellIndex++)
    {
    dax::cont::testing::CellConnections<CellTag> cellConnections =
      grid.GetCellConnections(cellIndex);

    for (int vertexIndex = 0;
         vertexIndex < cellConnections.NUM_VERTICES;
         vertexIndex++)
      {
      dax::Id pointIndex = cellConnections[vertexIndex];
      numConnections[pointIndex]++;
      pointDataSums[pointIndex] += cellData[cellIndex];
      }
    }

  for (dax::Id pointIndex = 0;
       pointIndex < dax::Id(computedPointData.size());
       pointIndex++)
    {
    dax::Scalar expectedValue =
      pointDataSums[pointIndex] / numConnections[pointIndex];
    dax::Scalar computedValue = computedPointData[pointIndex];
    DAX_TEST_ASSERT(test_equal(computedValue, expectedValue),
                    "Got bad average at point");
    }
}

//-----------------------------------------------------------------------------
struct TestCellDataToPointDataWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
  {
    typedef typename GridType::CellTag CellTag;

    dax::cont::testing::TestGrid<GridType> grid(DIM);


    std::vector<dax::Scalar> field(grid->GetNumberOfCells());
    for (dax::Id cellIndex = 0;
         cellIndex < grid->GetNumberOfCells();
         cellIndex++)
      {
      field[cellIndex] = cellIndex;
      }

    dax::cont::ArrayHandle<dax::Scalar> fieldHandle
                                        = dax::cont::make_ArrayHandle(field);

    dax::cont::ArrayHandleConstant<dax::Id> keyGenCounts =
      dax::cont::make_ArrayHandleConstant<dax::Id>(
            dax::CellTraits<CellTag>::NUM_VERTICES, grid->GetNumberOfCells());

    dax::cont::ArrayHandle<dax::Id> keyHandle;
    dax::cont::ArrayHandle<dax::Scalar> valueHandle;
    dax::cont::ArrayHandle<dax::Scalar> resultHandle;

    std::cout << "Running CellDataToPointDataGenerateKeys worklet" << std::endl;

    dax::cont::DispatcherGenerateKeysValues<
        dax::worklet::CellDataToPointDataGenerateKeys,
        dax::cont::ArrayHandleConstant<dax::Id> > generateKeys(keyGenCounts);

    generateKeys.Invoke(grid.GetRealGrid(),fieldHandle,keyHandle,valueHandle);

    std::cout << "Running CellDataToPointDataReduceKeys worklet" << std::endl;

    dax::cont::DispatcherReduceKeysValues<
      dax::worklet::CellDataToPointDataReduceKeys> reduceKeys(keyHandle);

    reduceKeys.Invoke(valueHandle, resultHandle);

    std::cout << "Checking result" << std::endl;
    std::vector<dax::Scalar> pointData(resultHandle.GetNumberOfValues());
    resultHandle.CopyInto(pointData.begin());

    verifyPointData(grid, field, pointData);
  }
};

//-----------------------------------------------------------------------------
void TestCellDataToPointData()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes(
                                           TestCellDataToPointDataWorklet());
  }


} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletCellDataToPointData(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestCellDataToPointData);
}
