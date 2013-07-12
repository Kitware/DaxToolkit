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

// These macros help tease out when the default template arguments to
// ArrayHandle are inappropriately used.
#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_ERROR
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR

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
#include <dax/cont/GenerateKeysValues.h>
#include <dax/cont/ReduceKeysValues.h>
#include <dax/cont/Scheduler.h>

#include <algorithm>
#include <vector>

namespace {

const dax::Id DIM = 3;

typedef dax::cont::ArrayContainerControlTagBasic ArrayContainer;
typedef dax::cont::DeviceAdapterTagSerial DeviceAdapter;

//-----------------------------------------------------------------------------
template<typename TestGrid>
void verifyPointData(const TestGrid &grid,
                     const std::vector<dax::Scalar> cellData,
                     const std::vector<dax::Scalar> computedPointData)
{
  typedef typename TestGrid::CellTag CellTag;

  std::vector<int> numConnections(grid->GetNumberOfPoints());
  std::fill(numConnections.begin(), numConnections.end(), 0);

  std::vector<dax::Scalar> pointDataSums(grid->GetNumberOfPoints());
  std::fill(pointDataSums.begin(), pointDataSums.end(), 0);

  for (dax::Id cellIndex = 0; cellIndex < grid->GetNumberOfCells(); cellIndex++)
    {
    dax::exec::CellVertices<CellTag> cellConnections =
      grid.GetCellConnections(cellIndex);

    std::cout << "Cell " << cellIndex << "(" << cellData[cellIndex] << ") : ";
    for (int vertexIndex = 0;
         vertexIndex < cellConnections.NUM_VERTICES;
         vertexIndex++)
      {
      dax::Id pointIndex = cellConnections[vertexIndex];
      numConnections[pointIndex]++;
      pointDataSums[pointIndex] += cellData[cellIndex];
      std::cout << "{" << cellConnections[vertexIndex] << ", " << computedPointData[pointIndex] << "}, ";
      }
    std::cout << std::endl;
    }

  for (dax::Id pointIndex = 0;
       pointIndex < dax::Id(computedPointData.size());
       pointIndex++)
    {
    dax::Scalar expectedValue =
      pointDataSums[pointIndex] / numConnections[pointIndex];
    dax::Scalar computedValue = computedPointData[pointIndex];
    std::cout << "pointDataSums: " << pointDataSums[pointIndex] << std::endl;
    std::cout << "numConnections: " << numConnections[pointIndex] << std::endl;
    std::cout << "Expected value: " << expectedValue << std::endl;
    std::cout << "Computed value: " << computedValue << std::endl;
    DAX_TEST_ASSERT(test_equal(computedValue, expectedValue),
                    "Got bad average at point");
    }
}
template<class IteratorType>
void PrintArray(IteratorType beginIter, IteratorType endIter)
{
  for (IteratorType iter = beginIter; iter != endIter; iter++)
    {
    std::cout << " " << *iter;
    }
  std::cout << std::endl;
}
//-----------------------------------------------------------------------------
struct TestCellDataToPointDataWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
  {
    typedef typename GridType::CellTag CellTag;

    dax::cont::testing::TestGrid<GridType, ArrayContainer, DeviceAdapter>
      grid(DIM);


    std::vector<dax::Scalar> field(grid->GetNumberOfCells());
    for (dax::Id cellIndex = 0;
         cellIndex < grid->GetNumberOfCells();
         cellIndex++)
      {
      field[cellIndex] = cellIndex;
      }

    dax::cont::ArrayHandle<dax::Scalar,ArrayContainer,DeviceAdapter> fieldHandle
        = dax::cont::make_ArrayHandle(field, ArrayContainer(), DeviceAdapter());

    dax::cont::ArrayHandleConstant<dax::Id,DeviceAdapter> keyGenCounts =
      dax::cont::make_ArrayHandleConstant(dax::CellTraits<CellTag>::NUM_VERTICES,
                                          grid->GetNumberOfCells(),
                                          DeviceAdapter());

    dax::cont::ArrayHandle<dax::Id,ArrayContainer,DeviceAdapter> keyHandle;
    dax::cont::ArrayHandle<dax::Scalar,ArrayContainer,DeviceAdapter> valueHandle;

    dax::cont::ArrayHandle<dax::Scalar,ArrayContainer,DeviceAdapter>
      resultHandle;

    std::cout << "Running CellDataToPointDataGenerateKeys worklet" << std::endl;
    dax::cont::Scheduler<DeviceAdapter> scheduler;

    dax::cont::GenerateKeysValues<
      dax::worklet::CellDataToPointDataGenerateKeys,
      dax::cont::ArrayHandleConstant<dax::Id,DeviceAdapter> >
          generateKeys(keyGenCounts);

    scheduler.Invoke(generateKeys,
                     grid.GetRealGrid(),
                     fieldHandle,
                     keyHandle,
                     valueHandle);

    std::cout << "Running CellDataToPointDataReduceKeys worklet" << std::endl;

    dax::worklet::CellDataToPointDataReduceKeys CD2PD;

    dax::cont::ReduceKeysValues<
      dax::worklet::CellDataToPointDataReduceKeys,
      dax::cont::ArrayHandle<dax::Id,ArrayContainer,DeviceAdapter> >
        reduceKeys(keyHandle, CD2PD);

/*    std::vector<dax::Scalar> keyData(keyHandle.GetNumberOfValues());
    keyHandle.CopyInto(keyData.begin());

    std::cout << "keyData: ";
    for(int iCtr = 0; iCtr < keyData.size(); iCtr++)
        std::cout << keyData[iCtr] << ", ";
    std::cout << std::endl;
    std::cout << std::endl;

    std::vector<dax::Scalar> valueData(valueHandle.GetNumberOfValues());
    valueHandle.CopyInto(valueData.begin());

    std::cout << "valueData: ";
    for(int iCtr = 0; iCtr < valueData.size(); iCtr++)
        std::cout << valueData[iCtr] << ", ";
    std::cout << std::endl;
    std::cout << std::endl;
*/
    scheduler.Invoke(reduceKeys,
                     valueHandle,
                     resultHandle);

/*
    std::cout << "CountSize: " << resultHandle.GetReductionCounts().GetNumberOfValues() << std::endl;
    std::cout << "OffsetSize: " << resultHandle.GetReductionOffsets().GetNumberOfValues() << std::endl;
    std::cout << "IndexSize: " << resultHandle.GetReductionIndices().GetNumberOfValues() << std::endl;
    std::cout << "Counts: ";
    PrintArray(
          reduceKeys.GetReductionCounts().GetPortalConstControl().GetIteratorBegin(),
          reduceKeys.GetReductionCounts().GetPortalConstControl().GetIteratorEnd());

    std::vector<dax::Scalar> valueData(valueHandle.GetNumberOfValues());
    valueHandle.CopyInto(valueData.begin());

    std::cout << "valueData: ";
    for(int iCtr = 0; iCtr < valueData.size(); iCtr++)
        std::cout << valueData[iCtr] << ", ";
    std::cout << std::endl;
    std::cout << std::endl;
  */

    std::cout << "Checking result" << std::endl;
    std::vector<dax::Scalar> pointData(resultHandle.GetNumberOfValues());
    resultHandle.CopyInto(pointData.begin());

    std::cout << "pointData: ";
    for(unsigned int iCtr = 0; iCtr < pointData.size(); iCtr++)
        std::cout << pointData[iCtr] << ", ";
    std::cout << std::endl;
    std::cout << std::endl;

    verifyPointData(grid, field, pointData);
  }
};

//-----------------------------------------------------------------------------
void TestCellDataToPointData()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes(
        TestCellDataToPointDataWorklet(),
        ArrayContainer(),
        DeviceAdapter());
  }


} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletCellDataToPointData(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestCellDataToPointData);
}
