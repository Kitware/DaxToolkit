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

#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/testing/TestingGridGenerator.h>
#include <dax/cont/testing/Testing.h>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <dax/CellTag.h>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleConstant.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/GenerateKeysValues.h>

#include <dax/exec/WorkletGenerateKeysValues.h>

#include <dax/cont/testing/Testing.h>
#include <vector>


namespace {

const dax::Id DIM = 4;
const dax::Id COUNTS = 7;

struct TestGenerateKeysValuesCellPermutationWorklet
    : public dax::exec::WorkletGenerateKeysValues
{
  typedef void ControlSignature(Topology, Field(In,Cell), Field(Out));
  typedef _3 ExecutionSignature(_2, VisitIndex);

  DAX_EXEC_EXPORT
  dax::Scalar operator()(dax::Id index, dax::Id visitIndex) const
  {
    return 10*index + 1000*visitIndex;
  }
};

struct TestGenerateKeysValuesPointPermutationWorklet
    : public dax::exec::WorkletGenerateKeysValues
{
  typedef void ControlSignature(Topology, Field(In,Point), Field(Out));
  typedef _3 ExecutionSignature(_2, VisitIndex);

  template<class InCellTag>
  DAX_EXEC_EXPORT
  dax::Scalar operator()(
      const dax::exec::CellField<dax::Id,InCellTag> &pointScalars,
      dax::Id visitIndex) const
  {
    return pointScalars[visitIndex%dax::CellTraits<InCellTag>::NUM_VERTICES];
  }
};

//-----------------------------------------------------------------------------
struct TestGenerateKeysValuesPermutation
{
  //----------------------------------------------------------------------------
  template <typename InGridType>
  void operator()(const InGridType&) const
    {
    typedef dax::CellTagVertex OutGridTag;

    dax::cont::testing::TestGrid<InGridType> inGenerator(DIM);
    InGridType inGrid = inGenerator.GetRealGrid();

    // Perhaps there should be a better way to specify a constant value for
    // the number of cells generated per location.
    typedef dax::cont::ArrayHandleConstant<dax::Id> CellCountArrayType;
    CellCountArrayType cellCounts =
        dax::cont::make_ArrayHandleConstant(dax::Id(COUNTS),
                                            inGrid.GetNumberOfCells());

    dax::cont::ArrayHandle<dax::Id> outField;

    std::vector<dax::Id> cellFieldData(inGrid.GetNumberOfCells());
    for (dax::Id cellIndex = 0;
         cellIndex < inGrid.GetNumberOfCells();
         cellIndex++)
      {
      cellFieldData[cellIndex] = cellIndex + 1;
      }
    dax::cont::ArrayHandle<dax::Id> cellField =
        dax::cont::make_ArrayHandle(cellFieldData);

    std::cout << "Trying cell field permutation" << std::endl;
    dax::cont::Scheduler<> scheduler;

    dax::cont::GenerateKeysValues<
        TestGenerateKeysValuesCellPermutationWorklet,
        CellCountArrayType> cellPermutationWorklet(cellCounts);
    scheduler.Invoke(cellPermutationWorklet, inGrid, cellField, outField);

    this->CheckCellPermutation(outField.GetPortalConstControl());

    std::vector<dax::Id> pointFieldData(inGrid.GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < inGrid.GetNumberOfPoints();
         pointIndex++)
      {
      pointFieldData[pointIndex] = pointIndex + 1;
      }
    dax::cont::ArrayHandle<dax::Id> pointField =
        dax::cont::make_ArrayHandle(pointFieldData);

    std::cout << "Trying point field permutation" << std::endl;

    dax::cont::GenerateKeysValues<
        TestGenerateKeysValuesPointPermutationWorklet,
        CellCountArrayType> pointPermutationWorklet(cellCounts);
    scheduler.Invoke(pointPermutationWorklet, inGrid, pointField, outField);

    this->CheckPointPermutation(inGenerator, outField.GetPortalConstControl());
  }

private:
  //----------------------------------------------------------------------------
  template<class OutValuesPortal>
  void CheckCellPermutation(const OutValuesPortal &outValues) const
  {
    std::cout << "Checking cell field permutation" << std::endl;
    for (dax::Id index = 0; index < outValues.GetNumberOfValues(); index++)
      {
      dax::Id expectedValue = 10*((index/COUNTS)+1) + 1000*(index%COUNTS);
      dax::Id actualValue = outValues.Get(index);
//      std::cout << index << " - " << actualValue << ", " << expectedValue << std::endl;
      DAX_TEST_ASSERT(actualValue == expectedValue,
                      "Got bad value testing cell permutation.");
      }
  }

  //----------------------------------------------------------------------------
  template<class InGridType, class OutValuesPortal>
  void CheckPointPermutation(
      const dax::cont::testing::TestGrid<InGridType> &inGenerator,
      const OutValuesPortal &outValues) const
  {
    std::cout << "Checking point field permutation" << std::endl;
    for (dax::Id index = 0; index < outValues.GetNumberOfValues(); index++)
      {
      dax::Id actualValue = outValues.Get(index);
      dax::Id cellIndex = index/COUNTS;
      dax::Id visitIndex = index%COUNTS;
      typedef typename InGridType::CellTag InCellTag;
      int vertexIndex = visitIndex%dax::CellTraits<InCellTag>::NUM_VERTICES;
      dax::Id expectedValue =
          inGenerator.GetCellConnections(cellIndex)[vertexIndex] + 1;
//      std::cout << index << " - " << actualValue << ", " << expectedValue << std::endl;
      DAX_TEST_ASSERT(actualValue == expectedValue,
                      "Got bad value testing cell permutation.");
      }
  }
};


//-----------------------------------------------------------------------------
void RunTestGenerateKeysValuesPermutation()
  {
//  dax::cont::testing::GridTesting::TryAllGridTypes(
//        TestGenerateKeysValuesPermutation());
  TestGenerateKeysValuesPermutation()(dax::cont::UniformGrid<>());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestGenerateKeysValuesPermutation(int, char *[])
{
  return dax::cont::testing::Testing::Run(RunTestGenerateKeysValuesPermutation);
}
