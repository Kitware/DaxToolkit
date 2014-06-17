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

#include <dax/CellTag.h>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleConstant.h>
#include <dax/cont/DispatcherGenerateInterpolatedCells.h>
#include <dax/cont/UniformGrid.h>

#include <dax/exec/WorkletInterpolatedCell.h>

#include <iostream>
#include <vector>


namespace {

const dax::Id DIM = 4;
const dax::Id COUNTS = 7;

struct TestInterpolatedCellCellPermutationWorklet
    : public dax::exec::WorkletInterpolatedCell
{
  typedef void ControlSignature(TopologyIn, GeometryOut, FieldCellIn);
  typedef void ExecutionSignature(AsVertices(_2), _3, VisitIndex);

  DAX_EXEC_EXPORT
  void operator()(dax::exec::InterpolatedCellPoints<dax::CellTagVertex> &outCell,
                  dax::Id index,
                  dax::Id) const
  {
    outCell.SetInterpolationPoint(0, index,
                                     index,
                                     1.0f);
  }
};

struct TestInterpolatedCellPointPermutationWorklet
    : public dax::exec::WorkletInterpolatedCell
{
  typedef void ControlSignature(TopologyIn, GeometryOut, FieldPointIn);
  typedef void ExecutionSignature(AsVertices(_2), _3, VisitIndex);

  template<class InCellTag>
  DAX_EXEC_EXPORT
  void operator()(dax::exec::InterpolatedCellPoints<dax::CellTagVertex> &outCell,
                  const dax::exec::CellField<dax::Id,InCellTag> &,
                  dax::Id visitIndex) const
  {
   outCell.SetInterpolationPoint(0, visitIndex, visitIndex, 1.0f);
  }
};

//-----------------------------------------------------------------------------
struct TestInterpolatedCellPermutation
{
  //----------------------------------------------------------------------------
  template <typename InGridType>
  void operator()(const InGridType&) const
    {
    typedef dax::CellTagVertex OutGridTag;
    typedef dax::cont::UnstructuredGrid<OutGridTag> OutGridType;

    dax::cont::testing::TestGrid<InGridType> inGenerator(DIM);
    InGridType inGrid = inGenerator.GetRealGrid();
    OutGridType outGrid;

    // Perhaps there should be a better way to specify a constant value for
    // the number of cells generated per location.
    typedef dax::cont::ArrayHandleConstant<dax::Id> CellCountArrayType;
    CellCountArrayType cellCounts =
        dax::cont::make_ArrayHandleConstant(dax::Id(COUNTS),
                                                 inGrid.GetNumberOfCells());

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

    dax::cont::DispatcherGenerateInterpolatedCells<
        TestInterpolatedCellCellPermutationWorklet,
        CellCountArrayType> cellPermutationWorklet(cellCounts);
    cellPermutationWorklet.SetRemoveDuplicatePoints(true);
    cellPermutationWorklet.Invoke(inGrid, outGrid, cellField);

    std::cout << "inGrid numCells: " << inGrid.GetNumberOfCells() << std::endl;

    this->CheckCellPermutation(
          outGrid.GetCellConnections().GetPortalConstControl());

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

    dax::cont::DispatcherGenerateInterpolatedCells<
        TestInterpolatedCellPointPermutationWorklet,
        CellCountArrayType> pointPermutationWorklet(cellCounts);
    pointPermutationWorklet.SetRemoveDuplicatePoints(true);
    pointPermutationWorklet.Invoke(inGrid, outGrid, pointField);

    this->CheckPointPermutation(
          outGrid.GetCellConnections().GetPortalConstControl());
  }

private:
  //----------------------------------------------------------------------------
  template<class OutConnectionsPortal>
  void CheckCellPermutation(const OutConnectionsPortal &outConnections) const
  {
    std::cout << "Checking cell field permutation" << std::endl;
    for (dax::Id index = 0; index < outConnections.GetNumberOfValues(); index++)
      {
      dax::Id expectedValue = index/COUNTS;
      dax::Id actualValue = outConnections.Get(index);
      DAX_TEST_ASSERT(actualValue == expectedValue,
                      "Got bad value testing cell permutation.");
      }
  }

  //----------------------------------------------------------------------------
  template<class OutConnectionsPortal>
  void CheckPointPermutation( const OutConnectionsPortal &outConnections) const
  {
    std::cout << "Checking point field permutation" << std::endl;
    for (dax::Id index = 0; index < outConnections.GetNumberOfValues(); index++)
      {
      dax::Id actualValue = outConnections.Get(index);
      dax::Id expectedValue = index%COUNTS;
      DAX_TEST_ASSERT(actualValue == expectedValue,
                      "Got bad value testing cell permutation.");
      }
  }
};


//-----------------------------------------------------------------------------
void RunTestInterpolatedCellPermutation()
  {
  TestInterpolatedCellPermutation()(dax::cont::UniformGrid<>());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestInterpolatedCellPermutation(int, char *[])
{
  return dax::cont::testing::Testing::Run(RunTestInterpolatedCellPermutation);
}
