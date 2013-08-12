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

#include <dax/worklet/CellAverage.h>

#include <dax/Types.h>
#include <dax/VectorTraits.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapterSerial.h>
#include <dax/cont/Scheduler.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

//----------------------------------------------------------------------------
template<typename CellTag>
void verifyAverage(const dax::exec::CellVertices<CellTag> &cellVertices,
                   const dax::Scalar& computedAverage)
{
  dax::Scalar expectedAverage = 0.0;
  for(int vertexIndex=0; vertexIndex<cellVertices.NUM_VERTICES; ++vertexIndex)
  {
    expectedAverage += cellVertices[vertexIndex];
  }
  expectedAverage /= cellVertices.NUM_VERTICES;

  DAX_TEST_ASSERT(test_equal(computedAverage, expectedAverage),
                  "Got bad average");
}

//-----------------------------------------------------------------------------
struct TestCellAverageWorklet
{
  //----------------------------------------------------------------------------
  template<typename GridType>
  void operator()(const GridType&) const
    {
    typedef typename GridType::CellTag CellTag;
    dax::cont::testing::TestGrid<
        GridType,
        dax::cont::ArrayContainerControlTagBasic> grid(DIM);


    std::vector<dax::Scalar> field(grid->GetNumberOfPoints());
    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      field[pointIndex] = pointIndex;
      }

    dax::cont::ArrayHandle<dax::Scalar> fieldHandle =
        dax::cont::make_ArrayHandle(field);

    dax::cont::ArrayHandle<dax::Scalar> resultHandle;

    std::cout << "Running CellAverage worklet" << std::endl;
    dax::cont::Scheduler< > scheduler;
    scheduler.Invoke(dax::worklet::CellAverage(),
                     grid.GetRealGrid(),
                     fieldHandle,
                     resultHandle);

    std::cout << "Checking result" << std::endl;
    std::vector<dax::Scalar> averages(grid->GetNumberOfCells());
    resultHandle.CopyInto(averages.begin());
    for (dax::Id cellIndex = 0;
         cellIndex < grid->GetNumberOfCells();
         cellIndex++)
      {
      verifyAverage<CellTag>(grid.GetCellConnections(cellIndex),
                             averages[cellIndex]);
      }
    }
};

//-----------------------------------------------------------------------------
void TestCellAverage()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes( TestCellAverageWorklet() );
  }


} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletCellAverage(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestCellAverage);
}
