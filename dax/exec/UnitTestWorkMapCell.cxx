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
#include <dax/exec/WorkMapCell.h>

#include <dax/exec/Field.h>

#include <dax/exec/internal/TestingTopologyGenerator.h>

#include <dax/internal/Testing.h>

#include <algorithm>
#include <vector>

namespace {

struct TestMapCellFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &topology)
  {
    typedef typename TopologyGenType::CellType CellType;
    typedef typename TopologyGenType::ExecutionAdapter ExecutionAdapter;

    // Create a point field of indices (+1) so that we can check both the
    // cell connectivity and point lookup.
    dax::Id numPoints = topology.GetNumberOfPoints();
    std::vector<dax::Scalar> pointData(numPoints);
    for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
      pointData[pointIndex] = pointIndex + 1;
      }
    dax::exec::FieldPointIn<dax::Scalar,ExecutionAdapter> pointField =
        dax::exec::internal::CreateField<dax::exec::FieldPointIn>(topology,
                                                                  pointData);

    // Create a cell field that we can check inputs and outputs.
    dax::Id numCells = topology.GetNumberOfCells();
    std::vector<dax::Scalar> cellData(numCells);
    dax::exec::FieldCellOut<dax::Scalar,ExecutionAdapter> cellField =
        dax::exec::internal::CreateField<dax::exec::FieldCellOut>(topology,
                                                                  cellData);

    for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
      {
      // Clear out cell field array so we can check it.
      std::fill(cellData.begin(), cellData.end(), -1.0);
      cellData[cellIndex] = cellIndex;

      dax::exec::WorkMapCell<CellType,ExecutionAdapter> work =
          dax::exec::internal::CreateWorkMapCell(topology, cellIndex);
      DAX_TEST_ASSERT(work.GetCellIndex() == cellIndex,
                      "Work object got wrong index");

      // Test cell field access.
      dax::Scalar scalarValue = work.GetFieldValue(cellField);
      DAX_TEST_ASSERT(scalarValue == cellIndex,
                      "Did not get expected scalar value");
      work.SetFieldValue(cellField, dax::Scalar(-2));
      DAX_TEST_ASSERT(cellData[cellIndex] == -2,
                      "Field value not set as expected.");

      // Test point field access and connectivity.
      dax::Tuple<dax::Scalar,CellType::NUM_POINTS> pointScalars =
          work.GetFieldValues(pointField);
      dax::Tuple<dax::Id,CellType::NUM_POINTS> cellConnections =
          topology.GetCellConnections(cellIndex);
      for (dax::Id index = 0; index < CellType::NUM_POINTS; index++)
        {
        DAX_TEST_ASSERT(pointScalars[index] == (cellConnections[index] + 1),
                        "Did not get expected point lookup.");
        }
      }
  }
};

void TestMapCell()
{
  dax::exec::internal::TryAllTopologyTypes(TestMapCellFunctor());
}

} // anonymous namespace

int UnitTestWorkMapCell(int, char *[])
{
  return dax::internal::Testing::Run(TestMapCell);
}
