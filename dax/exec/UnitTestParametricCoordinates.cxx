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
#include <dax/exec/ParametricCoordinates.h>

#include <dax/exec/WorkMapCell.h>

#include <dax/exec/internal/TestExecutionAdapter.h>

#include <dax/internal/Testing.h>

namespace {

/// An (invalid) error handler to pass to work constructors.
dax::exec::internal::ErrorHandler<TestExecutionAdapter> ErrorHandler
  = dax::exec::internal::ErrorHandler<TestExecutionAdapter>(NULL, NULL);

}  // Anonymous namespace

template<class WorkType, class CellType, class ExecutionAdapter>
static void CompareCoordinates(
    const WorkType &work,
    const CellType &cell,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
    dax::Vector3 truePCoords,
    dax::Vector3 trueWCoords)
{
  dax::Vector3 computedWCoords
      = dax::exec::parametricCoordinatesToWorldCoordinates(work,
                                                           cell,
                                                           coordField,
                                                           truePCoords);
  DAX_TEST_ASSERT(computedWCoords == trueWCoords,
                  "Computed wrong world coords from parametric coords.");

  dax::Vector3 computedPCoords
      = dax::exec::worldCoordinatesToParametricCoordinates(work,
                                                           cell,
                                                           coordField,
                                                           trueWCoords);
  DAX_TEST_ASSERT(computedPCoords == truePCoords,
                  "Computed wrong parametric coords from world coords.");
}

template<class ExecutionAdapter>
static void TestPCoordsVoxel(
  const dax::exec::WorkMapCell<dax::exec::CellVoxel, ExecutionAdapter> &work,
  const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField)
{
  const dax::Vector3 cellVertexToParametricCoords[8] = {
    dax::make_Vector3(0, 0, 0),
    dax::make_Vector3(1, 0, 0),
    dax::make_Vector3(1, 1, 0),
    dax::make_Vector3(0, 1, 0),
    dax::make_Vector3(0, 0, 1),
    dax::make_Vector3(1, 0, 1),
    dax::make_Vector3(1, 1, 1),
    dax::make_Vector3(0, 1, 1)
  };
  const dax::exec::CellVoxel &cell = work.GetCell();

  // Check the coordinates at all vertices
  dax::Tuple<dax::Vector3, 8> cellVertexToWorldCoords
      = work.GetFieldValues(coordField);
  for (dax::Id vertexIndex = 0; vertexIndex < 8; vertexIndex++)
    {
    dax::Vector3 truePCoords = cellVertexToParametricCoords[vertexIndex];
    dax::Vector3 trueWCoords = cellVertexToWorldCoords[vertexIndex];
    CompareCoordinates(work, cell, coordField, truePCoords, trueWCoords);
    }

  dax::Vector3 centerCoords = dax::make_Vector3(0.0, 0.0, 0.0);
  for (dax::Id vertexIndex = 0; vertexIndex < 8; vertexIndex++)
    {
    centerCoords = centerCoords + cellVertexToWorldCoords[vertexIndex];
    }
  centerCoords = (1.0/8.0) * centerCoords;
  CompareCoordinates(
        work, cell, coordField, dax::make_Vector3(0.5,0.5,0.5), centerCoords);
}

static void TestPCoordsVoxel()
{
  std::cout << "Testing TestPCoords<CellVoxel>" << std::endl;

  dax::exec::FieldCoordinatesIn<TestExecutionAdapter> coordField;

  {
  dax::exec::internal::TopologyUniform gridstruct;
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  for (dax::Id flatIndex = 0; flatIndex < 1000; flatIndex++)
    {
    dax::exec::WorkMapCell<dax::exec::CellVoxel, TestExecutionAdapter>
        work(gridstruct, flatIndex, ErrorHandler);
    TestPCoordsVoxel(work, coordField);
    }
  }

  {
  dax::exec::internal::TopologyUniform gridstruct;
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  for (dax::Id flatIndex = 0; flatIndex < 1500; flatIndex++)
    {
    dax::exec::WorkMapCell<dax::exec::CellVoxel, TestExecutionAdapter>
        work(gridstruct, flatIndex, ErrorHandler);
    TestPCoordsVoxel(work, coordField);
    }
  }
}

static void TestPCoords()
{
  TestPCoordsVoxel();
}

int UnitTestParametricCoordinates(int, char *[])
{
  return dax::internal::Testing::Run(TestPCoords);
}
