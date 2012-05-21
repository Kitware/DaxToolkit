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
#include <dax/exec/Cell.h>

#include <dax/exec/internal/TestExecutionAdapter.h>

#include <dax/internal/Testing.h>
#include <vector>

namespace
{
static void CheckPointIndex(dax::Id pointFlatIndex,
                            const dax::Id3 &pointIjkIndex,
                            const dax::Extent3 &extent)
{
  dax::Id3 compaWeld = dax::flatIndexToIndex3(pointFlatIndex, extent);

  DAX_TEST_ASSERT(compaWeld == pointIjkIndex,
                  "Bad point index.");
}

static void CheckPointIndex(const dax::Id &hexPointIndex,
                            const dax::Id &voxPointIndex)
{
  DAX_TEST_ASSERT(hexPointIndex == voxPointIndex,
                  "Bad point index.");
}
}

// This function is available in the global scope so that it can be used
// in other tests such as UnitTestWorkMapCellHexahedron.
dax::exec::internal::TopologyUnstructured<dax::exec::CellHexahedron, TestExecutionAdapter>
  make_ugrid(const dax::exec::internal::TopologyUniform& uniform,
             std::vector<dax::Vector3>& points,
             std::vector<dax::Id>& connections
             )
{
  //copy the point info over to the unstructured grid
  points.clear();
  dax::Id numberOfPoints = dax::exec::internal::numberOfPoints(uniform);
  for(dax::Id i=0; i < numberOfPoints; ++i)
    {
    points.push_back(dax::exec::internal::pointCoordiantes(uniform,i));
    }

  //copy the cell connection information over
  connections.clear();
  dax::Id numberOfCells = dax::exec::internal::numberOfCells(uniform);
  for(dax::Id i=0; i < numberOfCells; ++i)
    {
    dax::exec::CellVoxel vox(uniform,i);
    for(dax::Id j=0; j < vox.GetNumberOfPoints(); ++j)
      {
      connections.push_back(vox.GetPointIndex(j));
      }
    }

  dax::exec::internal::TopologyUnstructured<dax::exec::CellHexahedron, TestExecutionAdapter>
      ugrid(&connections[0], numberOfPoints, numberOfCells);
  return ugrid;
}

// This function is available in the global scope so that it can be used
// in other tests such as UnitTestWorkMapCellVoxel.
void TestCellVoxel(const dax::exec::CellVoxel cell,
                   const dax::exec::internal::TopologyUniform &gridstruct,
                   dax::Id cellFlatIndex)
{
  dax::Id3 cellIjkIndex
      = dax::flatIndexToIndex3Cell(cellFlatIndex, gridstruct.Extent);

  DAX_TEST_ASSERT(cell.GetNumberOfPoints() == 8,
                  "CellVoxel has wrong number of points");

  CheckPointIndex(cell.GetPointIndex(0), cellIjkIndex, gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(1),
                  cellIjkIndex + dax::make_Id3(1,0,0),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(2),
                  cellIjkIndex + dax::make_Id3(1,1,0),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(3),
                  cellIjkIndex + dax::make_Id3(0,1,0),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(4),
                  cellIjkIndex + dax::make_Id3(0,0,1),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(5),
                  cellIjkIndex + dax::make_Id3(1,0,1),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(6),
                  cellIjkIndex + dax::make_Id3(1,1,1),
                  gridstruct.Extent);
  CheckPointIndex(cell.GetPointIndex(7),
                  cellIjkIndex + dax::make_Id3(0,1,1),
                  gridstruct.Extent);

  DAX_TEST_ASSERT(cell.GetOrigin() == gridstruct.Origin,
                  "CellVoxel has wrong origin");

  DAX_TEST_ASSERT(cell.GetSpacing() == gridstruct.Spacing,
                  "CellVoxel has wrong spacing");

  DAX_TEST_ASSERT(cell.GetExtent().Min == gridstruct.Extent.Min,
                  "CellVoxel has wrong extent");
  DAX_TEST_ASSERT(cell.GetExtent().Max == gridstruct.Extent.Max,
                  "CellVoxel has wrong extent");

  DAX_TEST_ASSERT(cell.GetIndex() == cellFlatIndex,
                  "CellVoxel has wrong index");
}

// This function is available in the global scope so that it can be used
// in other tests such as UnitTestWorkMapCellHexahedron.
void TestCellHexahedron(const dax::exec::CellHexahedron cell,
                        const dax::exec::CellVoxel voxel)
{
  DAX_TEST_ASSERT(cell.GetNumberOfPoints() == voxel.GetNumberOfPoints(),
                 "CellHexahedron has wrong number of points");

  DAX_TEST_ASSERT(cell.GetIndex() == voxel.GetIndex(),
                  "CellHexahedron has different index for cell");

  CheckPointIndex(cell.GetPointIndex(0), voxel.GetPointIndex(0));
  CheckPointIndex(cell.GetPointIndex(1), voxel.GetPointIndex(1));
  CheckPointIndex(cell.GetPointIndex(2), voxel.GetPointIndex(2));
  CheckPointIndex(cell.GetPointIndex(3), voxel.GetPointIndex(3));
  CheckPointIndex(cell.GetPointIndex(4), voxel.GetPointIndex(4));
  CheckPointIndex(cell.GetPointIndex(5), voxel.GetPointIndex(5));
  CheckPointIndex(cell.GetPointIndex(6), voxel.GetPointIndex(6));
  CheckPointIndex(cell.GetPointIndex(7), voxel.GetPointIndex(7));
}

static void TestCellVoxel()
{
  dax::exec::internal::TopologyUniform gridstruct;

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  for (dax::Id flatIndex = 0; flatIndex < 1000; flatIndex++)
    {
    dax::exec::CellVoxel cell(gridstruct, flatIndex);
    TestCellVoxel(cell, gridstruct, flatIndex);
    }

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  for (dax::Id flatIndex = 0; flatIndex < 1500; flatIndex++)
    {
    dax::exec::CellVoxel cell(gridstruct, flatIndex);
    TestCellVoxel(cell, gridstruct, flatIndex);
    }
}

static void TestCellHexahedron()
{
  std::vector<dax::Id> topo;
  std::vector<dax::Vector3> points;
  dax::exec::internal::TopologyUniform gridstruct;
  dax::exec::internal::TopologyUnstructured<
      dax::exec::CellHexahedron, TestExecutionAdapter> ugrid;
  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  ugrid = make_ugrid(gridstruct,points,topo);

  for (dax::Id flatIndex = 0; flatIndex < 1000; flatIndex++)
    {
    dax::exec::CellHexahedron hex(ugrid, flatIndex);
    dax::exec::CellVoxel vox(gridstruct, flatIndex);
    TestCellHexahedron(hex,vox);
    }

  gridstruct.Origin = dax::make_Vector3(0, 0, 0);
  gridstruct.Spacing = dax::make_Vector3(1, 1, 1);
  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  ugrid = make_ugrid(gridstruct,points,topo);

  for (dax::Id flatIndex = 0; flatIndex < 1500; flatIndex++)
    {
    dax::exec::CellHexahedron cell(ugrid, flatIndex);
    dax::exec::CellVoxel vox(gridstruct, flatIndex);
    TestCellHexahedron(cell,vox);
    }
}

static void TestCells()
{
  TestCellVoxel();
  TestCellHexahedron();
}

int UnitTestCells(int, char *[])
{
  return dax::internal::Testing::Run(TestCells);
}
