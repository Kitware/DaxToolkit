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

#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/Cell.h>

#include <dax/exec/internal/TestExecutionAdapter.h>

#include <dax/internal/Testing.h>
#include <vector>

namespace {

template<typename Grid>
static void TestGridSize(const Grid &gridstruct,
                         dax::Id numPoints,
                         dax::Id numCells)
{
  dax::Id computedNumPoints = dax::exec::internal::numberOfPoints(gridstruct);
  DAX_TEST_ASSERT(computedNumPoints == numPoints,
                  "Structured grid returned wrong number of points");

  dax::Id computedNumCells = dax::exec::internal::numberOfCells(gridstruct);
  DAX_TEST_ASSERT(computedNumCells == numCells,
                  "Structured grid return wrong number of cells");
}

static void TestUniformGrid()
{
  std::cout << "Testing Structured grid size." << std::endl;

  dax::exec::internal::TopologyUniform gridstruct;
  gridstruct.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  gridstruct.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);

  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  TestGridSize(gridstruct, 1331, 1000);

  gridstruct.Extent.Min = dax::make_Id3(5, -9, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 6, 13);
  TestGridSize(gridstruct, 1936, 1500);
}

static void TestUnstructuredGrid()
{
  std::cout << "Testing Unstructured grid size." << std::endl;
  {
    dax::exec::internal::TopologyUnstructured<
        dax::exec::CellHexahedron, TestExecutionAdapter> ugrid;
    TestGridSize(ugrid,0,0);
  }


  //to simplify the process of creating a hexahedron unstrucutured
  //grid I am going to copy the ids and points from a uniform grid.
  dax::exec::internal::TopologyUniform uniform;
  uniform.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  uniform.Spacing = dax::make_Vector3(1.0, 1.0, 0.0);

  //make the grid only contain 8 cells
  uniform.Extent.Min = dax::make_Id3(0, 0, 0);
  uniform.Extent.Max = dax::make_Id3(2, 2, 1);
  TestGridSize(uniform,18,4);

  //copy the point info over to the unstructured grid
  std::vector<dax::Vector3> points;
  dax::Id numPoints = dax::exec::internal::numberOfPoints(uniform);
  for(dax::Id i=0; i < numPoints; ++i)
    {
    points.push_back(dax::exec::internal::pointCoordiantes(uniform,i));
    }

  //copy the cell connection information over
  std::vector<dax::Id> connections;
  dax::Id numCells = dax::exec::internal::numberOfCells(uniform);
  for(dax::Id i=0; i < numCells; ++i)
    {
    dax::exec::CellVoxel vox(uniform,i);
    for(dax::Id j=0; j < vox.GetNumberOfPoints(); ++j)
      {
      connections.push_back(vox.GetPointIndex(j));
      }
    }

  dax::exec::internal::TopologyUnstructured<
      dax::exec::CellHexahedron, TestExecutionAdapter> ugrid(&points[0],
                                                             numPoints,
                                                             &connections[0],
                                                             numCells);

  TestGridSize(ugrid,18,4);
}

static void TestGridSizes()
{
  TestUniformGrid();
  TestUnstructuredGrid();
}

}

int UnitTestGridTopologies(int, char *[])
{
  return dax::internal::Testing::Run(TestGridSizes);
}
