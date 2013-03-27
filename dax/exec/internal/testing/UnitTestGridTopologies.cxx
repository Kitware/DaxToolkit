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

#include <dax/exec/internal/ArrayPortalFromIterators.h>

#include <dax/exec/internal/testing/TestingTopologyGenerator.h>
#include <dax/internal/testing/Testing.h>
#include <vector>

namespace {

template<typename Grid>
static void TestGridSize(const Grid &gridstruct,
                         dax::Id numPoints,
                         dax::Id numCells)
{
  dax::Id computedNumPoints = gridstruct.GetNumberOfPoints();
  DAX_TEST_ASSERT(computedNumPoints == numPoints,
                  "Structured grid returned wrong number of points");

  dax::Id computedNumCells = gridstruct.GetNumberOfCells();
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

  typedef dax::exec::internal::ArrayPortalFromIterators<
      std::vector<dax::Id>::iterator> ConnectionsPortal;
  typedef dax::exec::internal::TopologyUnstructured<
      dax::CellTagHexahedron,ConnectionsPortal> GridType;

  {
    GridType ugrid;
    TestGridSize(ugrid,0,0);
  }

  // Use the TestTopology generator to create a non-trival unstructured grid.
  dax::exec::internal::TestTopology<GridType> generator;
  TestGridSize(generator.GetTopology(),
               generator.GetNumberOfPoints(),
               generator.GetNumberOfCells());
}

struct TestTopologyFunctor {
  template<class TopologyType>
  void operator()(
      const dax::exec::internal::TestTopology<TopologyType> &generator)
  {
    TestGridSize(generator.GetTopology(),
                 generator.GetNumberOfPoints(),
                 generator.GetNumberOfCells());
  }
};

static void TestAllGrids()
{
  std::cout << "Pedantic test of all grid types." << std::endl;
  dax::exec::internal::TryAllTopologyTypes(TestTopologyFunctor());
}

static void TestGridSizes()
{
  TestUniformGrid();
  TestUnstructuredGrid();
  TestAllGrids();
}

}

int UnitTestGridTopologies(int, char *[])
{
  return dax::internal::Testing::Run(TestGridSizes);
}
