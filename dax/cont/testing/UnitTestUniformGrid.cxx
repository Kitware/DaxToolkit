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

#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/testing/TestingGridGenerator.h>
#include <dax/cont/internal/testing/Testing.h>

namespace {

void TestUniformGrid()
{
  const dax::Id DIM = 5;

  dax::cont::internal::TestGrid<dax::cont::UniformGrid<> > gridGen(DIM);
  dax::cont::UniformGrid<> grid = gridGen.GetRealGrid();

  std::cout << "Test basic information." << std::endl;
  DAX_TEST_ASSERT(grid.GetNumberOfCells() == (DIM-1)*(DIM-1)*(DIM-1),
                  "Wrong number of cells.");
  DAX_TEST_ASSERT(grid.GetNumberOfPoints() == DIM*DIM*DIM,
                  "Wrong number of points.");

  std::cout << "Test point indices." << std::endl;
  dax::Id index = 0;
  dax::Id3 ijk;
  for (ijk[2] = 0; ijk[2] < DIM; ijk[2]++)
    {
    for (ijk[1] = 0; ijk[1] < DIM; ijk[1]++)
      {
      for (ijk[0] = 0; ijk[0] < DIM; ijk[0]++)
        {
        DAX_TEST_ASSERT(grid.ComputePointIndex(ijk) == index,
                        "Unexpected point index.");
        DAX_TEST_ASSERT(grid.ComputePointLocation(index) == ijk,
                        "Unexpected point location.");
        index++;
        }
      }
    }

  std::cout << "Test cell indices." << std::endl;
  index = 0;
  for (ijk[2] = 0; ijk[2] < DIM-1; ijk[2]++)
    {
    for (ijk[1] = 0; ijk[1] < DIM-1; ijk[1]++)
      {
      for (ijk[0] = 0; ijk[0] < DIM-1; ijk[0]++)
        {
        DAX_TEST_ASSERT(grid.ComputeCellIndex(ijk) == index,
                        "Unexpected cell index.");
        DAX_TEST_ASSERT(grid.ComputeCellLocation(index) == ijk,
                        "Unexpected cell location.");
        index++;
        }
      }
    }

  std::cout << "Test point coordinates portal.";
  dax::cont::UniformGrid<>::PointCoordinatesType coords =
      grid.GetPointCoordinates();
  dax::cont::UniformGrid<>::PointCoordinatesType::PortalConstControl
      coordsPortal = coords.GetPortalConstControl();
  for (index = 0; index < grid.GetNumberOfPoints(); index++)
    {
    dax::Vector3 gridCoords = grid.ComputePointCoordinates(index);
    dax::Vector3 portalCoords = coordsPortal.Get(index);
    DAX_TEST_ASSERT(gridCoords == portalCoords,
                    "Point coordinates seem wrong.");
    }

  std::cout << "Test PrepareForInput" << std::endl;
  dax::cont::UniformGrid<>::TopologyStructConstExecution topology =
      grid.PrepareForInput();
  DAX_TEST_ASSERT(topology.Origin == grid.GetOrigin(),
                  "Topology origin wrong.");
  DAX_TEST_ASSERT(topology.Spacing == grid.GetSpacing(),
                  "Topology spacing wrong.");
  DAX_TEST_ASSERT(topology.Extent.Min == grid.GetExtent().Min,
                  "Topology extent wrong.");
  DAX_TEST_ASSERT(topology.Extent.Max == grid.GetExtent().Max,
                  "Topology extent wrong.");
}

} // anonymous namespace

int UnitTestUniformGrid(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestUniformGrid);
}
