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

#include <dax/exec/internal/TestingTopologyGenerator.h>

#include <dax/internal/Testing.h>
#include <vector>

namespace
{

struct TestCellFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &topology)
  {
    typedef typename TopologyGenType::CellType CellType;
    typedef typename CellType::PointConnectionsType PointConnectionsType;

    dax::Id numCells = topology.GetNumberOfCells();
    for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
      {
      CellType cell = topology.GetCell(cellIndex);
      DAX_TEST_ASSERT(cell.GetIndex() == cellIndex,
                      "Cell has wrong index.");
      DAX_TEST_ASSERT(cell.GetNumberOfPoints() == CellType::NUM_POINTS,
                      "Cell has wrong number of points");

      PointConnectionsType cellConnections = cell.GetPointIndices();
      PointConnectionsType expectedConnections =
          topology.GetCellConnections(cellIndex);
      DAX_TEST_ASSERT(test_equal(cellConnections, expectedConnections),
                      "Cell has unexpected connections.");
      }
  }
};

void TestCells()
{
  dax::exec::internal::TryAllTopologyTypes(TestCellFunctor());
}

} // anonymous namespace

int UnitTestCells(int, char *[])
{
  return dax::internal::Testing::Run(TestCells);
}
