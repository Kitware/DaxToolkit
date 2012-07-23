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

#include <dax/exec/VectorOperations.h>
#include <dax/exec/WorkMapCell.h>

#include <dax/exec/internal/TestingTopologyGenerator.h>

#include <dax/internal/Testing.h>

namespace {

template<class CellType>
static void CompareCoordinates(
    const CellType &cell,
    const dax::Tuple<dax::Vector3,CellType::NUM_POINTS> &vertexCoords,
    dax::Vector3 truePCoords,
    dax::Vector3 trueWCoords)
{
  dax::Vector3 computedWCoords
      = dax::exec::ParametricCoordinatesToWorldCoordinates(cell,
                                                           vertexCoords,
                                                           truePCoords);
  DAX_TEST_ASSERT(test_equal(computedWCoords, trueWCoords),
                  "Computed wrong world coords from parametric coords.");

  dax::Vector3 computedPCoords
      = dax::exec::WorldCoordinatesToParametricCoordinates(cell,
                                                           vertexCoords,
                                                           trueWCoords);
  DAX_TEST_ASSERT(test_equal(computedPCoords, truePCoords, 0.01),
                  "Computed wrong parametric coords from world coords.");
}

struct Add
{
  template<typename T>
  T operator()(const T &a, const T &b) const {
    return a + b;
  }
};

template<class CellType>
void TestPCoordsSpecial(
    const CellType &cell,
    const dax::Tuple<dax::Vector3,CellType::NUM_POINTS> &vertexCoords)
{
  const dax::Id NUM_POINTS = CellType::NUM_POINTS;

  for (dax::Id vertexIndex = 0; vertexIndex < NUM_POINTS; vertexIndex++)
    {
    dax::Vector3 pcoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[vertexIndex];
    dax::Vector3 wcoords = vertexCoords[vertexIndex];
    CompareCoordinates(cell, vertexCoords, pcoords, wcoords);
    }

  dax::Vector3 wcoords =
      dax::exec::VectorReduce(vertexCoords, Add())
      * (dax::Scalar(1)/NUM_POINTS);

  CompareCoordinates(cell,
                     vertexCoords,
                     dax::exec::ParametricCoordinates<CellType>::Center(),
                     wcoords);
}

template<class CellType>
void TestPCoordsSample(
    const CellType &cell,
    const dax::Tuple<dax::Vector3,CellType::NUM_POINTS> &vertexCoords)
{
  dax::Vector3 pcoords;
  for (pcoords[2] = 0.0;
       pcoords[2] <= ((CellType::TOPOLOGICAL_DIMENSIONS > 2) ? 1.0 : 0.0);
       pcoords[2] += 0.25)
    {
    for (pcoords[1] = 0.0;
         pcoords[1] <= ((CellType::TOPOLOGICAL_DIMENSIONS > 1) ? 1.0 : 0.0);
         pcoords[1] += 0.25)
      {
      for (pcoords[0] = 0.0;
           pcoords[0] <= ((CellType::TOPOLOGICAL_DIMENSIONS > 0) ? 1.0 : 0.0);
           pcoords[0] += 0.25)
        {
        // If you convert to world coordinates and back, you should get the
        // same value.
        dax::Vector3 wcoords =
            dax::exec::ParametricCoordinatesToWorldCoordinates(cell,
                                                               vertexCoords,
                                                               pcoords);
        dax::Vector3 computedPCoords =
            dax::exec::WorldCoordinatesToParametricCoordinates(cell,
                                                               vertexCoords,
                                                               wcoords);

        DAX_TEST_ASSERT(test_equal(pcoords, computedPCoords, 0.01),
                        "pcoord/wcoord transform not symmetrical");
        }
      }
    }
}

template<class CellType>
static void TestPCoords(
    const CellType &cell,
    const dax::Tuple<dax::Vector3,CellType::NUM_POINTS> &vertexCoords)
{
  TestPCoordsSpecial(cell, vertexCoords);
  TestPCoordsSample(cell, vertexCoords);
}

struct TestPCoordsFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &topology) {
    dax::Id numCells = topology.GetNumberOfCells();
    for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
      {
      TestPCoords(topology.GetCell(cellIndex),
                  topology.GetCellVertexCoordinates(cellIndex));
      }
  }
};

void TestAllPCoords()
{
  dax::exec::internal::TryAllTopologyTypes(TestPCoordsFunctor());
}

} // Anonymous namespace

int UnitTestParametricCoordinates(int, char *[])
{
  return dax::internal::Testing::Run(TestAllPCoords);
}
