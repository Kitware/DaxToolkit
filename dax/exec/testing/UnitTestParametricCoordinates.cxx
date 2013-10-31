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

#include <dax/exec/internal/testing/TestingTopologyGenerator.h>

#include <dax/testing/Testing.h>

namespace {

template<class CellTag>
static void CompareCoordinates(
    const dax::exec::CellField<dax::Vector3,CellTag> &vertexCoords,
    dax::Vector3 truePCoords,
    dax::Vector3 trueWCoords,
    CellTag)
{
  dax::Vector3 computedWCoords
      = dax::exec::ParametricCoordinatesToWorldCoordinates(vertexCoords,
                                                           truePCoords,
                                                           CellTag());
  DAX_TEST_ASSERT(test_equal(computedWCoords, trueWCoords),
                  "Computed wrong world coords from parametric coords.");

  dax::Vector3 computedPCoords
      = dax::exec::WorldCoordinatesToParametricCoordinates(vertexCoords,
                                                           trueWCoords,
                                                           CellTag());
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

template<class CellTag>
void TestPCoordsSpecial(
    const dax::exec::CellField<dax::Vector3,CellTag> &vertexCoords,
    CellTag)
{
  const dax::Id NUM_VERTICES = vertexCoords.NUM_VERTICES;

  for (dax::Id vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
    {
    dax::Vector3 pcoords =
        dax::exec::ParametricCoordinates<CellTag>::Vertex()[vertexIndex];
    dax::Vector3 wcoords = vertexCoords[vertexIndex];
    CompareCoordinates(vertexCoords, pcoords, wcoords, CellTag());
    }

  dax::Vector3 wcoords =
      dax::exec::VectorReduce(vertexCoords, Add())
      * (dax::Scalar(1)/NUM_VERTICES);

  CompareCoordinates(vertexCoords,
                     dax::exec::ParametricCoordinates<CellTag>::Center(),
                     wcoords,
                     CellTag());
}

template<class CellTag>
void TestPCoordsSample(
    const dax::exec::CellField<dax::Vector3,CellTag> &vertexCoords,
    CellTag)
{
  const int TOPOLOGICAL_DIMENSIONS =
      dax::CellTraits<CellTag>::TOPOLOGICAL_DIMENSIONS;
  dax::Vector3 pcoords;
  for (pcoords[2] = 0.0;
       pcoords[2] <= ((TOPOLOGICAL_DIMENSIONS > 2) ? 1.0 : 0.0);
       pcoords[2] += 0.25)
    {
    for (pcoords[1] = 0.0;
         pcoords[1] <= ((TOPOLOGICAL_DIMENSIONS > 1) ? 1.0 : 0.0);
         pcoords[1] += 0.25)
      {
      for (pcoords[0] = 0.0;
           pcoords[0] <= ((TOPOLOGICAL_DIMENSIONS > 0) ? 1.0 : 0.0);
           pcoords[0] += 0.25)
        {
        // If you convert to world coordinates and back, you should get the
        // same value.
        dax::Vector3 wcoords =
            dax::exec::ParametricCoordinatesToWorldCoordinates(vertexCoords,
                                                               pcoords,
                                                               CellTag());
        dax::Vector3 computedPCoords =
            dax::exec::WorldCoordinatesToParametricCoordinates(vertexCoords,
                                                               wcoords,
                                                               CellTag());

        DAX_TEST_ASSERT(test_equal(pcoords, computedPCoords, 0.01),
                        "pcoord/wcoord transform not symmetrical");
        }
      }
    }
}

template<class CellTag>
static void TestPCoords(
    const dax::exec::CellField<dax::Vector3,CellTag> &vertexCoords,
    CellTag)
{
  TestPCoordsSpecial(vertexCoords, CellTag());
  TestPCoordsSample(vertexCoords, CellTag());
}

struct TestPCoordsFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &topology) {
    dax::Id numCells = topology.GetNumberOfCells();
    for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
      {
      TestPCoords(topology.GetCellVertexCoordinates(cellIndex),
                  typename TopologyGenType::CellTag());
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
  return dax::testing::Testing::Run(TestAllPCoords);
}
