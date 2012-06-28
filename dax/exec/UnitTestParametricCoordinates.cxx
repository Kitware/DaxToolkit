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

template<class WorkType, class CellType, class ExecutionAdapter>
static void CompareCoordinates(
    const WorkType &work,
    const CellType &cell,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField,
    dax::Vector3 truePCoords,
    dax::Vector3 trueWCoords)
{
  dax::Vector3 computedWCoords
      = dax::exec::ParametricCoordinatesToWorldCoordinates(work,
                                                           cell,
                                                           coordField,
                                                           truePCoords);
  DAX_TEST_ASSERT(test_equal(computedWCoords, trueWCoords),
                  "Computed wrong world coords from parametric coords.");

  dax::Vector3 computedPCoords
      = dax::exec::WorldCoordinatesToParametricCoordinates(work,
                                                           cell,
                                                           coordField,
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

template<class CellType, class ExecutionAdapter>
void TestPCoordsSpecial(
    const dax::exec::WorkMapCell<CellType, ExecutionAdapter> &work,
    const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField)
{
  const dax::Id NUM_POINTS = CellType::NUM_POINTS;

  CellType cell = work.GetCell();
  dax::Tuple<dax::Vector3, NUM_POINTS> worldCoordinates =
      work.GetFieldValues(coordField);

  for (dax::Id vertexIndex = 0; vertexIndex < NUM_POINTS; vertexIndex++)
    {
    dax::Vector3 pcoords =
        dax::exec::ParametricCoordinates<CellType>::Vertex()[vertexIndex];
    dax::Vector3 wcoords = worldCoordinates[vertexIndex];
    CompareCoordinates(work, cell, coordField, pcoords, wcoords);
    }

  dax::Vector3 wcoords =
      dax::exec::VectorReduce(worldCoordinates, Add())
      * (dax::Scalar(1)/NUM_POINTS);

  CompareCoordinates(work,
                     cell,
                     coordField,
                     dax::exec::ParametricCoordinates<CellType>::Center(),
                     wcoords);
}

template<class CellType, class ExecutionAdapter>
static void TestPCoords(
  const dax::exec::WorkMapCell<CellType, ExecutionAdapter> &work,
  const dax::exec::FieldCoordinatesIn<ExecutionAdapter> &coordField)
{
  TestPCoordsSpecial(work, coordField);
}

struct TestPCoordsFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &topology) {
    typedef typename TopologyGenType::CellType CellType;
    typedef typename TopologyGenType::ExecutionAdapter ExecutionAdapter;

    dax::exec::FieldCoordinatesIn<TestExecutionAdapter> coordField =
        topology.GetCoordinates();

    dax::Id numCells = topology.GetNumberOfCells();
    for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
      {
      dax::exec::WorkMapCell<CellType,ExecutionAdapter> work =
          dax::exec::internal::CreateWorkMapCell(topology, cellIndex);
      TestPCoords(work, coordField);
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
