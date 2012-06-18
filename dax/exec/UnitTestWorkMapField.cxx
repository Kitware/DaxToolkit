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
#include <dax/exec/WorkMapField.h>

#include <dax/exec/Field.h>

#include <dax/exec/internal/TestingTopologyGenerator.h>

#include <dax/internal/Testing.h>

#include <algorithm>
#include <vector>

namespace {

template <class TopologyGenType>
void BasicTestMapField(const TopologyGenType &topology)
{
  typedef typename TopologyGenType::CellType CellType;
  typedef typename TopologyGenType::ExecutionAdapter ExecutionAdapter;

  std::cout << "Basic field test." << std::endl;

  dax::Id numPoints = topology.GetNumberOfPoints();
  std::vector<dax::Scalar> fieldData(numPoints);

  dax::exec::FieldPointOut<dax::Scalar,ExecutionAdapter> field =
      dax::exec::internal::CreateField<dax::exec::FieldPointOut>(topology,
                                                                 fieldData);

  for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
    {
    // Clear out field array so we can check it.
    std::fill(fieldData.begin(), fieldData.end(), -1.0);
    fieldData[pointIndex] = pointIndex;

    dax::exec::WorkMapField<CellType,ExecutionAdapter> work =
        dax::exec::internal::CreateWorkMapField(topology, pointIndex);

    DAX_TEST_ASSERT(work.GetIndex() == pointIndex,
                    "Work object returned wrong index.");

    dax::Scalar scalarValue = work.GetFieldValue(field);
    DAX_TEST_ASSERT(scalarValue == pointIndex,
                    "Did not get expected scalar value");

    work.SetFieldValue(field, dax::Scalar(-2));
    DAX_TEST_ASSERT(fieldData[pointIndex] == -2,
                    "Field value not set as expected.");
    }
}

template<class TopologyGenType>
void TestMapField(const TopologyGenType &topology)
{
  BasicTestMapField(topology);
}

// Specialization for regular grids so we can also check coordinate compute.
template<>
void TestMapField(const dax::exec::internal::TestTopology<
                      dax::exec::internal::TopologyUniform> &topology)
{
  BasicTestMapField(topology);

  std::cout << "Test coordinates" << std::endl;

  typedef dax::exec::internal::TestTopology<
      dax::exec::internal::TopologyUniform> TopologyGenType;
  typedef TopologyGenType::CellType CellType;
  typedef TopologyGenType::ExecutionAdapter ExecutionAdapter;

  const dax::exec::internal::TopologyUniform gridstruct =topology.GetTopology();

  dax::exec::FieldCoordinatesIn<ExecutionAdapter> fieldCoords
      = topology.GetCoordinates();

  dax::Id numPoints = topology.GetNumberOfPoints();
  for (dax::Id pointFlatIndex = 0; pointFlatIndex < numPoints; pointFlatIndex++)
    {
    dax::Id3 pointIjkIndex = dax::flatIndexToIndex3(pointFlatIndex,
                                                    gridstruct.Extent);

    dax::Vector3 expectedCoords
        = dax::make_Vector3(static_cast<dax::Scalar>(pointIjkIndex[0]),
                            static_cast<dax::Scalar>(pointIjkIndex[1]),
                            static_cast<dax::Scalar>(pointIjkIndex[2]));
    expectedCoords = gridstruct.Origin + expectedCoords * gridstruct.Spacing;

    dax::exec::WorkMapField<CellType,ExecutionAdapter> work =
        dax::exec::internal::CreateWorkMapField(topology, pointFlatIndex);

    dax::Vector3 coords = work.GetFieldValue(fieldCoords);

    DAX_TEST_ASSERT(expectedCoords == coords,
                    "Did not get expected point coordinates.");
    }
}

struct TestMapFieldFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &topology) const
  {
    TestMapField(topology);
  }
};

void RunTestMapField()
{
  dax::exec::internal::TryAllTopologyTypes(TestMapFieldFunctor());
}

} // anonymous namespace

int UnitTestWorkMapField(int, char *[])
{
  return dax::internal::Testing::Run(RunTestMapField);
}
