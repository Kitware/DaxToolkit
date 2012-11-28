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

#include <dax/exec/internal/testing/TestingTopologyGenerator.h>

#include <dax/CellTag.h>

#include <dax/exec/CellVertices.h>

#include <dax/exec/internal/ArrayPortalFromIterators.h>
#include <dax/exec/internal/GridTopologies.h>

#include <dax/internal/testing/Testing.h>

namespace {

void TestVoxelGrid()
{
  std::cout << "Testing voxel grid." << std::endl;
  typedef dax::exec::internal::TopologyUniform TopologyType;

  dax::exec::internal::TestTopology<TopologyType> generator;
  TopologyType topology = generator.GetTopology();

  // Basic topology checks.
  DAX_TEST_ASSERT(topology.Spacing[0] >= 0, "Bad spacing");
  DAX_TEST_ASSERT(topology.Spacing[1] >= 0, "Bad spacing");
  DAX_TEST_ASSERT(topology.Spacing[2] >= 0, "Bad spacing");

  DAX_TEST_ASSERT(topology.Extent.Max[0] >= topology.Extent.Min[0],
                  "Bad extent");
  DAX_TEST_ASSERT(topology.Extent.Max[1] >= topology.Extent.Min[1],
                  "Bad extent");
  DAX_TEST_ASSERT(topology.Extent.Max[2] >= topology.Extent.Min[2],
                  "Bad extent");
}

template<class CellTag>
void TestUnstructuredGrid(CellTag)
{
  typedef dax::exec::internal::ArrayPortalFromIterators<
      std::vector<dax::Id>::iterator> ConnectionsPortal;
  typedef dax::exec::internal::TopologyUnstructured
      <CellTag,ConnectionsPortal> TopologyType;

  dax::exec::internal::TestTopology<TopologyType> generator;
  TopologyType topology = generator.GetTopology();

  DAX_TEST_ASSERT(topology.NumberOfPoints > 0, "Bad number of points");
  DAX_TEST_ASSERT(topology.NumberOfCells > 0, "Bad number of cells");

  // Check that all cell connections are to valid points.
  dax::Id connectionIndex = 0;
  for (dax::Id cellIndex=0; cellIndex < topology.NumberOfCells; cellIndex++)
    {
    dax::exec::CellVertices<CellTag> expectedConnections =
        generator.GetCellConnections(cellIndex);
    for (dax::Id vertexIndex=0;
         vertexIndex < expectedConnections.NUM_VERTICES;
         vertexIndex++)
      {
      dax::Id connection = topology.CellConnections.Get(connectionIndex);
      DAX_TEST_ASSERT(connection >= 0, "Bad cell connection");
      DAX_TEST_ASSERT(connection < topology.NumberOfPoints,
                      "Bad cell connection");
      DAX_TEST_ASSERT(connection == expectedConnections[vertexIndex],
                      "Bad cell connection");
      connectionIndex++;
      }
    }
}

void TestHexahedronGrid()
{
  std::cout << "Testing hexahedron grid." << std::endl;
  typedef dax::CellTagHexahedron CellTag;
  TestUnstructuredGrid(CellTag());
}

void TestTetrahedronGrid()
{
  std::cout << "Testing tetrahedron grid." << std::endl;
  typedef dax::CellTagTetrahedron CellTag;
  TestUnstructuredGrid(CellTag());
}

void TestTriangleGrid()
{
  std::cout << "Testing triangle grid." << std::endl;
  typedef dax::CellTagTriangle CellTag;
  TestUnstructuredGrid(CellTag());
}

struct TestTemplatedTopology {
  template<class TopologyGenerator>
  void operator()(TopologyGenerator &daxNotUsed(topologyGenerator)) const {
    std::cout << "Test disabled.  Original test checked created fields and "
              << "work objects.  When fields objects were removed, these "
              << "features ceased to exist." << std::endl;
#if 0
    typedef typename TopologyGenerator::ExecutionAdapter ExecutionAdapter;
    std::vector<dax::Vector3> pointArray;

    dax::exec::FieldCoordinatesIn<ExecutionAdapter> fieldCoord =
        topologyGenerator.GetCoordinates();

    std::cout << "Set point field." << std::endl;
    dax::exec::FieldPointOut<dax::Vector3, ExecutionAdapter> fieldOut =
        dax::exec::internal::CreateField<dax::exec::FieldPointOut>(
          topologyGenerator, pointArray);
    dax::Id numPoints = topologyGenerator.GetNumberOfPoints();
    for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
      this->CopyField(dax::exec::internal::CreateWorkMapField(topologyGenerator,
                                                              pointIndex),
                      fieldCoord,
                      fieldOut);
      }

    std::cout << "Check point field via cells." << std::endl;
    dax::exec::FieldPointIn<dax::Vector3, ExecutionAdapter> fieldIn =
        dax::exec::internal::CreateField<dax::exec::FieldPointIn>(
          topologyGenerator, pointArray);
    dax::Id numCells = topologyGenerator.GetNumberOfCells();
    for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
      {
      this->TestFieldsEqual(
            dax::exec::internal::CreateWorkMapCell(topologyGenerator,
                                                   cellIndex),
            fieldCoord,
            fieldIn);
      }
#endif
  }
#if 0
private:
  template<class WorkType, class FieldInType, class FieldOutType>
  void CopyField(const WorkType &work,
                 FieldInType fieldIn,
                 FieldOutType fieldOut) const
  {
    work.SetFieldValue(fieldOut, work.GetFieldValue(fieldIn));
  }

  template<class WorkType, class FieldType1, class FieldType2>
  void TestFieldsEqual(const WorkType &work,
                       FieldType1 field1,
                       FieldType2 field2) const
  {
    typedef typename WorkType::CellType CellType;

    dax::Tuple<dax::Vector3,CellType::NUM_POINTS> values1 =
        work.GetFieldValues(field1);
    dax::Tuple<dax::Vector3,CellType::NUM_POINTS> values2 =
        work.GetFieldValues(field2);
    for (dax::Id index = 0; index < CellType::NUM_POINTS; index++)
      {
      DAX_TEST_ASSERT(test_equal(values1[index], values2[index]),
                      "Did not get same coordinates in different work type.");
      }
  }
#endif
};

void TestTopologyGenerator()
{
  std::cout << "*** Independent grid tests." << std::endl;
  TestVoxelGrid();
  TestHexahedronGrid();
  TestTetrahedronGrid();
  TestTriangleGrid();

  std::cout << "*** Templated grid tests." << std::endl;
  dax::exec::internal::TryAllTopologyTypes(TestTemplatedTopology());
}

} // anonymous namespace

int UnitTestTopologyGenerator(int, char *[])
{
  return dax::internal::Testing::Run(TestTopologyGenerator);
}
