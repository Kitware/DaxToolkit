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
#include <dax/exec/CellVertices.h>

#include <dax/CellTag.h>

#include <dax/exec/internal/testing/TestingTopologyGenerator.h>

#include <dax/internal/testing/Testing.h>

namespace
{

dax::Id CreateUniqueOffset()
{
  static dax::Id nextOffset = 1;
  return nextOffset++;
}

struct TestValueGenerator
{
  TestValueGenerator() : Offset(CreateUniqueOffset() * 1000) {  }
  dax::Id operator[](dax::Id index) { return index + this->Offset; }
private:
  const dax::Id Offset;
};

template<class CellTag>
struct CellVerticesTests
{
  typedef dax::exec::CellVertices<CellTag> CVType;
  const static dax::Id NUM_VERTICES = CVType::NUM_VERTICES;

  static void TestSizesMatch()
  {
    std::cout << "Testing sizes." << std::endl;

    DAX_TEST_ASSERT(NUM_VERTICES == CVType::NUM_VERTICES,
                    "Test has wrong number of vertices?");
    DAX_TEST_ASSERT(
          CVType::NUM_VERTICES == dax::CellTraits<CellTag>::NUM_VERTICES,
          "CellVertices reports wrong number of points.");
    DAX_TEST_ASSERT(NUM_VERTICES == CVType::PointIndicesType::NUM_COMPONENTS,
                    "CellVertices tuple has wrong number of components.");
  }

  static void TestSet1GetAll()
  {
    std::cout << "Testing setting components individually, getting tuple."
              << std::endl;

    TestValueGenerator values;
    CVType vertices;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      vertices.SetPointIndex(vertexIndex, values[vertexIndex]);
      }

    const typename CVType::PointIndicesType &pointIndices =
        vertices.GetPointIndices();
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(pointIndices[vertexIndex] == values[vertexIndex],
                      "Got wrong point index.");
      }
  }

  static void TestSetAllGet1()
  {
    std::cout << "Testing setting tuple, getting components individually."
              << std::endl;

    TestValueGenerator values;
    typename CVType::PointIndicesType pointIndices;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      pointIndices[vertexIndex] = values[vertexIndex];
      }

    CVType vertices;
    vertices.SetPointIndices(pointIndices);

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(
            vertices.GetPointIndex(vertexIndex) == values[vertexIndex],
            "Got wrong point index.");
      }
  }

  static void TestConstructors()
  {
    std::cout << "Testing setting with constructors."
              << std::endl;

    TestValueGenerator values;
    typename CVType::PointIndicesType pointIndices;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      pointIndices[vertexIndex] = values[vertexIndex];
      }

    CVType vertices(pointIndices);
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(
            vertices.GetPointIndex(vertexIndex) == values[vertexIndex],
            "Got wrong point index.");
      }

    CVType verticesCopy(vertices);
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(
            verticesCopy.GetPointIndex(vertexIndex) == values[vertexIndex],
            "Got wrong point index.");
      }
  }

  static void TestAll()
  {
    TestSizesMatch();
    TestSet1GetAll();
    TestSetAllGet1();
    TestConstructors();
  }

};

//struct TestCellVerticesFunctor
//{
//  template<class TopologyGenType>
//  void operator()(const TopologyGenType &topology)
//  {
//    typedef typename TopologyGenType::CellType CellType;
//    typedef typename CellType::PointConnectionsType PointConnectionsType;

//    dax::Id numCells = topology.GetNumberOfCells();
//    for (dax::Id cellIndex = 0; cellIndex < numCells; cellIndex++)
//      {
//      CellType cell = topology.GetCell(cellIndex);
//      DAX_TEST_ASSERT(cell.GetNumberOfPoints() == CellType::NUM_POINTS,
//                      "Cell has wrong number of points");

//      PointConnectionsType cellConnections = cell.GetPointIndices();
//      PointConnectionsType expectedConnections =
//          topology.GetCellConnections(cellIndex);
//      DAX_TEST_ASSERT(test_equal(cellConnections, expectedConnections),
//                      "Cell has unexpected connections.");
//      }
//  }
//};

void TestCellVertices()
{
  // Enable all tests once the TryAllTopologyTypes uses cell tags.
//  dax::exec::internal::TryAllTopologyTypes(TestCellVerticesFunctor());
  CellVerticesTests<dax::CellTagHexahedron>::TestAll();
}

} // anonymous namespace

int UnitTestCellVertices(int, char *[])
{
  return dax::internal::Testing::Run(TestCellVertices);
}
