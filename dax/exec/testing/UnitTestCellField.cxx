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
#include <dax/exec/CellField.h>

#include <dax/CellTag.h>
#include <dax/VectorTraits.h>

#include <dax/exec/internal/testing/TestingTopologyGenerator.h>

#include <dax/internal/testing/Testing.h>

namespace
{

dax::Id CreateUniqueOffset()
{
  static dax::Id nextOffset = 1;
  return nextOffset++;
}

template<typename T>
struct TestValueGenerator
{
  typedef typename dax::VectorTraits<T>::ComponentType ComponentType;

  TestValueGenerator() : Offset(CreateUniqueOffset() * 10000) {  }
  T operator[](dax::Id index) {
    T value;
    for (int component = 0;
         component < dax::VectorTraits<T>::NUM_COMPONENTS;
         component++)
      {
      dax::VectorTraits<T>::SetComponent(
            value,
            component,
            static_cast<ComponentType>(this->Offset+10*index+component));
      }
    return value;
  }
private:
  const dax::Id Offset;
};

template<typename T, class CellTag>
struct CellFieldTests
{
  typedef dax::exec::CellField<T,CellTag> CVType;
  const static dax::Id NUM_VERTICES = CVType::NUM_VERTICES;

  static void TestSizesMatch()
  {
    std::cout << "  Testing sizes." << std::endl;

    DAX_TEST_ASSERT(NUM_VERTICES == CVType::NUM_VERTICES,
                    "Test has wrong number of vertices?");
    DAX_TEST_ASSERT(
          CVType::NUM_VERTICES == dax::CellTraits<CellTag>::NUM_VERTICES,
          "CellField reports wrong number of points.");
    DAX_TEST_ASSERT(NUM_VERTICES == CVType::ValuesTupleType::NUM_COMPONENTS,
                    "CellField tuple has wrong number of components.");
  }

  static void TestSet1GetAll()
  {
    std::cout << "  Testing setting components individually, getting tuple."
              << std::endl;

    TestValueGenerator<T> values;
    CVType vertices;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      vertices.SetValue(vertexIndex, values[vertexIndex]);
      }

    const typename CVType::ValuesTupleType &valueTuple = vertices.GetValues();
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(valueTuple[vertexIndex] == values[vertexIndex],
                      "Got wrong point index.");
      }
  }

  static void TestSetAllGet1()
  {
    std::cout << "  Testing setting tuple, getting components individually."
              << std::endl;

    TestValueGenerator<T> values;
    typename CVType::ValuesTupleType valueTuple;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      valueTuple[vertexIndex] = values[vertexIndex];
      }

    CVType vertices;
    vertices.SetValues(valueTuple);

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(
            vertices.GetValue(vertexIndex) == values[vertexIndex],
            "Got wrong point index.");
      }
  }

  static void TestConstructors()
  {
    std::cout << "  Testing setting with constructors."
              << std::endl;

    TestValueGenerator<T> values;
    typename CVType::ValuesTupleType valueTuple;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      valueTuple[vertexIndex] = values[vertexIndex];
      }

    CVType vertices(valueTuple);
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(
            vertices.GetValue(vertexIndex) == values[vertexIndex],
            "Got wrong point index.");
      }

    CVType verticesCopy(vertices);
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(
            verticesCopy.GetValue(vertexIndex) == values[vertexIndex],
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

template<class CellTag>
struct TestCellFieldFunctorType
{
  template<typename T>
  void operator()(T)
  {
    CellFieldTests<T, CellTag>::TestAll();
  }
};

//struct TestCellFieldFunctor
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

void TestCellField()
{
  // Enable all tests once the TryAllTopologyTypes uses cell tags.
//  dax::exec::internal::TryAllTopologyTypes(TestCellFieldFunctor());
  dax::internal::Testing::TryAllTypes(
        TestCellFieldFunctorType<dax::CellTagHexahedron>());
}

} // anonymous namespace

int UnitTestCellField(int, char *[])
{
  return dax::internal::Testing::Run(TestCellField);
}
