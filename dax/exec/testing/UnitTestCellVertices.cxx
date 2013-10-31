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
#include <dax/VectorTraits.h>

#include <dax/exec/internal/testing/TestingTopologyGenerator.h>

#include <dax/testing/Testing.h>
#include <dax/testing/VectorTraitsTests.h>

#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>

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

struct ComponentsChecker
{
  template<class CellVerticesType>
  void operator()(const CellVerticesType &)
  {
    dax::testing::TestVectorComponentsTag<CellVerticesType>();
  }
};
struct ComponentsCheckerVertex
{
  template<class CellVerticesType>
  void operator()(const CellVerticesType &)
  {
    dax::testing::TestScalarComponentsTag<CellVerticesType>();
  }
};

template<class CellTag>
struct CellVerticesVectorTraitsTests
{
  void operator()()
  {
    std::cout << "Testing compliance with VectorTraits." << std::endl;

    TestValueGenerator values;
    dax::exec::CellVertices<CellTag> field;

    for (int vertexIndex = 0;
         vertexIndex < dax::CellTraits<CellTag>::NUM_VERTICES;
         vertexIndex++)
      {
      field[vertexIndex] = values[vertexIndex];
      }

    dax::testing::TestVectorType(field);

    typename boost::mpl::if_<
        boost::is_same<CellTag, dax::CellTagVertex>,
        ComponentsCheckerVertex,
        ComponentsChecker>::type()(field);
  }
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
    DAX_TEST_ASSERT(NUM_VERTICES == CVType::TupleType::NUM_COMPONENTS,
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
      vertices[vertexIndex] = values[vertexIndex];
      }

    const typename CVType::TupleType &pointIndices = vertices.GetAsTuple();
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
    typename CVType::TupleType pointIndices;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      pointIndices[vertexIndex] = values[vertexIndex];
      }

    CVType vertices;
    vertices.SetFromTuple(pointIndices);

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(vertices[vertexIndex] == values[vertexIndex],
                      "Got wrong point index.");
      }
  }

  static void TestConstructors()
  {
    std::cout << "Testing setting with constructors."
              << std::endl;

    TestValueGenerator values;
    typename CVType::TupleType pointIndices;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      pointIndices[vertexIndex] = values[vertexIndex];
      }

    CVType vertices(pointIndices);
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(vertices[vertexIndex] == values[vertexIndex],
                      "Got wrong point index.");
      }

    CVType verticesCopy(vertices);
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(verticesCopy[vertexIndex] == values[vertexIndex],
                      "Got wrong point index.");
      }
  }

  static void TestVectorTraits()
  {
    CellVerticesVectorTraitsTests<CellTag>()();
  }

  static void TestAll()
  {
    TestSizesMatch();
    TestSet1GetAll();
    TestSetAllGet1();
    TestConstructors();
    TestVectorTraits();
  }

};

struct TestCellVerticesFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &)
  {
    typedef typename TopologyGenType::CellTag CellTag;
    CellVerticesTests<CellTag>::TestAll();
  }
};

void TestCellVertices()
{
  dax::exec::internal::TryAllTopologyTypes(TestCellVerticesFunctor());
}

} // anonymous namespace

int UnitTestCellVertices(int, char *[])
{
  return dax::testing::Testing::Run(TestCellVertices);
}
