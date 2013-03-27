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

struct ComponentsChecker
{
  template<class CellFieldType>
  void operator()(const CellFieldType &)
  {
    dax::testing::TestVectorComponentsTag<CellFieldType>();
  }
};
struct ComponentsCheckerVertex
{
  template<class CellFieldType>
  void operator()(const CellFieldType &)
  {
    dax::testing::TestScalarComponentsTag<CellFieldType>();
  }
};

template<typename T, class CellTag>
struct CellFieldVectorTraitsTests
{
  void operator()()
  {
    // The vector traits test assumes scalar components, which we don't
    // guarantee.  Specialize for that.
    std::cout << "Skipping traits test for non-scalar field." << std::endl;
  }
};
template<class CellTag>
struct CellFieldVectorTraitsTests<dax::Scalar, CellTag>
{
  void operator()()
  {
    std::cout << "  Testing compliance with VectorTraits." << std::endl;

    TestValueGenerator<dax::Scalar> values;
    dax::exec::CellField<dax::Scalar,CellTag> field;

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
struct CellFieldVectorTraitsTests<dax::Id, CellTag>
{
  void operator()()
  {
    std::cout << "  Testing compliance with VectorTraits." << std::endl;

    TestValueGenerator<dax::Id> values;
    dax::exec::CellField<dax::Id,CellTag> field;

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

template<typename T, class CellTag>
struct CellFieldTests
{
  typedef dax::exec::CellField<T,CellTag> CFType;
  const static dax::Id NUM_VERTICES = CFType::NUM_VERTICES;

  static void TestSizesMatch()
  {
    std::cout << "  Testing sizes." << std::endl;

    DAX_TEST_ASSERT(NUM_VERTICES == CFType::NUM_VERTICES,
                    "Test has wrong number of vertices?");
    DAX_TEST_ASSERT(
          CFType::NUM_VERTICES == dax::CellTraits<CellTag>::NUM_VERTICES,
          "CellField reports wrong number of points.");
    DAX_TEST_ASSERT(NUM_VERTICES == CFType::TupleType::NUM_COMPONENTS,
                    "CellField tuple has wrong number of components.");
  }

  static void TestSet1GetAll()
  {
    std::cout << "  Testing setting components individually, getting tuple."
              << std::endl;

    TestValueGenerator<T> values;
    CFType field;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      field[vertexIndex] = values[vertexIndex];
      }

    const typename CFType::TupleType &valueTuple = field.GetAsTuple();
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
    typename CFType::TupleType valueTuple;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      valueTuple[vertexIndex] = values[vertexIndex];
      }

    CFType field;
    field.SetFromTuple(valueTuple);

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(field[vertexIndex] == values[vertexIndex],
                      "Got wrong point index.");
      }
  }

  static void TestConstructors()
  {
    std::cout << "  Testing setting with constructors."
              << std::endl;

    TestValueGenerator<T> values;
    typename CFType::TupleType valueTuple;

    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      valueTuple[vertexIndex] = values[vertexIndex];
      }

    CFType field(valueTuple);
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(field[vertexIndex] == values[vertexIndex],
                      "Got wrong point index.");
      }

    CFType fieldCopy(field);
    for (int vertexIndex = 0; vertexIndex < NUM_VERTICES; vertexIndex++)
      {
      DAX_TEST_ASSERT(fieldCopy[vertexIndex] == values[vertexIndex],
                      "Got wrong point index.");
      }
  }

  static void TestVectorTraits()
  {
    CellFieldVectorTraitsTests<T, CellTag>()();
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

template<class CellTag>
struct TestCellFieldFunctorType
{
  template<typename T>
  void operator()(T)
  {
    CellFieldTests<T, CellTag>::TestAll();
  }
};

struct TestCellFieldFunctor
{
  template<class TopologyGenType>
  void operator()(const TopologyGenType &)
  {
    typedef typename TopologyGenType::CellTag CellTag;
    dax::internal::Testing::TryAllTypes(TestCellFieldFunctorType<CellTag>());
  }
};

void TestCellField()
{
  dax::exec::internal::TryAllTopologyTypes(TestCellFieldFunctor());
}

} // anonymous namespace

int UnitTestCellField(int, char *[])
{
  return dax::internal::Testing::Run(TestCellField);
}
