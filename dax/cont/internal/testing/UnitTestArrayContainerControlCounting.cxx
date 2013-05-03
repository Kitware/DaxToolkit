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

#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_ERROR

#include <dax/cont/internal/ArrayContainerControlCounting.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;

//increments by two instead of one wrapper
template<typename T>
struct CountByTwo
{
  CountByTwo(): Value() {}
  explicit CountByTwo(T t): Value(t) {}

  bool operator==(const T& other) const
    { return Value == other; }

  bool operator==(const CountByTwo<T>& other) const
    { return Value == other.Value; }

  CountByTwo<T> operator+(dax::Id count) const
  { return CountByTwo<T>(Value+(count*2)); }

  CountByTwo<T>& operator++()
    { ++Value; ++Value; return *this; }

  friend std::ostream& operator<< (std::ostream& os, const CountByTwo<T>& obj)
    { os << obj.Value; return os; }
  T Value;
};



template< typename ValueType>
struct TemplatedTests
{
  typedef dax::cont::internal::ArrayContainerControlTagCounting ContainerTagType;
  typedef dax::cont::internal::ArrayContainerControl<ValueType,ContainerTagType>
                  ArrayContainerType;
  typedef dax::cont::internal::ArrayPortalCounting<ValueType> PortalType;


  void TestAccess( ValueType startingValue ) const
  {
  PortalType portal(startingValue,ARRAY_SIZE);

  DAX_TEST_ASSERT(portal.GetNumberOfValues() == ARRAY_SIZE,
                  "portal has wrong size.");

  typedef typename PortalType::IteratorType Iterator;
  dax::Id count = 0;
  ValueType properValue = startingValue;
  for (Iterator i = portal.GetIteratorBegin();
       i != portal.GetIteratorEnd();
       ++i)
    {
    DAX_TEST_ASSERT( ValueType(*i) == properValue,
                    "portal iteration has unexpected value.");
    ++count;
    ++properValue;
    }

  DAX_TEST_ASSERT(count == ARRAY_SIZE, "portal iteration did go long enough.");
  }

  void TestAllocation() const
  {
    ArrayContainerType arrayContainer;

    try{ arrayContainer.GetNumberOfValues();
      DAX_TEST_ASSERT(false == true,
                      "Counting Value Container GetNumberOfValues method didn't throw error.");
      }
    catch(dax::cont::ErrorControlBadValue e){}

    try{ arrayContainer.Allocate(ARRAY_SIZE);
      DAX_TEST_ASSERT(false == true,
                    "Counting Value Container Allocate method didn't throw error.");
      }
    catch(dax::cont::ErrorControlBadValue e){}

    try
      {
      arrayContainer.Shrink(ARRAY_SIZE);
      DAX_TEST_ASSERT(true==false,
                      "Counting Value shrink do a larger size was possible. This can't be allowed.");
      }
    catch(dax::cont::ErrorControlBadValue){}

    try
      {
      arrayContainer.ReleaseResources();
      DAX_TEST_ASSERT(true==false,
                      "Can't Release a Counting Value array");
      }
    catch(dax::cont::ErrorControlBadValue){}
  }

  void operator()(const ValueType t)
  {
    TestAccess(t);
    TestAllocation();
  }
};

struct TestFunctor
{
  template <typename T>
  void operator()(const T t)
  {
    TemplatedTests<T> tests;
    tests(t);
  }
};

void TestArrayContainerCounting()
{
  TestFunctor()( dax::Id(0) );
  TestFunctor()( dax::Id(50) );
  TestFunctor()( dax::Scalar(2.5) );
  TestFunctor()( CountByTwo<dax::Id>(12) );
  TestFunctor()( CountByTwo<dax::Scalar>(-40.2f) );
}


} // annonymous namespace

int UnitTestArrayContainerControlCounting(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestArrayContainerCounting);
}
