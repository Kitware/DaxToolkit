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

#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/ArrayHandleCounting.h>

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
  typedef dax::cont::ArrayHandleCounting<ValueType> ArrayHandleType;

  typedef dax::cont::ArrayHandle<ValueType,
    typename dax::cont::internal::ArrayHandleCountingTraits<ValueType>::Tag>
  ArrayHandleType2;

  typedef typename ArrayHandleType::PortalConstControl PortalType;

  void operator()( const ValueType startingValue )
  {
  ArrayHandleType arrayConst(startingValue, ARRAY_SIZE);

  ArrayHandleType arrayMake = dax::cont::make_ArrayHandleCounting(startingValue,ARRAY_SIZE);

  ArrayHandleType2 arrayHandle =
      ArrayHandleType2(PortalType(startingValue, ARRAY_SIZE));

  DAX_TEST_ASSERT(arrayConst.GetNumberOfValues() == ARRAY_SIZE,
                  "Counting array using constructor has wrong size.");

  DAX_TEST_ASSERT(arrayMake.GetNumberOfValues() == ARRAY_SIZE,
                  "Counting array using make has wrong size.");

  DAX_TEST_ASSERT(arrayHandle.GetNumberOfValues() == ARRAY_SIZE,
                "Counting array using raw array handle + tag has wrong size.");

  ValueType properValue = startingValue;
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    DAX_TEST_ASSERT(arrayConst.GetPortalConstControl().Get(index) == properValue,
                    "Counting array using constructor has unexpected value.");
    DAX_TEST_ASSERT(arrayMake.GetPortalConstControl().Get(index) == properValue,
                    "Counting array using make has unexpected value.");

    DAX_TEST_ASSERT(arrayHandle.GetPortalConstControl().Get(index) == properValue,
                  "Counting array using raw array handle + tag has unexpected value.");
    ++properValue;
    }
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

void TestArrayHandleCounting()
{
  TestFunctor()(dax::Id(0));
  TestFunctor()(dax::Scalar(0));
  TestFunctor()( CountByTwo<dax::Id>(12) );
  TestFunctor()( CountByTwo<dax::Scalar>(1.2f) );
}


} // annonymous namespace

int UnitTestArrayHandleCounting(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestArrayHandleCounting);
}
