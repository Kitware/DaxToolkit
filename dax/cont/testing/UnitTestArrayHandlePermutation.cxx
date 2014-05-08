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

//This sets up the ArrayHandle semantics to allocate pointers and share memory
//between control and execution.
#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL


#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/ArrayHandlePermutation.h>
#include <dax/cont/internal/ArrayPortalFromIterators.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;

template<typename T>
struct CountByThree
{
  CountByThree(): Value() {}
  explicit CountByThree(T t): Value(t) {}

  bool operator==(const T& other) const
    { return Value == other; }

  bool operator==(const CountByThree<T>& other) const
    { return Value == other.Value; }

  CountByThree<T> operator+(dax::Id count) const
  { return CountByThree<T>(Value+(count*3)); }

  CountByThree<T>& operator++()
    { ++Value; ++Value; ++Value; return *this; }

  friend std::ostream& operator<< (std::ostream& os, const CountByThree<T>& obj)
    { os << obj.Value; return os; }
  T Value;
};


template< typename ValueType>
struct TemplatedTests
{
  typedef dax::cont::ArrayHandleCounting<ValueType> CountingArrayHandleType;

  typedef dax::cont::ArrayHandlePermutation<
            dax::cont::ArrayHandle<dax::Id>, //key type
            CountingArrayHandleType > ArrayPermHandleType;

  typedef dax::cont::ArrayHandlePermutation<
            dax::cont::ArrayHandleCounting<dax::Id>, //key type
            CountingArrayHandleType > ArrayCountPermHandleType;

  void operator()( const ValueType startingValue )
  {
    dax::Id everyOtherBuffer[ARRAY_SIZE/2];
    dax::Id fullBuffer[ARRAY_SIZE];

    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      everyOtherBuffer[index/2] = index; //1,3,5,7,9
      fullBuffer[index] = index;
      }

  {
  //verify the different constructors work
  CountingArrayHandleType counting(startingValue,ARRAY_SIZE);
  dax::cont::ArrayHandleCounting<dax::Id> keys(dax::Id(0),ARRAY_SIZE);

  ArrayCountPermHandleType permutation_constructor(keys,counting);

  ArrayCountPermHandleType make_permutation_constructor =
      dax::cont::make_ArrayHandlePermutation(
              dax::cont::make_ArrayHandleCounting(dax::Id(0),ARRAY_SIZE),
              dax::cont::make_ArrayHandleCounting(startingValue, ARRAY_SIZE));

  }

  //make a short permutation array, verify its length and values
  {
  dax::cont::ArrayHandle<dax::Id> keys =
      dax::cont::make_ArrayHandle(everyOtherBuffer, ARRAY_SIZE/2);
  CountingArrayHandleType values(startingValue,ARRAY_SIZE);

  ArrayPermHandleType permutation =
      dax::cont::make_ArrayHandlePermutation(keys,values);

  typename ArrayPermHandleType::PortalConstExecution permPortal =
                                      permutation.PrepareForInput();

  //now lets try to actually get some values of these permutation arrays
  ValueType correct_value = startingValue;
  for(int i=0; i < ARRAY_SIZE/2; ++i)
    {
    ++correct_value; //the permutation should have every other value
    ValueType v = permPortal.Get(i);
    DAX_TEST_ASSERT(v == correct_value, "Count By Three permutation wrong");
    ++correct_value;
    }

  }

  //make a long permutation array, verify its length and values
  {
  dax::cont::ArrayHandle<dax::Id> keys =
      dax::cont::make_ArrayHandle(fullBuffer, ARRAY_SIZE);
  CountingArrayHandleType values(startingValue,ARRAY_SIZE);

  ArrayPermHandleType permutation =
      dax::cont::make_ArrayHandlePermutation(keys,values);

  typename ArrayPermHandleType::PortalConstExecution permPortal =
                                      permutation.PrepareForInput();

  //now lets try to actually get some values of these permutation arrays
  ValueType correct_value = startingValue;
  for(int i=0; i < ARRAY_SIZE; ++i)
    {
    ValueType v = permPortal.Get(i);
    DAX_TEST_ASSERT(v == correct_value, "Full permutation wrong");
    ++correct_value;
    }
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

void TestArrayHandlePermutation()
{
  TestFunctor()( dax::Id(0) );
  TestFunctor()( dax::Scalar(0) );
  TestFunctor()( CountByThree<dax::Id>(12) );
  TestFunctor()( CountByThree<dax::Scalar>(1.2f) );
}



} // annonymous namespace

int UnitTestArrayHandlePermutation(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestArrayHandlePermutation);
}
