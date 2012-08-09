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

#include <dax/cont/ArrayContainerControlBasic.h>

#include <dax/VectorTraits.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/Testing.h>

namespace
{

const dax::Id ARRAY_SIZE = 10;

template <typename T>
struct TemplatedTests
{
  typedef dax::cont::internal::ArrayContainerControl<
      T, dax::cont::ArrayContainerControlTagBasic> ArrayContainerType;
  typedef typename ArrayContainerType::ValueType ValueType;
  typedef typename ArrayContainerType::PortalType PortalType;
  typedef typename PortalType::IteratorType IteratorType;

  void SetContainer(ArrayContainerType &array, ValueType value)
  {
    for (IteratorType iter = array.GetPortal().GetIteratorBegin();
         iter != array.GetPortal().GetIteratorEnd();
         iter ++)
      {
      *iter = value;
      }
  }

  bool CheckContainer(ArrayContainerType &array, ValueType value)
  {
    for (IteratorType iter = array.GetPortal().GetIteratorBegin();
         iter != array.GetPortal().GetIteratorEnd();
         iter ++)
      {
      if (!test_equal(*iter, value)) return false;
      }
    return true;
  }

  static const typename dax::VectorTraits<ValueType>::ComponentType
      STOLEN_ARRAY_VALUE = 4529;

  /// Returned value should later be passed to StealArray2.  It is best to
  /// put as much between the two test parts to maximize the chance of a
  /// deallocated array being overridden (and thus detected).
  ValueType *StealArray1()
  {
    ValueType *stolenArray;

    ValueType stolenArrayValue
        = dax::cont::VectorFill<ValueType>(STOLEN_ARRAY_VALUE);

    ArrayContainerType stealMyArray;
    stealMyArray.Allocate(ARRAY_SIZE);
    SetContainer(stealMyArray, stolenArrayValue);

    DAX_TEST_ASSERT(stealMyArray.GetNumberOfValues() == ARRAY_SIZE,
                    "Array not properly allocated.");
    // This call steals the array and prevents deallocation.
    stolenArray = stealMyArray.StealArray();
    DAX_TEST_ASSERT(stealMyArray.GetNumberOfValues() == 0,
                    "StealArray did not let go of array.");

    return stolenArray;
  }
  void StealArray2(ValueType *stolenArray)
  {
    ValueType stolenArrayValue
        = dax::cont::VectorFill<ValueType>(STOLEN_ARRAY_VALUE);

    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      DAX_TEST_ASSERT(test_equal(stolenArray[index], stolenArrayValue),
                      "Stolen array did not retain values.");
      }
    delete[] stolenArray;
  }

  void BasicAllocation()
  {
    ArrayContainerType arrayContainer;
    DAX_TEST_ASSERT(arrayContainer.GetNumberOfValues() == 0,
                    "New array container not zero sized.");

    arrayContainer.Allocate(ARRAY_SIZE);
    DAX_TEST_ASSERT(arrayContainer.GetNumberOfValues() == ARRAY_SIZE,
                    "Array not properly allocated.");

    const ValueType BASIC_ALLOC_VALUE = dax::cont::VectorFill<ValueType>(548);
    SetContainer(arrayContainer, BASIC_ALLOC_VALUE);
    DAX_TEST_ASSERT(CheckContainer(arrayContainer, BASIC_ALLOC_VALUE),
                    "Array not holding value.");

    arrayContainer.Allocate(ARRAY_SIZE * 2);
    DAX_TEST_ASSERT(arrayContainer.GetNumberOfValues() == ARRAY_SIZE * 2,
                    "Array not reallocated correctly.");

    arrayContainer.ReleaseResources();
    DAX_TEST_ASSERT(arrayContainer.GetNumberOfValues() == 0,
                    "Array not released correctly.");
  }

  void operator()()
  {
    ValueType *stolenArray = StealArray1();

    BasicAllocation();

    StealArray2(stolenArray);
  }
};

struct TestFunctor
{
  template <typename T>
  void operator()(T)
  {
    TemplatedTests<T> tests;
    tests();

  }
};

void TestArrayContainerControlBasic()
{
  dax::internal::Testing::TryAllTypes(TestFunctor());
}

} // Anonymous namespace

int UnitTestArrayContainerControlBasic(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestArrayContainerControlBasic);
}
