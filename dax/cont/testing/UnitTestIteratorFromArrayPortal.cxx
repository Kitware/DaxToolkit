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

#include <dax/cont/IteratorFromArrayPortal.h>

#include <dax/VectorTraits.h>
#include <dax/cont/ArrayPortalFromIterators.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/testing/Testing.h>

namespace {

template<typename T>
struct TemplatedTests
{
  static const dax::Id ARRAY_SIZE = 10;

  typedef T ValueType;
  typedef typename dax::VectorTraits<ValueType>::ComponentType ComponentType;

  ValueType ExpectedValue(dax::Id index, ComponentType value) {
    return dax::cont::VectorFill<ValueType>(index + value);
  }

  template<class IteratorType>
  void FillIterator(IteratorType begin, IteratorType end, ComponentType value) {
    dax::Id index = 0;
    for (IteratorType iter = begin; iter != end; iter++)
      {
      *iter = ExpectedValue(index, value);
      index++;
      }
  }

  template<class IteratorType>
  bool CheckIterator(IteratorType begin,
                     IteratorType end,
                     ComponentType value) {
    dax::Id index = 0;
    for (IteratorType iter = begin; iter != end; iter++)
      {
      if (*iter != ExpectedValue(index, value)) return false;
      index++;
      }
    return true;
  }

  static const ComponentType ORIGINAL_VALUE = 239;

  template<class ArrayPortalType>
  void TestIteratorRead(ArrayPortalType portal)
  {
    typedef dax::cont::IteratorFromArrayPortal<ArrayPortalType> IteratorType;

    IteratorType begin = dax::cont::make_IteratorBegin(portal);
    IteratorType end = dax::cont::make_IteratorEnd(portal);
    DAX_TEST_ASSERT(std::distance(begin, end) == ARRAY_SIZE,
                    "Distance between begin and end incorrect.");

    std::cout << "    Check forward iteration." << std::endl;
    DAX_TEST_ASSERT(CheckIterator(begin, end, ORIGINAL_VALUE),
                    "Forward iteration wrong");

    std::cout << "    Check backward iteration." << std::endl;
    IteratorType middle = end;
    for (dax::Id index = portal.GetNumberOfValues()-1; index >= 0; index--)
      {
      middle--;
      ValueType value = *middle;
      DAX_TEST_ASSERT(value == ExpectedValue(index, ORIGINAL_VALUE),
                      "Backward iteration wrong");
      }

    std::cout << "    Check advance" << std::endl;
    middle = begin + ARRAY_SIZE/2;
    DAX_TEST_ASSERT(std::distance(begin, middle) == ARRAY_SIZE/2,
                    "Bad distance to middle.");
    DAX_TEST_ASSERT(*middle == ExpectedValue(ARRAY_SIZE/2, ORIGINAL_VALUE),
                    "Bad value at middle.");
  }

  template<class ArrayPortalType>
  void TestIteratorWrite(ArrayPortalType portal)
  {
    typedef dax::cont::IteratorFromArrayPortal<ArrayPortalType> IteratorType;

    IteratorType begin = dax::cont::make_IteratorBegin(portal);
    IteratorType end = dax::cont::make_IteratorEnd(portal);

    static const ComponentType WRITE_VALUE = 873;

    std::cout << "    Write values to iterator." << std::endl;
    FillIterator(begin, end, WRITE_VALUE);

    std::cout << "    Check values in portal." << std::endl;
    DAX_TEST_ASSERT(CheckIterator(portal.GetIteratorBegin(),
                                  portal.GetIteratorEnd(),
                                  WRITE_VALUE),
                    "Did not get correct values when writing to iterator.");
  }

  void operator()()
  {
    ValueType array[ARRAY_SIZE];

    FillIterator(array, array+ARRAY_SIZE, ORIGINAL_VALUE);

    dax::cont::ArrayPortalFromIterators<ValueType *>
        portal(array, array+ARRAY_SIZE);
    dax::cont::ArrayPortalFromIterators<const ValueType *>
        const_portal(array, array+ARRAY_SIZE);

    std::cout << "  Test read from iterator." << std::endl;
    TestIteratorRead(portal);

    std::cout << "  Test read from const iterator." << std::endl;
    TestIteratorRead(const_portal);

    std::cout << "  Test write to iterator." << std::endl;
    TestIteratorWrite(portal);
  }
};

struct TestFunctor
{
  template<typename T>
  void operator()(T)
  {
    TemplatedTests<T> tests;
    tests();
  }
};

void TestArrayIteratorFromArrayPortal()
{
  dax::internal::Testing::TryAllTypes(TestFunctor());
}

} // Anonymous namespace

int UnitTestIteratorFromArrayPortal(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestArrayIteratorFromArrayPortal);
}
