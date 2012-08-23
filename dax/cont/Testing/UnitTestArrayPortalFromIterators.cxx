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

#include <dax/cont/ArrayPortalFromIterators.h>

#include <dax/VectorTraits.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/Testing.h>

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

  void operator()()
  {
    ValueType array[ARRAY_SIZE];

    static const ComponentType ORIGINAL_VALUE = 239;
    FillIterator(array, array+ARRAY_SIZE, ORIGINAL_VALUE);

    dax::cont::ArrayPortalFromIterators<ValueType *>
        portal(array, array+ARRAY_SIZE);
    dax::cont::ArrayPortalFromIterators<const ValueType *>
        const_portal(array, array+ARRAY_SIZE);

    DAX_TEST_ASSERT(portal.GetNumberOfValues() == ARRAY_SIZE,
                    "Portal array size wrong.");
    DAX_TEST_ASSERT(const_portal.GetNumberOfValues() == ARRAY_SIZE,
                    "Const portal array size wrong.");

    std::cout << "  Check inital value." << std::endl;
    DAX_TEST_ASSERT(CheckIterator(portal.GetIteratorBegin(),
                                  portal.GetIteratorEnd(),
                                  ORIGINAL_VALUE),
                    "Portal iterator has bad value.");
    DAX_TEST_ASSERT(CheckIterator(const_portal.GetIteratorBegin(),
                                  const_portal.GetIteratorEnd(),
                                  ORIGINAL_VALUE),
                    "Const portal iterator has bad value.");

    static const ComponentType SET_VALUE = 562;

    std::cout << "  Check get/set methods." << std::endl;
    for (dax::Id index = 0; index < ARRAY_SIZE; index++)
      {
      DAX_TEST_ASSERT(portal.Get(index)
                      == ExpectedValue(index, ORIGINAL_VALUE),
                      "Bad portal value.");
      DAX_TEST_ASSERT(const_portal.Get(index)
                      == ExpectedValue(index, ORIGINAL_VALUE),
                      "Bad const portal value.");

      portal.Set(index, ExpectedValue(index, SET_VALUE));
      }

    std::cout << "  Make sure set has correct value." << std::endl;
    DAX_TEST_ASSERT(CheckIterator(portal.GetIteratorBegin(),
                                  portal.GetIteratorEnd(),
                                  SET_VALUE),
                    "Portal iterator has bad value.");
    DAX_TEST_ASSERT(CheckIterator(array, array+ARRAY_SIZE, SET_VALUE),
                    "Array has bad value.");
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

void TestArrayPortalFromIterators()
{
  dax::internal::Testing::TryAllTypes(TestFunctor());
}

} // Anonymous namespace

int UnitTestArrayPortalFromIterators(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestArrayPortalFromIterators);
}
