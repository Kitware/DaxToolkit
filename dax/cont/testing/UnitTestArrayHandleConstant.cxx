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
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR

#include <dax/cont/ArrayHandleConstant.h>

#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;

template< typename ValueType>
struct TemplatedTests
{
  typedef dax::cont::ArrayHandleConstant<ValueType> ArrayHandleType;

  typedef dax::cont::ArrayHandle<ValueType,
    typename dax::cont::internal::ArrayHandleConstantTraits<ValueType>::Tag>
  ArrayHandleType2;

  typedef typename ArrayHandleType::PortalConstControl PortalType;

  void operator()(const ValueType constantValue) const
  {
  ArrayHandleType handle(constantValue,ARRAY_SIZE);

  ArrayHandleType make_handle =
        dax::cont::make_ArrayHandleConstant(constantValue, ARRAY_SIZE);

  ArrayHandleType2 superclass_handle =
      ArrayHandleType2(PortalType(constantValue, ARRAY_SIZE));


  DAX_TEST_ASSERT(handle.GetNumberOfValues() == ARRAY_SIZE,
                  "Constant array using constructor has wrong size.");

  DAX_TEST_ASSERT(make_handle.GetNumberOfValues() == ARRAY_SIZE,
                  "Constant array using make has wrong size.");

  DAX_TEST_ASSERT(superclass_handle.GetNumberOfValues() == ARRAY_SIZE,
                  "Constant array using array handle + portal has wrong size.");


  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    DAX_TEST_ASSERT(make_handle.GetPortalConstControl().Get(index) == constantValue,
                    "Constant array using make helper has unexpected value.");
    DAX_TEST_ASSERT(handle.GetPortalConstControl().Get(index) == constantValue,
                    "Constant array using constructor  has unexpected value.");
    DAX_TEST_ASSERT(superclass_handle.GetPortalConstControl().Get(index) == constantValue,
                    "Constant array using raw array handle + tag has unexpected value.");
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

void TestArrayHandleConstant()
{
  dax::testing::Testing::TryAllTypes(TestFunctor());
}


} // annonymous namespace

int UnitTestArrayHandleConstant(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestArrayHandleConstant);
}
