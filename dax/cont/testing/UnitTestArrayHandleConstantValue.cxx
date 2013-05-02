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

#include <dax/cont/internal/ArrayContainerControlConstantValue.h>
#include <dax/cont/ArrayHandleConstantValue.h>

#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;
const dax::Id CONSTANT_VALUE = 100;
void TestConstantValueArray()
{
  typedef dax::cont::ArrayHandleConstantValue<dax::Id> ArrayHandleType;

  std::cout << "Creating array." << std::endl;
  ArrayHandleType arrayConst(CONSTANT_VALUE,ARRAY_SIZE);
  ArrayHandleType arrayMake =
      dax::cont::make_ArrayHandleConstantValue(CONSTANT_VALUE,
                                               ARRAY_SIZE);
  DAX_TEST_ASSERT(arrayConst.GetNumberOfValues() == ARRAY_SIZE,
                  "ConstantValue Array using constructor has wrong size.");

  DAX_TEST_ASSERT(arrayMake.GetNumberOfValues() == ARRAY_SIZE,
                  "ConstantValue Array using make has wrong size.");

  std::cout << "Testing values" << std::endl;
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    DAX_TEST_ASSERT(arrayConst.GetPortalConstControl().Get(index) == CONSTANT_VALUE,
                    "ConstantValue Array using constructor has unexpected value.");
    DAX_TEST_ASSERT(arrayMake.GetPortalConstControl().Get(index) == CONSTANT_VALUE,
                    "ConstantValue Array using make has unexpected value.");
    }
}

} // annonymous namespace

int UnitTestArrayHandleConstantValue(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestConstantValueArray);
}
