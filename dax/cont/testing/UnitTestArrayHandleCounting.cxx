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

#include <dax/cont/internal/ArrayContainerControlCounting.h>
#include <dax/cont/ArrayHandleCounting.h>

#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;

void TestCountingArray()
{
  std::cout << "Creating counting array." << std::endl;
  dax::cont::ArrayHandleCounting< > arrayConst(ARRAY_SIZE);
  dax::cont::ArrayHandleCounting< > arrayMake = dax::cont::make_ArrayHandleCounting(ARRAY_SIZE);
  DAX_TEST_ASSERT(arrayConst.GetNumberOfValues() == ARRAY_SIZE,
                  "Counting array using constructor has wrong size.");

  DAX_TEST_ASSERT(arrayMake.GetNumberOfValues() == ARRAY_SIZE,
                  "Counting array using make has wrong size.");

  std::cout << "Testing values" << std::endl;
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    DAX_TEST_ASSERT(arrayConst.GetPortalConstControl().Get(index) == index,
                    "Counting array using constructor has unexpected value.");
    DAX_TEST_ASSERT(arrayMake.GetPortalConstControl().Get(index) == index,
                    "Counting array using make has unexpected value.");
    }
}

} // annonymous namespace

int UnitTestArrayHandleCounting(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestCountingArray);
}
