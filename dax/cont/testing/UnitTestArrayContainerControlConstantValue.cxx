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

#include <dax/cont/ArrayContainerControlConstantValue.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/cont/internal/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;
const dax::Id CONSTANT_VALUE = 100;
void TestConstantValueArray()
{
  std::cout << "Creating array." << std::endl;
  dax::cont::ArrayPortalConstantValue portal(CONSTANT_VALUE,ARRAY_SIZE);
  dax::cont::ArrayHandle<dax::Id, dax::cont::ArrayContainerControlTagConstantValue>
      array(portal);
  DAX_TEST_ASSERT(array.GetNumberOfValues() == ARRAY_SIZE,
                  "Array has wrong size.");

  std::cout << "Testing values" << std::endl;
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    DAX_TEST_ASSERT(array.GetPortalConstControl().Get(index) == CONSTANT_VALUE,
                    "Array has unexpected value.");
    }
}

} // annonymous namespace

int UnitTestArrayContainerControlConstantValue(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestConstantValueArray);
}
