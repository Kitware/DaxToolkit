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
#include <dax/cont/ArrayContainerControlBasic.h>

#include <dax/cont/internal/ArrayContainerControlPermutation.h>
#include <dax/cont/internal/ArrayPortalFromIterators.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleCounting.h>

#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;
const dax::Id VALUE_ARRAY_SIZE = 3;
void TestPermutationArray()
{
  typedef dax::cont::ArrayContainerControlTagBasic Container;

  typedef dax::cont::ArrayHandle<dax::Id,Container>::PortalControl IdPortalType;

  typedef dax::cont::internal::ArrayPortalCounting<dax::Id> ReadOnlyValuePortalType;
  typedef dax::cont::ArrayHandleCounting<dax::Id,Container> ReadOnlyArrayHandle;


  dax::Id arrayKey[ARRAY_SIZE];
  for (dax::Id i=0; i<ARRAY_SIZE;++i)
    {
    arrayKey[i] = i % VALUE_ARRAY_SIZE;
    }
  dax::Id arrayValue[VALUE_ARRAY_SIZE];


  // Make  readWrite array from portals
  IdPortalType keyPortal(arrayKey,arrayKey + ARRAY_SIZE);
  IdPortalType readWriteValuePortal(arrayValue,arrayValue + VALUE_ARRAY_SIZE);
  dax::cont::internal::ArrayPortalPermutation <IdPortalType,IdPortalType>
      readWritePortal(keyPortal, readWriteValuePortal);

  // Make readOnly array from portals
  ReadOnlyValuePortalType readOnlyValuePortaly(0,VALUE_ARRAY_SIZE);
  dax::cont::internal::ArrayPortalPermutation <IdPortalType,ReadOnlyValuePortalType>
      readOnlyPortal(keyPortal, readOnlyValuePortaly);

  // copy readOnlyArray to readWriteArray (i.e readWriteArray = readOnlyArray)
  for (dax::Id index=0; index < ARRAY_SIZE; index++)
    {
    readWritePortal.Set(index, readOnlyPortal.Get(index));
    }

  DAX_TEST_ASSERT(readWritePortal.GetNumberOfValues() == ARRAY_SIZE,
                  "readWritePortal has wrong size.");

  DAX_TEST_ASSERT(readOnlyPortal.GetNumberOfValues() == ARRAY_SIZE,
                  "readOnlyPortal has wrong size.");
}

} // annonymous namespace

int UnitTestArrayContainerControlPermutation(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestPermutationArray);
}
