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

#include <dax/cont/internal/ArrayContainerControlCounting.h>
#include <dax/cont/internal/ArrayContainerControlPermutation.h>
#include <dax/cont/ArrayContainerControlImplicit.h>
#include <dax/cont/internal/ArrayPortalFromIterators.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/cont/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;
const dax::Id VALUE_ARRAY_SIZE = 3;
void TestPermutationArray()
{
  typedef ::dax::cont::internal::ArrayPortalFromIterators < dax::Id* > IdArrayPortal;
  dax::Id arrayKey[ARRAY_SIZE];
  for (dax::Id i=0; i<ARRAY_SIZE;++i)
    {
    arrayKey[i] = i % VALUE_ARRAY_SIZE;
    }
  dax::Id arrayValue[VALUE_ARRAY_SIZE];

  typedef IdArrayPortal KeyPortalType;
  typedef IdArrayPortal ReadWriteValuePortalType;
  typedef dax::cont::internal::ArrayPortalCounting<dax::Id> ReadOnlyValuePortalType;


  // Make  readWrite array from portals
  KeyPortalType keyPortal(arrayKey,arrayKey + ARRAY_SIZE);
  ReadWriteValuePortalType readWriteValuePortal(arrayValue,arrayValue + VALUE_ARRAY_SIZE);
  dax::cont::internal::ArrayPortalPermutation <KeyPortalType,ReadWriteValuePortalType>
      readWritePortal(keyPortal, readWriteValuePortal);

  dax::cont::ArrayHandle<dax::Id,
                         dax::cont::internal::ArrayContainerControlTagPermutation
        <KeyPortalType,ReadWriteValuePortalType> >
      readWriteArray(readWritePortal);

  // Make readOnly array from portals
  ReadOnlyValuePortalType readOnlyValuePortaly(0,VALUE_ARRAY_SIZE);
  dax::cont::internal::ArrayPortalPermutation <KeyPortalType,ReadOnlyValuePortalType>
      readOnlyPortal(keyPortal, readOnlyValuePortaly);

  dax::cont::ArrayHandle<dax::Id,
                         dax::cont::internal::ArrayContainerControlTagPermutation
        <KeyPortalType,ReadOnlyValuePortalType> >
      readOnlyArray(readOnlyPortal);

  // copy readOnlyArray to readWriteArray (i.e readWriteArray = readOnlyArray)
  for (dax::Id index=0; index < ARRAY_SIZE; index++)
    {
    readWriteArray.GetPortalConstControl ().Set (index,
                                           readOnlyArray.GetPortalConstControl ().Get(index));
    }

  DAX_TEST_ASSERT(readWriteArray.GetNumberOfValues() == ARRAY_SIZE,
                  "ReadWriteArray has wrong size.");

  DAX_TEST_ASSERT(readOnlyArray.GetNumberOfValues() == ARRAY_SIZE,
                  "ReadOnlyArray has wrong size.");

  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    DAX_TEST_ASSERT(readOnlyArray.GetPortalConstControl().Get(index) ==  readOnlyValuePortaly.Get(index%VALUE_ARRAY_SIZE),
                    "ReadOnlyArray has unexpected value.");
    DAX_TEST_ASSERT(readWriteArray.GetPortalConstControl().Get(index) ==  readOnlyValuePortaly.Get(index%VALUE_ARRAY_SIZE),
                    "ReadWriteArray has unexpected value.");
    }
}

} // annonymous namespace

int UnitTestArrayContainerControlPermutation(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestPermutationArray);
}
