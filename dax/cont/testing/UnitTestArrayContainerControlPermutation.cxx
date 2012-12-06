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

#include <dax/cont/ArrayContainerControlCounting.h>
#include <dax/cont/ArrayContainerControlPermutation.h>
#include <dax/cont/ArrayContainerControlImplicit.h>
#include <dax/cont/ArrayPortalFromIterators.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/cont/internal/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;
const dax::Id VALUE_ARRAY_SIZE = 3;
void TestPermutationArray()
{
  typedef ::dax::cont::ArrayPortalFromIterators < dax::Id* > IdArrayPortal;
  dax::Id arrayKey[ARRAY_SIZE];
  for (dax::Id i=0; i<ARRAY_SIZE;++i)
    {
    arrayKey[i] = i % VALUE_ARRAY_SIZE;
    }

  typedef IdArrayPortal KeyPortalType;
  typedef dax::cont::ArrayPortalCounting ValuePortalType;

  KeyPortalType keyPortal(arrayKey,arrayKey + ARRAY_SIZE);
  ValuePortalType valuePortal(VALUE_ARRAY_SIZE);

  dax::cont::ArrayPortalPermutation <KeyPortalType,ValuePortalType>
      portal(keyPortal,valuePortal);

  dax::cont::ArrayHandle<dax::Id,
                         dax::cont::ArrayContainerControlTagPermutation
        <KeyPortalType,ValuePortalType> >
      array(portal);

  DAX_TEST_ASSERT(array.GetNumberOfValues() == ARRAY_SIZE,
                  "Array has wrong size.");

  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    DAX_TEST_ASSERT(array.GetPortalConstControl().Get(index) == valuePortal.Get(index%VALUE_ARRAY_SIZE),
                    "Array has unexpected value.");
    }
}

} // annonymous namespace

int UnitTestArrayContainerControlPermutation(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestPermutationArray);
}
