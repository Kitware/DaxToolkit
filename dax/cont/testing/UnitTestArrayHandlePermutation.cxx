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

//This sets up the ArrayHandle semantics to allocate pointers and share memory
//between control and execution.
#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL


#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/ArrayHandlePermutation.h>
#include <dax/cont/ArrayPortalFromIterators.h>
#include <dax/cont/ArrayHandle.h>

#include <dax/cont/internal/testing/Testing.h>

namespace {

const dax::Id ARRAY_SIZE = 10;
const dax::Id VALUE_ARRAY_SIZE = 3;
void TestPermutationArray()
{
  // Make Key array
  dax::Id arrayKey[ARRAY_SIZE];
  for (dax::Id i=0; i<ARRAY_SIZE;++i)
    {
    arrayKey[i] = i % VALUE_ARRAY_SIZE;
    }

  // Make Value array
  dax::Id arrayValue[VALUE_ARRAY_SIZE];

  // Make KeyPortal
  typedef ::dax::cont::ArrayPortalFromIterators < dax::Id* > KeyPortalType;
  KeyPortalType keyPortal(arrayKey,
                          arrayKey + ARRAY_SIZE);

  // Make readWritePermutationArray (using constructor) from Key Portal and
  // ValuePortal
  typedef KeyPortalType ReadWriteValuePortalType;
  ReadWriteValuePortalType readWriteValuePortal(arrayValue,
                                                arrayValue + VALUE_ARRAY_SIZE);
  dax::cont::ArrayHandlePermutation <KeyPortalType,ReadWriteValuePortalType>
      readWriteArray(keyPortal, readWriteValuePortal);




  typedef dax::cont::ArrayHandle<dax::Id> StandardKeyHandle;
  StandardKeyHandle keyHandle = dax::cont::make_ArrayHandle(arrayKey,ARRAY_SIZE);

  // Make readOnlyPermutationArray (using make_ArrayHandlePermutation) from
  // keyPortal and valueArray
  typedef dax::cont::ArrayHandleCounting ReadOnlyArrayType;
  ReadOnlyArrayType readOnlyValueArray(VALUE_ARRAY_SIZE);

  typedef dax::cont::ArrayHandlePermutation <
                                    StandardKeyHandle::PortalConstControl,
                                    ReadOnlyArrayType::PortalConstControl> ReadPermType;

  ReadPermType readOnlyArray =
                dax::cont::make_ArrayHandlePermutation(keyHandle,readOnlyValueArray);

  // copy readOnlyArray to readWriteArray (i.e readWriteArray = readOnlyArray)
  for (dax::Id index=0; index < ARRAY_SIZE; index++)
    {
    readWriteArray.GetPortalConstControl().
      Set (index, readOnlyArray.GetPortalConstControl ().Get(index));
    }

  DAX_TEST_ASSERT(readWriteArray.GetNumberOfValues() == ARRAY_SIZE,
                  "ReadWriteArray has wrong size.");

  DAX_TEST_ASSERT(readOnlyArray.GetNumberOfValues() == ARRAY_SIZE,
                  "ReadOnlyArray has wrong size.");

  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    DAX_TEST_ASSERT(readOnlyArray.GetPortalConstControl().Get(index)
                    ==  readOnlyValueArray.GetPortalConstControl ().Get(index%VALUE_ARRAY_SIZE),
                    "ReadOnlyArray has unexpected value.");
    DAX_TEST_ASSERT(readWriteArray.GetPortalConstControl().Get(index)
                    ==  readOnlyValueArray.GetPortalConstControl ().Get(index%VALUE_ARRAY_SIZE),
                    "ReadWriteArray has unexpected value.");
    }
}

} // annonymous namespace

int UnitTestArrayHandlePermutation(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestPermutationArray);
}
