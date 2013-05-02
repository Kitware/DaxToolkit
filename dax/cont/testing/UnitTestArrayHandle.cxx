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

#include <dax/cont/ArrayHandle.h>

#include <dax/cont/testing/Testing.h>

#include <algorithm>

namespace
{
const dax::Id ARRAY_SIZE = 10;

dax::Scalar TestValue(dax::Id index)
{
  return static_cast<dax::Scalar>(10.0*index + 0.01*(index+1));
}

template<class IteratorType>
bool CheckValues(IteratorType begin, IteratorType end)
{
  dax::Id index = 0;
  for (IteratorType iter = begin; iter != end; iter++)
    {
    if (!test_equal(*iter, TestValue(index))) return false;
    index++;
    }
  return true;
}

template<typename T>
bool CheckValues(const dax::cont::ArrayHandle<T> &handle)
{
  return CheckValues(handle.GetPortalConstControl().GetIteratorBegin(),
                     handle.GetPortalConstControl().GetIteratorEnd());
}

void TestArrayHandle()
{
  std::cout << "Create array handle." << std::endl;
  dax::Scalar array[ARRAY_SIZE];
  dax::Scalar arrayCopy[ARRAY_SIZE];
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    array[index] = TestValue(index);
    }

  dax::cont::ArrayHandle<dax::Scalar>::PortalControl arrayPortal(
        &array[0], &array[ARRAY_SIZE]);

  dax::cont::ArrayHandle<dax::Scalar>
      arrayHandle(arrayPortal);

  DAX_TEST_ASSERT(arrayHandle.GetNumberOfValues() == ARRAY_SIZE,
                  "ArrayHandle has wrong number of entries.");

  std::cout << "Check basic array." << std::endl;
  DAX_TEST_ASSERT(CheckValues(arrayHandle),
                  "Array values not set correctly.");
  std::fill_n(arrayCopy, ARRAY_SIZE, 0);
  arrayHandle.CopyInto(arrayCopy);
  DAX_TEST_ASSERT(CheckValues(arrayCopy, arrayCopy+ARRAY_SIZE),
                  "Could not copy values back into array.");

  std::cout << "Check out execution array behavior." << std::endl;
  {
  dax::cont::ArrayHandle<dax::Scalar>::PortalConstExecution executionPortal
      = arrayHandle.PrepareForInput();
  DAX_TEST_ASSERT(CheckValues(executionPortal.GetIteratorBegin(),
                              executionPortal.GetIteratorEnd()),
                  "Array not copied to execution correctly.");
  }

  {
  bool gotException = false;
  try
    {
    arrayHandle.PrepareForInPlace();
    }
  catch (dax::cont::Error &error)
    {
    std::cout << "Got expected error: " << error.GetMessage() << std::endl;
    gotException = true;
    }
  DAX_TEST_ASSERT(gotException,
                  "PrepareForInPlace did not fail for const array.");
  }

  {
  dax::cont::ArrayHandle<dax::Scalar>::PortalExecution executionPortal
      = arrayHandle.PrepareForOutput(ARRAY_SIZE*2);
  dax::Id index = 0;
  for (dax::Scalar *iter = executionPortal.GetIteratorBegin();
       iter != executionPortal.GetIteratorEnd();
       iter++)
    {
    *iter = TestValue(index);
    index++;
    }
  }
  DAX_TEST_ASSERT(arrayHandle.GetNumberOfValues() == ARRAY_SIZE*2,
                  "Array not allocated correctly.");
  DAX_TEST_ASSERT(CheckValues(arrayHandle),
                  "Array values not retrieved from execution.");

  std::cout << "Try shrinking the array." << std::endl;
  arrayHandle.Shrink(ARRAY_SIZE);
  DAX_TEST_ASSERT(arrayHandle.GetNumberOfValues() == ARRAY_SIZE,
                  "Array size did not shrink correctly.");
  DAX_TEST_ASSERT(CheckValues(arrayHandle),
                  "Array values not retrieved from execution.");

  std::cout << "Try in place operation." << std::endl;
  {
  dax::cont::ArrayHandle<dax::Scalar>::PortalExecution executionPortal
      = arrayHandle.PrepareForInPlace();
  for (dax::Scalar *iter = executionPortal.GetIteratorBegin();
       iter != executionPortal.GetIteratorEnd();
       iter++)
    {
    *iter += 1;
    }
  }
  arrayHandle.CopyInto(array);
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    DAX_TEST_ASSERT(test_equal(array[index], TestValue(index) + 1),
                    "Did not get result from in place operation.");
    }
}

}


int UnitTestArrayHandle(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestArrayHandle);
}
