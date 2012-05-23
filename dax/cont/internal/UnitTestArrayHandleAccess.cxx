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

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/DeviceAdapterSerial.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/ArrayHandleAccess.h>

#include <dax/cont/internal/Testing.h>

namespace {

void TestConstAccess(const dax::cont::ArrayHandle<dax::Scalar> &array)
{
  const dax::cont::internal::ArrayManagerExecutionShareWithControl
      <dax::Scalar, dax::cont::ArrayContainerControlTagBasic> &manager =
      dax::cont::internal::ArrayHandleAccess::GetArrayManagerExecution(array);

  // This manager's iterators should be the same as the control iterators.
  DAX_TEST_ASSERT(array.GetIteratorConstControlBegin()
                  == manager.GetIteratorConstBegin(),
                  "Array and manager iterators do not agree.");
}

void TestAccess()
{
  dax::cont::ArrayHandle<dax::Scalar> array;
  array.PrepareForOutput(10);

  dax::cont::internal::ArrayManagerExecutionShareWithControl
      <dax::Scalar, dax::cont::ArrayContainerControlTagBasic> &manager =
      dax::cont::internal::ArrayHandleAccess::GetArrayManagerExecution(array);

  // This manager's iterators should be the same as the control iterators.
  DAX_TEST_ASSERT(array.GetIteratorControlBegin() == manager.GetIteratorBegin(),
                  "Array and manager iterators do not agree.");
}

void TestFail()
{
  DAX_TEST_FAIL("I expect this error.");
}

void BadTestAssert()
{
  DAX_TEST_ASSERT(0 == 1, "I expect this error.");
}

void BadAssert()
{
  DAX_ASSERT_CONT(0 == 1);
}

void GoodAssert()
{
  DAX_TEST_ASSERT(1 == 1, "Always true.");
  DAX_ASSERT_CONT(1 == 1);
}

} // anonymous namespace

int UnitTestArrayHandleAccess(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestAccess);
}
