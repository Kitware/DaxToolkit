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

#include <dax/exec/internal/ErrorMessageBuffer.h>

#include <dax/internal/Testing.h>

namespace {

void TestErrorMessageBuffer()
{
  char messageBuffer[100];

  std::cout << "Testing buffer large enough for message." << std::endl;
  messageBuffer[0] = '\0';
  dax::exec::internal::ErrorMessageBuffer largeBuffer(messageBuffer, 100);
  DAX_TEST_ASSERT(!largeBuffer.IsErrorRaised(), "Message created with error.");

  largeBuffer.RaiseError("Hello World");
  DAX_TEST_ASSERT(largeBuffer.IsErrorRaised(), "Error not reported.");
  DAX_TEST_ASSERT(strcmp(messageBuffer, "Hello World") == 0,
                  "Did not record error message.");

  std::cout << "Testing truncated error message." << std::endl;
  messageBuffer[0] = '\0';
  dax::exec::internal::ErrorMessageBuffer smallBuffer(messageBuffer, 9);
  DAX_TEST_ASSERT(!smallBuffer.IsErrorRaised(), "Message created with error.");

  smallBuffer.RaiseError("Hello World");
  DAX_TEST_ASSERT(smallBuffer.IsErrorRaised(), "Error not reported.");
  DAX_TEST_ASSERT(strcmp(messageBuffer, "Hello Wo") == 0,
                  "Did not record error message.");
}

} // anonymous namespace

int UnitTestErrorMessageBuffer(int, char *[])
{
  return (dax::internal::Testing::Run(TestErrorMessageBuffer));
}
