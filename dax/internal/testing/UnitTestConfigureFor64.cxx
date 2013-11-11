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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#include <dax/internal/ConfigureFor64.h>

#include <dax/Types.h>

#include <dax/testing/Testing.h>

// Size of 64 bits.
#define EXPECTED_SIZE 8

#if DAX_SIZE_ID != EXPECTED_SIZE
#error DAX_SIZE_ID an unexpected size.
#endif

#if DAX_SIZE_SCALAR != EXPECTED_SIZE
#error DAX_SIZE_SCALAR an unexpected size.
#endif

namespace {

void TestTypeSizes()
{
  DAX_TEST_ASSERT(DAX_SIZE_ID == EXPECTED_SIZE,
                  "DAX_SIZE_ID an unexpected size.");
  DAX_TEST_ASSERT(sizeof(dax::Id) == EXPECTED_SIZE,
                  "dax::Id an unexpected size.");
  DAX_TEST_ASSERT(DAX_SIZE_SCALAR == EXPECTED_SIZE,
                  "DAX_SIZE_SCALAR an unexpected size.");
  DAX_TEST_ASSERT(sizeof(dax::Scalar) == EXPECTED_SIZE,
                  "dax::Scalar an unexpected size.");
}

}

int UnitTestConfigureFor64(int, char *[])
{
  return dax::testing::Testing::Run(TestTypeSizes);
}
