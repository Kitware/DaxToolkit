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

#include <dax/testing/VectorTraitsTests.h>

#include <dax/testing/Testing.h>

namespace {

static const dax::Id MAX_VECTOR_SIZE = 5;
static const dax::Id VectorInit[MAX_VECTOR_SIZE] = { 42, 54, 67, 12, 78 };

struct TestVectorTypeFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    DAX_TEST_ASSERT(Traits::NUM_COMPONENTS <= MAX_VECTOR_SIZE,
                    "Need to update test for larger vectors.");
    T vector;
    for (int index = 0; index < Traits::NUM_COMPONENTS; index++)
      {
      Traits::SetComponent(vector, index, VectorInit[index]);
      }
    dax::testing::TestVectorType(vector);
  }
};

void TestVectorTraits()
{
  TestVectorTypeFunctor test;
  dax::testing::Testing::TryAllTypes(test);
  std::cout << "dax::Tuple<dax::Scalar, 5>" << std::endl;
  test(dax::Tuple<dax::Scalar,5>());

  dax::testing::TestVectorComponentsTag<dax::Id3>();
  dax::testing::TestVectorComponentsTag<dax::Vector3>();
  dax::testing::TestVectorComponentsTag<dax::Vector4>();
  dax::testing::TestScalarComponentsTag<dax::Id>();
  dax::testing::TestScalarComponentsTag<dax::Scalar>();
}

} // anonymous namespace

int UnitTestVectorTraits(int, char *[])
{
  return dax::testing::Testing::Run(TestVectorTraits);
}
