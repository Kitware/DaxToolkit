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

// This teases out a bug where math functions in the std namespace conflict
// with math functions in other namespaces.
using namespace std;

#include <dax/math/Sign.h>

#include <dax/internal/testing/Testing.h>

namespace {

template<typename VectorType>
void TestAbs(VectorType negative, VectorType positive)
{
  std::cout << "  Testing absolute value "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  DAX_TEST_ASSERT(test_equal(dax::math::Abs(negative), positive),
                  "Absolute value not flipping negative.");
  DAX_TEST_ASSERT(test_equal(dax::math::Abs(positive), positive),
                  "Absolute value not leaving positive.");
}

void TestIsNegative(dax::Scalar negative, dax::Scalar positive)
{
  std::cout << "Testing IsNegative" << std::endl;

  DAX_TEST_ASSERT(dax::math::IsNegative(negative),
                  "Did not detect negative.");
  DAX_TEST_ASSERT(!dax::math::IsNegative(positive),
                  "Did not detect positive.");
  DAX_TEST_ASSERT(!dax::math::IsNegative(dax::Scalar(0.0)),
                  "Did not detect non-negative.");
}

void TestCopySign(dax::Scalar negative, dax::Scalar positive)
{
  std::cout << "Testing CopySign" << std::endl;

  DAX_TEST_ASSERT(test_equal(-negative,
                             dax::math::CopySign(negative, positive)),
                  "Did not copy positive sign.");
  DAX_TEST_ASSERT(test_equal(-positive,
                             dax::math::CopySign(positive, negative)),
                  "Did not copy negative sign.");
  DAX_TEST_ASSERT(test_equal(-dax::math::Abs(negative),
                             dax::math::CopySign(negative, -1)),
                  "Did not copy negative sign.");
  DAX_TEST_ASSERT(test_equal(-dax::math::Abs(positive),
                             dax::math::CopySign(positive, -1)),
                  "Did not copy negative sign.");
  DAX_TEST_ASSERT(test_equal(dax::math::Abs(negative),
                             dax::math::CopySign(negative, 1)),
                  "Did not copy positive sign.");
  DAX_TEST_ASSERT(test_equal(dax::math::Abs(positive),
                             dax::math::CopySign(positive, 1)),
                  "Did not copy positive sign.");
  DAX_TEST_ASSERT(test_equal(dax::Scalar(-1),
                             dax::math::CopySign(1, negative)),
                  "Did not copy negative sign.");
  DAX_TEST_ASSERT(test_equal(dax::Scalar(1),
                             dax::math::CopySign(1, positive)),
                  "Did not copy positive sign.");
}

void AbsInput(dax::Scalar &negative, dax::Scalar &positive, int component) {
  DAX_TEST_ASSERT(component < 4, "Need to update test for bigger vectors.");
  negative = dax::make_Vector4(0.0, -0.5, -3241.12, -4e12)[component];
  positive = dax::make_Vector4(0.0,  0.5,  3241.12,  4e12)[component];
}

void AbsInput(dax::Id &negative, dax::Id &positive, int component) {
  DAX_TEST_ASSERT(component < 3, "Need to update test for bigger vectors.");
  negative = dax::make_Id3(0, -23, -652)[component];
  positive = dax::make_Id3(0,  23,  652)[component];
}

struct TestSignFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    T negative, positive;
    for (int index = 0; index < Traits::NUM_COMPONENTS; index++)
      {
      AbsInput(Traits::GetComponent(negative, index),
               Traits::GetComponent(positive, index),
               index);
      }
    TestAbs(negative, positive);
  }
};

void TestSign()
{
  dax::internal::Testing::TryAllTypes(TestSignFunctor());
  TestIsNegative(-2.3, 4.5);
  TestCopySign(-2.3, 4.5);
}

} // anonymous namespace

int UnitTestMathSign(int, char *[])
{
  return dax::internal::Testing::Run(TestSign);
}
