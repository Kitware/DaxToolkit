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

#include <dax/testing/Testing.h>

namespace {

template<typename OutT, typename InT>
void Copy(OutT &out, const InT &in)
{
  typedef dax::VectorTraits<OutT> OutTraits;
  typedef dax::VectorTraits<InT> InTraits;
  DAX_TEST_ASSERT(OutTraits::NUM_COMPONENTS <= InTraits::NUM_COMPONENTS,
                  "Need to update test for bigger vectors.");
  for (int index = 0; index < OutTraits::NUM_COMPONENTS; index++)
    {
    OutTraits::SetComponent(out, index, InTraits::GetComponent(in, index));
    }
}

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

void TestSignBit(dax::Scalar negative, dax::Scalar positive)
{
  std::cout << "Testing SignBit" << std::endl;

  DAX_TEST_ASSERT(dax::math::SignBit(negative) != 0,
                  "Did not detect negative SignBit.");
  DAX_TEST_ASSERT(dax::math::SignBit(positive) == 0,
                  "Did not detect positive SignBit.");
  DAX_TEST_ASSERT(dax::math::SignBit(dax::Scalar(0.0)) == 0,
                  "Did not detect zero SignBit.");
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

template<typename T>
void AbsInput(T &negative, T &positive, dax::Scalar)
{
  Copy(negative, dax::make_Vector4(0.0, -0.5, -3241.12, -4e12));
  Copy(positive, dax::make_Vector4(0.0,  0.5,  3241.12,  4e12));
}

template<typename T>
void AbsInput(T &negative, T &positive, dax::Id)
{
  Copy(negative, dax::make_Id3(0, -23, -652));
  Copy(positive, dax::make_Id3(0,  23,  652));
}

struct TestSignFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    T negative, positive;
    AbsInput(negative, positive, typename Traits::ComponentType());
    TestAbs(negative, positive);
  }
};

void TestSign()
{
  dax::testing::Testing::TryAllTypes(TestSignFunctor());
  TestIsNegative(-2.3, 4.5);
  TestSignBit(-2.3, 4.5);
  TestCopySign(-2.3, 4.5);
}

} // anonymous namespace

int UnitTestMathSign(int, char *[])
{
  return dax::testing::Testing::Run(TestSign);
}
