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

#include <dax/exec/math/LinearAlgebra.h>

#include <dax/internal/Testing.h>

namespace {

namespace internal{

template<typename VectorType>
VectorType::ValueType norm(VectorType vt)
  {
  return 0;
  }

}

template<typename VectorType>
void TestVector(VectorType vector)
{
  //to do have to implement a norm and normalized call to verify the math ones
  //agianst
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Norm(vector), norm),
                  "Norm on zero length vector failed test.");

  DAX_TEST_ASSERT(test_equal(dax::exec::math::Normalized(vector), normalized),
                  "Normalized vector failed test.");

  dax::exec::math::Normalize(vector);
  DAX_TEST_ASSERT(test_equal(vector, normalized),
                  "Inplace Normalized vector failed test.");
}

const dax::Id MAX_VECTOR_SIZE = 4;

struct LinearAlgInitStruct {
  dax::Scalar zero;
  dax::Scalar normalized;
}  LinearAlgInit[MAX_VECTOR_SIZE] = {
  { 0, 1 },
  { 0, 1 },
  { 0, 1 },
  { 0, 1 },
};

struct TestLinearFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    DAX_TEST_ASSERT(Traits::NUM_COMPONENTS <= MAX_VECTOR_SIZE,
                    "Need to update test for larger vectors.");
    T zeroVector,normalizedVector;
    for (int index = 0; index < Traits::NUM_COMPONENTS; index++)
      {
      Traits::SetComponent(zeroVector, index, LinearAlgInit[index].zero);
      Traits::SetComponent(normalizedVector, index, LinearAlgInit[index].normalized);
      }
    TestVector(zeroVector);
    TestVector(normalizedVector);
  }
};

void TestLinearAlgebra()
{
  dax::internal::Testing::TryAllTypes(TestLinearFunctor(),
                                      dax::internal::Testing::TypeCheckRealVector());
}

} // anonymous namespace

int UnitTestMathLinearAlgebra(int, char *[])
{
  return dax::internal::Testing::Run(TestLinearAlgebra);
}
