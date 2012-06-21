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

#include <dax/exec/math/VectorAnalysis.h>

#include<dax/Types.h>
#include <dax/exec/VectorOperations.h>
#include <dax/internal/Testing.h>

#include <math.h>

namespace {

namespace internal{

template<typename VectorType>
dax::Scalar norm(const VectorType& vt)
{
  typedef dax::VectorTraits<VectorType> Traits;
  double total = 0.0;
  for (int i = 0; i < Traits::NUM_COMPONENTS; ++i)
    {
    total += Traits::GetComponent(vt,i) * Traits::GetComponent(vt,i);
    }
  return sqrt(total);
}

template<typename VectorType>
VectorType normalized(const VectorType& vt)
{
  typedef dax::VectorTraits<VectorType> Traits;
  double total = 0.0;
  for (int i = 0; i < Traits::NUM_COMPONENTS; ++i)
    {
    total += Traits::GetComponent(vt,i) * Traits::GetComponent(vt,i);
    }
  VectorType temp = vt;
  if(total)
    {
    for (int i = 0; i < Traits::NUM_COMPONENTS; ++i)
      {
      typename Traits::ComponentType norm = Traits::GetComponent(vt,i)/sqrt(total);
      Traits::SetComponent(temp,i,norm);
      }
    }
  return temp;
}

}

template<typename VectorType>
void TestVector(const VectorType& vector)
{
  //to do have to implement a norm and normalized call to verify the math ones
  //agianst
  DAX_TEST_ASSERT(test_equal(dax::exec::math::Norm(vector), internal::norm(vector)),
                  "Norm on zero length vector failed test.");

  DAX_TEST_ASSERT(test_equal(dax::exec::math::Normalized(vector), internal::normalized(vector)),
                  "Normalized vector failed test.");

  VectorType normalizedVector=vector;
  dax::exec::math::Normalize(normalizedVector);
  DAX_TEST_ASSERT(test_equal(normalizedVector, internal::normalized(vector)),
                  "Inplace Normalized vector failed test.");
}

const dax::Id MAX_VECTOR_SIZE = 4;

struct VectorAnalInitStruct {
  dax::Scalar zero;
  dax::Scalar normalized;
  dax::Scalar positive;
  dax::Scalar negative;
}  VectorAnalInit[MAX_VECTOR_SIZE] = {
  { 0, 1, 0.13, -2 },
  { 0, 1, 8, -4 },
  { 0, 1, 3.0, -1 },
  { 0, 1, 4, -2 },
};

struct TestLinearFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    DAX_TEST_ASSERT(Traits::NUM_COMPONENTS <= MAX_VECTOR_SIZE,
                    "Need to update test for larger vectors.");
    T zeroVector,normalizedVector,posVec,negVec;
    for (int index = 0; index < Traits::NUM_COMPONENTS; index++)
      {
      Traits::SetComponent(zeroVector, index, VectorAnalInit[index].zero);
      Traits::SetComponent(normalizedVector, index, VectorAnalInit[index].normalized);
      Traits::SetComponent(posVec, index, VectorAnalInit[index].positive);
      Traits::SetComponent(negVec, index, VectorAnalInit[index].negative);
      }
    TestVector(zeroVector);
    TestVector(normalizedVector);
    TestVector(posVec);
    TestVector(negVec);
  }
};

void TestVectorAnalysis()
{
  dax::internal::Testing::TryAllTypes(TestLinearFunctor(),
                                      dax::internal::Testing::TypeCheckReal());
}

} // anonymous namespace

int UnitTestMathVectorAnalysis(int, char *[])
{
  return dax::internal::Testing::Run(TestVectorAnalysis);
}
