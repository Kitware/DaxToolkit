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

#include <dax/math/VectorAnalysis.h>

#include<dax/Types.h>
#include <dax/exec/VectorOperations.h>
#include <dax/internal/testing/Testing.h>

#include <math.h>

namespace {

namespace internal{

template<typename VectorType>
dax::Scalar mag(const VectorType& vt)
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
VectorType normal(const VectorType& vt)
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
  std::cout << "Testing " << vector << std::endl;

  //to do have to implement a norm and normalized call to verify the math ones
  //against
  dax::Scalar magnitude = dax::math::Magnitude(vector);
  dax::Scalar magnitudeCompare = internal::mag(vector);
  DAX_TEST_ASSERT(test_equal(magnitude, magnitudeCompare),
                  "Magnitude failed test.");

  dax::Scalar magnitudeSquared = dax::math::MagnitudeSquared(vector);
  DAX_TEST_ASSERT(test_equal(magnitude*magnitude, magnitudeSquared),
                  "Magnitude squared test failed.");

  if (magnitudeSquared > 0)
    {
    dax::Scalar rmagnitude = dax::math::RMagnitude(vector);
    DAX_TEST_ASSERT(test_equal(1/magnitude, rmagnitude),
                    "Reciprical magnitude failed.");

    DAX_TEST_ASSERT(test_equal(dax::math::Normal(vector),
                               internal::normal(vector)),
                    "Normalized vector failed test.");

    VectorType normalizedVector=vector;
    dax::math::Normalize(normalizedVector);
    DAX_TEST_ASSERT(test_equal(normalizedVector, internal::normal(vector)),
                    "Inplace Normalized vector failed test.");
    }
}

void TestCross(const dax::Vector3 &x, const dax::Vector3 &y)
{
  dax::Vector3 cross = dax::math::Cross(x, y);

  std::cout << "Testing " << x << " x " << y << " = " << cross << std::endl;

  // The cross product result should be perpendicular to input vectors.
  DAX_TEST_ASSERT(test_equal(dax::dot(cross,x),dax::Scalar(0.0)),
                  "Cross product not perpendicular.");
  DAX_TEST_ASSERT(test_equal(dax::dot(cross,y),dax::Scalar(0.0)),
                  "Cross product not perpendicular.");

  // The length of cross product should be the lengths of the input vectors
  // times the sin of the angle between them.
  dax::Scalar sinAngle =
      dax::math::Magnitude(cross)
      * dax::math::RMagnitude(x)
      * dax::math::RMagnitude(y);

  // The dot product is likewise the lengths of the input vectors times the
  // cos of the angle between them.
  dax::Scalar cosAngle =
      dax::dot(x,y)
      * dax::math::RMagnitude(x)
      * dax::math::RMagnitude(y);

  // Test that these are the actual sin and cos of the same angle with a
  // basic trigonometric identity.
  DAX_TEST_ASSERT(test_equal(sinAngle*sinAngle+cosAngle*cosAngle,
                             dax::Scalar(1.0)),
                  "Bad cross product length.");

  // Test finding the normal to a triangle (similar to cross product).
  dax::Vector3 normal =
      dax::math::TriangleNormal(x, y, dax::make_Vector3(0, 0, 0));
  DAX_TEST_ASSERT(test_equal(dax::dot(normal, x-y), dax::Scalar(0.0)),
                  "Triangle normal is not really normal.");
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
  TestCross(dax::make_Vector3(1.0,0.0,0.0), dax::make_Vector3(0.0,1.0,0.0));
  TestCross(dax::make_Vector3(1.0,2.0,3.0), dax::make_Vector3(-3.0,-1.0,1.0));
  TestCross(dax::make_Vector3(0.0,0.0,1.0), dax::make_Vector3(0.001,0.01,2.0));
}

} // anonymous namespace

int UnitTestMathVectorAnalysis(int, char *[])
{
  return dax::internal::Testing::Run(TestVectorAnalysis);
}
