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

#include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/testing/Testing.h>

namespace {

/// Simple functions to be used in conjunction with the vector operations.
///
template <class T>
static T Square(T x) { return x*x; }
template <class T>
static T Add(T x, T y) { return x + y; }

template<class T>
struct Summation {
  T Sum;
  Summation() : Sum(0) { }
  void operator()(T x) { Sum += x; }
};

/// Compares operations through generic vector operations with some other
/// overloaded operations that should be tested separately in UnitTestTypes.
///
template <class VectorType>
static void TestVectorType(const VectorType &value)
{
  typedef typename dax::VectorTraits<VectorType> Traits;
  typedef typename Traits::ComponentType ComponentType;

  VectorType squaredVector = dax::cont::VectorMap(value, Square<ComponentType>);
  DAX_TEST_ASSERT(squaredVector == value*value,
                  "Got bad result for squaring vector components");

  ComponentType magSquared
      = dax::cont::VectorReduce(squaredVector, Add<ComponentType>);
  DAX_TEST_ASSERT(magSquared == dax::dot(value, value),
                  "Got bad result for summing vector components");

  {
  Summation<ComponentType> sum;
  dax::cont::VectorForEach(squaredVector, sum);
  DAX_TEST_ASSERT(sum.Sum == magSquared,
                  "Got bad result for summing with VectorForEach");
  }

  {
  // Repeat the last test with a const reference.
  Summation<ComponentType> sum;
  const VectorType &constSquaredVector = squaredVector;
  dax::cont::VectorForEach(constSquaredVector, sum);
  DAX_TEST_ASSERT(sum.Sum == magSquared,
                  "Got bad result for summing with VectorForEach");
  }

  {
  // Test the VectorFill function.
  ComponentType fillValue = Traits::GetComponent(value, 0);
  VectorType fillVector;
  dax::cont::VectorFill(fillVector, fillValue);
  Summation<ComponentType> sum;
  dax::cont::VectorForEach(fillVector, sum);
  DAX_TEST_ASSERT(sum.Sum == fillValue * Traits::NUM_COMPONENTS,
                  "Got bad result filling vector");
  }

  {
  // Test another form of the VectorFill function.
  ComponentType fillValue = Traits::GetComponent(value, 0);
  VectorType fillVector = dax::cont::VectorFill<VectorType>(fillValue);
  Summation<ComponentType> sum;
  dax::cont::VectorForEach(fillVector, sum);
  DAX_TEST_ASSERT(sum.Sum == fillValue * Traits::NUM_COMPONENTS,
                  "Got bad result filling vector");
  }
}

static const dax::Id MAX_VECTOR_SIZE = 4;
static const dax::Id VectorInit[MAX_VECTOR_SIZE] = { 42, 54, 67, 12 };

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
    TestVectorType(vector);
  }
};

static void TestVectorTypes()
{
  dax::internal::Testing::TryAllTypes(TestVectorTypeFunctor());
}

} // anonymous namespace

int UnitTestVectorOperations(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestVectorTypes);
}
