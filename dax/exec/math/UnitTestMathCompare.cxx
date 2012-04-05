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

#include <dax/exec/math/Compare.h>

#include <dax/internal/Testing.h>

namespace {

template<typename VectorType>
void TestMinMax(VectorType x, VectorType y)
{
  typedef dax::VectorTraits<VectorType> Traits;
  typedef typename Traits::ComponentType ComponentType;
  const dax::Id NUM_COMPONENTS = Traits::NUM_COMPONENTS;

  std::cout << "  Testing Min and Max: " << NUM_COMPONENTS << " components"
            << std::endl;

  VectorType min = dax::exec::math::Min(x, y);
  VectorType max = dax::exec::math::Max(x, y);

  for (dax::Id index = 0; index < NUM_COMPONENTS; index++)
    {
    ComponentType x_index = Traits::GetComponent(x, index);
    ComponentType y_index = Traits::GetComponent(y, index);
    ComponentType min_index = Traits::GetComponent(min, index);
    ComponentType max_index = Traits::GetComponent(max, index);
    if (x_index < y_index)
      {
      DAX_TEST_ASSERT(x_index == min_index, "Got wrong min.");
      DAX_TEST_ASSERT(y_index == max_index, "Got wrong max.");
      }
    else
      {
      DAX_TEST_ASSERT(x_index == max_index, "Got wrong max.");
      DAX_TEST_ASSERT(y_index == min_index, "Got wrong min.");
      }
    }
}

static const dax::Id MAX_VECTOR_SIZE = 4;
static const dax::Scalar VectorInitX[MAX_VECTOR_SIZE] = { -4, -1, 2, 0.0 };
static const dax::Scalar VectorInitY[MAX_VECTOR_SIZE] = { 7, -6, 5, -0.001 };

struct TestCompareFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    DAX_TEST_ASSERT(Traits::NUM_COMPONENTS <= MAX_VECTOR_SIZE,
                    "Need to update test for larger vectors.");
    T x, y;
    for (int index = 0; index < Traits::NUM_COMPONENTS; index++)
      {
      Traits::SetComponent(x, index, VectorInitX[index]);
      Traits::SetComponent(y, index, VectorInitY[index]);
      }
    TestMinMax(x, y);
  }
};

void TestCompare()
{
  dax::internal::Testing::TryAllTypes(TestCompareFunctor());
}

} // anonymous namespace

int UnitTestMathCompare(int, char *[])
{
  return dax::internal::Testing::Run(TestCompare);
}
