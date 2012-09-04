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

#include <functional>
#include <dax/math/Compare.h>
#include <dax/internal/testing/Testing.h>

namespace {

template<typename VectorType>
void TestMinMax(VectorType x, VectorType y)
{
  typedef dax::VectorTraits<VectorType> Traits;
  typedef typename Traits::ComponentType ComponentType;
  const dax::Id NUM_COMPONENTS = Traits::NUM_COMPONENTS;

  std::cout << "  Testing Min and Max: " << NUM_COMPONENTS << " components"
            << std::endl;

  VectorType min = dax::math::Min(x, y);
  VectorType max = dax::math::Max(x, y);

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

template<typename VectorType, typename Functor>
bool VerifyFunctor(VectorType x, VectorType y, Functor f)
{
  typedef dax::VectorTraits<VectorType> Traits;
  typedef typename Traits::ComponentType ComponentType;
  const dax::Id NUM_COMPONENTS = Traits::NUM_COMPONENTS;
  bool functorResult = true;
  for (dax::Id index = 0; index < NUM_COMPONENTS && functorResult; index++)
    {
    ComponentType x_value = Traits::GetComponent(x, index);
    ComponentType y_value = Traits::GetComponent(y, index);
    functorResult = f(x_value,y_value);
    }
  return functorResult;
}

template<typename VectorType, typename DaxCompFunctor, typename STLCompFunctor>
void TestCompFunctor(VectorType x, VectorType y,
                     DaxCompFunctor daxFunctor, STLCompFunctor stdFunctor, const char* name)
{
  typedef dax::VectorTraits<VectorType> Traits;
  typedef typename Traits::ComponentType ComponentType;

  //Test vector comparisons
  bool equal = daxFunctor(x, y);
  bool verified = VerifyFunctor(x,y,stdFunctor);
  DAX_TEST_ASSERT(equal == verified, name);

  //Test scalar comparisons for each component
  //this will test the specialization of the scalar dimension tag for comparison
  const dax::Id NUM_COMPONENTS = Traits::NUM_COMPONENTS;
  for (dax::Id index = 0; index < NUM_COMPONENTS; index++)
    {
    ComponentType x_value = Traits::GetComponent(x, index);
    ComponentType y_value = Traits::GetComponent(y, index);
    equal = daxFunctor(x_value, y_value);
    verified = stdFunctor(x_value,y_value);
    DAX_TEST_ASSERT(equal == verified, name);
    }
}


static const dax::Id MAX_VECTOR_SIZE = 4;
static const dax::Scalar VectorInitX[MAX_VECTOR_SIZE] = { -4, -1, 2, 0.0 };
static const dax::Scalar VectorInitY[MAX_VECTOR_SIZE] = { 7, -6, 5, -0.001 };

struct TestCompareFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    typedef typename Traits::ComponentType ComponentType;

    DAX_TEST_ASSERT(Traits::NUM_COMPONENTS <= MAX_VECTOR_SIZE,
                    "Need to update test for larger vectors.");
    T x, y;
    for (int index = 0; index < Traits::NUM_COMPONENTS; index++)
      {
      Traits::SetComponent(x, index, VectorInitX[index]);
      Traits::SetComponent(y, index, VectorInitY[index]);
      }
    TestMinMax(x, y);

    TestCompFunctor(x,y, dax::math::Equal(),
                    std::equal_to<ComponentType>(), "Equal");
    TestCompFunctor(y,y, dax::math::Equal(),
                    std::equal_to<ComponentType>(),  "Equal");

    TestCompFunctor(x,y, dax::math::NotEqual(),
                    std::not_equal_to<ComponentType>(),"Not Equal");
    TestCompFunctor(x,x, dax::math::NotEqual(),
                    std::not_equal_to<ComponentType>(),"Not Equal");

    TestCompFunctor(x,y, dax::math::Less(),
                    std::less<ComponentType>(),"Less");
    TestCompFunctor(y,x, dax::math::Less(),
                    std::less<ComponentType>(),"Less");
    TestCompFunctor(y,y, dax::math::Less(),
                    std::less<ComponentType>(),"Less");

    TestCompFunctor(y,x, dax::math::LessEqual(),
                    std::less_equal<ComponentType>(),"Less than or Equal");
    TestCompFunctor(x,y, dax::math::LessEqual(),
                    std::less_equal<ComponentType>(),"Less than or Equal");
    TestCompFunctor(x,x, dax::math::LessEqual(),
                    std::less_equal<ComponentType>(),"Less than or Equal");

    TestCompFunctor(y,x,dax::math::Greater(),
                    std::greater<ComponentType>(),"Greater");
    TestCompFunctor(y,x,dax::math::Greater(),
                    std::greater<ComponentType>(),"Greater");
    TestCompFunctor(x,x,dax::math::Greater(),
                    std::greater<ComponentType>(),"Greater");

    TestCompFunctor(x,y,dax::math::GreaterEqual(),
                     std::greater_equal<ComponentType>(),"Greater than or Equal");
    TestCompFunctor(y,x,dax::math::GreaterEqual(),
                     std::greater_equal<ComponentType>(),"Greater than or Equal");
    TestCompFunctor(y,y,dax::math::GreaterEqual(),
                     std::greater_equal<ComponentType>(),"Greater than or Equal");
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

