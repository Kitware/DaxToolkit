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

#include <dax/math/Trig.h>

#include <dax/exec/VectorOperations.h>
#include <dax/math/Exp.h>

#include <dax/testing/Testing.h>

namespace {

void TestPi()
{
  std::cout << "Testing Pi" << std::endl;
  DAX_TEST_ASSERT(test_equal(dax::math::Pi(), dax::Scalar(3.14159265)),
                  "Pi not correct.");
}

void TestArcTan2()
{
  std::cout << "Testing arc tan 2" << std::endl;

  DAX_TEST_ASSERT(test_equal(dax::math::ATan2(0.0, 1.0),
                             dax::Scalar(0.0)),
                  "ATan2 x+ axis.");
  DAX_TEST_ASSERT(test_equal(dax::math::ATan2(1.0, 0.0),
                             dax::Scalar(0.5*dax::math::Pi())),
                  "ATan2 y+ axis.");
  DAX_TEST_ASSERT(test_equal(dax::math::ATan2(-1.0, 0.0),
                             dax::Scalar(-0.5*dax::math::Pi())),
                  "ATan2 y- axis.");

  DAX_TEST_ASSERT(test_equal(dax::math::ATan2(1.0, 1.0),
                             dax::Scalar(0.25*dax::math::Pi())),
                  "ATan2 Quadrant 1");
  DAX_TEST_ASSERT(test_equal(dax::math::ATan2(1.0, -1.0),
                             dax::Scalar(0.75*dax::math::Pi())),
                  "ATan2 Quadrant 2");
  DAX_TEST_ASSERT(test_equal(dax::math::ATan2(-1.0, -1.0),
                             dax::Scalar(-0.75*dax::math::Pi())),
                  "ATan2 Quadrant 3");
  DAX_TEST_ASSERT(test_equal(dax::math::ATan2(-1.0, 1.0),
                             dax::Scalar(-0.25*dax::math::Pi())),
                  "ATan2 Quadrant 4");
}

template<typename VectorType>
void TestTriangle(VectorType angle,
                  VectorType opposite,
                  VectorType adjacent,
                  VectorType hypotenuse)
{
  std::cout << "  Testing triangle "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  DAX_TEST_ASSERT(test_equal(dax::math::Sin(angle), opposite/hypotenuse),
                  "Sin failed test.");
  DAX_TEST_ASSERT(test_equal(dax::math::Cos(angle), adjacent/hypotenuse),
                  "Cos failed test.");
  DAX_TEST_ASSERT(test_equal(dax::math::Tan(angle), opposite/adjacent),
                  "Tan failed test.");

  DAX_TEST_ASSERT(test_equal(dax::math::ASin(opposite/hypotenuse), angle),
                  "Arc Sin failed test.");
  DAX_TEST_ASSERT(test_equal(dax::math::ACos(adjacent/hypotenuse), angle),
                  "Arc Cos failed test.");
  DAX_TEST_ASSERT(test_equal(dax::math::ATan(opposite/adjacent), angle),
                  "Arc Tan failed test.");
}

template<typename VectorType>
void TestHyperbolic(VectorType x)
{
  std::cout << "  Testing hyperbolic "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  VectorType zero;
  dax::exec::VectorFill(zero, dax::Scalar(0.0));
  VectorType minusX = zero - x;

  using namespace dax::math;
  DAX_TEST_ASSERT(test_equal(SinH(x), 0.5f*(Exp(x) - Exp(minusX))),
                  "SinH does not match definition.");
  DAX_TEST_ASSERT(test_equal(CosH(x), 0.5f*(Exp(x) + Exp(minusX))),
                  "SinH does not match definition.");
  DAX_TEST_ASSERT(test_equal(TanH(x), SinH(x)/CosH(x)),
                  "TanH does not match definition");

  DAX_TEST_ASSERT(test_equal(ASinH(SinH(x)), x),
                  "SinH not inverting.");
  DAX_TEST_ASSERT(test_equal(ACosH(CosH(x)), x),
                  "CosH not inverting.");
  DAX_TEST_ASSERT(test_equal(ATanH(TanH(x)), x),
                  "TanH not inverting.");
}

const dax::Id MAX_VECTOR_SIZE = 4;

struct TriangleInitStruct {
  dax::Scalar Angle;
  dax::Scalar Opposite;
  dax::Scalar Adjacent;
  dax::Scalar Hypotenuse;
} TriangleInit[MAX_VECTOR_SIZE] = {
  { 0.643501108793284, 3.0, 4.0, 5.0 },
  { (1.0f/4.0f)*dax::math::Pi(), 1.0f, 1.0f, dax::math::Sqrt(2.0f) },
  { (1.0f/6.0f)*dax::math::Pi(), 1.0f, dax::math::Sqrt(3.0f), 2.0f },
  { (1.0f/3.0f)*dax::math::Pi(), dax::math::Sqrt(3.0f), 1.0f, 2.0f }
};

struct TestTrigFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    DAX_TEST_ASSERT(Traits::NUM_COMPONENTS <= MAX_VECTOR_SIZE,
                    "Need to update test for larger vectors.");
    T angle, opposite, adjacent, hypotenuse;
    for (int index = 0; index < Traits::NUM_COMPONENTS; index++)
      {
      Traits::SetComponent(angle, index, TriangleInit[index].Angle);
      Traits::SetComponent(opposite, index, TriangleInit[index].Opposite);
      Traits::SetComponent(adjacent, index, TriangleInit[index].Adjacent);
      Traits::SetComponent(hypotenuse, index, TriangleInit[index].Hypotenuse);
      }
    TestTriangle(angle, opposite, adjacent, hypotenuse);
    TestHyperbolic(angle);
  }
};

void TestTrig()
{
  TestPi();
  TestArcTan2();
  dax::testing::Testing::TryAllTypes(TestTrigFunctor(),
                                      dax::testing::Testing::TypeCheckReal());
}

} // anonymous namespace

int UnitTestMathTrig(int, char *[])
{
  return dax::testing::Testing::Run(TestTrig);
}
