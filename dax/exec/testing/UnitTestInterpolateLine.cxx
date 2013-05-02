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
#include <dax/exec/Interpolate.h>
#include <dax/Types.h>

#include <dax/testing/Testing.h>

namespace {

template<typename T,typename W>
T lerp (const T &a, const T &b, const W &w)
{
  return a + w *(b -a);
}

template <typename T> void TestInterpolateLine();

template <> void TestInterpolateLine <dax::Id> ()
{
}

template <> void TestInterpolateLine<dax::Id3>()
{
}

template <> void TestInterpolateLine<dax::Scalar>()
{
  dax::Scalar a = 4;
  dax::Scalar b = 2;
  dax::Scalar w = 0.5;
  dax::Scalar lhs = lerp(a,b,w);
  dax::Scalar rhs = dax::exec::InterpolateLine(a,b,w);
  if(lhs != rhs)
    {
    DAX_TEST_FAIL("Sclars do not InterpolateLine() correctly");
    }
}

template <> void TestInterpolateLine<dax::Vector2>()
{
  dax::Vector2 a = dax::make_Vector2(2, 4);
  dax::Vector2 b = dax::make_Vector2(1, 2);

  dax::Scalar w = 0.5;
  dax::Vector2 lhs = lerp(a,b,w);
  dax::Vector2 rhs = dax::exec::InterpolateLine(a,b,w);
  if(lhs != rhs)
    {
    DAX_TEST_FAIL("Vectors with Scalar weight do not InterpolateLine() correctly");
    }

  dax::Vector2 w1 = dax::make_Vector2(0.5,0.5);
  dax::Vector2 lhs1 = lerp(a,b,w1);
  dax::Vector2 rhs1 = dax::exec::InterpolateLine(a,b,w1);
  if(lhs1 != rhs1)
    {
    DAX_TEST_FAIL("Vectors with Vector weight do not InterpolateLine() correctly");
    }

  if(rhs != rhs1)
    {
    DAX_TEST_FAIL("Both vector outputs return different output");
    }
}

template <> void TestInterpolateLine<dax::Vector3>()
{
  dax::Vector3 a = dax::make_Vector3(2, 4, 6);
  dax::Vector3 b = dax::make_Vector3(1, 2, 3);

  dax::Scalar w = 0.5;
  dax::Vector3 lhs = lerp(a,b,w);
  dax::Vector3 rhs = dax::exec::InterpolateLine(a,b,w);
  if(lhs != rhs)
    {
    DAX_TEST_FAIL("Vectors with Scalar weight do not InterpolateLine() correctly");
    }

  dax::Vector3 w1 = dax::make_Vector3(0.5,0.5,0.5);
  dax::Vector3 lhs1 = lerp(a,b,w1);
  dax::Vector3 rhs1 = dax::exec::InterpolateLine(a,b,w1);
  if(lhs1 != rhs1)
    {
    DAX_TEST_FAIL("Vectors with Vector weight do not InterpolateLine() correctly");
    }

  if(rhs != rhs1)
    {
    DAX_TEST_FAIL("Both vector outputs return different output");
    }
}

template <> void TestInterpolateLine<dax::Vector4>()
{
  dax::Vector4 a = dax::make_Vector4(2, 4, 6, 8);
  dax::Vector4 b = dax::make_Vector4(1, 2, 3, 4);

  dax::Scalar w = 0.5;
  dax::Vector4 lhs = lerp(a,b,w);
  dax::Vector4 rhs = dax::exec::InterpolateLine(a,b,w);
  if(lhs != rhs)
    {
    DAX_TEST_FAIL("Vectors with Scalar weight do not InterpolateLine() correctly");
    }

  dax::Vector4 w1 = dax::make_Vector4(0.5,0.5,0.5,0.5);
  dax::Vector4 lhs1 = lerp(a,b,w1);
  dax::Vector4 rhs1 = dax::exec::InterpolateLine(a,b,w1);
  if(lhs1 != rhs1)
    {
    DAX_TEST_FAIL("Vectors with Vector weight do not InterpolateLine() correctly");
    }

  if(rhs != rhs1)
    {
    DAX_TEST_FAIL("Both vector outputs return different output");
    }
}

struct TestInterpolateLineFunctor
{
  template <typename T> void operator()(const T&) const{
    TestInterpolateLine<T>();
  }
};

void TestAllInterpolateLine()
{
  dax::testing::Testing::TryAllTypes(TestInterpolateLineFunctor());
}

} // Anonymous namespace

int UnitTestInterpolateLine(int, char *[])
{
  return dax::testing::Testing::Run(TestAllInterpolateLine);
}
