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

#include <dax/Types.h>

#include <dax/testing/Testing.h>

namespace {

//general type test
template <typename T> void TypeTest()
{
  //grab the number of elements of T
  T a, b, c;
  typename T::ComponentType s(5);

  for(int i=0; i < T::NUM_COMPONENTS; ++i)
    {
    a[i]=typename T::ComponentType((i+1)*2);
    b[i]=typename T::ComponentType(i+1);
    c[i]=typename T::ComponentType((i+1)*2);
    }

  //verify prefix and postfix increment and decrement
  ++c[T::NUM_COMPONENTS-1];
  c[T::NUM_COMPONENTS-1]++;
  --c[T::NUM_COMPONENTS-1];
  c[T::NUM_COMPONENTS-1]--;

  //make c nearly like a to verify == and != are correct.
  c[T::NUM_COMPONENTS-1]=(c[T::NUM_COMPONENTS-1]-1);

  T plus = a + b;
  T correct_plus;
  for(int i=0; i < T::NUM_COMPONENTS; ++i)
    { correct_plus[i] = a[i] + b[i]; }
  DAX_TEST_ASSERT(test_equal(plus, correct_plus),"Tuples not added correctly.");

  T minus = a - b;
  T correct_minus;
  for(int i=0; i < T::NUM_COMPONENTS; ++i)
    { correct_minus[i] = a[i] - b[i]; }
  DAX_TEST_ASSERT(test_equal(minus, correct_minus),"Tuples not subtracted correctly.");


  T mult = a * b;
  T correct_mult;
  for(int i=0; i < T::NUM_COMPONENTS; ++i)
    { correct_mult[i] = a[i] * b[i]; }
  DAX_TEST_ASSERT(test_equal(mult, correct_mult),"Tuples not multiplied correctly.");

  T div = a / b;
  T correct_div;
  for(int i=0; i < T::NUM_COMPONENTS; ++i)
    { correct_div[i] = a[i] / b[i]; }
  DAX_TEST_ASSERT(test_equal(div,correct_div),"Tuples not divided correctly.");

  mult = s * a;
  for(int i=0; i < T::NUM_COMPONENTS; ++i)
    { correct_mult[i] = s * a[i]; }
  DAX_TEST_ASSERT(test_equal(mult, correct_mult),
                  "Scalar and Tuple did not multiply correctly.");

  mult = a * s;
  DAX_TEST_ASSERT(test_equal(mult, correct_mult),
                  "Tuple and Scalar to not multiply correctly.");

  typename T::ComponentType d = dax::dot(a, b);
  typename T::ComponentType correct_d = 0;
  for(int i=0; i < T::NUM_COMPONENTS; ++i)
    {correct_d += a[i] * b[i]; }
  DAX_TEST_ASSERT(test_equal(d, correct_d), "dot(Tuple) wrong");

  DAX_TEST_ASSERT(!(a == b), "operator== wrong");
  DAX_TEST_ASSERT((a == a),  "operator== wrong");

  DAX_TEST_ASSERT((a != b),  "operator!= wrong");
  DAX_TEST_ASSERT(!(a != a), "operator!= wrong");

  //test against a tuple that shares some values
  DAX_TEST_ASSERT( !(c == a), "operator == wrong");
  DAX_TEST_ASSERT( !(a == c), "operator == wrong");

  DAX_TEST_ASSERT( (c != a), "operator != wrong");
  DAX_TEST_ASSERT( (a != c), "operator != wrong");
}

template<> void TypeTest<dax::Vector2>()
{
  dax::Vector2 a = dax::make_Vector2(2, 4);
  dax::Vector2 b = dax::make_Vector2(1, 2);
  dax::Scalar s = 5;

  dax::Vector2 plus = a + b;
  DAX_TEST_ASSERT(test_equal(plus, dax::make_Vector2(3, 6)),
                  "Vectors do not add correctly.");

  dax::Vector2 minus = a - b;
  DAX_TEST_ASSERT(test_equal(minus, dax::make_Vector2(1, 2)),
                  "Vectors to not subtract correctly.");

  dax::Vector2 mult = a * b;
  DAX_TEST_ASSERT(test_equal(mult, dax::make_Vector2(2, 8)),
                  "Vectors to not multiply correctly.");

  dax::Vector2 div = a / b;
  DAX_TEST_ASSERT(test_equal(div, dax::make_Vector2(2, 2)),
                  "Vectors to not divide correctly.");

  mult = s * a;
  DAX_TEST_ASSERT(test_equal(mult, dax::make_Vector2(10, 20)),
                  "Vector and scalar to not multiply correctly.");

  mult = a * s;
  DAX_TEST_ASSERT(test_equal(mult, dax::make_Vector2(10, 20)),
                  "Vector and scalar to not multiply correctly.");

  dax::Scalar d = dax::dot(a, b);
  DAX_TEST_ASSERT(test_equal(d, dax::Scalar(10)), "dot(Vector2) wrong");

  DAX_TEST_ASSERT(!(a == b), "operator== wrong");
  DAX_TEST_ASSERT((a == a),  "operator== wrong");

  DAX_TEST_ASSERT((a != b),  "operator!= wrong");
  DAX_TEST_ASSERT(!(a != a), "operator!= wrong");

  //test against a tuple that shares some values
  const dax::Vector2 c = dax::make_Vector2(2,3);
  DAX_TEST_ASSERT( !(c == a), "operator == wrong");
  DAX_TEST_ASSERT( !(a == c), "operator == wrong");

  DAX_TEST_ASSERT( (c != a), "operator != wrong");
  DAX_TEST_ASSERT( (a != c), "operator != wrong");
}

template<> void TypeTest<dax::Vector3>()
{
  dax::Vector3 a = dax::make_Vector3(2, 4, 6);
  dax::Vector3 b = dax::make_Vector3(1, 2, 3);
  dax::Scalar s = 5;

  dax::Vector3 plus = a + b;
  DAX_TEST_ASSERT(test_equal(plus, dax::make_Vector3(3, 6, 9)),
                  "Vectors do not add correctly.");

  dax::Vector3 minus = a - b;
  DAX_TEST_ASSERT(test_equal(minus, dax::make_Vector3(1, 2, 3)),
                  "Vectors to not subtract correctly.");

  dax::Vector3 mult = a * b;
  DAX_TEST_ASSERT(test_equal(mult, dax::make_Vector3(2, 8, 18)),
                  "Vectors to not multiply correctly.");

  dax::Vector3 div = a / b;
  DAX_TEST_ASSERT(test_equal(div, dax::make_Vector3(2, 2, 2)),
                  "Vectors to not divide correctly.");

  mult = s * a;
  DAX_TEST_ASSERT(test_equal(mult, dax::make_Vector3(10, 20, 30)),
                  "Vector and scalar to not multiply correctly.");

  mult = a * s;
  DAX_TEST_ASSERT(test_equal(mult, dax::make_Vector3(10, 20, 30)),
                  "Vector and scalar to not multiply correctly.");

  dax::Scalar d = dax::dot(a, b);
  DAX_TEST_ASSERT(test_equal(d, dax::Scalar(28)), "dot(Vector3) wrong");

  DAX_TEST_ASSERT(!(a == b), "operator== wrong");
  DAX_TEST_ASSERT((a == a),  "operator== wrong");

  DAX_TEST_ASSERT((a != b),  "operator!= wrong");
  DAX_TEST_ASSERT(!(a != a), "operator!= wrong");

  //test against a tuple that shares some values
  const dax::Vector3 c = dax::make_Vector3(2,4,5);
  DAX_TEST_ASSERT( !(c == a), "operator == wrong");
  DAX_TEST_ASSERT( !(a == c), "operator == wrong");

  DAX_TEST_ASSERT( (c != a), "operator != wrong");
  DAX_TEST_ASSERT( (a != c), "operator != wrong");
}

template<> void TypeTest<dax::Vector4>()
{
  dax::Vector4 a = dax::make_Vector4(2, 4, 6, 8);
  dax::Vector4 b = dax::make_Vector4(1, 2, 3, 4);
  dax::Scalar s = 5;

  dax::Vector4 plus = a + b;
  DAX_TEST_ASSERT(test_equal(plus, dax::make_Vector4(3, 6, 9, 12)),
                  "Vectors do not add correctly.");

  dax::Vector4 minus = a - b;
  DAX_TEST_ASSERT(test_equal(minus, dax::make_Vector4(1, 2, 3, 4)),
                  "Vectors to not subtract correctly.");

  dax::Vector4 mult = a * b;
  DAX_TEST_ASSERT(test_equal(mult, dax::make_Vector4(2, 8, 18, 32)),
                  "Vectors to not multiply correctly.");

  dax::Vector4 div = a / b;
  DAX_TEST_ASSERT(test_equal(div, dax::make_Vector4(2, 2, 2, 2)),
                  "Vectors to not divide correctly.");

  mult = s * a;
  DAX_TEST_ASSERT(test_equal(mult, dax::make_Vector4(10, 20, 30, 40)),
                  "Vector and scalar to not multiply correctly.");

  mult = a * s;
  DAX_TEST_ASSERT(test_equal(mult, dax::make_Vector4(10, 20, 30, 40)),
                  "Vector and scalar to not multiply correctly.");

  dax::Scalar d = dax::dot(a, b);
  DAX_TEST_ASSERT(test_equal(d, dax::Scalar(60)), "dot(Vector4) wrong");

  DAX_TEST_ASSERT(!(a == b), "operator== wrong");
  DAX_TEST_ASSERT((a == a),  "operator== wrong");

  DAX_TEST_ASSERT((a != b),  "operator!= wrong");
  DAX_TEST_ASSERT(!(a != a), "operator!= wrong");

  //test against a tuple that shares some values
  const dax::Vector4 c = dax::make_Vector4(2,4,6,9);
  DAX_TEST_ASSERT( !(c == a), "operator == wrong");
  DAX_TEST_ASSERT( !(a == c), "operator == wrong");

  DAX_TEST_ASSERT( (c != a), "operator != wrong");
  DAX_TEST_ASSERT( (a != c), "operator != wrong");
}

template<> void TypeTest<dax::Id3>()
{
  dax::Id3 a = dax::make_Id3(2, 4, 6);
  dax::Id3 b = dax::make_Id3(1, 2, 3);
  dax::Id s = 5;

  dax::Id3 plus = a + b;
  if ((plus[0] != 3) || (plus[1] != 6) || (plus[2] != 9))
    {
    DAX_TEST_FAIL("Vectors do not add correctly.");
    }

  dax::Id3 minus = a - b;
  if ((minus[0] != 1) || (minus[1] != 2) || (minus[2] != 3))
    {
    DAX_TEST_FAIL("Vectors to not subtract correctly.");
    }

  dax::Id3 mult = a * b;
  if ((mult[0] != 2) || (mult[1] != 8) || (mult[2] != 18))
    {
    DAX_TEST_FAIL("Vectors to not multiply correctly.");
    }

  dax::Id3 div = a / b;
  if ((div[0] != 2) || (div[1] != 2) || (div[2] != 2))
    {
    DAX_TEST_FAIL("Vectors to not divide correctly.");
    }

  mult = s * a;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    DAX_TEST_FAIL("Vector and scalar to not multiply correctly.");
    }

  mult = a * s;
  if ((mult[0] != 10) || (mult[1] != 20) || (mult[2] != 30))
    {
    DAX_TEST_FAIL("Vector and scalar to not multiply correctly.");
    }

  if (dax::dot(a, b) != 28)
    {
    DAX_TEST_FAIL("dot(Id3) wrong");
    }

  if (a == b)
    {
    DAX_TEST_FAIL("operator== wrong");
    }
  if (!(a == a))
    {
    DAX_TEST_FAIL("operator== wrong");
    }

  if (!(a != b))
    {
    DAX_TEST_FAIL("operator!= wrong");
    }
  if (a != a)
    {
    DAX_TEST_FAIL("operator!= wrong");
    }

  //test against a tuple that shares some values
  const dax::Id3 c = dax::make_Id3(2,4,5);
  if (c == a) { DAX_TEST_FAIL("operator== wrong"); }
  if (a == c) { DAX_TEST_FAIL("operator== wrong"); }

  if (!(c != a)) { DAX_TEST_FAIL("operator!= wrong"); }
  if (!(a != c)) { DAX_TEST_FAIL("operator!= wrong"); }
}

template<> void TypeTest<dax::Scalar>()
{
  dax::Scalar a = 4;
  dax::Scalar b = 2;

  dax::Scalar plus = a + b;
  if (plus != 6)
    {
    DAX_TEST_FAIL("Scalars do not add correctly.");
    }

  dax::Scalar minus = a - b;
  if (minus != 2)
    {
    DAX_TEST_FAIL("Scalars to not subtract correctly.");
    }

  dax::Scalar mult = a * b;
  if (mult != 8)
    {
    DAX_TEST_FAIL("Scalars to not multiply correctly.");
    }

  dax::Scalar div = a / b;
  if (div != 2)
    {
    DAX_TEST_FAIL("Scalars to not divide correctly.");
    }

  if (a == b)
    {
    DAX_TEST_FAIL("operator== wrong");
    }
  if (!(a != b))
    {
    DAX_TEST_FAIL("operator!= wrong");
    }

  if (dax::dot(a, b) != 8)
    {
    DAX_TEST_FAIL("dot(Scalar) wrong");
    }
}

template<> void TypeTest<dax::Id>()
{
  dax::Id a = 4;
  dax::Id b = 2;

  dax::Id plus = a + b;
  if (plus != 6)
    {
    DAX_TEST_FAIL("Scalars do not add correctly.");
    }

  dax::Id minus = a - b;
  if (minus != 2)
    {
    DAX_TEST_FAIL("Scalars to not subtract correctly.");
    }

  dax::Id mult = a * b;
  if (mult != 8)
    {
    DAX_TEST_FAIL("Scalars to not multiply correctly.");
    }

  dax::Id div = a / b;
  if (div != 2)
    {
    DAX_TEST_FAIL("Scalars to not divide correctly.");
    }

  if (a == b)
    {
    DAX_TEST_FAIL("operator== wrong");
    }
  if (!(a != b))
    {
    DAX_TEST_FAIL("operator!= wrong");
    }

  if (dax::dot(a, b) != 8)
    {
    DAX_TEST_FAIL("dot(Id) wrong");
    }
}

struct TypeTestFunctor
{
  template <typename T> void operator()(const T&) const {
    TypeTest<T>();
  }
};

void TestTypes()
{
  dax::testing::Testing::TryAllTypes(TypeTestFunctor());

  //try with some custom tuple types
  TypeTestFunctor()( dax::Tuple<dax::Scalar,6>() );
  TypeTestFunctor()( dax::Tuple<dax::Id,4>() );
  TypeTestFunctor()( dax::Tuple<unsigned char,4>() );
  TypeTestFunctor()( dax::Tuple<dax::Id,1>() );
  TypeTestFunctor()( dax::Tuple<dax::Scalar,1>() );
}

} // anonymous namespace

int UnitTestTypes(int, char *[])
{
  return dax::testing::Testing::Run(TestTypes);
}
