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

template <typename T> void TypeTest();

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

  DAX_TEST_ASSERT(!(a == b), "operator== wrong");
  DAX_TEST_ASSERT((a != b),"operator!= wrong");

  dax::Scalar d = dax::dot(a, b);
  DAX_TEST_ASSERT(test_equal(d, dax::Scalar(10)), "dot(Vector2) wrong");
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

  DAX_TEST_ASSERT(!(a == b), "operator== wrong");
  DAX_TEST_ASSERT((a != b),"operator!= wrong");

  dax::Scalar d = dax::dot(a, b);
  DAX_TEST_ASSERT(test_equal(d, dax::Scalar(28)), "dot(Vector3) wrong");
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

  DAX_TEST_ASSERT(!(a == b), "operator== wrong");
  DAX_TEST_ASSERT((a != b),"operator!= wrong");

  dax::Scalar d = dax::dot(a, b);
  DAX_TEST_ASSERT(test_equal(d, dax::Scalar(60)), "dot(Vector4) wrong");
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

  if (a == b)
    {
    DAX_TEST_FAIL("operator== wrong");
    }
  if (!(a != b))
    {
    DAX_TEST_FAIL("operator!= wrong");
    }

  if (dax::dot(a, b) != 28)
    {
    DAX_TEST_FAIL("dot(Id3) wrong");
    }
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
}

} // anonymous namespace

int UnitTestTypes(int, char *[])
{
  return dax::testing::Testing::Run(TestTypes);
}
