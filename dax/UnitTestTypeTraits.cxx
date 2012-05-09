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

#include <dax/TypeTraits.h>

#include <dax/VectorTraits.h>

#include <dax/internal/Testing.h>

namespace {

struct TypeTraitTest
{
  template <typename T> void operator()(T t) {
    // If you get compiler errors here, it could be a TypeTraits instance
    // has missing or malformed tags.
    this->TestDimensionality(t, typename dax::TypeTraits<T>::DimensionalityTag());
    this->TestNumeric(t, typename dax::TypeTraits<T>::NumericTag());
  }
private:

  template <typename T>
  void TestDimensionality(T, dax::TypeTraitsScalarTag) {
    std::cout << "  scalar" << std::endl;
    DAX_TEST_ASSERT(dax::VectorTraits<T>::NUM_COMPONENTS == 1,
                    "Scalar type does not have one component.");
  }
  template <typename T>
  void TestDimensionality(T, dax::TypeTraitsVectorTag) {
    std::cout << "  vector" << std::endl;
    DAX_TEST_ASSERT(dax::VectorTraits<T>::NUM_COMPONENTS > 1,
                    "Vector type does not have multiple components.");
  }

  template <typename T>
  void TestNumeric(T, dax::TypeTraitsIntegerTag) {
    std::cout << "  integer" << std::endl;
    typedef typename dax::VectorTraits<T>::ComponentType VT;
    VT value = VT(2.001);
    DAX_TEST_ASSERT(value == 2, "Integer does not round to integer.");
  }
  template <typename T>
  void TestNumeric(T, dax::TypeTraitsRealTag) {
    std::cout << "  real" << std::endl;
    typedef typename dax::VectorTraits<T>::ComponentType VT;
    VT value = VT(2.001);
    DAX_TEST_ASSERT(test_equal(float(value), float(2.001)),
                    "Real does not hold floaing point number.");
  }
};

static void TestTypeTraits()
{
  dax::internal::Testing::TryAllTypes(TypeTraitTest());
}

} // anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestTypeTraits(int, char *[])
{
  return dax::internal::Testing::Run(TestTypeTraits);
}
