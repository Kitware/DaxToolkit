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
#ifndef __dax_testing_VectorTraitsTest_h
#define __dax_testing_VectorTraitsTest_h

#include <dax/VectorTraits.h>

#include <dax/TypeTraits.h>

#include <dax/internal/testing/Testing.h>

namespace dax {
namespace testing {

namespace detail {

inline void CompareDimensionalityTags(dax::TypeTraitsScalarTag,
                                      dax::VectorTraitsTagSingleComponent)
{
  // If we are here, everything is fine.
}
inline void CompareDimensionalityTags(dax::TypeTraitsVectorTag,
                                      dax::VectorTraitsTagMultipleComponents)
{
  // If we are here, everything is fine.
}

} // namespace detail

/// Compares some manual arithmetic through type traits to arithmetic with
/// the Tuple class.
template <class T>
static void TestVectorType(const T &vector)
{
  typedef typename dax::VectorTraits<T> Traits;
  typedef typename Traits::ComponentType ComponentType;
  static const int NUM_COMPONENTS = Traits::NUM_COMPONENTS;

  {
  T result;
  const ComponentType multiplier = 4;
  for (int i = 0; i < NUM_COMPONENTS; i++)
    {
    Traits::SetComponent(result, i, multiplier*Traits::GetComponent(vector, i));
    }
  DAX_TEST_ASSERT(Traits::ToTuple(result) == multiplier*Traits::ToTuple(vector),
                  "Got bad result for scalar multiple");
  }

  {
  T result;
  const ComponentType multiplier = 7;
  for (int i = 0; i < NUM_COMPONENTS; i++)
    {
    Traits::GetComponent(result, i)
        = multiplier * Traits::GetComponent(vector, i);
    }
  DAX_TEST_ASSERT(Traits::ToTuple(result) == multiplier*Traits::ToTuple(vector),
                  "Got bad result for scalar multiple");
  }

  {
  ComponentType result = 0;
  for (int i = 0; i < NUM_COMPONENTS; i++)
    {
    ComponentType component
        = Traits::GetComponent(vector, i);
    result += component * component;
    }
  DAX_TEST_ASSERT(
        result == dax::dot(Traits::ToTuple(vector), Traits::ToTuple(vector)),
        "Got bad result for dot product");
  }

  // This will fail to compile if the tags are wrong.
  detail::CompareDimensionalityTags(
        typename dax::TypeTraits<T>::DimensionalityTag(),
        typename dax::VectorTraits<T>::HasMultipleComponents());
}

namespace detail {

inline void CheckVectorComponentsTag(dax::VectorTraitsTagMultipleComponents)
{
  // If we are running here, everything is fine.
}

} // namespace detail

/// Checks to make sure that the HasMultipleComponents tag is actually for
/// multiple components. Should only be called for vector classes that actually
/// have multiple components.
///
template<class T>
inline void TestVectorComponentsTag()
{
  // This will fail to compile if the tag is wrong
  // (i.e. not dax::VectorTraitsTagMultipleComponents)
  detail::CheckVectorComponentsTag(
        typename dax::VectorTraits<T>::HasMultipleComponents());
}

namespace detail {

inline void CheckScalarComponentsTag(dax::VectorTraitsTagSingleComponent)
{
  // If we are running here, everything is fine.
}

} // namespace detail

/// Checks to make sure that the HasMultipleComponents tag is actually for a
/// single component. Should only be called for "vector" classes that actually
/// have only a single component (that is, are really scalars).
///
template<class T>
inline void TestScalarComponentsTag()
{
  // This will fail to compile if the tag is wrong
  // (i.e. not dax::VectorTraitsTagSingleComponent)
  detail::CheckScalarComponentsTag(
        typename dax::VectorTraits<T>::HasMultipleComponents());
}

}
} // namespace dax::testing

#endif //__dax_testing_VectorTraitsTest_h
