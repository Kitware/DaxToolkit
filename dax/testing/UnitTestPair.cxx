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

#include <dax/Pair.h>
#include <dax/Types.h>
#include <dax/VectorTraits.h>

#include <dax/testing/Testing.h>

namespace {

//general pair test
template <typename T, typename U> void PairTest( )
{
  //test that all the constructors work properly
  {
  dax::Pair<T,U> no_params_pair;
  dax::Pair<T,U> copy_constructor_pair(no_params_pair);
  dax::Pair<T,U> assignment_pair = no_params_pair;

  DAX_TEST_ASSERT( (no_params_pair == copy_constructor_pair),
                  "copy constructor doesn't match default constructor");
  DAX_TEST_ASSERT( !(no_params_pair != copy_constructor_pair),
                  "operator != is working properly");


  DAX_TEST_ASSERT( (no_params_pair == assignment_pair),
                  "assignment constructor doesn't match default constructor");
  DAX_TEST_ASSERT( !(no_params_pair != assignment_pair),
                  "operator != is working properly");
  }

  //now lets give each item in the pair some values and do some in depth
  //comparisons
  T a;
  U b;
  for(int i=0; i < dax::VectorTraits<T>::NUM_COMPONENTS; ++i)
    { dax::VectorTraits<T>::SetComponent(a,i,(i+1)*2); }

  for(int i=0; i < dax::VectorTraits<U>::NUM_COMPONENTS; ++i)
    { dax::VectorTraits<U>::SetComponent(b,i,(i+1)); }

  //test the constructors now with real values
  {
  dax::Pair<T,U> pair_ab(a,b);
  dax::Pair<T,U> copy_constructor_pair(pair_ab);
  dax::Pair<T,U> assignment_pair = pair_ab;
  dax::Pair<T,U> make_p = dax::make_Pair(a,b);

  DAX_TEST_ASSERT( !(pair_ab != pair_ab),
                    "operator != isn't working properly for dax::pair" );
  DAX_TEST_ASSERT( (pair_ab == pair_ab),
                    "operator == isn't working properly for dax::pair" );

  DAX_TEST_ASSERT( (pair_ab == copy_constructor_pair),
                  "copy constructor doesn't match pair constructor");
  DAX_TEST_ASSERT( (pair_ab == assignment_pair),
                  "assignment constructor doesn't match pair constructor");
  DAX_TEST_ASSERT( (pair_ab == make_p),
                  "make_pair function doesn't match pair constructor");
  }


  //test the ordering operators <, >, <=, >=
  {
  //in all cases pair_ab2 is > pair_ab. these verify that if the second
  //argument of the pair is different we respond properly
  U b2(b);
  dax::VectorTraits<U>::SetComponent(b2,0,
         dax::VectorTraits<U>::GetComponent(b2,0)+1);

  dax::Pair<T,U> pair_ab2(a,b2);
  dax::Pair<T,U> pair_ab(a,b);

  DAX_TEST_ASSERT( !(pair_ab2 == pair_ab), "operator == failed" );
  DAX_TEST_ASSERT( (pair_ab2 != pair_ab), "operator != failed" );

  T a2(a);
  dax::VectorTraits<T>::SetComponent(a2,0,
       dax::VectorTraits<T>::GetComponent(a2,0)+1);
  dax::Pair<T,U> pair_a2b(a2,b);

  //this way can verify that if the first argument of the pair is different
  //we respond properly

  DAX_TEST_ASSERT( !(pair_a2b == pair_ab), "operator == failed" );
  DAX_TEST_ASSERT( (pair_a2b != pair_ab), "operator != failed" );
  }

}

template< typename FirstType >
struct DecideSecondType
{
  template <typename SecondType> void operator()(const SecondType&) const
  {
  PairTest<FirstType,SecondType>();
  }
};

struct DecideFirstType
{
  template <typename T> void operator()(const T&) const
  {
  //T is our first type for dax::Pair, now to figure out the second type
  dax::testing::Testing::TryAllTypes(DecideSecondType<T>());

  }
};

void TestPair()
{
  //we want to test each combination of standard dax types in a
  //dax::Pair, so to do that we dispatch twice on TryAllTypes, that
  //way to get all combinations ( id, vec3 ), (id, id), (id, vec4 ) etc
  dax::testing::Testing::TryAllTypes(DecideFirstType());
}

} // anonymous namespace

int UnitTestPair(int, char *[])
{
  return dax::testing::Testing::Run(TestPair);
}
