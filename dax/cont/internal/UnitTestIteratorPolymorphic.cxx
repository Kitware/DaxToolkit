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

#include <dax/cont/internal/IteratorPolymorphic.h>

#include <dax/cont/internal/Testing.h>

#include <algorithm>

#include <boost/iterator/counting_iterator.hpp>

namespace
{
const dax::Id ARRAY_SIZE = 10;

void TestBasicIteration()
{
  // Make a basic array.  Fill with 10 + index.
  dax::Scalar array[ARRAY_SIZE];
  std::copy(boost::counting_iterator<dax::Scalar>(10),
            boost::counting_iterator<dax::Scalar>(10+ARRAY_SIZE),
            array);

  // Make a polymorphic iterator out of the array.
  typedef dax::cont::internal::IteratorPolymorphic<dax::Scalar> IteratorType;
  IteratorType polyIterBegin
      = dax::cont::internal::make_IteratorPolymorphic(&array[0]);

  std::cout << "Checking basic polymorphic access." << std::endl;
  DAX_TEST_ASSERT(*polyIterBegin == 10, "Cannot dereference.");
  IteratorType polyIterMiddle = polyIterBegin + 5;
  DAX_TEST_ASSERT(*polyIterMiddle == 15, "Cannot advance.");
  polyIterMiddle--;
  DAX_TEST_ASSERT(*polyIterMiddle == 14, "Cannot decrement.");
  polyIterMiddle++;
  DAX_TEST_ASSERT(*polyIterMiddle == 15, "Cannot increment.");
  DAX_TEST_ASSERT((polyIterMiddle - polyIterBegin) == 5, "Cannot take distance.");
  *polyIterMiddle = 432;
  DAX_TEST_ASSERT(array[5] == 432, "Cannot set dereference.");
  DAX_TEST_ASSERT(polyIterBegin != polyIterMiddle, "Inequality wrong.");
  polyIterMiddle -= 5;
  DAX_TEST_ASSERT(polyIterBegin == polyIterMiddle, "Equality wrong.");

  std::cout << "Basic copy into polymorphic iterator." << std::endl;
  std::copy(boost::counting_iterator<dax::Scalar>(0),
            boost::counting_iterator<dax::Scalar>(ARRAY_SIZE),
            polyIterBegin);

  // Check results.
  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    DAX_TEST_ASSERT(array[i] == i, "Array has incorrect value copied.");
    }

  std::cout << "Basic copy from polymorphic iterator." << std::endl;
  dax::Scalar array2[ARRAY_SIZE];
  dax::cont::internal::IteratorPolymorphic<dax::Scalar> polyIterEnd
      = dax::cont::internal::make_IteratorPolymorphic(&array[10]);
  std::copy(polyIterBegin, polyIterEnd, &array2[0]);

  // Check results.
  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    DAX_TEST_ASSERT(array2[i] == i, "Array has incorrect value copied.");
    }
}

}

int UnitTestIteratorPolymorphic(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestBasicIteration);
}
