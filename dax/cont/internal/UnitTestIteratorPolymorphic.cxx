/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/internal/IteratorPolymorphic.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <algorithm>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>

namespace
{
const dax::Id ARRAY_SIZE = 10;

#define test_assert(condition, message) \
  test_assert_impl(condition, message, __FILE__, __LINE__);

static inline void test_assert_impl(bool condition,
                                    const std::string& message,
                                    const char *file,
                                    int line)
{
  if(!condition)
    {
    std::stringstream error;
    error << file << ":" << line << std::endl;
    error << message << std::endl;
    throw error.str();
    }
}

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
  test_assert(*polyIterBegin == 10, "Cannot dereference.");
  IteratorType polyIterMiddle = polyIterBegin + 5;
  test_assert(*polyIterMiddle == 15, "Cannot advance.");
  polyIterMiddle--;
  test_assert(*polyIterMiddle == 14, "Cannot decrement.");
  polyIterMiddle++;
  test_assert(*polyIterMiddle == 15, "Cannot increment.");
  test_assert((polyIterMiddle - polyIterBegin) == 5, "Cannot take distance.");
  *polyIterMiddle = 432;
  test_assert(array[5] == 432, "Cannot set dereference.");
  test_assert(polyIterBegin != polyIterMiddle, "Inequality wrong.");
  polyIterMiddle -= 5;
  test_assert(polyIterBegin == polyIterMiddle, "Equality wrong.");

  std::cout << "Basic copy into polymorphic iterator." << std::endl;
  std::copy(boost::counting_iterator<dax::Scalar>(0),
            boost::counting_iterator<dax::Scalar>(ARRAY_SIZE),
            polyIterBegin);

  // Check results.
  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    test_assert(array[i] == i, "Array has incorrect value copied.");
    }

  std::cout << "Basic copy from polymorphic iterator." << std::endl;
  dax::Scalar array2[ARRAY_SIZE];
  dax::cont::internal::IteratorPolymorphic<dax::Scalar> polyIterEnd
      = dax::cont::internal::make_IteratorPolymorphic(&array[10]);
  std::copy(polyIterBegin, polyIterEnd, &array2[0]);

  // Check results.
  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    test_assert(array2[i] == i, "Array has incorrect value copied.");
    }
}

}

int UnitTestIteratorPolymorphic(int, char *[])
{
  try
    {
    TestBasicIteration();
    }
  catch (std::string error)
    {
    std::cout
        << "Encountered error: " << std::endl
        << error << std::endl;
    return 1;
    }

  return 0;
}
