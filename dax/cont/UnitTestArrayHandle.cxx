/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/ArrayHandle.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

void TestArrayHandle()
{
  dax::Scalar array[ARRAY_SIZE];

  // Create an array handle.
  dax::cont::ArrayHandle<dax::Scalar>
      arrayHandle(&array[0], &array[ARRAY_SIZE]);

  test_assert(arrayHandle.GetNumberOfEntries() == ARRAY_SIZE,
              "ArrayHandle has wrong number of entries.");

  test_assert(arrayHandle.IsControlArrayValid(),
              "Control data not valid.");

  // Make sure that invalidating any copy will invalidate all copies.
  dax::cont::ArrayHandle<dax::Scalar> arrayHandleCopy;
  arrayHandleCopy = arrayHandle;
  arrayHandleCopy.InvalidateControlArray();
  test_assert(!arrayHandle.IsControlArrayValid(),
              "Invalidate did not propagate to copies.");
}

}


int UnitTestArrayHandle(int, char *[])
{
  try
    {
    TestArrayHandle();
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
