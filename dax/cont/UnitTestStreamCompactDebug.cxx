/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/ScheduleDebug.h>
#include <dax/cont/StreamCompactDebug.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const dax::Id ARRAY_SIZE = 500;

const dax::Id OFFSET = 1000;

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

struct InitArray
{
  DAX_EXEC_EXPORT void operator()(std::vector<dax::Id> &array, dax::Id index)
  {
    array[index] = OFFSET + index;
  }
};

struct MakeMask
{
  DAX_EXEC_EXPORT void operator()(std::vector<dax::Id> &array, dax::Id index)
  {
    array[index] = index%2;
  }
};

}

int TestCompact()
  {
  //test the version of compact that takes in input and uses it as a stencil
  std::vector<dax::Id> array(ARRAY_SIZE);
  std::vector<dax::Id> result;

  //construct the index array
  dax::cont::scheduleDebug(MakeMask(), array, ARRAY_SIZE);
  dax::cont::streamCompactDebug(array,result);

  test_assert(result.size() == array.size()/2,
              "result of compacation is an incorrect size.");

  std::cout << "Checking results." << std::endl;
  for (dax::Id index = 0; index < result.size(); index++)
    {
    dax::Id value = result[index];
    test_assert(value == (index*2),
                "Incorrect value in compaction result.");
    }

  }

bool TestCompactWithStencil()
  {

  }

int UnitTestStreamCompactDebug(int, char *[])
{

  if(!TestCompact())
    {
    std::cout << "Stream Compact without a stencil array failed" << std::endl;
    return 1;
    }
  if(!TestCompactWithStencil())
    {
    std::cout << "Stream Compact with a stencil array failed" << std::endl;
    return 1;
    }
  return 0;
}
