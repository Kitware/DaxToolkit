/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/internal/ArrayContainerExecutionCPU.h>
#include <dax/cont/ScheduleDebug.h>
#include <dax/cont/StreamCompactDebug.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

const dax::Id ARRAY_SIZE = 50000;

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
  DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id> array,
                                  dax::Id index,
                                  dax::exec::internal::ErrorHandler &)
  {
    array.SetValue(index,OFFSET + index);
  }
};

struct MarkOddNumbers
{
  DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id> array,
                                  dax::Id index,
                                  dax::exec::internal::ErrorHandler &)
  {
    array.SetValue(index,index%2);
  }
};

bool TestCompact()
  {
  //test the version of compact that takes in input and uses it as a stencil
  //and uses the index of each item as the value to place in the result vector
  dax::cont::internal::ArrayContainerExecutionCPU<dax::Id> array;
  dax::cont::internal::ArrayContainerExecutionCPU<dax::Id> result;
  typedef dax::cont::internal::ArrayContainerExecutionCPU<dax::Id>::const_iterator
      iterator;

  array.Allocate(ARRAY_SIZE);
  dax::internal::DataArray<dax::Id> rawArray = array.GetExecutionArray();

  //construct the index array
  dax::cont::scheduleDebug(MarkOddNumbers(), rawArray, ARRAY_SIZE);
  dax::cont::streamCompactDebug(array,result);

  std::stringstream buffer;
  buffer << "result of compacation has an incorrect size of:";
  buffer << result.GetNumberOfEntries();
  test_assert(result.GetNumberOfEntries() == array.GetNumberOfEntries()/2,
              buffer.str());

  dax::Id index=0;
  for(iterator i = result.begin();
      i != result.end();
      ++i,++index)
    {
    const dax::Id value = *i;
    test_assert(value == (index*2)+1,
                "Incorrect value in compaction result.");
    }
  return true;
  }

bool TestCompactWithStencil()
  {
  //test the version of compact that takes in input and a stencil
  dax::cont::internal::ArrayContainerExecutionCPU<dax::Id> array;
  dax::cont::internal::ArrayContainerExecutionCPU<dax::Id> stencil;
  dax::cont::internal::ArrayContainerExecutionCPU<dax::Id> result;
  typedef dax::cont::internal::ArrayContainerExecutionCPU<dax::Id>::const_iterator
      iterator;

  array.Allocate(ARRAY_SIZE);
  stencil.Allocate(ARRAY_SIZE);

  dax::internal::DataArray<dax::Id> rawArray = array.GetExecutionArray();
  dax::internal::DataArray<dax::Id> rawStencil = stencil.GetExecutionArray();

  //construct the index array
  dax::cont::scheduleDebug(InitArray(), rawArray, ARRAY_SIZE);
  dax::cont::scheduleDebug(MarkOddNumbers(), rawStencil, ARRAY_SIZE);
  dax::cont::streamCompactDebug(array,stencil,result);

  std::stringstream buffer;
  buffer << "result of compacation has an incorrect size of:";
  buffer << result.GetNumberOfEntries();
  test_assert(result.GetNumberOfEntries() == array.GetNumberOfEntries()/2,
              buffer.str());

  dax::Id index=0;
  for(iterator i = result.begin();
      i != result.end();
      ++i,++index)
    {
    const dax::Id value = *i;
    test_assert(value == (OFFSET + (index*2)+1),
                "Incorrect value in compaction result.");
    }
  return true;
  }

}

int UnitTestStreamCompactDebug(int, char *[])
{
  bool valid = false;
  try
    {
    valid = TestCompact();
    }
  catch (std::string error)
    {
    std::cout
        << "Encountered error: " << std::endl
        << error << std::endl;
    return 1;
    }
  if(!valid)
    {
    std::cout << "Stream Compact without a stencil array failed" << std::endl;
    return 1;
    }

  valid = false;
  try
    {
    valid = TestCompactWithStencil();
    }
  catch (std::string error)
    {
    std::cout
        << "Encountered error: " << std::endl
        << error << std::endl;
    return 1;
    }
  if(!valid)
    {
    std::cout << "Stream Compact with a stencil array failed" << std::endl;
    return 1;
    }
  return 0;
}
