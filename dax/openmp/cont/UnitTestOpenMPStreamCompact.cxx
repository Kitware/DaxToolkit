/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/openmp/cont/internal/ArrayContainerExecutionThrust.h>
#include <dax/openmp/cont/ScheduleThrust.h>
#include <dax/openmp/cont/StreamCompact.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

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

}

namespace ut_OpenMPStreamCompact
{

struct InitArray
{
  DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id>& array,
                                  dax::Id index)
  {
    array.SetValue(index,OFFSET + index);
  }
};

struct MarkOddNumbers
{
  DAX_EXEC_EXPORT void operator()(dax::internal::DataArray<dax::Id>& array,
                                  dax::Id index)
  {
    array.SetValue(index,index%2);
  }
};

bool TestCompact()
  {
  //test the version of compact that takes in input and uses it as a stencil
  //and uses the index of each item as the value to place in the result vector
  dax::openmp::cont::internal::ArrayContainerExecutionThrust<dax::Id> array;
  dax::openmp::cont::internal::ArrayContainerExecutionThrust<dax::Id> result;
  typedef dax::openmp::cont::internal::ArrayContainerExecutionThrust<dax::Id>::const_iterator
      iterator;

  array.Allocate(ARRAY_SIZE);
  dax::internal::DataArray<dax::Id> rawArray = array.GetExecutionArray();


  //construct the index array
  dax::openmp::cont::scheduleThrust(MarkOddNumbers(),rawArray, ARRAY_SIZE);
  dax::openmp::cont::streamCompact(array,result);

  std::stringstream buffer;
  buffer << "result of compacation has an incorrect size of:";
  buffer << result.GetNumberOfEntries();
  test_assert(result.GetNumberOfEntries() == array.GetNumberOfEntries()/2,
              buffer.str());

  dax::Id index=0;
  for(iterator i = result.GetBeginThrustIterator();
      i != result.GetEndThrustIterator();
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
  dax::openmp::cont::internal::ArrayContainerExecutionThrust<dax::Id> array;
  dax::openmp::cont::internal::ArrayContainerExecutionThrust<dax::Id> stencil;
  dax::openmp::cont::internal::ArrayContainerExecutionThrust<dax::Id> result;
  typedef dax::openmp::cont::internal::ArrayContainerExecutionThrust<dax::Id>::const_iterator
      iterator;

  array.Allocate(ARRAY_SIZE);
  stencil.Allocate(ARRAY_SIZE);


  dax::internal::DataArray<dax::Id> rawArray = array.GetExecutionArray();
  dax::internal::DataArray<dax::Id> rawStencil = stencil.GetExecutionArray();

  //construct the index array
  dax::openmp::cont::scheduleThrust(InitArray(), rawArray, ARRAY_SIZE);
  dax::openmp::cont::scheduleThrust(MarkOddNumbers(), rawStencil, ARRAY_SIZE);
  dax::openmp::cont::streamCompact(array,stencil,result);

  std::stringstream buffer;
  buffer << "result of compacation has an incorrect size of:";
  buffer << result.GetNumberOfEntries();
  test_assert(result.GetNumberOfEntries() == array.GetNumberOfEntries()/2,
              buffer.str());

  dax::Id index=0;
  for(iterator i = result.GetBeginThrustIterator();
      i != result.GetEndThrustIterator();
      ++i,++index)
    {
    const dax::Id value = *i;
    test_assert(value == (OFFSET + (index*2)+1),
                "Incorrect value in compaction result.");
    }
  return true;
  }

}

int UnitTestOpenMPStreamCompact(int, char *[])
{
  bool valid = false;
  try
    {
    valid = ut_OpenMPStreamCompact::TestCompact();
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
    valid = ut_OpenMPStreamCompact::TestCompactWithStencil();
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
