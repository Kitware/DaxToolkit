/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cuda/cont/ScheduleThrust.h>
#include <dax/cuda/cont/StreamCompact.h>

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

namespace ut_CudaStreamCompact
{

struct InitArray
{
  DAX_EXEC_EXPORT void operator()(dax::Id *array, dax::Id index)
  {
    array[index] = OFFSET + index;
  }
};

struct MarkOddNumbers
{
  DAX_EXEC_EXPORT void operator()(dax::Id *array, dax::Id index)
  {
    array[index] = index%2;
  }
};

bool TestCompact()
  {
  //test the version of compact that takes in input and uses it as a stencil
  //and uses the index of each item as the value to place in the result vector
  ::thrust::device_vector<dax::Id> array(ARRAY_SIZE);
  ::thrust::device_vector<dax::Id> result;

  array.resize(ARRAY_SIZE,dax::Id());
  dax::Id *rawArray = thrust::raw_pointer_cast(&array[0]);

  //construct the index array
  dax::cuda::cont::scheduleThrust(MarkOddNumbers(),rawArray, ARRAY_SIZE);
  dax::cuda::cont::streamCompact(array,result);

  test_assert(result.size() == array.size()/2,
              "result of compacation has an incorrect size.");

  ::thrust::host_vector<dax::Id> hresult(ARRAY_SIZE);
  hresult = result;
  for (dax::Id index = 0; index < static_cast<dax::Id>(hresult.size()); index++)
    {
    dax::Id value = hresult[index];
    test_assert(value == (index*2+1),
                "Incorrect value in compaction result.");
    }
  return true;
  }

bool TestCompactWithStencil()
  {
  //test the version of compact that takes in input and a stencil
  ::thrust::device_vector<dax::Id> array(ARRAY_SIZE);
  ::thrust::device_vector<dax::Id> stencil(ARRAY_SIZE);
  ::thrust::device_vector<dax::Id> result;

  array.resize(ARRAY_SIZE,dax::Id());
  stencil.resize(ARRAY_SIZE,dax::Id());

  dax::Id *rawArray = thrust::raw_pointer_cast(&array[0]);
  dax::Id *rawStencil = thrust::raw_pointer_cast(&stencil[0]);

  //construct the index array
  dax::cuda::cont::scheduleThrust(InitArray(), rawArray, ARRAY_SIZE);
  dax::cuda::cont::scheduleThrust(MarkOddNumbers(), rawStencil, ARRAY_SIZE);
  dax::cuda::cont::streamCompact(array,stencil,result);

  test_assert(result.size() == array.size()/2,
              "result of compacation has an incorrect size.");

  ::thrust::host_vector<dax::Id> hresult(ARRAY_SIZE);
  hresult = result;
  for (dax::Id index = 0; index < static_cast<dax::Id>(hresult.size()); index++)
    {
    dax::Id value = hresult[index];
    test_assert(value == (OFFSET + (index*2)+1),
                "Incorrect value in compaction result.");
    }
  return true;
  }

}

int UnitTestCudaStreamCompact(int, char *[])
{
  bool valid = false;
  try
    {
    valid = ut_CudaStreamCompact::TestCompact();
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
    valid = ut_CudaStreamCompact::TestCompactWithStencil();
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
