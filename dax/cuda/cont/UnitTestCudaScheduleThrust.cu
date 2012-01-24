/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cuda/cont/ScheduleThrust.h>

#include <dax/exec/internal/ErrorHandler.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

namespace {

const dax::Id ARRAY_SIZE = 500;

const dax::Id OFFSET = 1000;

}

namespace ut_CudaScheduleThrust {

struct ClearArray
{
  DAX_EXEC_EXPORT void operator()(dax::Id *array,
                                  dax::Id index,
                                  dax::exec::internal::ErrorHandler &)
  {
    array[index] = OFFSET;
  }
};

struct AddArray
{
  DAX_EXEC_EXPORT void operator()(dax::Id *array,
                                  dax::Id index,
                                  dax::exec::internal::ErrorHandler &)
  {
    array[index] += index;
  }
};

#define ERROR_MESSAGE "Got an error."

struct OneError
{
  DAX_EXEC_EXPORT void operator()(
      dax::Id *, dax::Id index, dax::exec::internal::ErrorHandler &errorHandler)
  {
    if (index == ARRAY_SIZE/2)
      {
      errorHandler.RaiseError(ERROR_MESSAGE);
      }
  }
};

struct AllError
{
  DAX_EXEC_EXPORT void operator()(
      dax::Id *, dax::Id, dax::exec::internal::ErrorHandler &errorHandler)
  {
    errorHandler.RaiseError(ERROR_MESSAGE);
  }
};

} // namespace ut_CudaSchedule

int UnitTestCudaScheduleThrust(int, char *[])
{
  using namespace ut_CudaScheduleThrust;

  thrust::device_vector<dax::Id> deviceArray(ARRAY_SIZE);
  dax::Id *rawDeviceArray = thrust::raw_pointer_cast(&deviceArray[0]);

  std::cout << "Running clear." << std::endl;
  dax::cuda::cont::scheduleThrust(ClearArray(), rawDeviceArray, ARRAY_SIZE);

  std::cout << "Running add." << std::endl;
  dax::cuda::cont::scheduleThrust(AddArray(), rawDeviceArray, ARRAY_SIZE);

  std::cout << "Checking results." << std::endl;
  thrust::host_vector<dax::Id> hostArray(ARRAY_SIZE);
  thrust::copy(deviceArray.begin(), deviceArray.end(), hostArray.begin());

  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    dax::Id value = hostArray[index];
    if (value != index + OFFSET)
      {
      std::cout << "Got bad value." << std::endl;
      return 1;
      }
    }

  std::cout << "Generating one error." << std::endl;
  char *message;
  message = dax::cuda::cont::scheduleThrust(OneError(),
                                            rawDeviceArray,
                                            ARRAY_SIZE);
  if (strcmp(message, ERROR_MESSAGE) != 0)
    {
    std::cout << "Did not get expected error message." << std::endl;
    }

  std::cout << "Generating lots of errors." << std::endl;
  message = dax::cuda::cont::scheduleThrust(AllError(),
                                            rawDeviceArray,
                                            ARRAY_SIZE);
  if (strcmp(message, ERROR_MESSAGE) != 0)
    {
    std::cout << "Did not get expected error message." << std::endl;
    }

  return 0;
}
