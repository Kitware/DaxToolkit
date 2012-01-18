/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/openmp/cont/ScheduleThrust.h>

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
  DAX_EXEC_EXPORT void operator()(dax::Id *array, dax::Id index)
  {
    array[index] = OFFSET;
  }
};

struct AddArray
{
  DAX_EXEC_EXPORT void operator()(dax::Id *array, dax::Id index)
  {
    array[index] += index;
  }
};

} // namespace ut_CuadSchedule

int UnitTestOpenMPScheduleThrust(int, char *[])
{
  using namespace ut_CudaScheduleThrust;

  thrust::device_vector<dax::Id> deviceArray(ARRAY_SIZE);
  dax::Id *rawDeviceArray = thrust::raw_pointer_cast(&deviceArray[0]);

  std::cout << "Running clear." << std::endl;
  dax::openmp::cont::scheduleThrust(ClearArray(), rawDeviceArray, ARRAY_SIZE);

  std::cout << "Running add." << std::endl;
  dax::openmp::cont::scheduleThrust(AddArray(), rawDeviceArray, ARRAY_SIZE);

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

  return 0;
}
