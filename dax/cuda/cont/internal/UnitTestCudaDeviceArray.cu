/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/Array.h>
#include <dax/cuda/cont/internal/DeviceArray.h>


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#define TEST_FAIL(msg)                                  \
  {                                                     \
    std::stringstream error;                            \
    error << __FILE__ << ":" << __LINE__ << std::endl;  \
    error msg;                                          \
    throw error.str();                                  \
  }

namespace {
const dax::Id ARRAY_SIZE = 10;
}

typedef dax::Scalar T;

static T StartValue(dax::Id index)
{
  return static_cast<T>(index);
}

__device__ __host__ static T AddOne(T value)
{
  return value + static_cast<T>(value);
}

__global__ void AddOneToArray(dax::internal::DataArray<T> array)
{
  dax::Id index = threadIdx.x;
  T value = array.GetValue(index);
  array.SetValue(index, AddOne(value));
}

static void TestDeviceDataArray()
{
  dax::cont::Array<T> inputArray(ARRAY_SIZE);
  //dax::internal::DataArray<T> inputArray(inputBuffer, ARRAY_SIZE);
  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    inputArray[i] = StartValue(i);
    }

  dax::cuda::cont::internal::DeviceArray<T> deviceArray;
  deviceArray = inputArray;

  AddOneToArray<<<1, ARRAY_SIZE>>>(deviceArray);

  dax::cont::Array<T> outputArray(ARRAY_SIZE);
  outputArray = deviceArray;

  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    T inValue = inputArray[i];
    T outValue = outputArray[i];
    if (outValue != AddOne(inValue))
      {
      TEST_FAIL(<< "Bad value in copied array.");
      }
    }
}

static void TestManagedDeviceDataArray()
{
  dax::cont::Array<T> inputArray(ARRAY_SIZE);
  //dax::internal::DataArray<T> inputArray(inputBuffer, ARRAY_SIZE);
  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    inputArray[i] = StartValue(i);
    }

  dax::cuda::cont::internal::DeviceArrayPtr<T> deviceArray(
        new dax::cuda::cont::internal::DeviceArray<T>(ARRAY_SIZE));
  (*deviceArray.get()) = inputArray;

  AddOneToArray<<<1, ARRAY_SIZE>>>(deviceArray);

  dax::cont::Array<T> outputArray(ARRAY_SIZE);
  outputArray = deviceArray.get();

  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    T inValue = inputArray[i];
    T outValue = outputArray[i];
    if (outValue != AddOne(inValue))
      {
      TEST_FAIL(<< "Bad value in copied array.");
      }
    }
}

int UnitTestCudaDeviceArray(int, char *[])
{
  try
    {
    TestDeviceDataArray();
    TestManagedDeviceDataArray();
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
