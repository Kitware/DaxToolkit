/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cuda/cont/internal/ManagedDeviceDataArray.h>

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

static void TestManagedDeviceDataArray()
{
  T inputBuffer[ARRAY_SIZE];
  dax::internal::DataArray<T> inputArray(inputBuffer, ARRAY_SIZE);
  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    inputArray.SetValue(i, StartValue(i));
    }

  dax::cuda::cont::internal::ManagedDeviceDataArrayPtr<T> deviceArray(
        new dax::cuda::cont::internal::ManagedDeviceDataArray<T>());
  deviceArray->CopyToDevice(inputArray);

  AddOneToArray<<<1, ARRAY_SIZE>>>(deviceArray->GetArray());

  T outputBuffer[ARRAY_SIZE];
  dax::internal::DataArray<T> outputArray(outputBuffer, ARRAY_SIZE);
  deviceArray->CopyToHost(outputArray);

  for (dax::Id i = 0; i < ARRAY_SIZE; i++)
    {
    T inValue = inputArray.GetValue(i);
    T outValue = outputArray.GetValue(i);
    if (outValue != AddOne(inValue))
      {
      TEST_FAIL(<< "Bad value in copied array.");
      }
    }
}

int UnitTestCudaManagedDeviceDataArray(int, char *[])
{
  try
    {
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
