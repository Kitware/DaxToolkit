/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cuda/exec/internal/CudaThreadIterator.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

namespace {

const dax::Id THREADS_PER_BLOCK = 16;
const dax::Id BLOCKS_PER_GRID = 8;
const dax::Id ARRAY_SIZE = 500;

const dax::Id THREADS_PER_GRID = THREADS_PER_BLOCK * BLOCKS_PER_GRID;

}

__global__ void CudaThreadIteratorKernel(dax::Id *array)
{
  dax::cuda::exec::internal::CudaThreadIterator iter(ARRAY_SIZE);

  dax::Id color = iter.GetIndex();

  for (; !iter.IsDone(); iter.Next())
    {
    array[iter.GetIndex()] = color;
    }
}

int UnitTestCudaThreadIterator(int, char *[])
{
  std::cout << "Running kernel." << std::endl;
  thrust::device_vector<dax::Id> deviceArray(ARRAY_SIZE);
  CudaThreadIteratorKernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(&deviceArray[0]));

  std::cout << "Checking results." << std::endl;
  thrust::host_vector<dax::Id> hostArray(ARRAY_SIZE);
  thrust::copy(deviceArray.begin(), deviceArray.end(), hostArray.begin());

  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    dax::Id value = hostArray[index];
    if (value != index%THREADS_PER_GRID)
      {
      std::cout << "Got bad value." << std::endl;
      return 1;
      }
    }

  return 0;
}
