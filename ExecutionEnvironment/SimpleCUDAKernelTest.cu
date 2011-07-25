/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#include <thrust/device_vector.h>
#include "DaxExecutionEnvironment.h"
#include <iostream>
using namespace std;

#define SIZE 256

__global__ void SimpleCUDAKernelTest(float3* data_array)
{
  DaxWorkMapField work;
  data_array[work.GetItem()].x = work.GetItem();
  data_array[work.GetItem()].y = 12;
  data_array[work.GetItem()].z = 13;
}

int main()
{
  thrust::host_vector<float3> host_vector(SIZE * SIZE * SIZE);
  thrust::device_vector<float3> device_vector;

  device_vector = host_vector;
  SimpleCUDAKernelTest<<<SIZE * SIZE * SIZE / 128, 128>>> ( thrust::raw_pointer_cast(
    device_vector.data()));
  host_vector = device_vector;
  for (int cc=0; cc < SIZE*SIZE*SIZE; cc++)
    {
    if (host_vector[cc].x != cc)
      {
      cout << "Mismatch at " << cc << " = " << host_vector[cc].x << endl;
      abort();
      }
    }
  return 0;
}
