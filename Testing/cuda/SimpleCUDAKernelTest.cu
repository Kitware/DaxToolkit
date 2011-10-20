/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/

#include <thrust/device_vector.h>
#include <dax/cuda/exec/ExecutionEnvironment.h>
#include <dax/exec/Cell.h>
#include <dax/exec/WorkMapField.h>
#include <dax/internal/GridStructures.h>
#include <iostream>
using namespace std;

#define SIZE 128

__global__ void SimpleCUDAKernelTest(float3* data_array)
{
  dax::internal::StructureUniformGrid grid;
  grid.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  grid.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);
  grid.Extent.Min = dax::make_Id3(0, 0, 0);
  grid.Extent.Max = dax::make_Id3(SIZE-1, SIZE-1, (SIZE/128)-1);
  dax::exec::WorkMapField<dax::exec::CellVoxel> work(
        grid, (blockIdx.x * blockDim.x) + threadIdx.x);
  data_array[work.GetIndex()].x = work.GetIndex();
  data_array[work.GetIndex()].y = 12;
  data_array[work.GetIndex()].z = 13;
}

int main()
{
  int deviceCount;
  cuDeviceGetCount(&deviceCount);
  cout << "deviceCount: " << deviceCount << endl;
  for (int device=0; device < deviceCount; device++)
    {
    CUdevice cuDevice;
    cuDeviceGet(&cuDevice, device);
    int major, minor;
    cuDeviceComputeCapability(&major, &minor, cuDevice);
    cout << "Device: " << device << " = " << major << "." << minor << endl;
    }

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
