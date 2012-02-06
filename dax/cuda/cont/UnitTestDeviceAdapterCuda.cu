/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cuda/cont/DeviceAdapterCuda.h>

#include <dax/cont/internal/TestingDeviceAdapter.h>
#include <dax/cuda/cont/internal/Testing.h>

int UnitTestDeviceAdapterCuda(int, char *[])
{
  int result =  dax::cont::internal::TestingDeviceAdapter
      <dax::cuda::cont::DeviceAdapterCuda>::Run();
  return dax::cuda::cont::internal::Testing::CheckCudaBeforeExit(result);
}
